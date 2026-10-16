"""Existing native decode with one length plan shared across all model layers.

Frozen logical histories differ by layer/head but their lengths do not. A single
GPU kernel maps every layer's logical supports through the current block table.
The native scheduler plan is shared, while each captured layer binds its own
persistent index row and original K/V alias. This is an engineering control,
not a new sparse attention algorithm or a novel cache mechanism.
"""
import hashlib
import math
import time

import flashinfer
import torch
import triton
import triton.language as tl

from .vllm_sparse_executor import cache_views


@triton.jit
def _map_all_layers(history, question_rows, prompt_lengths, history_lengths,
                    indptr, block_tables, cache_pitches, indices,
                    HISTORY_Q:tl.constexpr, HEADS:tl.constexpr, BATCH:tl.constexpr,
                    HISTORY_CAP:tl.constexpr, INDEX_CAP:tl.constexpr,
                    TABLE_STRIDE:tl.constexpr, PAGE:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(1)
    head=row%HEADS
    request=(row//HEADS)%BATCH
    layer=row//(HEADS*BATCH)
    question=tl.load(question_rows+request)
    histlen=tl.load(history_lengths+question)
    plen=tl.load(prompt_lengths+question)
    begin=tl.load(indptr+request*HEADS+head)
    end=tl.load(indptr+request*HEADS+head+1)
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    live=offset<end-begin
    hist=tl.load(history+((layer*HISTORY_Q+question)*HEADS+head)*HISTORY_CAP+offset,
                 live & (offset<histlen),other=0)
    logical=tl.where(offset<histlen,hist,plen+offset-histlen)
    page=tl.load(block_tables+request*TABLE_STRIDE+logical//PAGE,live,other=0)
    p0=tl.load(cache_pitches+layer*3)
    p1=tl.load(cache_pitches+layer*3+1)
    p2=tl.load(cache_pitches+layer*3+2)
    physical=page*p0+head*p1+(logical%PAGE)*p2
    tl.store(indices+layer*INDEX_CAP+begin+offset,physical,live)


class LayerSharedDecodePlan:
    def __init__(self,caches,histories,prompt_lengths,history_id,batch,method='e0',
                 max_per_request=16384,fixed_split_size=None,workspace=None):
        assert method in ('e0','e0_simt')
        self.caches=list(caches)
        self.kv=[cache_views(cache) for cache in self.caches]
        self.layers=len(caches);self.heads=caches[0].shape[1];self.batch=batch
        self.device=caches[0].device;self.method=method
        self.max_per_request=max_per_request
        self.prompt_lengths_cpu=list(map(int,prompt_lengths))
        self.history_lengths_cpu=[len(histories[0][q][0]) for q in range(len(prompt_lengths))]
        self.history_cap=max(self.history_lengths_cpu)
        self.questions=len(prompt_lengths)
        self.histories=torch.zeros((self.layers,self.questions,self.heads,self.history_cap),device=self.device,dtype=torch.int32)
        self.set_histories(histories,history_id)
        self.prompt_lengths=torch.tensor(prompt_lengths,device=self.device,dtype=torch.int32)
        self.history_lengths=torch.tensor(self.history_lengths_cpu,device=self.device,dtype=torch.int32)
        self.pitches=torch.tensor([[s//256 for s in c.stride()[:3]] for c in caches],device=self.device,dtype=torch.int32)
        self.index_cap=batch*self.heads*max_per_request
        self.indices=torch.empty((self.layers,self.index_cap),device=self.device,dtype=torch.int32)
        self.indptr=torch.empty(batch*self.heads+1,device=self.device,dtype=torch.int32)
        self.last=torch.ones(batch*self.heads,device=self.device,dtype=torch.int32)
        self.question_rows=torch.empty(batch,device=self.device,dtype=torch.int32)
        self.workspace=workspace if workspace is not None else torch.zeros(256*1024*1024,device=self.device,dtype=torch.uint8)
        self.wrapper=flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace,kv_layout='NHD',
            use_cuda_graph=True,use_tensor_cores=method=='e0',backend='auto',
            paged_kv_indptr_buffer=self.indptr,paged_kv_indices_buffer=self.indices[0],paged_kv_last_page_len_buffer=self.last)
        self.outputs=torch.empty((self.layers,batch,self.heads*8,128),device=self.device,dtype=torch.bfloat16)
        self.fixed_split_size=fixed_split_size
        self.plan_info=None;self.plan_calls=0;self.epoch=0;self.prepared=False

    def set_histories(self,histories,history_id):
        if not history_id:
            raise ValueError('Explicit immutable document/mask identity is required')
        assert len(histories)==self.layers
        host=torch.zeros(self.histories.shape,dtype=torch.int32)
        for layer in range(self.layers):
            for question in range(self.questions):
                for head in range(self.heads):
                    ids=torch.as_tensor(histories[layer][question][head],dtype=torch.int32,device='cpu')
                    assert len(ids)==self.history_lengths_cpu[question] and len(ids)==len(torch.unique(ids))
                    assert len(ids)>0 and int(ids.min())>=0 and int(ids.max())<self.prompt_lengths_cpu[question]
                    host[layer,question,head,:len(ids)]=ids
        self.histories.copy_(host)
        self.history_id=history_id
        self.history_payload_sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest()
        self.prepared=False

    def prepare(self,block_tables,question_rows,query_positions,history_id):
        self.prepared=False
        if history_id!=self.history_id:
            raise ValueError('History identity changed: explicitly install the new frozen histories before reuse')
        started=time.perf_counter()
        assert block_tables.is_cuda and block_tables.shape[0]>=self.batch
        assert len(question_rows)==len(query_positions)==self.batch
        lens=[self.history_lengths_cpu[q]+int(p)-self.prompt_lengths_cpu[q]+1 for q,p in zip(question_rows,query_positions)]
        assert all(int(p)>=self.prompt_lengths_cpu[q]-1 for q,p in zip(question_rows,query_positions))
        assert 0<min(lens)<=max(lens)<=self.max_per_request
        pointers=[0]
        for length in lens:
            for _ in range(self.heads): pointers.append(pointers[-1]+length)
        pointer_cpu=torch.tensor(pointers,dtype=torch.int32)
        self.indptr.copy_(pointer_cpu)
        self.question_rows.copy_(torch.tensor(question_rows,dtype=torch.int32))
        _map_all_layers[(triton.cdiv(max(lens),256),self.layers*self.batch*self.heads)](
            self.histories,self.question_rows,self.prompt_lengths,self.history_lengths,self.indptr,
            block_tables,self.pitches,self.indices,HISTORY_Q=self.questions,HEADS=self.heads,BATCH=self.batch,
            HISTORY_CAP=self.history_cap,INDEX_CAP=self.index_cap,TABLE_STRIDE=block_tables.stride(0),PAGE=self.caches[0].shape[2],BLOCK=256)
        mapped=time.perf_counter()
        # Same scheduler metadata for every layer. Restore the first row after
        # run_layer's capture-time binding; plan never gathers K/V payload.
        self.wrapper._paged_kv_indices_buf=self.indices[0]
        self.wrapper.plan(pointer_cpu,self.indices[0,:pointers[-1]],torch.ones(self.batch*self.heads,dtype=torch.int32),
            8,1,128,1,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,fixed_split_size=self.fixed_split_size)
        topology=tuple(self.wrapper._plan_info)
        changed=self.plan_info is not None and topology!=self.plan_info
        self.plan_info=topology;self.plan_calls+=1;self.epoch+=1;self.prepared=True
        self.last_prepare={'history_id':history_id,'history_payload_sha256':self.history_payload_sha256,
            'question_rows':list(question_rows),'query_positions':list(map(int,query_positions)),
            'support_lengths':lens,'native_plan_calls':1,'layers_sharing_plan':self.layers,
            'plan_topology_changed_requires_recapture':changed,'plan_info':list(topology),
            'gpu_map_submit_host_seconds':mapped-started,'native_plan_complete_host_seconds':time.perf_counter()-mapped,
            'complete_prepare_host_seconds':time.perf_counter()-started}
        return self.last_prepare

    def run_layer(self,layer,query,output=None):
        assert self.prepared and query.shape==(self.batch,self.heads*8,128)
        self.wrapper._paged_kv_indices_buf=self.indices[layer]
        out=self.outputs[layer]
        self.wrapper.run(query.reshape(self.batch*self.heads,8,128),self.kv[layer],out=out.view(self.batch*self.heads,8,128))
        if output is not None:
            output.copy_(out)
            return output
        return out
