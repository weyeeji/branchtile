"""Existing exact union with persistent GPU metadata and a shared native plan.

Frozen union histories are padded to one length across heads/layers. Per-query
custom masks make padding and inactive histories invisible. Original cache
payload is read through strided aliases. Shared complete pages must retain the
same physical identity across the current ready requests; invalidation fails
before graph replay. This is an engineering baseline, not a new algorithm.
"""
import hashlib
import time

import flashinfer
import torch
import triton
import triton.language as tl

from .vllm_sparse_executor import cache_views


@triton.jit
def _entries(logicals, owners, question_rows, prompt_lengths, tails, tables,
             pitches, indices, live_owners, HISTORY:tl.constexpr, HEADS:tl.constexpr,
             BATCH:tl.constexpr, TABLE_STRIDE:tl.constexpr, INDEX_CAP:tl.constexpr,
             LIVE, PAGE:tl.constexpr, BLOCK:tl.constexpr):
    group=tl.program_id(1);layer=group//HEADS;head=group%HEADS
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=x<LIVE
    pos=tl.load(logicals+group*HISTORY+x,valid & (x<HISTORY),other=0)
    bits=tl.load(owners+group*HISTORY+x,valid & (x<HISTORY),other=0)
    chosen=tl.full((BLOCK,),-1,tl.int32)
    for r in tl.static_range(BATCH):
        q=tl.load(question_rows+r)
        active=(bits & (1<<q))!=0
        chosen=tl.where((chosen<0)&active,r,chosen)
        start=tl.load(tails+r);end=tl.load(tails+r+1)
        suffix=(x>=HISTORY+start)&(x<HISTORY+end)
        pos=tl.where(suffix,tl.load(prompt_lengths+q)+x-HISTORY-start,pos)
        bits=tl.where(suffix,1<<q,bits)
        chosen=tl.where(suffix,r,chosen)
    active=chosen>=0
    pos=tl.where(active,pos,0);chosen=tl.maximum(chosen,0)
    page=tl.load(tables+chosen*TABLE_STRIDE+pos//PAGE,valid,other=0)
    p0=tl.load(pitches+layer*3);p1=tl.load(pitches+layer*3+1);p2=tl.load(pitches+layer*3+2)
    physical=page*p0+head*p1+(pos%PAGE)*p2
    tl.store(indices+layer*INDEX_CAP+head*LIVE+x,physical,valid)
    tl.store(live_owners+layer*INDEX_CAP+head*LIVE+x,tl.where(active,bits,0),valid)


@triton.jit
def _pack_masks(live_owners,question_rows,masks,INDEX_CAP:tl.constexpr,MASK_CAP:tl.constexpr,
                HEADS:tl.constexpr,BATCH:tl.constexpr,LIVE,BYTES,BLOCK:tl.constexpr):
    group=tl.program_id(1);layer=group//HEADS;head=group%HEADS
    byte=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    packed=tl.full((BLOCK,),0,tl.uint32)
    for bit in tl.static_range(8):
        x=byte*8+bit;request=x//LIVE;token=x%LIVE
        q=tl.load(question_rows+request,x<BATCH*LIVE,other=0)
        owner=tl.load(live_owners+layer*INDEX_CAP+head*LIVE+token,byte<BYTES,other=0)
        on=(x<BATCH*LIVE)&((owner&(1<<q))!=0)
        packed=packed|(on.to(tl.uint32)<<bit)
    tl.store(masks+layer*MASK_CAP+head*BYTES+byte,packed.to(tl.uint8),byte<BYTES)


class LayerSharedUnionPlan:
    def __init__(self,caches,histories,prompt_lengths,history_id,batch,common_end,
                 max_generated=8192,workspace=None):
        self.caches=list(caches);self.layers=len(caches);self.heads=caches[0].shape[1]
        self.batch=batch;self.device=caches[0].device;self.kv=[cache_views(c) for c in caches]
        self.questions=len(prompt_lengths);self.prompt_cpu=list(map(int,prompt_lengths))
        self.common_end=common_end;assert common_end%16==0
        self.max_generated=max_generated;self.history_id=history_id;assert history_id
        entries=[]
        for layer in range(self.layers):
            per_head=[]
            for head in range(self.heads):
                union={}
                for q in range(self.questions):
                    ids=list(map(int,histories[layer][q][head]));assert len(ids)==len(set(ids))
                    for pos in ids:
                        assert 0<=pos<self.prompt_cpu[q]
                        key=(-1 if pos<common_end else q,pos)
                        union[key]=union.get(key,0)|(1<<q)
                per_head.append([(key[1],bits) for key,bits in sorted(union.items())])
            entries.append(per_head)
        self.history=max(len(e) for layer in entries for e in layer)
        logical=torch.zeros((self.layers,self.heads,self.history),dtype=torch.int32)
        owner=torch.zeros_like(logical)
        for l in range(self.layers):
            for h in range(self.heads):
                n=len(entries[l][h]);logical[l,h,:n]=torch.tensor([a for a,b in entries[l][h]])
                owner[l,h,:n]=torch.tensor([b for a,b in entries[l][h]])
        self.history_payload_sha256=hashlib.sha256(logical.numpy().tobytes()+owner.numpy().tobytes()).hexdigest()
        self.logicals=logical.to(self.device);self.owners=owner.to(self.device)
        self.prompt_lengths=torch.tensor(prompt_lengths,device=self.device,dtype=torch.int32)
        self.pitches=torch.tensor([[s//256 for s in c.stride()[:3]] for c in caches],device=self.device,dtype=torch.int32)
        self.question_rows=torch.empty(batch,device=self.device,dtype=torch.int32)
        self.tails=torch.empty(batch+1,device=self.device,dtype=torch.int32)
        self.max_live=self.history+batch*max_generated;self.index_cap=self.heads*self.max_live
        self.mask_cap=self.heads*triton.cdiv(batch*self.max_live,8)
        self.indices=torch.empty((self.layers,self.index_cap),device=self.device,dtype=torch.int32)
        self.live_owners=torch.empty_like(self.indices)
        self.masks=torch.empty((self.layers,self.mask_cap),device=self.device,dtype=torch.uint8)
        self.indptr=torch.empty(self.heads+1,device=self.device,dtype=torch.int32)
        self.qindptr=torch.arange(self.heads+1,device=self.device,dtype=torch.int32)*batch
        self.last=torch.ones(self.heads,device=self.device,dtype=torch.int32)
        self.mask_indptr=torch.empty(self.heads+1,device=self.device,dtype=torch.int32)
        self.workspace=workspace if workspace is not None else torch.zeros(256*1024*1024,device=self.device,dtype=torch.uint8)
        self.wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace,kv_layout='NHD',backend='fa2',use_cuda_graph=True,
            qo_indptr_buf=self.qindptr,paged_kv_indptr_buf=self.indptr,paged_kv_indices_buf=self.indices[0],
            paged_kv_last_page_len_buf=self.last,custom_mask_buf=self.masks[0],mask_indptr_buf=self.mask_indptr)
        self.outputs=torch.empty((self.layers,batch,self.heads*8,128),device=self.device,dtype=torch.bfloat16)
        self.native_outputs=torch.empty((self.layers,self.heads*batch,8,128),device=self.device,dtype=torch.bfloat16)
        self.prepared=False;self.plan_info=None;self.plan_calls=0

    def prepare(self,tables,question_rows,query_positions,history_id):
        self.prepared=False;started=time.perf_counter()
        if history_id!=self.history_id:raise ValueError('Frozen union prefix/mask identity changed: construct a new plan')
        assert len(question_rows)==len(query_positions)==self.batch and len(set(question_rows))==self.batch
        # One current table identity check covers all layers and all logical
        # common-source tokens. It also detects eviction/remapping divergence.
        if self.batch>1 and not bool((tables[:self.batch,:self.common_end//16]==tables[0:1,:self.common_end//16]).all()):
            raise ValueError('Shared complete-page physical identity changed; union source must be rebuilt')
        tail_lengths=[int(p)-self.prompt_cpu[q]+1 for q,p in zip(question_rows,query_positions)]
        assert min(tail_lengths)>=0 and max(tail_lengths)<=self.max_generated
        tail=[0]
        for n in tail_lengths:tail.append(tail[-1]+n)
        live=self.history+tail[-1];mask_bytes=triton.cdiv(self.batch*live,8)
        self.question_rows.copy_(torch.tensor(question_rows,dtype=torch.int32));self.tails.copy_(torch.tensor(tail,dtype=torch.int32))
        _entries[(triton.cdiv(live,256),self.layers*self.heads)](self.logicals,self.owners,self.question_rows,self.prompt_lengths,
            self.tails,tables,self.pitches,self.indices,self.live_owners,HISTORY=self.history,HEADS=self.heads,BATCH=self.batch,
            TABLE_STRIDE=tables.stride(0),INDEX_CAP=self.index_cap,LIVE=live,PAGE=16,BLOCK=256)
        _pack_masks[(triton.cdiv(mask_bytes,256),self.layers*self.heads)](self.live_owners,self.question_rows,self.masks,
            INDEX_CAP=self.index_cap,MASK_CAP=self.mask_cap,HEADS=self.heads,BATCH=self.batch,LIVE=live,BYTES=mask_bytes,BLOCK=256)
        mapped=time.perf_counter()
        self.wrapper._paged_kv_indices_buf=self.indices[0];self.wrapper._custom_mask_buf=self.masks[0]
        self.wrapper.plan(torch.arange(self.heads+1,dtype=torch.int32)*self.batch,
            torch.arange(self.heads+1,dtype=torch.int32)*live,self.indices[0,:self.heads*live],torch.ones(self.heads,dtype=torch.int32),
            8,1,128,1,packed_custom_mask=self.masks[0,:self.heads*mask_bytes],causal=False,
            q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16)
        # Native custom-mask offsets are bytes, each head separately padded.
        # The public packed-mask path computes bit offsets in this installed
        # version; supply explicit byte offsets to the documented graph buffer.
        self.mask_indptr.copy_(torch.arange(self.heads+1,dtype=torch.int32)*mask_bytes)
        topology=tuple(self.wrapper._plan_info);changed=self.plan_info is not None and topology!=self.plan_info
        self.plan_info=topology;self.plan_calls+=1;self.prepared=True
        self.last_prepare={'history_id':history_id,'history_payload_sha256':self.history_payload_sha256,
            'question_rows':list(question_rows),'query_positions':list(map(int,query_positions)),
            'union_history_padded_tokens':self.history,'total_union_tokens_per_head':live,'generated_lengths':tail_lengths,
            'native_plan_calls':1,'layers_sharing_plan':self.layers,'plan_topology_changed_requires_recapture':changed,
            'plan_info':list(topology),'gpu_map_submit_host_seconds':mapped-started,
            'native_plan_complete_host_seconds':time.perf_counter()-mapped,'complete_prepare_host_seconds':time.perf_counter()-started}
        return self.last_prepare

    def run_layer(self,layer,query,output=None):
        assert self.prepared and query.shape==(self.batch,self.heads*8,128)
        self.wrapper._paged_kv_indices_buf=self.indices[layer];self.wrapper._custom_mask_buf=self.masks[layer]
        q=query.view(self.batch,self.heads,8,128).permute(1,0,2,3).contiguous().view(self.heads*self.batch,8,128)
        self.wrapper.run(q,self.kv[layer],out=self.native_outputs[layer])
        actual=self.native_outputs[layer].view(self.heads,self.batch,8,128).permute(1,0,2,3).reshape(query.shape)
        self.outputs[layer].copy_(actual)
        if output is not None:output.copy_(self.outputs[layer]);return output
        return self.outputs[layer]
