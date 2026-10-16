"""Exact paged owner attention for real vLLM BF16 caches.

Frozen prompt supports are grouped once. Generated tokens remain private and
are mapped every step. A graph has fixed buffer capacities, but task end points,
request identities and physical indices are live metadata. No KV payload copy.
This is a new integration control, not qualification of the older contiguous ABI.
"""
import hashlib
import itertools
import time

import torch
import triton
import triton.language as tl

from .vllm_sparse_executor import cache_views
from .branchtile.triton_kernel_paged_owner import producer, merge


@triton.jit
def _map(Logical, Owners, Questions, Prompts, Lengths, Tables, Pitches, Indices,
         HISTORY:tl.constexpr,CAP:tl.constexpr,HEADS:tl.constexpr,BATCH:tl.constexpr,
         TAIL_CAP:tl.constexpr,TABLE_STRIDE:tl.constexpr,PAGE:tl.constexpr,BLOCK:tl.constexpr):
    group=tl.program_id(1);layer=group//HEADS;head=group%HEADS
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    hist=x<HISTORY
    pos=tl.load(Logical+group*HISTORY+x,hist,other=0)
    bits=tl.load(Owners+group*HISTORY+x,hist,other=0)
    chosen=tl.full((BLOCK,),-1,tl.int32)
    for r in tl.static_range(BATCH):
        question=tl.load(Questions+r)
        chosen=tl.where((chosen<0)&hist&((bits&(1<<question))!=0),r,chosen)
        local=x-HISTORY-r*TAIL_CAP
        suffix=(local>=0)&(local<tl.load(Lengths+r))
        pos=tl.where(suffix,tl.load(Prompts+question)+local,pos)
        chosen=tl.where(suffix,r,chosen)
    live=(x<CAP)&(chosen>=0)
    page=tl.load(Tables+tl.maximum(chosen,0)*TABLE_STRIDE+pos//PAGE,live,other=0)
    physical=page*tl.load(Pitches+layer*3)+head*tl.load(Pitches+layer*3+1)+(pos%PAGE)*tl.load(Pitches+layer*3+2)
    tl.store(Indices+group*CAP+x,physical,live)


@triton.jit
def _tail_ends(Tasks, TailTasks, Lengths, TASKS:tl.constexpr,HEADS:tl.constexpr,BATCH:tl.constexpr):
    layer=tl.program_id(0)
    for h in tl.static_range(HEADS):
        for r in tl.static_range(BATCH):
            task=tl.load(TailTasks+(layer*HEADS+h)*BATCH+r)
            start=tl.load(Tasks+(layer*TASKS+task)*6+2)
            tl.store(Tasks+(layer*TASKS+task)*6+3,start+tl.load(Lengths+r))


class LayerSharedPagedOwnerPlan:
    def __init__(self,caches,histories,prompt_lengths,history_id,batch,common_end,
                 max_generated=8192,chunk_tokens=128,block_n=64,workspace=None):
        started=time.perf_counter()
        self.caches=list(caches);self.kv=[cache_views(c) for c in caches]
        self.layers=len(caches);self.heads=caches[0].shape[1];self.batch=batch
        self.device=caches[0].device;self.prompt_cpu=list(map(int,prompt_lengths))
        self.questions=len(prompt_lengths);self.history_id=history_id
        self.common_end=common_end;self.max_generated=max_generated;self.block_n=block_n
        assert self.questions<=4 and common_end%caches[0].shape[2]==0
        assert block_n in (32,64) and history_id and chunk_tokens>0
        self.labels=list(range(1,1<<self.questions));self.regions=len(self.labels)+batch
        entries=[];bounds=[]
        for layer in range(self.layers):
            he=[];hb=[]
            for head in range(self.heads):
                union={}
                for q in range(self.questions):
                    ids=list(map(int,histories[layer][q][head]));assert len(ids)==len(set(ids))
                    for pos in ids:
                        assert 0<=pos<self.prompt_cpu[q]
                        key=(-1 if pos<common_end else q,pos)
                        union[key]=union.get(key,0)|(1<<q)
                row=[];ranges=[]
                for bits in self.labels:
                    begin=len(row)
                    row.extend((key[1],bits) for key,value in sorted(union.items()) if value==bits)
                    ranges.append((begin,len(row)))
                he.append(row);hb.append(ranges)
            entries.append(he);bounds.append(hb)
        self.history=max(len(e) for layer in entries for e in layer)
        self.cap=self.history+batch*max_generated
        logical=torch.zeros((self.layers,self.heads,self.history),dtype=torch.int32)
        owner=torch.zeros_like(logical)
        task_rows=[];tail_rows=[]
        for l in range(self.layers):
            tasks=[];tails=[]
            for h in range(self.heads):
                n=len(entries[l][h])
                logical[l,h,:n]=torch.tensor([a for a,b in entries[l][h]])
                owner[l,h,:n]=torch.tensor([b for a,b in entries[l][h]])
                for region,(begin,end) in enumerate(bounds[l][h]):
                    for start in range(begin,end,chunk_tokens):
                        tasks.append([h,region,start,min(start+chunk_tokens,end),0,len(tasks)*32])
                ht=[]
                for r in range(batch):
                    ht.append(len(tasks));start=self.history+r*max_generated
                    tasks.append([h,len(self.labels)+r,start,start,8,len(tasks)*32])
                tails.append(ht)
            task_rows.append(tasks);tail_rows.append(tails)
        self.task_count=max(map(len,task_rows))
        base=torch.zeros((self.layers,self.task_count,6),dtype=torch.int32)
        for l,tasks in enumerate(task_rows):base[l,:len(tasks)]=torch.tensor(tasks)
        self.history_payload_sha256=hashlib.sha256(logical.numpy().tobytes()+owner.numpy().tobytes()).hexdigest()
        self.logicals=logical.to(self.device);self.owners=owner.to(self.device)
        self.prompt_lengths=torch.tensor(prompt_lengths,device=self.device,dtype=torch.int32)
        self.pitches=torch.tensor([[s//256 for s in c.stride()[:3]] for c in caches],device=self.device,dtype=torch.int32)
        self.question_rows=torch.empty(batch,device=self.device,dtype=torch.int32)
        self.tail_lengths=torch.empty(batch,device=self.device,dtype=torch.int32)
        self.tail_tasks=torch.tensor(tail_rows,device=self.device,dtype=torch.int32)
        self.indices=torch.empty((self.layers,self.heads,self.cap),device=self.device,dtype=torch.int32)
        self.tasks=base.to(self.device)
        self.row_map=torch.empty((self.layers,self.heads,self.regions,32),device=self.device,dtype=torch.int32)
        self.max_contrib=max(sum(1 for t in tasks if t[0]==h) for tasks in task_rows for h in range(self.heads))
        self.merge_indices=torch.empty((self.layers,self.heads,batch,self.max_contrib),device=self.device,dtype=torch.int32)
        self.counts=torch.empty((self.layers,self.heads,batch),device=self.device,dtype=torch.int32)
        self.subsets={}
        # All request permutations are explicit; a completed request can leave
        # from any position, and scheduler compaction may reorder survivors.
        for subset in itertools.permutations(range(self.questions),batch):
            ts=base.clone();rm=torch.zeros(self.row_map.shape,dtype=torch.int32)
            mi=torch.full(self.merge_indices.shape,-1,dtype=torch.int32);co=torch.zeros(self.counts.shape,dtype=torch.int32)
            for l,tasks in enumerate(task_rows):
                for h in range(self.heads):
                    for region,bits in enumerate(self.labels):
                        active=[r for r,q in enumerate(subset) if bits&(1<<q)]
                        rows=[r*8+g for r in active for g in range(8)]
                        rm[l,h,region,:len(rows)]=torch.tensor(rows,dtype=torch.int32)
                    for r in range(batch):rm[l,h,len(self.labels)+r,:8]=torch.arange(r*8,(r+1)*8)
                for t,desc in enumerate(tasks):
                    h,region,begin,end,_,packed=desc
                    active=([r for r,q in enumerate(subset) if self.labels[region]&(1<<q)]
                            if region<len(self.labels) else [region-len(self.labels)])
                    ts[l,t,4]=len(active)*8
                    for local,r in enumerate(active):
                        idx=int(co[l,h,r]);mi[l,h,r,idx]=packed+local*8;co[l,h,r]+=1
            self.subsets[subset]=tuple(x.to(self.device) for x in (ts,rm,mi,co))
        self.u=torch.empty((self.layers,self.task_count*32,128),device=self.device,dtype=torch.float32)
        self.m=torch.empty((self.layers,self.task_count*32),device=self.device,dtype=torch.float32);self.l=torch.empty_like(self.m)
        self.outputs=torch.empty((self.layers,batch,self.heads*8,128),device=self.device,dtype=torch.bfloat16)
        self.lse=torch.empty((self.layers,self.heads,batch*8),device=self.device,dtype=torch.float32)
        self.subset=None;self.prepared=False;self.prepare_seconds=time.perf_counter()-started

    def prepare(self,tables,question_rows,query_positions,history_id):
        started=time.perf_counter();self.prepared=False
        if history_id!=self.history_id:raise ValueError('History changed: rebuild plan and recapture graph')
        subset=tuple(question_rows)
        assert subset in self.subsets and len(query_positions)==self.batch
        if self.batch>1 and not bool((tables[:self.batch,:self.common_end//16]==tables[0:1,:self.common_end//16]).all()):
            raise ValueError('Shared complete-page physical identity diverged: no replay permitted')
        lens=[int(p)-self.prompt_cpu[q]+1 for q,p in zip(subset,query_positions)]
        assert min(lens)>=0 and max(lens)<=self.max_generated
        changed=subset!=self.subset
        if changed:
            for target,source in zip((self.tasks,self.row_map,self.merge_indices,self.counts),self.subsets[subset]):target.copy_(source)
            self.question_rows.copy_(torch.tensor(question_rows,dtype=torch.int32));self.subset=subset
        self.tail_lengths.copy_(torch.tensor(lens,dtype=torch.int32))
        # Fixed suffix slots are skipped when dead; no stale values are read.
        _map[(triton.cdiv(self.cap,256),self.layers*self.heads)](self.logicals,self.owners,self.question_rows,
            self.prompt_lengths,self.tail_lengths,tables,self.pitches,self.indices,HISTORY=self.history,CAP=self.cap,
            HEADS=self.heads,BATCH=self.batch,TAIL_CAP=self.max_generated,TABLE_STRIDE=tables.stride(0),PAGE=16,BLOCK=256)
        _tail_ends[(self.layers,)](self.tasks,self.tail_tasks,self.tail_lengths,TASKS=self.task_count,HEADS=self.heads,BATCH=self.batch)
        self.prepared=True
        return {'history_id':history_id,'history_payload_sha256':self.history_payload_sha256,
                'question_rows':list(subset),'query_positions':list(map(int,query_positions)),
                'generated_lengths':lens,'subset_changed':changed,'plan_topology_changed_requires_recapture':False,
                'complete_prepare_host_seconds':time.perf_counter()-started,'native_plan_calls':0,
                'new_paged_owner_abi':True,'payload_copy_bytes':0,'fixed_task_capacity_per_layer':self.task_count,
                'suffix_strategy':'one dynamic-length private task per active request/head; all generated tokens included',
                'graph_topology_contract':'Fixed buffers and task capacity; values updated every step. New frozen supports, larger capacities, or divergent shared pages require rebuild/recapture.'}

    def run_layer(self,layer,query,output=None):
        assert self.prepared and query.shape==(self.batch,self.heads*8,128) and query.stride(-1)==1
        k,v=self.kv[layer]
        producer[(self.task_count,)](query,k,v,self.indices[layer],self.row_map[layer],self.tasks[layer],
            self.u[layer],self.m[layer],self.l[layer],ROWS=self.batch*8,TOKENS=self.cap,REGIONS=self.regions,
            HEADS=self.heads,QS0=query.stride(0),QS1=query.stride(1),N=self.block_n,D=128,num_warps=4,num_stages=1)
        out=self.outputs[layer] if output is None else output
        assert out.is_contiguous()
        merge[(self.heads,self.batch*8)](self.u[layer],self.m[layer],self.l[layer],self.merge_indices[layer],self.counts[layer],
            out,self.lse[layer],BATCH=self.batch,GQA=8,ROWS=self.batch*8,MAX_CONTRIB=self.max_contrib,
            D=128,HEADS=self.heads,GROUP=16,num_warps=4)
        if output is not None:self.outputs[layer].copy_(output)
        return out
