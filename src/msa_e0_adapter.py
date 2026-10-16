"""Exact-support E0 gather adapter for FlashInfer's native MSA SM120 kernel.

This is a mechanism adapter, not a reproduction of a complete MSA algorithm.
Every call copies the selected K/V payload. No source block is expanded.
"""
from __future__ import annotations

import time
import torch


def index_plan(owners,batch):
    if owners.ndim!=2 or owners.dtype not in (torch.int32,torch.int64):
        raise ValueError('owners must be integer [head,source_token]')
    if not 1<=batch<=4:raise ValueError('Explicit request batch must be in [1,4]')
    owners=owners.cpu().long();heads,n=owners.shape
    if int(owners.min())<0 or int(owners.max())>=1<<batch:raise ValueError('Owner bit outside batch')
    indices=[];lengths=[];pointers=[0]
    for head in range(heads):
        for request in range(batch):
            ids=torch.where((owners[head]&(1<<request))!=0)[0]
            if ids.numel()==0:raise ValueError('This adapter qualifies nonempty request/head histories only')
            indices.append(ids+head*n);lengths.append(len(ids));pointers.append(pointers[-1]+len(ids))
    blocks=[(length+127)//128 for length in lengths]
    q2k=torch.full((1,heads*batch,max(blocks)),-1,dtype=torch.int32)
    for row,count in enumerate(blocks):q2k[0,row,:count]=torch.arange(count,dtype=torch.int32)
    return {'indices':torch.cat(indices),'cu_seqlens_k':torch.tensor(pointers,dtype=torch.int32),
        'q2k_indices':q2k,'lengths':lengths,'blocks':blocks,'payload_padding_tokens':0,
        'masked_tail_lanes':sum(128*count-length for count,length in zip(blocks,lengths))}


class MSAGatherE0:
    def __init__(self,q,k,v,owners,batch,*,partial_dtype=torch.float32):
        if q.ndim!=3 or k.ndim!=3 or v.shape!=k.shape:raise ValueError('Expected q[H,B*G,D], k/v[H,N,D]')
        h,n,d=k.shape
        if d!=128 or q.shape[0]!=h or q.shape[2]!=d or q.shape[1]%batch:raise ValueError('Unsupported Q/K dimensions')
        if not all(t.is_contiguous() for t in [q,k,v]):raise ValueError('Contiguous source required')
        if not q.is_cuda or k.device!=q.device or v.device!=q.device:raise ValueError('Same CUDA device required')
        if q.dtype not in [torch.bfloat16,torch.float16] or k.dtype!=q.dtype or v.dtype!=q.dtype:raise ValueError('Equal BF16/FP16 source required')
        g=q.shape[1]//batch
        if g>16:raise ValueError('Native MSA supports GQA<=16')
        if owners.shape!=(h,n):raise ValueError('Owner shape mismatch')
        started=time.perf_counter();plan=index_plan(owners,batch)
        self.plan_cpu=plan
        self.indices=plan['indices'].to(q.device)
        self.cu=plan['cu_seqlens_k'].to(q.device)
        self.q2k=plan['q2k_indices'].to(q.device)
        self.q=q.view(h*batch,g,d)
        self.k_source,self.v_source=k.view(h*n,d),v.view(h*n,d)
        self.k=torch.empty((len(self.indices),d),dtype=k.dtype,device=k.device)
        self.v=torch.empty_like(self.k)
        self.partial_dtype=partial_dtype;self.original_shape=q.shape
        self.payload_copy_bytes_per_call=2*self.k.numel()*self.k.element_size()
        self.metadata_bytes=sum(t.numel()*t.element_size() for t in [self.indices,self.cu,self.q2k])
        torch.cuda.synchronize(q.device);self.plan_seconds=time.perf_counter()-started

    def gather(self):
        torch.index_select(self.k_source,0,self.indices,out=self.k)
        torch.index_select(self.v_source,0,self.indices,out=self.v)

    def attention(self):
        from flashinfer.msa_ops import msa_sparse_decode_attention
        output,lse=msa_sparse_decode_attention(self.q,self.k[:,None,:],self.v[:,None,:],self.q2k,
            cu_seqlens_k=self.cu,seqlen_q=1,causal=True,return_softmax_lse=True,
            partial_dtype=self.partial_dtype)
        return output.view(self.original_shape),lse.view(self.original_shape[:2])

    def run(self):
        self.gather()
        return self.attention()
