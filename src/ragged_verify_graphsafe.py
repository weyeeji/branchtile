"""Explicit graph-safe and finite-shape controls for ragged causal attention.

This is execution plumbing for G0, not a new scheduler. The finite-shape path
only updates buffers when page counts and query partition stay unchanged.
"""
import time

import flashinfer
import torch

from src.ragged_shared_verify_plan_v2 import RaggedSharedVerifyPlanV2, ragged_metadata
from src.vllm_dense_cascade_plan import fused_cache_hnd_views


class RaggedGraphSafePlan(RaggedSharedVerifyPlanV2):
    def __init__(self,cache,groups,prefixes,total_queries,max_seq_lens,tables_cpu,k_scale=1.,v_scale=1.):
        self.cache=cache;self.device=cache.device;self.kv=fused_cache_hnd_views(cache)
        self.k_scale=float(k_scale);self.v_scale=float(v_scale)
        self.groups=list(groups);self.prefixes=list(prefixes);self.total_queries=total_queries
        self.workspace=torch.empty(512*1024**2,device=self.device,dtype=torch.uint8)
        self.wrappers={};self.buffers={}
        b=len(groups);g=len(prefixes)
        capacity=ragged_metadata(groups,prefixes,[1]*b,max_seq_lens,tables_cpu)
        for name in ('prefix','tail','full'):
            n=g if name=='prefix' else b
            q=torch.empty(n+1,device=self.device,dtype=torch.int32)
            ind=torch.empty(n+1,device=self.device,dtype=torch.int32)
            ids=torch.empty(len(capacity[name+'_ids']),device=self.device,dtype=torch.int32)
            last=torch.empty(n,device=self.device,dtype=torch.int32)
            self.buffers[name]=(q,ind,ids,last)
            self.wrappers[name]=flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace,kv_layout='HND',backend='fa2',use_cuda_graph=True,
                qo_indptr_buf=q,paged_kv_indptr_buf=ind,paged_kv_indices_buf=ids,paged_kv_last_page_len_buf=last)
        self.outputs={n:torch.empty((total_queries,16,128),device=self.device,dtype=torch.bfloat16) for n in self.wrappers}
        self.lse={n:torch.empty((total_queries,16),device=self.device,dtype=torch.float32) for n in self.wrappers}
        self.last_signature=None

    def prepare(self,counts,lens,tables_cpu):
        started=time.perf_counter();assert sum(counts)==self.total_queries
        metadata=ragged_metadata(self.groups,self.prefixes,counts,lens,tables_cpu)
        for name,w in self.wrappers.items():
            qkey='prefix_query_indptr' if name=='prefix' else 'query_indptr'
            q=torch.tensor(metadata[qkey],dtype=torch.int32)
            ind=torch.tensor(metadata[name+'_indptr'],dtype=torch.int32)
            last=torch.tensor(metadata[name+'_last'],dtype=torch.int32)
            ids_cpu=torch.tensor(metadata[name+'_ids'],dtype=torch.int32)
            ids=self.buffers[name][2]
            assert ids_cpu.numel()<=ids.numel()
            ids[:ids_cpu.numel()].copy_(ids_cpu)
            w.plan(q,ind,ids[:ids_cpu.numel()],last,16,2,128,16,causal=name!='prefix',
                q_data_type=torch.bfloat16,kv_data_type=self.kv[0].dtype,sm_scale=128**-.5)
        signature=tuple(tuple(w._plan_info) for w in self.wrappers.values())
        changed=self.last_signature is not None and signature!=self.last_signature
        self.last_signature=signature;self.metadata=metadata
        return dict(prepare_host_seconds=time.perf_counter()-started,signature_changed=changed,
                    signature=[list(x) for x in signature],internal_cuda_graph=True,plan_calls=3,
                    workspace_bytes=512*1024**2,query_counts=counts,seq_lens=lens)


class FiniteShapeVerifyPlan(RaggedSharedVerifyPlanV2):
    def update(self,groups,prefixes,counts,lens,tables_cpu):
        """Update physical pages and last-page lengths only under stable schedule.

        Return False before mutation if counts/page counts changed: caller must
        replan and capture/fallback, and charge that cost explicitly.
        """
        started=time.perf_counter()
        metadata=ragged_metadata(groups,prefixes,counts,lens,tables_cpu)
        keys=('query_indptr','prefix_query_indptr','prefix_indptr','tail_indptr','full_indptr')
        if any(metadata[k]!=self.metadata[k] for k in keys):
            return dict(reusable=False,reason='Query partition or page counts changed; replan required',
                        prepare_host_seconds=time.perf_counter()-started)
        for name,w in self.wrappers.items():
            # These are the buffers actually referenced by the captured kernel;
            # the CPU constructor inputs are not used for replay.
            w._paged_kv_indices_buf.copy_(torch.tensor(metadata[name+'_ids'],dtype=torch.int32))
            w._paged_kv_last_page_len_buf.copy_(torch.tensor(metadata[name+'_last'],dtype=torch.int32))
        self.metadata=metadata
        return dict(reusable=True,prepare_host_seconds=time.perf_counter()-started,plan_calls=0,
                    physical_ids_and_last_page_updated=True,internal_cuda_graph=False,
                    scope='Only fixed query partition and fixed page counts; no unlimited suffix-growth reuse')
