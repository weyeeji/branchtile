"""CUDA Graph executor contract for exact sparse reads of real strided cache.

Planning remains outside replay and is charged. Native planner topology must
remain unchanged; callers must recapture when this explicit invariant fails.
This operator contract alone is not a vLLM graph integration or speedup result.
"""
import flashinfer
import torch

from .vllm_sparse_executor import PagedSparseExecutor


class GraphPagedSparseExecutor(PagedSparseExecutor):
    def __init__(self, device, method, batch, heads=2, max_tokens=16384, workspace=None):
        if method not in ('e0', 'e0_simt', 'union'):
            raise ValueError(method)
        self.method, self.fixed_batch, self.fixed_heads = method, batch, heads
        self.workspace = workspace if workspace is not None else torch.zeros(256*1024*1024, device=device, dtype=torch.uint8)
        native_batch = heads if method == 'union' else batch*heads
        indptr = torch.zeros(native_batch+1, device=device, dtype=torch.int32)
        indices = torch.zeros(batch*heads*max_tokens, device=device, dtype=torch.int32)
        last = torch.ones(native_batch, device=device, dtype=torch.int32)
        if method != 'union':
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace, kv_layout='NHD', use_tensor_cores=method != 'e0_simt', backend='auto',
                use_cuda_graph=True, paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices, paged_kv_last_page_len_buffer=last)
        else:
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace, kv_layout='NHD', backend='fa2', use_cuda_graph=True,
                qo_indptr_buf=torch.arange(heads+1, device=device, dtype=torch.int32)*batch,
                paged_kv_indptr_buf=indptr, paged_kv_indices_buf=indices, paged_kv_last_page_len_buf=last,
                custom_mask_buf=torch.zeros((heads*batch*batch*max_tokens+7)//8, device=device, dtype=torch.uint8),
                mask_indptr_buf=torch.zeros(heads+1, device=device, dtype=torch.int32))
        self.planned = False
        self.fixed_plan_info = None
        self.fixed_binding = None
        self.persistent_output = self.persistent_lse = None

    def plan(self, query, cache, physical_ids):
        assert len(physical_ids) == self.fixed_batch and cache.shape[1] == self.fixed_heads
        binding = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()), tuple(query.shape), query.dtype)
        if self.fixed_binding is not None and binding != self.fixed_binding:
            raise RuntimeError('Cache/query binding changed; create a new graph executor')
        super().plan(query, cache, physical_ids)
        topology = tuple(self.wrapper._plan_info)
        if self.fixed_plan_info is not None and topology != self.fixed_plan_info:
            self.planned = False
            raise RuntimeError(f'Native plan topology changed; recapture required: {self.fixed_plan_info} -> {topology}')
        self.fixed_binding, self.fixed_plan_info = binding, topology
        if self.persistent_output is None:
            self.persistent_output, self.persistent_lse = self.out, self.lse
        self.out, self.lse = self.persistent_output, self.persistent_lse
        return self
