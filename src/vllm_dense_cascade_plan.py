"""Existing dense shared-prefix cascade over original page16 vLLM KV aliases.

One native prefix prefill plan and one private-tail plan are shared across all
layers. This is a mature mechanism control, not a new attention algorithm.
"""
import time

import flashinfer
import torch


def fused_cache_hnd_views(cache):
    """[physical_page,H,page16,2D] -> strided HND K/V aliases, including FP8 bytes."""
    if cache.ndim != 4 or cache.shape[1:] != (2, 16, 256) or cache.stride(-1) != 1:
        raise ValueError(f'Expected fused [pages,2,16,256] storage, got {cache.shape}, {cache.stride()}')
    if cache.dtype == torch.uint8:
        typed = cache.view(torch.float8_e4m3fn)
    elif cache.dtype in (torch.bfloat16, torch.float8_e4m3fn):
        typed = cache
    else:
        raise ValueError(f'Unsupported cache dtype {cache.dtype}')
    k, v = typed[..., :128], typed[..., 128:]
    assert k.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert v.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    return k, v


class LayerSharedDenseCascadePlan:
    def __init__(self, caches, batch, common_prefix_tokens, max_tail_tokens=16384,
                 workspace=None, k_scales=None, v_scales=None):
        self.caches = list(caches)
        assert self.caches and 1 <= batch <= 8
        assert common_prefix_tokens > 0 and common_prefix_tokens % 16 == 0
        self.batch = int(batch)
        self.common_prefix_tokens = int(common_prefix_tokens)
        self.prefix_pages = self.common_prefix_tokens // 16
        self.max_tail_tokens = int(max_tail_tokens)
        self.max_tail_pages = (self.max_tail_tokens + 15) // 16
        self.device = self.caches[0].device
        self.kv = [fused_cache_hnd_views(c) for c in self.caches]
        self.kv_dtype = self.kv[0][0].dtype
        assert all(k.dtype == self.kv_dtype for k, _ in self.kv)
        self.layers = len(self.caches)
        self.k_scales = [1.0] * self.layers if k_scales is None else list(map(float, k_scales))
        self.v_scales = [1.0] * self.layers if v_scales is None else list(map(float, v_scales))
        assert len(self.k_scales) == len(self.v_scales) == self.layers
        assert all(s > 0 for s in self.k_scales + self.v_scales)
        self.workspace = workspace if workspace is not None else torch.empty(256*1024*1024, device=self.device, dtype=torch.uint8)
        self.prefix_qindptr = torch.tensor([0, batch], device=self.device, dtype=torch.int32)
        self.prefix_indptr = torch.tensor([0, self.prefix_pages], device=self.device, dtype=torch.int32)
        self.prefix_indices = torch.zeros(self.prefix_pages, device=self.device, dtype=torch.int32)
        self.prefix_last = torch.tensor([16], device=self.device, dtype=torch.int32)
        self.tail_qindptr = torch.arange(batch+1, device=self.device, dtype=torch.int32)
        self.tail_indptr = torch.arange(batch+1, device=self.device, dtype=torch.int32)
        self.tail_indices = torch.zeros(batch*self.max_tail_pages, device=self.device, dtype=torch.int32)
        self.tail_last = torch.ones(batch, device=self.device, dtype=torch.int32)
        self.prefix_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout='HND', backend='fa2', use_cuda_graph=True,
            qo_indptr_buf=self.prefix_qindptr, paged_kv_indptr_buf=self.prefix_indptr,
            paged_kv_indices_buf=self.prefix_indices, paged_kv_last_page_len_buf=self.prefix_last)
        self.tail_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout='HND', backend='fa2', use_cuda_graph=True,
            qo_indptr_buf=self.tail_qindptr, paged_kv_indptr_buf=self.tail_indptr,
            paged_kv_indices_buf=self.tail_indices, paged_kv_last_page_len_buf=self.tail_last)
        self.prefix_wrapper.plan(torch.tensor([0, batch], dtype=torch.int32),
            torch.tensor([0, self.prefix_pages], dtype=torch.int32), self.prefix_indices,
            torch.tensor([16], dtype=torch.int32), 16, 2, 128, 16,
            causal=False, q_data_type=torch.bfloat16, kv_data_type=self.kv_dtype,
            sm_scale=128**-.5)
        self.outputs = torch.empty((self.layers, batch, 16, 128), device=self.device, dtype=torch.bfloat16)
        self.tail_outputs = torch.empty_like(self.outputs)
        self.prefix_lse = torch.empty((self.layers, batch, 16), device=self.device, dtype=torch.float32)
        self.tail_lse = torch.empty_like(self.prefix_lse)
        self.tail_plan_info = None
        self.prepared = False
        self.prepare_count = 0
        dummy = torch.zeros((batch, self.prefix_pages+1), device=self.device, dtype=torch.int32)
        self.prepare(dummy, [self.common_prefix_tokens+1]*batch, require_stable_topology=False)

    def prepare(self, block_tables, seq_lens, require_stable_topology=True):
        self.prepared = False
        started = time.perf_counter()
        assert block_tables.is_cuda and block_tables.shape[0] >= self.batch
        lens = list(map(int, seq_lens))
        assert len(lens) == self.batch
        tails = [length-self.common_prefix_tokens for length in lens]
        if not all(0 < tail <= self.max_tail_tokens for tail in tails):
            raise ValueError(f'Private tails outside (0,{self.max_tail_tokens}]: {tails}')
        if block_tables.shape[1] < max((length+15)//16 for length in lens):
            raise ValueError('Block table too short for complete sequences')
        # Check actual physical identity, not merely logical token equality. The
        # host synchronization is deliberately charged in preparation receipts.
        shared = block_tables[:self.batch, :self.prefix_pages]
        if not bool((shared == shared[:1]).all().item()):
            raise ValueError('Shared-prefix logical pages no longer have identical physical KV addresses')
        self.prefix_indices.copy_(shared[0])
        page_counts = [(tail+15)//16 for tail in tails]
        pointers = [0]
        for count in page_counts:
            pointers.append(pointers[-1]+count)
        for row, count in enumerate(page_counts):
            self.tail_indices[pointers[row]:pointers[row+1]].copy_(block_tables[row, self.prefix_pages:self.prefix_pages+count])
        pointer_cpu = torch.tensor(pointers, dtype=torch.int32)
        last_cpu = torch.tensor([(tail-1)%16+1 for tail in tails], dtype=torch.int32)
        self.tail_wrapper.plan(torch.arange(self.batch+1, dtype=torch.int32), pointer_cpu,
            self.tail_indices[:pointers[-1]], last_cpu, 16, 2, 128, 16,
            causal=False, q_data_type=torch.bfloat16, kv_data_type=self.kv_dtype,
            sm_scale=128**-.5, disable_split_kv=True)
        topology = tuple(self.tail_wrapper._plan_info)
        changed = self.tail_plan_info is not None and topology != self.tail_plan_info
        if require_stable_topology and changed:
            raise RuntimeError('Tail native plan topology changed; existing graph replay is not valid')
        self.tail_plan_info = topology
        self.prepare_count += 1
        self.prepared = True
        return {'batch': self.batch, 'seq_lens': lens, 'common_prefix_tokens': self.common_prefix_tokens,
                'private_tail_tokens': tails, 'private_tail_pages': page_counts,
                'actual_shared_physical_prefix_checked': True, 'tail_plan_topology_changed': changed,
                'tail_plan_info': list(topology), 'prepare_count': self.prepare_count,
                'prefix_plan_calls_this_prepare': 0, 'tail_plan_calls_this_prepare': 1,
                'layers_sharing_plan': self.layers, 'complete_prepare_host_seconds': time.perf_counter()-started,
                'source_payload_copy': False}

    def run_layer(self, layer, query, output=None):
        assert self.prepared and query.shape == (self.batch, 16, 128)
        out = self.outputs[layer]
        tail = self.tail_outputs[layer]
        self.prefix_wrapper.run(query, self.kv[layer], out=out, lse=self.prefix_lse[layer],
                                return_lse=True, k_scale=self.k_scales[layer], v_scale=self.v_scales[layer])
        self.tail_wrapper.run(query, self.kv[layer], out=tail, lse=self.tail_lse[layer],
                              return_lse=True, k_scale=self.k_scales[layer], v_scale=self.v_scales[layer])
        flashinfer.merge_state_in_place(out, self.prefix_lse[layer], tail, self.tail_lse[layer])
        if output is not None:
            output.copy_(out)
            return output
        return out
