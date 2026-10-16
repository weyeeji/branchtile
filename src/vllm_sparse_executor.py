"""Exact native sparse attention over vLLM 0.31's real BF16 fused KV cache.

K/V are strided aliases of [physical_page,H,page_tokens,2D]. Logical supports
are mapped through the request's actual block table; no KV payload is gathered.
This is an existing E0 / masked-union mechanism, not a new attention algorithm.
"""
import math
import time

import flashinfer
import torch


def cache_views(cache, head_dim=128):
    if cache.ndim != 4 or cache.shape[-1] != 2 * head_dim:
        raise ValueError('Expected vLLM fused cache [physical_pages,H,page_tokens,2D]')
    if cache.dtype != torch.bfloat16 or not cache.is_cuda or not cache.is_contiguous():
        raise ValueError('Qualification currently requires contiguous CUDA BF16 fused storage')
    flat = cache.view(-1, 2 * head_dim)
    k = flat[:, :head_dim].view(-1, 1, 1, head_dim)
    v = flat[:, head_dim:].view(-1, 1, 1, head_dim)
    assert k.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert v.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert k.stride(0) == v.stride(0) == 2 * head_dim
    return k, v


def physical_supports(block_tables, logical_supports, heads, page_size):
    """[request][head] sorted unique logical positions -> physical-token IDs."""
    tables = torch.as_tensor(block_tables, dtype=torch.int64, device='cpu')
    result = []
    for request, by_head in enumerate(logical_supports):
        if len(by_head) != heads:
            raise ValueError('Wrong KV head count')
        mapped = []
        for head, positions in enumerate(by_head):
            positions = torch.as_tensor(positions, device='cpu', dtype=torch.int64)
            assert len(positions) == len(torch.unique(positions))
            assert not len(positions) or int(positions.min()) >= 0
            page_ids = tables[request, positions // page_size]
            assert not len(page_ids) or int(page_ids.min()) >= 0
            mapped.append((page_ids * heads * page_size + head * page_size + positions % page_size).to(torch.int32))
        result.append(mapped)
    return result


class PagedSparseExecutor:
    """Reusable native wrapper/workspace. plan() includes per-step CPU metadata.

    Query and output: [B,H*G,D], already in the real scheduled-request order.
    ``physical_ids``: [B][H] unique token IDs into cache_views(cache).
    Both paths preserve original query supports; union uses exact custom masks.
    """
    def __init__(self, device, method='e0', workspace_bytes=256 * 1024 * 1024):
        if method not in ('e0', 'union'):
            raise ValueError(method)
        self.method = method
        self.workspace = torch.empty(workspace_bytes, device=device, dtype=torch.uint8)
        if method == 'e0':
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace, kv_layout='NHD', use_tensor_cores=True, backend='auto')
        else:
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, kv_layout='NHD', backend='fa2')
        self.planned = False

    def plan(self, query, cache, physical_ids):
        started = time.perf_counter()
        b, h, d = len(physical_ids), cache.shape[1], cache.shape[-1] // 2
        if query.ndim != 3 or query.shape[0] != b or query.shape[2] != d or query.shape[1] % h:
            raise ValueError('Expected actual query [B,H*G,D]')
        g = query.shape[1] // h
        kv = cache_views(cache, d)
        self.b, self.h, self.g, self.d = b, h, g, d
        self.source_cache, self.kv = cache, kv
        pointers, indices, masks = [0], [], []
        for request in range(b):
            assert len(physical_ids[request]) == h
            for head in range(h):
                ids = torch.as_tensor(physical_ids[request][head], dtype=torch.int32, device='cpu')
                assert len(ids) == len(torch.unique(ids)) and len(ids) > 0
                assert int(ids.min()) >= 0 and int(ids.max()) < kv[0].shape[0]
        if self.method == 'e0':
            for request in range(b):
                for head in range(h):
                    ids = physical_ids[request][head]
                    indices.append(ids)
                    pointers.append(pointers[-1] + len(ids))
        else:
            for head in range(h):
                combined = torch.unique(torch.cat([physical_ids[request][head] for request in range(b)]), sorted=True)
                indices.append(combined)
                pointers.append(pointers[-1] + len(combined))
                mask = torch.zeros((b, len(combined)), dtype=torch.bool)
                for request in range(b):
                    mask[request, torch.searchsorted(combined, physical_ids[request][head])] = True
                masks.append(mask.flatten())
        self.indices = torch.cat(indices).to(query.device)
        self.indptr = torch.tensor(pointers, dtype=torch.int32, device=query.device)
        self.last = torch.ones(len(pointers) - 1, dtype=torch.int32, device=query.device)
        if self.method == 'e0':
            self.wrapper.plan(self.indptr, self.indices, self.last, g, 1, d, 1,
                              q_data_type=query.dtype, kv_data_type=cache.dtype)
        else:
            self.qindptr = torch.arange(h + 1, dtype=torch.int32, device=query.device) * b
            self.mask = torch.cat(masks).to(query.device)
            self.wrapper.plan(self.qindptr, self.indptr, self.indices, self.last, g, 1, d, 1,
                              causal=False, custom_mask=self.mask, pos_encoding_mode='NONE',
                              q_data_type=query.dtype, kv_data_type=cache.dtype)
        self.out = torch.empty((b * h, g, d), device=query.device, dtype=query.dtype)
        self.lse = torch.empty((b * h, g), device=query.device, dtype=torch.float32)
        self.planned = True
        self.prepare_wall_ms = (time.perf_counter() - started) * 1000
        return self

    def run(self, query, output=None, return_lse=False):
        assert self.planned and query.shape == (self.b, self.h * self.g, self.d)
        q = query.reshape(self.b, self.h, self.g, self.d)
        if self.method == 'union':
            q = q.permute(1, 0, 2, 3).contiguous()
        q = q.reshape(self.b * self.h, self.g, self.d)
        self.wrapper.run(q, self.kv, out=self.out, lse=self.lse if return_lse else None, return_lse=return_lse)
        if self.method == 'union':
            actual = self.out.view(self.h, self.b, self.g, self.d).permute(1, 0, 2, 3).reshape(query.shape)
            lse = self.lse.view(self.h, self.b, self.g).permute(1, 0, 2).reshape(self.b, self.h * self.g)
        else:
            actual = self.out.view(query.shape)
            lse = self.lse.view(self.b, self.h * self.g)
        if output is not None:
            output.copy_(actual)
            actual = output
        return (actual, lse * math.log(2)) if return_lse else actual
