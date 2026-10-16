"""Exploratory B<=8 native union wrapper; widened admission bound only.

Each original KV head is a native prefill batch item, with B query positions,
G query heads, and one KV head. A per-query custom mask preserves the original
request supports, including common prefix and complete private suffix tokens.
The source K/V payload is never gathered or copied. Planning/packing is timed
separately from the preallocated run() call.
"""
import math
import time

import flashinfer
import torch


class NativeUnionPrefill:
    """q [H,B*G,D], k/v [H,N,D], owners [H,N] integer request bitmasks.

    ``batch`` is explicit because an absent highest request bit does not imply
    that request has left. Per-head zero-length unions are rejected: the native
    paged custom-mask indptr computation in 0.7.2 does not support that case.
    Individual empty request masks within a nonempty union remain representable.

    BF16 FA2 also rounds probabilities before accumulating its denominator;
    several small-support fixtures exceed this project's frozen LSE tolerance,
    identically to direct native single-request FA2. Qualify every candidate
    shape/configuration against FP64 before timing; do not assume this wrapper
    universally meets that stricter LSE contract.
    """

    lse_log_base = 2

    def __init__(self, q, k, v, owners, batch, *, fixed_split_size=None,
                 disable_split_kv=False, normalize_lse=False):
        if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
            raise ValueError("Expected q[H,B*G,D] and equal k/v[H,N,D]")
        h, n, d = k.shape
        if not isinstance(batch, int) or not 1 <= batch <= 8:
            raise ValueError("batch must be the explicit active request count in [1,8]")
        if q.shape[0] != h or q.shape[2] != d or q.shape[1] % batch:
            raise ValueError("Query shape is incompatible with the requested batch")
        if h == 0 or n == 0 or q.shape[1] == 0:
            raise ValueError("Source and query dimensions must be nonempty")
        if not all(t.is_contiguous() for t in (q, k, v)):
            raise ValueError("Contiguous inputs are required for zero-copy payload views")
        if not q.is_cuda or k.device != q.device or v.device != q.device:
            raise ValueError("Q/K/V must occupy the same CUDA device")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q/K/V must have the same dtype")
        if owners.shape != (h, n) or owners.dtype not in (torch.int32, torch.int64):
            raise ValueError("owners must be integer request bitmasks shaped [H,N]")
        if fixed_split_size is not None and fixed_split_size <= 0:
            raise ValueError("fixed_split_size must be positive when supplied")
        start = time.perf_counter()
        owner_cpu = owners.to(device="cpu", dtype=torch.int64)
        if int(owner_cpu.min()) < 0 or int(owner_cpu.max()) >= (1 << batch):
            raise ValueError("Owner bit outside the explicit request batch")
        g = q.shape[1] // batch
        query_bits = (1 << torch.arange(batch, dtype=torch.int64))[:, None]
        pointers, indices, masks = [0], [], []
        for head in range(h):
            ids = torch.where(owner_cpu[head] != 0)[0]
            if ids.numel() == 0:
                raise ValueError("Native paged custom-mask prefill requires a nonempty union for every head")
            pointers.append(pointers[-1] + len(ids))
            indices.append((ids + head * n).to(torch.int32))
            # Row-major [request, union position], concatenated in head order.
            masks.append(((query_bits & owner_cpu[head, ids][None]) != 0).flatten())
        self.qo_indptr = torch.arange(h + 1, dtype=torch.int32, device=q.device) * batch
        self.indptr = torch.tensor(pointers, dtype=torch.int32, device=q.device)
        self.indices = torch.cat(indices).to(q.device)
        self.last = torch.ones(h, dtype=torch.int32, device=q.device)
        self.custom_mask = torch.cat(masks).to(q.device)
        self.union_lengths = [pointers[i + 1] - pointers[i] for i in range(h)]
        self.workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="fa2")
        self.wrapper.plan(self.qo_indptr, self.indptr, self.indices, self.last,
                          g, 1, d, 1, causal=False, custom_mask=self.custom_mask,
                          pos_encoding_mode="NONE", q_data_type=q.dtype, kv_data_type=k.dtype,
                          fixed_split_size=fixed_split_size, disable_split_kv=disable_split_kv)
        self.q = q.view(h * batch, g, d)
        self.kv = (k.view(h * n, 1, 1, d), v.view(h * n, 1, 1, d))
        self.out = torch.empty_like(self.q)
        self.lse = torch.empty((h * batch, g), device=q.device, dtype=torch.float32)
        self.lse_ln = torch.empty_like(self.lse) if normalize_lse else None
        self.lse_log_base = math.e if normalize_lse else 2
        self.original_shape = q.shape
        self.batch, self.gqa = batch, g
        self.resolved_backend = "fa2"
        torch.cuda.synchronize(q.device)
        self.plan_ms = (time.perf_counter() - start) * 1000

    def run(self):
        self.wrapper.run(self.q, self.kv, out=self.out, lse=self.lse, return_lse=True)
        lse = self.lse
        if self.lse_ln is not None:
            # Match the existing native decode contract, including this launch
            # in any captured/timed complete call that requests natural-log LSE.
            torch.mul(self.lse, math.log(2), out=self.lse_ln)
            lse = self.lse_ln
        return self.out.view(self.original_shape), lse.view(self.original_shape[:2])
