"""Native FlashInfer sparse decode, sharing the physical KV source.

Flatten (KV-head, request) to native batch and expose each original token as
one page. This exactly supports head-specific arbitrary retained token indices.
There is no payload packing/copy: K and V are views of the source tensors.
Planning cost is explicit and independent of the repeatedly invoked run().
"""
import time
import torch
import flashinfer


class NativeSparseDecode:
    lse_log_base = 2  # FlashInfer state_t::get_lse; normalize for FP64 comparisons.
    def __init__(self, q, k, v, supports, *, use_tensor_cores=True,
                 fixed_split_size=None, disable_split_kv=False):
        """q [H,B*G,D], k/v [H,N,D], supports nested [H][B] CPU indices."""
        h, n, d = k.shape
        b = len(supports[0])
        g = q.shape[1] // b
        assert q.shape == (h, b * g, d) and v.shape == k.shape
        assert all(t.is_contiguous() for t in (q, k, v))
        start = time.perf_counter()
        pointers, indices = [0], []
        for head in range(h):
            for request in range(b):
                ids = torch.as_tensor(supports[head][request], dtype=torch.int32, device="cpu")
                assert len(ids) == len(torch.unique(ids)), "Duplicate support changes semantics"
                assert len(ids) == 0 or (int(ids.min()) >= 0 and int(ids.max()) < n)
                indices.append(ids + head * n)
                pointers.append(pointers[-1] + len(ids))
        self.indptr = torch.tensor(pointers, dtype=torch.int32, device=q.device)
        self.indices = torch.cat(indices).to(q.device)
        self.last = torch.tensor([int(pointers[i + 1] > pointers[i]) for i in range(h * b)],
                                 dtype=torch.int32, device=q.device)
        # Page-size-one + small fixed splits can require >256 MiB even for B4.
        # Capacity is reported rather than silently excluding legal split choices.
        self.workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", use_tensor_cores=use_tensor_cores,
            backend="auto")
        self.wrapper.plan(self.indptr, self.indices, self.last, g, 1, d, 1,
                          q_data_type=q.dtype, kv_data_type=k.dtype,
                          fixed_split_size=fixed_split_size,
                          disable_split_kv=disable_split_kv)
        self.q = q.view(h * b, g, d)
        self.kv = (k.view(h * n, 1, 1, d), v.view(h * n, 1, 1, d))
        self.out = torch.empty_like(self.q)
        self.lse = torch.empty((h * b, g), device=q.device, dtype=torch.float32)
        self.original_shape = q.shape
        torch.cuda.synchronize()
        self.plan_ms = (time.perf_counter() - start) * 1000

    def run(self):
        self.wrapper.run(self.q, self.kv, out=self.out, lse=self.lse, return_lse=True)
        return self.out.view(self.original_shape), self.lse.view(self.original_shape[:2])
