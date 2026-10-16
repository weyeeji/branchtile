"""Metadata, immutable-source attention API, and exact FP32 reference."""
from dataclasses import dataclass
import math
import time
import torch
import triton
import triton.language as tl
from .tilelang_kernel import producer, fixed_producer


@dataclass
class RegionPlan:
    indices: torch.Tensor
    offsets: torch.Tensor
    row_map: torch.Tensor
    counts: torch.Tensor
    owner_rows: torch.Tensor
    labels: list[int]
    batch: int
    gqa: int
    prepare_ms: float

    @property
    def rows(self):
        return self.batch * self.gqa

    @property
    def tensors(self):
        return (self.indices, self.offsets, self.row_map, self.counts, self.owner_rows)

    def statistics(self, block_n=64, splits=1):
        lengths = self.offsets[:, 1:] - self.offsets[:, :-1]
        compact_tiles = fixed_tiles = 0
        for s in range(splits):
            shard = lengths * (s + 1) // splits - lengths * s // splits
            blocks = triton.cdiv(shard, block_n)
            compact_tiles += int((blocks * triton.cdiv(self.counts, 16)).sum())
            fixed_tiles += int((blocks * triton.cdiv(self.rows, 16)).sum())
        return dict(regions=len(self.labels), owner_labels=self.labels,
                    compact_m16_tiles=compact_tiles, fixed_m16_tiles=fixed_tiles,
                    metadata_bytes=sum(t.numel()*t.element_size() for t in self.tensors),
                    prepare_ms=self.prepare_ms)


def make_plan(owners: torch.Tensor, batch: int = 4, gqa: int = 8, device=None):
    """owners[KV-head, source-token] is a request bitmask, zero means unused.

    CPU preparation is intentionally explicit and separately measured. Active
    row maps are one-to-one, in range, and every source-token appears once.
    """
    start = time.perf_counter()
    assert 1 <= batch <= 4 and gqa in (4, 8) and batch * gqa <= 32
    destination = device or owners.device
    owners = owners.to(device="cpu", dtype=torch.int64)
    assert owners.ndim == 2 and owners.numel() > 0
    assert int(owners.min()) >= 0 and int(owners.max()) < (1 << batch)
    heads, tokens = owners.shape
    labels = sorted(set(owners.flatten().tolist()) - {0}) or [1]
    regions = len(labels)
    indices = torch.zeros((heads, tokens), dtype=torch.int32)
    offsets = torch.zeros((heads, regions + 1), dtype=torch.int32)
    row_map = torch.zeros((heads, regions, 32), dtype=torch.int32)
    counts = torch.zeros((heads, regions), dtype=torch.int32)
    owner_rows = torch.zeros((heads, regions, batch*gqa), dtype=torch.int32)
    for h in range(heads):
        packed = []
        for r, label in enumerate(labels):
            ids = torch.where(owners[h] == label)[0]
            packed.extend(ids.tolist())
            offsets[h, r+1] = len(packed)
            active = [b*gqa + q for b in range(batch) if label & (1 << b) for q in range(gqa)]
            assert len(active) == len(set(active)) and all(0 <= x < batch*gqa for x in active)
            row_map[h, r, :len(active)] = torch.tensor(active, dtype=torch.int32)
            counts[h, r] = len(active)
            owner_rows[h, r, active] = 1
        assert len(packed) == len(set(packed)) == int((owners[h] != 0).sum())
        indices[h, :len(packed)] = torch.tensor(packed, dtype=torch.int32)
    arrays = [x.to(destination) for x in (indices, offsets, row_map, counts, owner_rows)]
    if str(destination).startswith("cuda"):
        torch.cuda.synchronize(destination)
    return RegionPlan(*arrays, labels, batch, gqa, (time.perf_counter()-start)*1000)


@triton.jit
def _merge(U, M, L, O, LSE, ROWS: tl.constexpr, D: tl.constexpr,
           SPLITS: tl.constexpr, BS: tl.constexpr):
    h = tl.program_id(0)
    r = tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    off = (h*SPLITS + s)*ROWS + r
    den = tl.load(L + off, s < SPLITS, other=0)
    m = tl.load(M + off, s < SPLITS, other=-float('inf'))
    m = tl.where(den > 0, m, -float('inf'))
    mx = tl.max(m, 0)
    valid = tl.sum(den, 0) > 0
    weight = tl.where(den > 0, tl.exp2((m-mx)*1.4426950408889634), 0)
    divisor = tl.sum(weight*den, 0)
    u = tl.load(U + off[:, None]*D+d[None, :], s[:, None] < SPLITS, other=0)
    value = tl.sum(u*weight[:, None], 0) / tl.where(valid, divisor, 1)
    tl.store(O+(h*ROWS+r)*D+d, value)
    tl.store(LSE+h*ROWS+r, tl.where(valid, mx+tl.log(divisor), -float('inf')))


class Attention:
    """A preallocated, CUDA-graph-compatible full producer + merge call."""
    def __init__(self, plan: RegionPlan, splits=1, block_n=64, threads=128, compact=True, region_partials=False):
        self.plan, self.splits = plan, splits
        heads, tokens = plan.indices.shape
        rows, dim = plan.rows, 128
        if compact:
            self.kernel = producer(heads, tokens, rows, dim, len(plan.labels),
                                   splits, block_n, threads, True, region_partials)
        else:
            self.kernel = fixed_producer(heads, tokens, rows, dim, len(plan.labels),
                                         splits, block_n, threads, region_partials)
        splits = splits * len(plan.labels) if region_partials else splits
        kwargs = dict(device=plan.indices.device, dtype=torch.float32)
        self.u = torch.empty((heads, splits, rows, dim), **kwargs)
        self.m = torch.empty((heads, splits, rows), **kwargs)
        self.l = torch.empty_like(self.m)
        self.output = torch.empty((heads, rows, dim), device=plan.indices.device, dtype=torch.bfloat16)
        self.lse = torch.empty((heads, rows), **kwargs)
        self.splits = splits
        self.shape = (heads, rows, dim)

    def __call__(self, q, k, v):
        self.kernel(q, k, v, *self.plan.tensors, self.u, self.m, self.l)
        _merge[(self.shape[0], self.shape[1])](self.u, self.m, self.l, self.output, self.lse,
                    self.shape[1], self.shape[2], self.splits, triton.next_power_of_2(self.splits), num_warps=4)
        return self.output, self.lse


def reference(q, k, v, owners, batch=4, gqa=8, compute_dtype=torch.float64):
    """Exact masks, FP64 by default (no TF32); empty rows return 0/-inf."""
    owners = owners.to(q.device)
    scores = torch.einsum("hmd,hnd->hmn", q.to(compute_dtype), k.to(compute_dtype)) / math.sqrt(q.shape[-1])
    row_owner = 1 << (torch.arange(batch*gqa, device=q.device) // gqa)
    keep = (owners[:, None, :] & row_owner[None, :, None]) != 0
    scores = scores.masked_fill(~keep, -float("inf"))
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1).nan_to_num()
    return torch.einsum("hmn,hnd->hmd", p, v.to(compute_dtype)), lse


def synthetic_owners(heads, tokens, pattern="cross", batch=4, seed=0):
    patterns = {
        "cross": [3, 5, 6, 9, 10, 12],
        "full": [(1 << batch)-1],
        "disjoint": [1 << b for b in range(batch)],
        "tree": [15, 3, 12, 1, 2, 4, 8],
        "mixed": list(range(1, 1 << batch)),
        "empty": [0],
    }
    labels = [x for x in patterns[pattern] if x < (1 << batch)]
    if not labels:
        labels = [(1 << batch)-1]
    generator = torch.Generator().manual_seed(seed)
    result = torch.empty((heads, tokens), dtype=torch.int64)
    for h in range(heads):
        # Position-randomized supports prevent a contiguous-KV-only shortcut.
        values = torch.tensor(labels)[torch.arange(tokens) % len(labels)]
        result[h] = values[torch.randperm(tokens, generator=generator)]
    return result
