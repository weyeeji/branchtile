"""Version 2 candidate with explicit shared-buffer reuse across row shapes."""
import torch
from .attention import Attention as BaseAttention
from .tilelang_kernel_b8 import producer as fixed_producer
from .attention import RegionPlan
import time
from .tilelang_kernel_b8 import producer


class Attention(BaseAttention):
    def __init__(self, plan, splits=1, block_n=64, threads=128, compact=True,
                 region_partials=False):
        self.plan = plan
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


def make_plan(owners: torch.Tensor, batch: int = 8, gqa: int = 8, device=None):
    """owners[KV-head, source-token] is a request bitmask, zero means unused.

    CPU preparation is intentionally explicit and separately measured. Active
    row maps are one-to-one, in range, and every source-token appears once.
    """
    start = time.perf_counter()
    assert batch == 8 and gqa == 8
    destination = device or owners.device
    owners = owners.to(device="cpu", dtype=torch.int64)
    assert owners.ndim == 2 and owners.numel() > 0
    assert int(owners.min()) >= 0 and int(owners.max()) < (1 << batch)
    heads, tokens = owners.shape
    labels = sorted(set(owners.flatten().tolist()) - {0}) or [1]
    regions = len(labels)
    indices = torch.zeros((heads, tokens), dtype=torch.int32)
    offsets = torch.zeros((heads, regions + 1), dtype=torch.int32)
    row_map = torch.zeros((heads, regions, 64), dtype=torch.int32)
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

