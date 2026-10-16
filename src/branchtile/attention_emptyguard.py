"""Empty-shard guard ablation; constructor/call API matches attention_v2.

compact=True differs only by skipping empty region shards. compact=False retains
the same unmodified fixed_producer control as v2. No new stream, state format,
mask, launch geometry, or merge behavior is introduced."""
import torch
from .attention import Attention as BaseAttention
from .tilelang_kernel import fixed_producer
from .tilelang_kernel_emptyguard import producer


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
