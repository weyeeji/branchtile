"""Bound BranchTile dynamic wrappers; preparation and full producer+merge are explicit."""
import time
from branchtile.attention import make_plan
from branchtile.attention_v2 import Attention as DynamicV2
from branchtile.attention_v3 import Attention as DynamicV3


class DynamicAttention:
    def __init__(self, q, k, v, owners, batch, *, version, splits, block_n):
        started = time.perf_counter()
        self.q, self.k, self.v = q, k, v
        self.plan = make_plan(owners, batch, q.shape[1] // batch, q.device)
        self.metadata_ms = self.plan.prepare_ms
        kernel = {"v2": DynamicV2, "v3": DynamicV3}[version]
        self.op = kernel(self.plan, splits=splits, block_n=block_n, threads=128, compact=True)
        self.prepare_ms = (time.perf_counter() - started) * 1000
        self.payload_aliases_source = True
        self.logical_kv_token_visits = int((owners != 0).sum())
        self.workspace_bytes = sum(t.numel() * t.element_size() for t in (self.op.u, self.op.m, self.op.l))

    def run(self):
        return self.op(self.q, self.k, self.v)
