"""Shared-arena B4..8 direct-input quad plan; reuses the qualified direct launch path."""
from src.xqa_raw_gqa4_quad_shared_runtime import RawGQA4SharedQuadPlan
from src.xqa_raw_direct_quad_runtime import RawGQA4DirectQuadPlan


class RawGQA4SharedDirectQuadPlan(RawGQA4SharedQuadPlan):
    def run(self):
        return RawGQA4DirectQuadPlan.run(self)
