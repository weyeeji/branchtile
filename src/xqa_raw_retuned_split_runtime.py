"""Bounded B8 performance mode: fixed SM-coverage split rule, not default exactness.

The generated arithmetic and direct input map are unchanged. Only split count
(and hence per-row token order and final reduction shape) changes. Outputs must
be qualified against native XQA forced to the SAME count; default native stays
the primary performance and numerical reference.
"""
import torch
from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA
from src.xqa_raw_direct_input_runtime import RawPairDirectInputPlan


def coverage_splits(sm_count, batch, kv_heads, max_table_tokens):
    assert sm_count > 0 and batch == 8 and kv_heads == 2 and max_table_tokens > 0
    groups = (batch + 1) // 2
    return min(max(1, sm_count // (groups * kv_heads)), (max_table_tokens + 255) // 256)


class RawPairRetunedSplitPlan(RawPairDirectInputPlan):
    def __init__(self, modules, query, cache, block_table, seq_lens, prefix_tokens,
                 workspace=None, raw_state=None):
        assert query.shape == (8, 16, 128), "This bounded variant only admits actual B8 GQA8"
        super().__init__(modules, query, cache, block_table, seq_lens, prefix_tokens,
                         workspace=workspace, raw_state=raw_state)
        self.original_splits = self.splits
        self.splits = coverage_splits(self.sm, self.b, 2, self.max_len)
        self.prefix_sm = self.splits * self.groups * 2
        self.tail_sm = self.splits * self.b * 2
        size = self.groups * 2 * self.splits * STATE_FLOATS_PER_CTA
        if raw_state is not None:
            assert raw_state.numel() >= size, "Retuned S requires more raw-state storage"
        self.state = (torch.empty(size, device=query.device, dtype=torch.float32)
                      if raw_state is None else raw_state.view(-1)[:size])
