"""Research-only direct pair path for B1..5, including negative controls.

This does not change the production B6..21 policy. All batches execute the
candidate, even when its lack of reuse or extra state traffic makes it slower.
"""
import torch
from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA
from src.xqa_raw_pair_general_runtime import RawPairGeneralPlan


class RawPairDirectSmallPlan(RawPairGeneralPlan):
    def __init__(self, modules, query, cache, block_table, seq_lens, prefix_tokens,
                 workspace=None, raw_state=None):
        super().__init__(modules, query, cache, block_table, seq_lens, prefix_tokens,
                         workspace=workspace, raw_state=raw_state)
        assert 1 <= self.b <= 5
        self.groups = (self.b + 1) // 2
        self.paired_b = self.b
        self.prefix_sm = self.splits * self.groups * 2
        self.tail_sm = self.splits * self.b * 2
        self.orphan_sm = 0
        size = self.groups * 2 * self.splits * STATE_FLOATS_PER_CTA
        if raw_state is not None:
            assert raw_state.numel() >= size
        self.state = (torch.empty(size, device=query.device, dtype=torch.float32)
                      if raw_state is None else raw_state.view(-1)[:size])
        self.unused_prefix_o = torch.empty((self.groups, 1, 32, 128),
                                          device=query.device, dtype=query.dtype)

    def prepare(self, seq_lens):
        assert len(seq_lens) == self.b
        assert all(self.prefix_tokens < x <= self.max_len for x in seq_lens)
        paired = self.b // 2 * 2
        n = self.prefix_tokens // 16
        if paired:
            assert bool((self.table[:paired:2, :n] == self.table[1:paired:2, :n]).all())
        self.lengths.copy_(torch.tensor(seq_lens, device=self.q.device,
                                       dtype=torch.uint32).reshape(self.b, 1))

    def run(self):
        self.modules[16].xqa_wrapper(
            False, self.prefix_sm, 2, 0, 1., None, self.unused_prefix_o, 1., self.q.unsqueeze(1),
            None, self.wrapper_k, self.wrapper_v, None, None, self.table, self.max_len,
            self.lengths, self.groups, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 1)
        self.modules[8].xqa_wrapper(
            False, self.tail_sm, 2, 0, 1., None, self.output.unsqueeze(1), 1., self.q.unsqueeze(1),
            None, self.wrapper_k, self.wrapper_v, None, None, self.table, self.max_len,
            self.lengths, self.b, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 3)
        return self.output
