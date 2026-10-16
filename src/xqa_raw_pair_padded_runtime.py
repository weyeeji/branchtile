"""V2: odd batches pad one zero-Q prefix row; tail retains original B and S.

B<=4 uses native fallback. Frozen CUDA modules and the older negative-results
runtime remain unchanged. Fixed shape per plan/graph, preparation is explicit.
"""
import torch
from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA
from src.xqa_raw_pair_general_runtime import RawPairGeneralPlan


class RawPairPaddedPlan(RawPairGeneralPlan):
    def __init__(self, modules, query, cache, block_table, seq_lens, prefix_tokens,
                 workspace=None, raw_state=None):
        super().__init__(modules, query, cache, block_table, seq_lens, prefix_tokens,
                         workspace=workspace, raw_state=raw_state)
        if self.b <= 4:
            self.paired_b, self.groups = 0, 0
            self.prefix_sm, self.tail_sm = 0, 0
            self.state = self.state[:0]
            return
        self.paired_b = self.b
        self.groups = (self.b + 1) // 2
        self.prefix_sm = self.splits * self.groups * 2
        self.tail_sm = self.splits * self.b * 2
        self.orphan_sm = 0
        if self.b % 2:
            self.padded_q = torch.zeros((self.b + 1, 16, 128), device=query.device, dtype=query.dtype)
            self.pair_table = torch.empty_like(block_table[::2])
            self.pair_lengths = torch.empty((self.groups, 1), device=query.device, dtype=torch.uint32)
            self.packed = torch.empty((self.groups, 1, 32, 128), device=query.device, dtype=query.dtype)
            self.unused_prefix_o = torch.empty_like(self.packed)
            size = self.groups * 2 * self.splits * STATE_FLOATS_PER_CTA
            if raw_state is not None:
                assert raw_state.numel() >= size
            self.state = (torch.empty(size, device=query.device, dtype=torch.float32)
                          if raw_state is None else raw_state.view(-1)[:size])

    def prepare(self, seq_lens):
        assert len(seq_lens) == self.b, "Batch changes require a new fixed-shape plan/graph"
        assert all(self.prefix_tokens < x <= self.max_len for x in seq_lens)
        if self.b > 4:
            paired = self.b // 2 * 2
            n = self.prefix_tokens // 16
            assert bool((self.table[:paired:2, :n] == self.table[1:paired:2, :n]).all()), "prefix physical pages differ"
        self.lengths.copy_(torch.tensor(seq_lens, device=self.q.device, dtype=torch.uint32).reshape(self.b, 1))

    def run(self):
        if self.b <= 4:
            return super().run()
        if self.b % 2:
            self.padded_q[:self.b].copy_(self.q)
            # The last dummy row stays zero for the lifetime of the fixed graph.
            packing_q = self.padded_q
        else:
            packing_q = self.q
        self.packed.view(self.groups, 2, 2, 8, 128).copy_(
            packing_q.view(self.groups, 2, 2, 8, 128).permute(0, 2, 1, 3, 4))
        self.pair_table.copy_(self.table[::2])
        self.pair_lengths.copy_(self.lengths[::2])
        self.modules[16].xqa_wrapper(
            False, self.prefix_sm, 2, 0, 1., None, self.unused_prefix_o, 1., self.packed,
            None, self.wrapper_k, self.wrapper_v, None, None, self.pair_table, self.max_len,
            self.pair_lengths, self.groups, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 1)
        self.modules[8].xqa_wrapper(
            False, self.tail_sm, 2, 0, 1., None, self.output.unsqueeze(1), 1., self.q.unsqueeze(1),
            None, self.wrapper_k, self.wrapper_v, None, None, self.table, self.max_len,
            self.lengths, self.b, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 3)
        return self.output
