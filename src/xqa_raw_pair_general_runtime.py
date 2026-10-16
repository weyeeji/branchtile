"""Fixed-shape general-batch adapter for the frozen raw XQA modules.

Even requests pair; an odd last request uses native XQA with the ORIGINAL split
count. B<4 falls back entirely. A batch-size change requires a new plan/graph.
"""
import torch
from flashinfer.xqa import xqa
from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA


class RawPairGeneralPlan:
    def __init__(self, modules, query, cache, block_table, seq_lens, prefix_tokens,
                 workspace=None, raw_state=None):
        self.modules, self.q, self.cache, self.table = modules, query, cache, block_table
        b, heads, dim = query.shape
        assert b >= 1 and heads == 16 and dim == 128
        assert query.dtype == cache.dtype == torch.bfloat16
        assert query.is_contiguous() and block_table.is_contiguous()
        assert block_table.dtype == torch.int32 and block_table.shape[0] == b
        assert cache.ndim == 4 and cache.shape[1:] == (2, 16, 256)
        assert prefix_tokens >= 0 and prefix_tokens % 16 == 0
        self.b, self.prefix_tokens = b, prefix_tokens
        self.paired_b = b // 2 * 2 if b >= 4 else 0
        self.groups = self.paired_b // 2
        self.max_len = block_table.shape[1] * 16
        self.sm = torch.cuda.get_device_properties(query.device).multi_processor_count
        self.splits = min(max(1, self.sm // (b * 2)), (self.max_len + 255) // 256)
        self.prefix_sm = self.splits * self.groups * 2
        self.tail_sm = self.splits * self.paired_b * 2
        self.orphan_sm = self.splits * 2
        self.lengths = torch.empty((b, 1), device=query.device, dtype=torch.uint32)
        self.output = torch.empty_like(query)
        if workspace is not None:
            assert workspace.dtype == torch.uint8 and workspace.is_contiguous()
            assert workspace.device == query.device and workspace.numel() >= 64 * 1024 * 1024
        self.workspace = (torch.zeros(64 * 1024 * 1024, device=query.device, dtype=torch.uint8)
                          if workspace is None else workspace)
        self.sem, self.scratch = self.workspace[:8 * 1024 * 1024], self.workspace[8 * 1024 * 1024:]
        state_size = self.groups * 2 * self.splits * STATE_FLOATS_PER_CTA
        if raw_state is not None:
            assert raw_state.dtype == torch.float32 and raw_state.is_contiguous()
            assert raw_state.device == query.device and raw_state.numel() >= state_size
        self.state = (torch.empty(state_size, device=query.device, dtype=torch.float32)
                      if raw_state is None else raw_state.view(-1)[:state_size])
        self.k, self.v = cache[..., :128], cache[..., 128:]
        self.wrapper_k, self.wrapper_v = self.k.transpose(1, 2), self.v.transpose(1, 2)
        if self.paired_b:
            self.pair_table = torch.empty_like(block_table[:self.paired_b:2])
            self.pair_lengths = torch.empty((self.groups, 1), device=query.device, dtype=torch.uint32)
            self.packed = torch.empty((self.groups, 1, 32, 128), device=query.device, dtype=query.dtype)
            self.unused_prefix_o = torch.empty_like(self.packed)
        self.prepare(seq_lens)

    def prepare(self, seq_lens):
        assert len(seq_lens) == self.b, "Batch changes require a new fixed-shape plan/graph"
        assert all(self.prefix_tokens < x <= self.max_len for x in seq_lens)
        if self.paired_b:
            n = self.prefix_tokens // 16
            assert bool((self.table[:self.paired_b:2, :n] == self.table[1:self.paired_b:2, :n]).all()), "prefix physical pages differ"
        self.lengths.copy_(torch.tensor(seq_lens, device=self.q.device, dtype=torch.uint32).reshape(self.b, 1))

    def run(self):
        if not self.paired_b:
            xqa(self.q.unsqueeze(1), self.k, self.v, self.table, self.lengths, self.output.unsqueeze(1),
                self.scratch, self.sem, 2, 16, q_scale=1., kv_scale=1., kv_layout="HND",
                sm_count=self.sm, enable_pdl=True)
            return self.output
        n = self.paired_b
        self.packed.view(self.groups, 2, 2, 8, 128).copy_(
            self.q[:n].view(self.groups, 2, 2, 8, 128).permute(0, 2, 1, 3, 4))
        self.pair_table.copy_(self.table[:n:2])
        self.pair_lengths.copy_(self.lengths[:n:2])
        self.modules[16].xqa_wrapper(
            False, self.prefix_sm, 2, 0, 1., None, self.unused_prefix_o, 1., self.packed,
            None, self.wrapper_k, self.wrapper_v, None, None, self.pair_table, self.max_len,
            self.pair_lengths, self.groups, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 1)
        self.modules[8].xqa_wrapper(
            False, self.tail_sm, 2, 0, 1., None, self.output[:n].unsqueeze(1), 1., self.q[:n].unsqueeze(1),
            None, self.wrapper_k, self.wrapper_v, None, None, self.table[:n], self.max_len,
            self.lengths[:n], n, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 3)
        if n != self.b:
            # This is a one-request kernel with forced ORIGINAL S, not its auto-S.
            xqa(self.q[n:].unsqueeze(1), self.k, self.v, self.table[n:], self.lengths[n:],
                self.output[n:].unsqueeze(1), self.scratch, self.sem, 2, 16,
                q_scale=1., kv_scale=1., kv_layout="HND", sm_count=self.orphan_sm, enable_pdl=False)
        return self.output
