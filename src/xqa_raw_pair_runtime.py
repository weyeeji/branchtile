"""Small BF16 pair-continuation adapter; preparation is explicit and charged.

Qualification/replay only. No production scheduler integration is implied.
"""
import torch

from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA


class RawPairPlan:
    def __init__(self, modules, query, cache, block_table, seq_lens, prefix_tokens,
                 workspace=None, raw_state=None):
        self.modules = modules
        self.q, self.cache, self.table = query, cache, block_table
        b, heads, d = query.shape
        assert b == 8 and heads == 16 and d == 128
        assert query.dtype == cache.dtype == torch.bfloat16
        assert query.is_contiguous() and block_table.is_contiguous()
        assert block_table.dtype == torch.int32 and block_table.shape[0] == b
        assert cache.ndim == 4 and cache.shape[1:] == (2, 16, 256)
        assert prefix_tokens >= 0 and prefix_tokens % 16 == 0
        self.b, self.groups, self.prefix_tokens = b, b // 2, prefix_tokens
        self.max_len = block_table.shape[1] * 16
        self.sm = torch.cuda.get_device_properties(query.device).multi_processor_count
        self.splits = min(max(1, self.sm // (b * 2)), (self.max_len + 255) // 256)
        self.paired_sm = self.splits * self.groups * 2
        self.lengths = torch.empty((b, 1), device=query.device, dtype=torch.uint32)
        self.pair_lengths = torch.empty((self.groups, 1), device=query.device, dtype=torch.uint32)
        self.pair_table = torch.empty_like(block_table[::2])
        self.packed = torch.empty((self.groups, 1, 32, 128), device=query.device, dtype=query.dtype)
        self.unused_prefix_o = torch.empty_like(self.packed)
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
        self.state = (torch.full((state_size,), float("nan"), device=query.device, dtype=torch.float32)
                      if raw_state is None else raw_state.view(-1)[:state_size])
        self.k = cache[..., :128].transpose(1, 2)
        self.v = cache[..., 128:].transpose(1, 2)
        self.prepare(seq_lens)

    def prepare(self, seq_lens):
        assert len(seq_lens) == self.b
        assert all(self.prefix_tokens < x <= self.max_len for x in seq_lens)
        n = self.prefix_tokens // 16
        # Fail before any buffer updates; physical identity, not equal content.
        assert bool((self.table[::2, :n] == self.table[1::2, :n]).all()), "prefix physical pages differ"
        self.lengths.copy_(torch.tensor(seq_lens, device=self.q.device, dtype=torch.uint32).reshape(self.b, 1))

    def run(self):
        self.packed.view(self.groups, 2, 2, 8, 128).copy_(
            self.q.view(self.groups, 2, 2, 8, 128).permute(0, 2, 1, 3, 4))
        self.pair_table.copy_(self.table[::2])
        self.pair_lengths.copy_(self.lengths[::2])
        self.modules[16].xqa_wrapper(
            False, self.paired_sm, 2, 0, 1., None, self.unused_prefix_o, 1., self.packed,
            None, self.k, self.v, None, None, self.pair_table, self.max_len,
            self.pair_lengths, self.groups, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 1)
        self.modules[8].xqa_wrapper(
            False, self.sm, 2, 0, 1., None, self.output.unsqueeze(1), 1., self.q.unsqueeze(1),
            None, self.k, self.v, None, None, self.table, self.max_len,
            self.lengths, self.b, 1., None, 1, None, None,
            self.sem, self.scratch, False, self.state, self.prefix_tokens // 256, 3)
        return self.output
