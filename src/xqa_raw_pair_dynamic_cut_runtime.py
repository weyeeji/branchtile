"""Standalone graph-stable cut adapter; callers still validate cohort ownership."""
import torch
from src.xqa_raw_pair_padded_runtime import RawPairPaddedPlan


class DeviceCutModule:
    def __init__(self, module, cut):
        self.module, self.cut = module, cut

    def xqa_wrapper(self, *args):
        # The frozen host-scalar adapter supplies its cut at position -2.
        # This ABI-compatible Python shim substitutes one stable device address.
        return self.module.xqa_wrapper(*args[:-2], self.cut, args[-1])


class RawPairDynamicCutPlan(RawPairPaddedPlan):
    def __init__(self, modules, query, cache, block_table, seq_lens, prefix_tokens,
                 workspace=None, raw_state=None, device_cut=None):
        self.device_cut = (torch.tensor([prefix_tokens // 256], device=query.device, dtype=torch.int32)
                           if device_cut is None else device_cut)
        assert self.device_cut.device == query.device and self.device_cut.dtype == torch.int32
        assert self.device_cut.numel() == 1 and self.device_cut.is_contiguous()
        wrapped = {g: DeviceCutModule(m, self.device_cut) for g, m in modules.items()}
        super().__init__(wrapped, query, cache, block_table, seq_lens, prefix_tokens,
                         workspace=workspace, raw_state=raw_state)

    def change_prefix(self, prefix_tokens, seq_lens):
        """Outside replay, on its stream; physical validation and upload are charged."""
        assert prefix_tokens > 0 and prefix_tokens % 16 == 0
        assert len(seq_lens) == self.b and min(seq_lens) > prefix_tokens
        pages = prefix_tokens // 16
        assert bool((self.table[:, :pages] == self.table[0, :pages]).all())
        self.prefix_tokens = prefix_tokens
        self.device_cut.fill_(prefix_tokens // 256)
        self.prepare(seq_lens)
