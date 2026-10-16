"""B8-only experimental D64 prefix partition, original direct tail and metadata."""
from src.xqa_raw_direct_input_runtime import RawPairDirectInputPlan


class RawPairDirectDSlicePlan(RawPairDirectInputPlan):
    def __init__(self, modules, query, cache, block_table, seq_lens, prefix_tokens,
                 workspace=None, raw_state=None):
        assert query.shape == (8, 16, 128), "First bounded gate is real or synthetic B8 only"
        super().__init__(modules, query, cache, block_table, seq_lens, prefix_tokens,
                         workspace=workspace, raw_state=raw_state)
