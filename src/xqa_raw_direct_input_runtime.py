"""Two-launch raw sharing: native input addresses, no prefix packing/copy kernels."""
from src.xqa_raw_pair_padded_runtime import RawPairPaddedPlan


class RawPairDirectInputPlan(RawPairPaddedPlan):
    def run(self):
        if self.b <= 4:
            return super().run()
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
