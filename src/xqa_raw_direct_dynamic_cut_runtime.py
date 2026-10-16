"""Direct-input execution through one stable device-cut ABI shim."""
from src.xqa_raw_pair_dynamic_cut_runtime import RawPairDynamicCutPlan
from src.xqa_raw_direct_input_runtime import RawPairDirectInputPlan


class RawPairDirectDynamicCutPlan(RawPairDynamicCutPlan):
    def run(self):
        assert self.b>4, 'Combined candidate requires B>=5; the model dispatcher keeps smaller batches native'
        return RawPairDirectInputPlan.run(self)
