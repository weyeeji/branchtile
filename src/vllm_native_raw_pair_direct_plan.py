"""Direct-input bank with per-layer native contiguous Q/O aliases during capture.

Graph capture freezes the concrete pointers for each layer. The mutable Python
plan only selects launch arguments at capture time; shared scratch is serial.
"""
import torch
from src.vllm_native_raw_pair_general_plan import LayerSharedRawPairGeneralBank
from src.xqa_raw_direct_input_runtime import RawPairDirectInputPlan


class LayerSharedRawPairDirectBank(LayerSharedRawPairGeneralBank):
    def __init__(self, modules, caches, native_table, native_lengths, prefix_tokens, enabled_batches):
        super().__init__(modules, caches, native_table, native_lengths, prefix_tokens,
                         enabled_batches, plan_factory=RawPairDirectInputPlan)

    def run_layer(self, layer, query, output):
        b = query.shape[0]
        assert query.shape == output.shape == (b, 16, 128)
        assert query.dtype == output.dtype == torch.bfloat16
        assert query.is_contiguous() and output.is_contiguous(), 'Native Q/O address contract; no stride fallback'
        assert query.device == output.device == self.caches[layer].device
        p = self.plans[b]
        self.query_layouts.setdefault(b, {})[layer] = {
            'shape': list(query.shape), 'stride': list(query.stride()),
            'output_stride': list(output.stride()),
            'Q_canonicalization_copy_charged': False,
            'Q_native_alias': True, 'O_native_alias': True}
        p.q, p.output = query, output
        p.k, p.v, p.wrapper_k, p.wrapper_v = self.views[layer]
        return p.run()

    def receipt(self):
        report = super().receipt()
        report['copy_policy'] = 'Native contiguous per-layer Q/O aliases; prefix directly reads Q and native table/length rows. No candidate input/output copy launch. State export and native tail/merge remain charged.'
        report['address_contract'] = 'Concrete Q/O and cache addresses selected per layer during graph capture; serial shared workspace/state. Noncontiguous Q/O rejected.'
        return report
