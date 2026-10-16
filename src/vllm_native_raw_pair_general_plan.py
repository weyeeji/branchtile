"""Serial layer/batch bank for dummy-padded native-state continuation.

Only small shape-specific Q/metadata buffers scale with enabled batch sizes.
All 48 layers and all batch shapes share one workspace and one raw state arena.
This bank must never be replayed concurrently on separate streams.
"""
import torch
from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA
from src.xqa_raw_pair_padded_runtime import RawPairPaddedPlan


def state_capacity(enabled_batches, sm_count, table_columns):
    assert enabled_batches and all(isinstance(b, int) and b >= 5 for b in enabled_batches)
    assert sm_count > 0 and table_columns > 0
    extent = table_columns * 16
    by_batch = {}
    for b in enabled_batches:
        splits = min(max(1, sm_count // (b * 2)), (extent + 255) // 256)
        groups = (b + 1) // 2
        by_batch[b] = {'native_splits': splits, 'prefix_groups': groups,
                       'state_floats': groups * 2 * splits * STATE_FLOATS_PER_CTA}
    return max(x['state_floats'] for x in by_batch.values()), by_batch


class LayerSharedRawPairGeneralBank:
    def __init__(self, modules, caches, native_table, native_lengths, prefix_tokens, enabled_batches,
                 plan_factory=RawPairPaddedPlan):
        assert len(caches) == 48
        self.caches = list(caches)
        self.enabled_batches = tuple(sorted(set(enabled_batches)))
        assert native_table.dtype == torch.int32 and native_table.is_contiguous()
        assert all(c.dtype == torch.bfloat16 and c.shape[1:] == (2, 16, 256) for c in caches)
        assert max(self.enabled_batches) <= native_table.shape[0]
        assert native_lengths.dtype == torch.int32 and native_lengths.is_contiguous()
        assert native_lengths.device == native_table.device and native_lengths.numel() >= max(self.enabled_batches)
        self.native_lengths = native_lengths
        self.prefix_tokens = prefix_tokens
        device = caches[0].device
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        count, self.capacity_by_batch = state_capacity(self.enabled_batches, sm, native_table.shape[1])
        self.workspace = torch.zeros(64 * 1024 * 1024, device=device, dtype=torch.uint8)
        self.raw_state = torch.empty(count, device=device, dtype=torch.float32)
        self.plans = {}
        self.views = [(c[..., :128], c[..., 128:], c[..., :128].transpose(1, 2),
                       c[..., 128:].transpose(1, 2)) for c in caches]
        self.query_layouts = {}
        for b in self.enabled_batches:
            q = torch.empty((b, 16, 128), device=device, dtype=torch.bfloat16)
            self.plans[b] = plan_factory(modules, q, caches[0], native_table[:b],
                [prefix_tokens + 1] * b, prefix_tokens, workspace=self.workspace, raw_state=self.raw_state)
            p = self.plans[b]
            assert p.groups == (b + 1) // 2 and p.paired_b == b
            assert p.splits == self.capacity_by_batch[b]['native_splits']
            assert p.state.data_ptr() == self.raw_state.data_ptr()
            # XQA interprets the same positive int32 native lengths as uint32.
            # No recomputation, H2D transfer, new metadata kernel, or host sync.
            p.lengths = native_lengths[:b].view(torch.uint32).reshape(b, 1)

    def validate_aliases(self, actual_native_table, actual_native_lengths, b):
        p = self.plans[b]
        assert actual_native_table.shape == p.table.shape
        assert actual_native_table.stride() == p.table.stride()
        assert actual_native_table.data_ptr() == p.table.data_ptr(), 'Require original full-width gathered table alias'
        assert actual_native_lengths.dtype == torch.int32 and actual_native_lengths.is_contiguous()
        assert actual_native_lengths.numel() == b
        assert actual_native_lengths.data_ptr() == p.lengths.data_ptr(), 'Require original native GPU lengths alias'

    def bind_prefix(self, actual_native_table):
        b = actual_native_table.shape[0]
        prefix = actual_native_table[:, :self.prefix_tokens // 16]
        assert prefix.shape == (b, self.prefix_tokens // 16)
        if not bool((prefix == prefix[:1]).all().item()):
            raise ValueError('Initial cohort does not share every physical prefix page')
        return {'physical_prefix_checked': True, 'rows': b,
                'prefix_pages': self.prefix_tokens // 16,
                'scope': 'Once per cohort; CPU request/slot/overwrite epoch guards retain this binding for survivors.'}

    def run_layer(self, layer, query, output):
        b = query.shape[0]
        assert query.shape == output.shape == (b, 16, 128)
        assert query.dtype == output.dtype == torch.bfloat16
        p = self.plans[b]
        self.query_layouts.setdefault(b, {})[layer] = {'shape': list(query.shape),
            'stride': list(query.stride()), 'output_stride': list(output.stride()),
            'Q_canonicalization_copy_charged': True}
        p.q.copy_(query)
        p.k, p.v, p.wrapper_k, p.wrapper_v = self.views[layer]
        # Replay retains concrete pointers selected here during graph capture.
        output.copy_(p.run())
        return output

    def receipt(self):
        return {'enabled_batches': list(self.enabled_batches), 'layers': 48,
            'workspace_allocation_sets': 1, 'raw_state_allocation_sets': 1,
            'workspace_bytes': self.workspace.numel(), 'raw_state_bytes': self.raw_state.numel() * 4,
            'capacity_by_batch': self.capacity_by_batch,
            'plans': {b: {'native_batch': b, 'dummy_prefix_rows': b % 2,
                'native_table_shape': list(p.table.shape), 'native_table_stride': list(p.table.stride()),
                'native_launch_extent': p.max_len, 'native_splits': p.splits,
                'prefix_SM': p.prefix_sm, 'tail_SM': p.tail_sm, 'native_SM': p.sm,
                'prefix_tiles': p.prefix_tokens // 256} for b, p in self.plans.items()},
            'query_layouts': self.query_layouts,
            'lengths_policy': 'Alias native GPU int32 seq_lens as uint32; no host copy or extra length kernel.',
            'copy_policy': 'All per-layer canonical Q, odd padding, packing, pair metadata refresh and O copies charged. No KV payload copy.',
            'runtime_candidate_PDL': False,
            'concurrency': 'One stream; all shapes/layers share state and scratch; concurrent replay forbidden.'}
