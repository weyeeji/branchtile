"""B8 model adapter reusing the qualified pair runtime and one state/workspace.

Every existing runtime copy remains in the model graph and is charged. Host
physical-prefix validation is once per admitted step, never once per layer.
"""
import torch
from src.xqa_raw_pair_runtime import RawPairPlan


class LayerSharedNativeRawPairPlan:
    def __init__(self, modules, caches, native_table, prefix_tokens):
        assert len(caches) == 48
        self.caches = list(caches)
        assert all(c.dtype == torch.bfloat16 and c.shape[1:] == (2, 16, 256) for c in caches)
        assert native_table.shape[0] == 8 and native_table.dtype == torch.int32 and native_table.is_contiguous()
        self.q = torch.empty((8, 16, 128), device=caches[0].device, dtype=torch.bfloat16)
        self.inner = RawPairPlan(modules, self.q, caches[0], native_table,
                                 [prefix_tokens + 1] * 8, prefix_tokens)
        self.views = [(c[..., :128].transpose(1, 2), c[..., 128:].transpose(1, 2)) for c in caches]
        self.query_layouts = {}

    def prepare(self, actual_native_table, seq_lens):
        assert actual_native_table.shape == self.inner.table.shape
        assert actual_native_table.stride() == self.inner.table.stride()
        assert actual_native_table.data_ptr() == self.inner.table.data_ptr(), 'Must alias original gathered full-width native table'
        self.inner.prepare(seq_lens)

    def run_layer(self, layer, query, output):
        assert query.shape == output.shape == (8, 16, 128)
        assert query.dtype == output.dtype == torch.bfloat16
        self.query_layouts[layer] = {'shape': list(query.shape), 'stride': list(query.stride()),
                                    'output_stride': list(output.stride()), 'Q_canonicalization_copy_charged': True}
        self.q.copy_(query)
        self.inner.k, self.inner.v = self.views[layer]
        # Python chooses layer aliases while capturing. Replay nodes retain them.
        result = self.inner.run()
        output.copy_(result)
        return output

    def receipt(self):
        p = self.inner
        return {'workspace_bytes': p.workspace.numel(), 'raw_state_bytes': p.state.numel()*4,
                'workspace_allocation_sets': 1, 'raw_state_allocation_sets': 1, 'layers': 48,
                'native_table_shape': list(p.table.shape), 'native_table_stride': list(p.table.stride()),
                'native_launch_extent': p.max_len, 'native_splits': p.splits,
                'prefix_SM': p.paired_sm, 'native_SM': p.sm,
                'prefix_tiles': p.prefix_tokens//256, 'shared_tile_tokens': p.prefix_tokens//256*256,
                'query_layouts': self.query_layouts,
                'copy_policy': 'Per layer Q canonicalization, Q packing, pair table/length refresh, O copy all charged; no KV payload copy.',
                'runtime_candidate_PDL': False, 'concurrency': 'One graph stream; no concurrent replay using this shared state.'}
