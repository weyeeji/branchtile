"""Export actual dense B8 inputs after the unchanged fixed-state qualification.

Export overhead is outside all paired diagnostic timings. Source KV stays in
place; only the CPU artifact is compacted into common-prefix/private-tail views.
"""
import hashlib
import json
import re
from pathlib import Path
import torch

from scripts.vllm_cached_o_b8_diagnostic_worker import CachedOB8DiagnosticWorker


class DenseSharedBF16CaptureWorker(CachedOB8DiagnosticWorker):
    @torch.inference_mode()
    def _measure_cached_o(self, desc):
        super()._measure_cached_o(desc)
        state = self.cached_o
        assert state['result']['final_inputs_tables_KV_hidden_unchanged']
        b = desc.num_reqs
        assert b == 8 and len(state['live_block_tables']) == 1
        table = state['live_block_tables'][0][:b]
        positions = self.model_runner.input_buffers.positions[:b].cpu().long()
        lengths = positions + 1
        table_cpu = table.cpu().long()
        complete_pages = int(lengths.min()) // 16
        same = (table_cpu[:, :complete_pages] == table_cpu[0:1, :complete_pages]).all(0)
        mismatch = (~same).nonzero().flatten()
        common_pages = int(mismatch[0]) if mismatch.numel() else complete_pages
        assert common_pages > 0
        scales = {}
        for name, parent in self.model_runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if match:
                scales[int(match[1])] = {'k_scale': float(parent.attn._k_scale_float),
                                        'v_scale': float(parent.attn._v_scale_float)}
        assert sorted(scales) == list(range(48))
        assert all(cache.dtype == torch.bfloat16 for cache in state['cache'].values()), 'Requires actual native BF16 KV, never cast FP8 artifacts'
        directory = self.native_probe['output'] / 'dense_capture'
        directory.mkdir(exist_ok=False)
        q_file = directory / 'all_layer_qo.pt'
        torch.save({'q': {l: state['saved_q'][l, b].cpu() for l in range(48)},
                    'o': {l: state['saved_o'][l, b].cpu() for l in range(48)},
                    'positions': positions, 'layer_scales': scales,
                    'source_native_kv_dtype': 'torch.bfloat16'}, q_file)
        files = []
        for layer in (0, 15, 31, 47):
            cache = state['cache'][layer]
            # Verified native_precision_receipt ABI: [page, head, page_token, K/V fused D].
            assert cache.ndim == 4 and tuple(cache.shape[1:]) == (2, 16, 256), (cache.shape, cache.stride())
            pages = table[0, :common_pages].long()
            prefix = cache[pages].reshape(common_pages, 2, 16, 2, 128).permute(3, 0, 2, 1, 4).reshape(2, common_pages * 16, 2, 128).cpu()
            tails = []
            for row, length in enumerate(lengths.tolist()):
                logical = torch.arange(common_pages * 16, length, device=cache.device)
                physical = table[row, logical // 16].long()
                tails.append(cache[physical, :, logical % 16, :].reshape(-1, 2, 2, 128).permute(2, 0, 1, 3).cpu())
            path = directory / f'layer{layer}.pt'
            torch.save({'q': state['saved_q'][layer, b].cpu(),
                        'o': state['saved_o'][layer, b].cpu(),
                        'prefix_kv': prefix, 'tail_kv': tails, 'positions': positions,
                        'original_block_table': table_cpu,
                        'original_cache_shape': list(cache.shape),
                        'original_cache_stride': list(cache.stride()),
                        'original_cache_dtype': str(cache.dtype),
                        'kv_semantic_dtype': 'float8_e4m3fn' if cache.dtype == torch.uint8 else str(cache.dtype),
                        'common_complete_pages': common_pages,
                        **scales[layer], 'rank': self.native_probe['rank'], 'layer': layer,
                        'source_native_kv_dtype': 'torch.bfloat16', 'payload_cast_performed': False}, path)
            files.append({'path': str(path.relative_to(Path.cwd())), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'layer': layer})
        receipt = {'status': 'completed', 'rank': self.native_probe['rank'],
                   'batch': b, 'lengths': lengths.tolist(), 'common_complete_pages': common_pages,
                   'source_native_kv_dtype': 'torch.bfloat16', 'payload_cast_performed': False,
                   'layer_scales': scales,
                   'source_cache_layouts': {str(layer): {'shape': list(cache.shape), 'stride': list(cache.stride()), 'dtype': str(cache.dtype)} for layer,cache in state['cache'].items()},
                   'q_file': str(q_file.relative_to(Path.cwd())),
                   'q_sha256': hashlib.sha256(q_file.read_bytes()).hexdigest(), 'layer_files': files,
                   'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'scope': 'Actual native third B8 decode state; CPU compact artifact only. Export costs excluded from diagnostic timing, no online compact-copy speed claim.'}
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        state['result']['dense_capture'] = receipt
