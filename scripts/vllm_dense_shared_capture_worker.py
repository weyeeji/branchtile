"""Export actual dense B8 inputs after the unchanged fixed-state qualification.

Export overhead is outside all paired diagnostic timings. Source KV stays in
place; only the CPU artifact is compacted into common-prefix/private-tail views.
"""
import hashlib
import json
from pathlib import Path
import torch

from scripts.vllm_cached_o_b8_diagnostic_worker import CachedOB8DiagnosticWorker


class DenseSharedCaptureWorker(CachedOB8DiagnosticWorker):
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
        directory = self.native_probe['output'] / 'dense_capture'
        directory.mkdir(exist_ok=True)
        q_file = directory / 'all_layer_qo.pt'
        torch.save({'q': {l: state['saved_q'][l, b].cpu() for l in range(48)},
                    'o': {l: state['saved_o'][l, b].cpu() for l in range(48)},
                    'positions': positions}, q_file)
        files = []
        for layer in (0, 15, 31, 47):
            cache = state['cache'][layer]
            # The current native XQA ABI is [page, K/V, page_token, head, D].
            assert cache.ndim == 5 and tuple(cache.shape[1:]) == (2, 16, 2, 128)
            pages = table[0, :common_pages].long()
            prefix = cache[pages].permute(1, 0, 2, 3, 4).reshape(2, common_pages * 16, 2, 128).cpu()
            tails = []
            for row, length in enumerate(lengths.tolist()):
                logical = torch.arange(common_pages * 16, length, device=cache.device)
                physical = table[row, logical // 16].long()
                tails.append(cache[physical, :, logical % 16].permute(1, 0, 2, 3).cpu())
            path = directory / f'layer{layer}.pt'
            torch.save({'q': state['saved_q'][layer, b].cpu(),
                        'o': state['saved_o'][layer, b].cpu(),
                        'prefix_kv': prefix, 'tail_kv': tails, 'positions': positions,
                        'original_block_table': table_cpu,
                        'original_cache_shape': list(cache.shape),
                        'original_cache_stride': list(cache.stride()),
                        'original_cache_dtype': str(cache.dtype),
                        'kv_semantic_dtype': 'float8_e4m3fn' if cache.dtype == torch.uint8 else str(cache.dtype),
                        'common_complete_pages': common_pages}, path)
            files.append({'path': str(path.relative_to(Path.cwd())), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'layer': layer})
        receipt = {'status': 'completed', 'rank': self.native_probe['rank'],
                   'batch': b, 'lengths': lengths.tolist(), 'common_complete_pages': common_pages,
                   'q_file': str(q_file.relative_to(Path.cwd())),
                   'q_sha256': hashlib.sha256(q_file.read_bytes()).hexdigest(), 'layer_files': files,
                   'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'scope': 'Actual native third B8 decode state; CPU compact artifact only. Export costs excluded from diagnostic timing, no online compact-copy speed claim.'}
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        state['result']['dense_capture'] = receipt
