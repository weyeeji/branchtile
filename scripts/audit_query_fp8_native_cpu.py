"""CPU screen on all 48-layer Q of the completed native FP8 B8 captures."""
import collections
import hashlib
import json
from pathlib import Path

import torch
from audit_query_fp8_components import component

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    rows = []; sources = {}
    for document in (1, 7):
        directory = ROOT / f'results/dense_capture_b8_train{document}_fp8_v2'
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text())
        assert summary['status'] == 'completed'
        sources[str(summary_path.relative_to(ROOT))] = sha(summary_path)
        for rank in (0, 1):
            receipt_path = directory / f'rank{rank}/dense_capture/receipt.json'
            receipt = json.loads(receipt_path.read_text())
            assert receipt['status'] == 'completed' and receipt['batch'] == 8
            path = ROOT / receipt['q_file']
            assert sha(path) == receipt['q_sha256']
            sources[str(path.relative_to(ROOT))] = receipt['q_sha256']
            sources[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
            obj = torch.load(path, map_location='cpu', weights_only=True)
            assert len(obj['q']) == 48
            for layer in range(48):
                q = obj['q'][layer]
                assert q.shape == (8, 16, 128) and q.dtype == torch.bfloat16
                x = q.float().reshape(-1, 128)
                hi, sh = component(x)
                lo, sl = component(x - hi)
                err = x - (hi + lo)
                rows.append({'document': document, 'rank': rank, 'layer': layer,
                             'query_vectors': len(x), 'exact_numeric_vectors': int((err == 0).all(-1).sum()),
                             'nonexact_values': int((err != 0).sum()), 'max_abs_error': float(err.abs().max()),
                             'relative_l2': float(err.norm() / x.norm()),
                             'query_abs_max': float(x.abs().max()),
                             'smallest_nonzero_query_abs': float(x.abs()[x != 0].min()),
                             'high_scale_log2_histogram': dict(collections.Counter(torch.log2(sh).flatten().int().tolist())),
                             'low_scale_log2_histogram': dict(collections.Counter(torch.log2(sl).flatten().int().tolist()))})
    assert len(rows) == 192 and not torch.cuda.is_initialized()
    totals = {'layers_times_rank_times_document': len(rows),
              'query_vectors': sum(r['query_vectors'] for r in rows),
              'exact_numeric_vectors': sum(r['exact_numeric_vectors'] for r in rows),
              'nonexact_values': sum(r['nonexact_values'] for r in rows),
              'max_abs_error': max(r['max_abs_error'] for r in rows),
              'max_layer_relative_l2': max(r['relative_l2'] for r in rows)}
    result = {'schema': 1, 'status': 'CPU_native_FP8_B8_Q_representation_screen', 'cuda_initialized': False,
              'source_sha256': sources,
              'script_sha256': {p: sha(ROOT / p) for p in ('scripts/audit_query_fp8_native_cpu.py', 'scripts/audit_query_fp8_components.py')},
              'totals': totals, 'rows': rows,
              'limitations': ['One actual third B8 decode state per document and rank; not all natural generation steps.',
                              'Q representation only; different two-MMA accumulation requires independent GPU QK and attention checks.',
                              'Approximate FP8 decomposition, no claim of universal exact BF16 representation or measured performance.']}
    target = ROOT / 'results/platform/query_fp8_native_components_20261017.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(target.relative_to(ROOT)), **totals, 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
