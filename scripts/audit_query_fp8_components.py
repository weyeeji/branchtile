"""CPU-only representation screen of a candidate, not GPU or attention quality."""
import collections
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component(x):
    scale = torch.pow(2., torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True) / 448.))).clamp_min(2. ** -126)
    return (x / scale).to(torch.float8_e4m3fn).float() * scale, scale


def main():
    torch.set_num_threads(2)
    rows = []
    for directory in ('main30b_tp2_train1_b8_256_export_retry1', 'main30b_tp2_train7_b8_256_export'):
        for path in sorted((ROOT / 'results/captures' / directory).rglob('decode*.pt')):
            obj = torch.load(path, map_location='cpu', weights_only=True)
            q = obj['query']
            assert q.dtype == torch.bfloat16 and q.shape[-1] == 128
            x = q.float().reshape(-1, 128)
            assert torch.isfinite(x).all()
            hi, s1 = component(x)
            lo, s2 = component(x - hi)
            reconstructed = hi + lo
            err = x - reconstructed
            # Equality is numeric, not signed-zero bitwise equivalence.
            exact_rows = (err == 0).all(-1)
            rows.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                         'query_vectors': len(x), 'exact_numeric_vectors': int(exact_rows.sum()),
                         'nonexact_values': int((err != 0).sum()), 'max_abs_error': err.abs().max().item(),
                         'relative_l2': (torch.linalg.vector_norm(err) / torch.linalg.vector_norm(x)).item(),
                         'query_abs_max': x.abs().max().item(),
                         'scale_high_log2_histogram': dict(collections.Counter(torch.log2(s1).view(-1).int().tolist())),
                         'scale_low_log2_histogram': dict(collections.Counter(torch.log2(s2).view(-1).int().tolist()))})
    assert rows and not torch.cuda.is_initialized()
    totals = {'files': len(rows), 'query_vectors': sum(r['query_vectors'] for r in rows),
              'exact_numeric_vectors': sum(r['exact_numeric_vectors'] for r in rows),
              'nonexact_values': sum(r['nonexact_values'] for r in rows),
              'max_abs_error': max(r['max_abs_error'] for r in rows),
              'max_file_relative_l2': max(r['relative_l2'] for r in rows)}
    result = {'schema': 1, 'status': 'CPU_representation_screen_only', 'script_sha256': sha(Path(__file__)),
              'recipe': 'FP32 x from BF16 Q; s1=2^ceil(log2(maxabs(x)/448)); hi=s1*e4m3(x/s1); repeat per-row on residual for lo.',
              'totals': totals, 'rows': rows,
              'limitations': ['Historical BF16-KV captures of two real B8 training groups, not new native FP8 trajectories.',
                              'Numeric reconstruction equality does not imply matching floating-point QK accumulation, attention output, logits, or natural task quality.',
                              'Two FP8 GEMMs have different accumulation order; this test has no GPU timing or kernel implementation.',
                              'Representation need not be exact for arbitrary BF16 vectors; runtime range/fallback cost remains unresolved.',
                              'Multi-component low-precision products are prior art; only a specific measured attention dataflow could provide a new contribution.']}
    target = ROOT / 'results/platform/query_fp8_components_20261017.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(target.relative_to(ROOT)), **totals, 'cuda_initialized': torch.cuda.is_initialized()}))


if __name__ == '__main__':
    main()
