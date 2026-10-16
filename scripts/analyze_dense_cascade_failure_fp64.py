"""CPU-only streamed FP64 adjudication of a preserved cascade pairwise failure."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def reference(payload, chunk, threads):
    import torch
    torch.set_num_threads(threads)
    assert not torch.cuda.is_initialized()
    cache = payload['cache']
    if cache.dtype == torch.uint8:
        assert payload.get('kv_semantic_dtype') in ('fp8_e4m3', 'float8_e4m3fn'), 'uint8 requires an explicit semantic dtype receipt'
        cache = cache.view(torch.float8_e4m3fn)
    assert cache.ndim == 4 and tuple(cache.shape[1:]) == (2, 16, 256)
    q = payload['query'].double().reshape(-1, 16, 128)
    tables = payload['block_tables'].long()
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float64)
    for row, length in enumerate(payload['seq_lens']):
        for head in range(2):
            query = q[row, head * 8:(head + 1) * 8]
            maximum = torch.full((8,), -torch.inf, dtype=torch.float64)
            normalizer = torch.zeros(8, dtype=torch.float64)
            numerator = torch.zeros((8, 128), dtype=torch.float64)
            for start in range(0, length, chunk):
                positions = torch.arange(start, min(start + chunk, length))
                values = cache[tables[row, positions // 16], head, positions % 16].double()
                scores = (query @ values[:, :128].T) * (128 ** -.5 * payload['k_scale'])
                updated = torch.maximum(maximum, scores.max(dim=1).values)
                old_scale = torch.exp(maximum - updated)
                probabilities = torch.exp(scores - updated[:, None])
                numerator = numerator * old_scale[:, None] + probabilities @ (values[:, 128:] * payload['v_scale'])
                normalizer = normalizer * old_scale + probabilities.sum(dim=1)
                maximum = updated
            out[row, head * 8:(head + 1) * 8] = numerator / normalizer[:, None]
            lse[row, head * 8:(head + 1) * 8] = maximum + normalizer.log()
    assert torch.isfinite(out).all() and not torch.cuda.is_initialized()
    return out, lse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly disable CUDA for this CPU audit'
    assert args.chunk > 0 and args.threads > 0
    out = Path(args.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    started = time.perf_counter()
    path = Path(args.snapshot)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['schema'] == 'dense_cascade_failure_v1'
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path = path.with_suffix('.json')
    if receipt_path.exists():
        assert json.loads(receipt_path.read_text())['sha256'] == source_sha
    expected, lse = reference(payload, args.chunk, args.threads)
    candidate = payload['candidate_o'].double().reshape_as(expected)
    native = payload['native_o'].double().reshape_as(expected)
    atol, rtol = payload['atol'], payload['rtol']
    pair_failed = ~torch.isclose(candidate, native, atol=atol, rtol=rtol)
    assert torch.equal(pair_failed, payload['failed_mask'].reshape_as(pair_failed))
    report = {'status': 'completed', 'snapshot': str(path), 'snapshot_sha256': source_sha,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'layer': payload['layer'], 'rank': payload['rank'], 'seq_lens': payload['seq_lens'],
        'chunk': args.chunk, 'threads': args.threads, 'atol': atol, 'rtol': rtol,
        'original_pairwise_gate_passed': False, 'pairwise_failed_values': int(pair_failed.sum()),
        'candidate_and_native': {}, 'failed_coordinates': [],
        'interpretation': 'Independent mathematical-reference diagnosis; original pairwise qualification remains failed. No threshold is changed. FP64 uses current Q and complete captured paged KV with actual scales.'}
    assert report['pairwise_failed_values'] > 0
    for name, value in [('candidate', candidate), ('native', native)]:
        error = (value - expected).abs()
        passed = torch.isclose(value, expected, atol=atol, rtol=rtol)
        report['candidate_and_native'][name] = {'passed_fp64': bool(passed.all()),
            'failed_fp64_values': int((~passed).sum()), 'max_abs_fp64': float(error.max()),
            'relative_l2_fp64': float(error.norm() / expected.norm()),
            'max_normalized_error': float((error / (atol + rtol * expected.abs())).max()),
            'failed_fp64_within_pairwise_failure': int((~passed & pair_failed).sum())}
    for coord in pair_failed.nonzero().tolist():
        idx = tuple(coord)
        c, n, e = [float(x[idx]) for x in (candidate, native, expected)]
        report['failed_coordinates'].append({'index': coord, 'candidate': c, 'native': n,
            'fp64': e, 'pairwise_tolerance': atol + rtol * abs(n),
            'fp64_tolerance': atol + rtol * abs(e), 'candidate_error': c - e, 'native_error': n - e})
    report['seconds'] = time.perf_counter() - started
    report['cuda_initialized'] = torch.cuda.is_initialized()
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'failed_coordinates'}, indent=2))


if __name__ == '__main__':
    main()
