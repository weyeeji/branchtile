"""Independent CPU recomputation of fixed O-only output-contract diagnostics."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted((ROOT / 'results/operator_output_contract_wave27').glob('*.json'))
    assert len(paths) == 8
    ledger = [json.loads(line) for line in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines()]
    pairs = [('native_O_only', 'owner_O_with_internal_lnLSE'),
             ('native_O_lnLSE', 'owner_O_with_internal_lnLSE'),
             ('native_O_lnLSE', 'native_O_only'),
             ('native_O_only', 'native_O_only_AA')]
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        assert data['status'] == 'completed' and data['uid'] != 0
        for filename, digest in data['source_sha256'].items():
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
        terminal = [e for e in ledger if e['name'] == path.stem and e['event'] == 'end']
        assert len(terminal) == 1 and terminal[0]['exit_code'] == 0 and not terminal[0]['timed_out']
        assert all(c.get('passed', c.get('graph_O_passed', False)) for c in data['checks'].values())
        blocks = data['args']['blocks']
        assert len(data['measurements']) == blocks * 4
        records = {}
        for cache in ('hot', 'cold'):
            raw = [r for r in data['measurements'] if r['cache'] == cache]
            assert len(raw) == blocks * 2
            assert {(r['block'], r['direction']) for r in raw} == {(b, d) for b in range(blocks) for d in ('random', 'reverse')}
            medians = {method: statistics.median(r['us'][method] for r in raw) for method in raw[0]['us']}
            assert medians == data['summary'][cache]['median_us']
            draws = np.random.default_rng(71).integers(0, blocks, (10000, blocks))
            paired = {}
            for num, den in pairs:
                values = np.array([[r['us'][num] / r['us'][den] for r in raw if r['block'] == b] for b in range(blocks)])
                ci = np.quantile(np.median(values[draws], axis=(1, 2)), [.025, .975])
                result = {'median': float(np.median(values)), 'bootstrap95': ci.tolist()}
                key = num + '_over_' + den
                assert result == data['summary'][cache]['paired_ratios'][key]
                paired[key] = result
            records[cache] = {'median_us': medians, 'paired_ratios': paired,
                              'ln_conversion_increment_us': medians['native_O_lnLSE'] - medians['native_O_log2LSE'],
                              'raw_lse_increment_us': medians['native_O_log2LSE'] - medians['native_O_only'],
                              'unique_native_O_only_us': sorted(set(r['us']['native_O_only'] for r in raw))}
        rows.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'provenance': data['provenance'], 'results': records,
                     'qualified_contracts': list(data['checks']), 'flush_allocated_bytes': data['flush_allocated_bytes'],
                     'ledger_terminal': terminal[0], 'measurement_rows': len(data['measurements'])})
    native_key = 'native_O_only_over_owner_O_with_internal_lnLSE'
    report = {'schema': 1, 'status': 'passed', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'jobs': rows, 'completed_jobs': len(rows), 'paired_statistics_recomputed': len(rows) * 2 * len(pairs),
              'measurement_rows': sum(r['measurement_rows'] for r in rows),
              'hot_native_O_only_strictly_faster_cells': sum(r['results']['hot']['paired_ratios'][native_key]['bootstrap95'][1] < 1 for r in rows),
              'cold_native_O_only_strictly_faster_cells': sum(r['results']['cold']['paired_ratios'][native_key]['bootstrap95'][1] < 1 for r in rows),
              'limitations': ['Fixed native tensor-core=True and owner C128/N64, not oracle selection.',
                              'Same page-one sparse captured inputs, not dense deployed XQA; two KV heads from one correlated development document.',
                              'Preparing metadata and constructing graphs are excluded from operator timings.',
                              'Owner still computes LSE internally for its segmented algorithm; native O-only avoids application-unneeded final LSE conversion.',
                              'Cold CUDA-event quantization can collapse bootstrap intervals; no equivalence claim from CI [1,1].',
                              'This tests the current fixed owner implementation; it does not establish impossibility of future shared-attention optimizations.']}
    out = ROOT / 'results/operator_output_contract_wave27_audit_v1'
    out.mkdir(exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'completed_jobs', 'paired_statistics_recomputed', 'measurement_rows', 'hot_native_O_only_strictly_faster_cells', 'cold_native_O_only_strictly_faster_cells')}))


if __name__ == '__main__':
    main()
