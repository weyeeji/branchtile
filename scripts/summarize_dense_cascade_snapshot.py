"""CPU paired audit of same-input/history native versus actual cascade full graphs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    path = ROOT / args.input
    data = json.loads(path.read_text())
    assert data['status'] == 'completed'
    assert len(data['workers']) == 2
    assert sorted(w['rank'] for w in data['workers']) == [0, 1]
    for filename, value in data['source_sha256'].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == value, filename
    workers = []
    for receipt in sorted(data['workers'], key=lambda w:w['rank']):
        raw = Path(receipt['path'])
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == receipt['sha256']
        workers.append(json.loads(raw.read_text()))
    for worker in workers:
        assert worker['status'] == 'completed' and worker['final_native_inputs_tables_written_KV_hidden_bitwise_restored']
        assert worker['all_layers_qualified']
        assert [c['layer'] for c in worker['layer_checks']] == list(range(48))
        assert worker['output_differences']['checked_and_production_hidden_and_written_KV_bitwise_equal']
        for check in worker['layer_checks']:
            assert check['qualification_passed']
            if not check['same_Q_native_O_passed']:
                assert check['reference_computed'] and check['pair_failed_but_reference_passed']
                assert all(check[key]['passed'] and check[key]['max_normalized_tolerance'] <= 1 for key in ('candidate_vs_fp64','native_vs_fp64'))
        assert worker['plan_receipt']['actual_shared_physical_prefix_checked']
        assert not worker['plan_receipt']['tail_plan_topology_changed']
    assert workers[0]['positions'] == workers[1]['positions']
    assert workers[0]['input_tokens'] == workers[1]['input_tokens']
    indexed = []
    for worker in workers:
        rows = {(r['cache'], r['block'], r['direction']): r for r in worker['measurements']}
        assert len(rows) == len(worker['measurements']) == data['args']['blocks'] * 4
        assert all(set(row['cuda_us']) == {'native', 'cascade', 'native_AA'} for row in rows.values())
        assert all(math.isfinite(value) and value > 0 for row in rows.values() for value in row['cuda_us'].values())
        indexed.append(rows)
    assert indexed[0].keys() == indexed[1].keys()
    output = {'input': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'qualification': '48 layer same-Q screen, failed screens adjudicated for both arms against FP64 at unchanged tolerances; checked/production graph identity and native restoration.',
              'positions': workers[0]['positions'], 'statistics': {},
              'output_differences_per_rank': [w['output_differences'] for w in workers],
              'plan_receipts_per_rank': [w['plan_receipt'] for w in workers],
              'scope': 'One frozen real input/history; both methods actually compute full attention and complete model graph. Candidate output and subsequent routing can differ.',
              'limitations': ['Per-pair maximum rank event approximates slowest TP graph duration, not request walltime.',
                              'Excludes prefill, dynamic per-step planning and graph initialization; those remain charged in separate generation experiments.',
                              'Both methods write current-token KV before reading; final native hidden and written slots bitwise restored.',
                              'No claim of equal internal MoE routes or natural task accuracy; candidate-vs-native logits and argmax differences retained.',
                              'Artificial flush is a diagnostic cache condition.']}
    for cache in ('model_warm', 'artificial_cold'):
        chosen = sorted(k for k in indexed[0] if k[0] == cache)
        methods = ('native', 'cascade', 'native_AA')
        scopes = {'rank0': {k: indexed[0][k]['cuda_us'] for k in chosen},
                  'rank1': {k: indexed[1][k]['cuda_us'] for k in chosen},
                  'max_rank_event': {k: {m: max(idx[k]['cuda_us'][m] for idx in indexed) for m in methods} for k in chosen}}
        summaries = {}
        for name, rows in scopes.items():
            medians = {m: statistics.median(r[m] for r in rows.values()) for m in methods}
            blocks = sorted({k[1] for k in rows})
            draws = np.random.default_rng(811).integers(0, len(blocks), (10000, len(blocks)))
            ratios = {}
            for numerator, denominator in (('native', 'cascade'), ('native', 'native_AA')):
                values = np.array([[rows[k][numerator] / rows[k][denominator] for k in rows if k[1] == block] for block in blocks])
                ratios[numerator + '_over_' + denominator] = {'median': float(np.median(values)),
                    'bootstrap95': np.quantile(np.median(values[draws], axis=(1, 2)), [.025, .975]).tolist()}
            summaries[name] = {'median_cuda_us': medians, 'paired_ratios': ratios,
                               'ratio_of_medians_removed_fraction': 1 - medians['cascade'] / medians['native']}
        output['statistics'][cache] = summaries
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['statistics'], indent=2))


if __name__ == '__main__':
    main()
