"""CPU paired audit of qualified fixed-state cached-O interventions."""
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
    workers = sorted(data['workers'], key=lambda w: w['rank'])
    for worker in workers:
        assert worker['status'] == 'completed' and worker['final_inputs_tables_KV_hidden_unchanged']
        assert len(worker['layer_checks']) == 48
        assert all(c['q_bitwise_equal'] and c['o_bitwise_equal'] for c in worker['layer_checks'])
        assert len(worker['checks']) == 2
        for check in worker['checks']:
            assert check['hidden_bitwise_equal'] and check['local_vocab_logits_bitwise_equal']
            assert check['written_KV_slots_bitwise_equal_all48']
            assert check['gathered_logits_bitwise_equal'] is True if worker['rank'] == 0 else check['gathered_logits_bitwise_equal'] in (True, None)
    assert workers[0]['positions'] == workers[1]['positions']
    assert workers[0]['input_tokens'] == workers[1]['input_tokens']
    indexed = []
    for worker in workers:
        rows = {(r['cache'], r['block'], r['direction']): r for r in worker['measurements']}
        assert len(rows) == len(worker['measurements']) == data['args']['blocks'] * 4
        assert all(set(row['cuda_us']) == {'native', 'cached_o', 'native_AA'} for row in rows.values())
        assert all(math.isfinite(value) and value > 0 for row in rows.values() for value in row['cuda_us'].values())
        indexed.append(rows)
    assert indexed[0].keys() == indexed[1].keys()
    output = {'input': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'qualification': '48 layer Q/O, all hidden outputs, local vocabulary logits, rank0 gathered logits and written KV slots bitwise equal; fixed inputs/tables unchanged.',
              'positions': workers[0]['positions'], 'statistics': {},
              'scope': 'One frozen real state; cached O is unavailable for future tokens. No actual serving speedup or quality claim.',
              'limitations': ['Per-pair maximum of rank CUDA event durations approximates slowest TP graph interval; it is not a separately measured scheduler or request wall time.',
                              'All model operations including KV write remain; cached O replaces attention using a copy. Dependency structure changes.',
                              'Graph build, witness capture, qualification and warmups excluded from paired graph timings; their costs make this unavailable as a serving method.',
                              'Warm versus artificial flush are diagnostic cache conditions, not separate natural workloads.']}
    for cache in ('model_warm', 'artificial_cold'):
        chosen = sorted(k for k in indexed[0] if k[0] == cache)
        methods = ('native', 'cached_o', 'native_AA')
        scopes = {'rank0': {k: indexed[0][k]['cuda_us'] for k in chosen},
                  'rank1': {k: indexed[1][k]['cuda_us'] for k in chosen},
                  'max_rank_event': {k: {m: max(idx[k]['cuda_us'][m] for idx in indexed) for m in methods} for k in chosen}}
        summaries = {}
        for name, rows in scopes.items():
            medians = {m: statistics.median(r[m] for r in rows.values()) for m in methods}
            blocks = sorted({k[1] for k in rows})
            draws = np.random.default_rng(811).integers(0, len(blocks), (10000, len(blocks)))
            ratios = {}
            for numerator, denominator in (('native', 'cached_o'), ('native', 'native_AA')):
                values = np.array([[rows[k][numerator] / rows[k][denominator] for k in rows if k[1] == block] for block in blocks])
                ratios[numerator + '_over_' + denominator] = {'median': float(np.median(values)),
                    'bootstrap95': np.quantile(np.median(values[draws], axis=(1, 2)), [.025, .975]).tolist()}
            summaries[name] = {'median_cuda_us': medians, 'paired_ratios': ratios,
                               'ratio_of_medians_removed_fraction': 1 - medians['cached_o'] / medians['native']}
        output['statistics'][cache] = summaries
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['statistics'], indent=2))


if __name__ == '__main__':
    main()
