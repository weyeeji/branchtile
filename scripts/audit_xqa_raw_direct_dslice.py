"""CPU-only independent numerical/resource/paired-timing audit of bounded D-slice."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def paired(blocks, numerator, denominator):
    values = [b[numerator] / b[denominator] for b in blocks]
    rng = random.Random(20261017)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return {'paired_median_ratio': statistics.median(values), 'bootstrap_95ci': [boot[49], boot[1949]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(a.output)
    assert not output.exists()
    report = {'status': 'audited', 'cases': [], 'development_survivor': True,
              'scope': 'Only bounded GQA8 B8 standalone complete-call evidence; no whole-model claim.'}
    for name in a.inputs:
        path = Path(name)
        r = json.loads(path.read_text())
        row = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'status': r['status'], 'identity': r.get('build_identity')}
        assert r['status'] in ('failed', 'qualified_and_measured'), r['status']
        row['source_hashes_match'] = all(hashlib.sha256((root / k).read_bytes()).hexdigest() == v
                                        for k, v in r['source_sha256'].items())
        assert row['source_hashes_match']
        if r['status'] == 'failed':
            row['error'] = r.get('error')
            row['saved_checks'] = [c.get('checks') for c in r['cases']]
            report['development_survivor'] = False
            report['cases'].append(row)
            continue
        assert len(r['cases']) == 1 and r['cases'][0]['batch'] == 8
        c = r['cases'][0]
        assert all(c['checks'].values())
        assert set(c['raw_prefix_mismatch_fields'].values()) == {0}
        assert all(all(v for k, v in cut.items() if k != 'cut_tiles') for cut in c['cut_checks'])
        row['checks'] = c['checks']
        row['cut_checks'] = c['cut_checks']
        row['timing'] = {}
        for regime in ('hot', 'cold'):
            t = c[regime]
            assert len(t['blocks_ms']) == 30
            assert all(set(b) == {'native', 'native_AA', 'candidate', 'direct_control'} for b in t['blocks_ms'])
            ratios = {'direct_over_dslice': paired(t['blocks_ms'], 'direct_control', 'candidate'),
                      'native_over_candidate': paired(t['blocks_ms'], 'native', 'candidate'),
                      'AA': paired(t['blocks_ms'], 'native', 'native_AA')}
            for key, value in ratios.items():
                assert value == t[key], (key, value, t[key])
            row['timing'][regime] = {'median_us': {k: 1000 * statistics.median(b[k] for b in t['blocks_ms']) for k in t['median_ms']}, **ratios}
        row['development_survivor'] = (row['timing']['hot']['direct_over_dslice']['bootstrap_95ci'][0] > 1.02
                                      and row['timing']['cold']['direct_over_dslice']['bootstrap_95ci'][0] >= .98)
        report['development_survivor'] &= row['development_survivor']
        trace = Path(c['trace'])
        events = json.loads(trace.read_text())['traceEvents']
        scopes = [e for e in events if e.get('name', '').startswith('dslice_arm_')
                  and e.get('ph') == 'X' and e.get('cat') == 'user_annotation']
        assert len(scopes) == 4
        row['trace'] = {'path': str(trace), 'sha256': hashlib.sha256(trace.read_bytes()).hexdigest(), 'arms': {}}
        for scope in scopes:
            arm = scope['name'].removeprefix('dslice_arm_')
            kernels = [e for e in events if e.get('cat') == 'kernel' and scope['ts'] <= e['ts'] < scope['ts'] + scope['dur']]
            expected = 1 if arm in ('native', 'native_AA') else 2
            assert len(kernels) == expected, (arm, len(kernels), expected)
            assert all('kernel_mha(' in e['name'] and e['args'].get('graph node id', 0) for e in kernels)
            grids = [e['args']['grid'] for e in kernels]
            native_grid = c['grids']['native']
            assert grids == ([native_grid] if expected == 1 else [c['grids']['dslice_prefix' if arm == 'candidate' else 'direct_prefix'], native_grid])
            row['trace']['arms'][arm] = [{'grid': e['args']['grid'], 'registers_per_thread': e['args']['registers per thread'],
                                         'shared_bytes': e['args']['shared memory'], 'profile_duration_us': e['dur']} for e in kernels]
        report['cases'].append(row)
    report['decision'] = ('Bounded operator survivor; memcheck/regression/broader evidence required before integration.'
                          if report['development_survivor'] else 'Close current D-slice variant; do not extend to quad or widen tile search.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'decision': report['decision'], 'output': str(output)}))


if __name__ == '__main__':
    main()
