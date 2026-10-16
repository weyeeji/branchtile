"""Audit the four frozen B1..5 direct-path controls; do not select a policy."""
import hashlib
import json
from pathlib import Path


def main():
    files = sorted(Path('results/xqa_raw_direct_small_v1').glob('*train*.json'))
    assert len(files) == 4
    rows = []
    for path in files:
        data = json.loads(path.read_text())
        assert data['status'] == 'qualified_and_measured'
        assert data['build_identity'] == 'c28391a037f29141'
        for source, expected in data['source_sha256'].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
        for case in data['cases']:
            assert all(value is True for value in case['checks'].values())
            rows.append({
                'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'batch': case['batch'], 'prefix': case['prefix'],
                'extra_tail': data['args']['extra_tail'], 'captured_rows': case['rows'],
                'original_S': case['S'],
                'hot_native_over_candidate': case['hot']['native_over_candidate'],
                'hot_AA': case['hot']['AA'], 'numerical_checks': case['checks']})
    memcheck = Path('logs/raw_direct_small_b1_b3_memcheck_v1.memcheck.log')
    assert 'ERROR SUMMARY: 0 errors' in memcheck.read_text()
    report = {
        'status': 'four_complete_controls_verified', 'rows': rows,
        'memcheck': {'path': str(memcheck), 'sha256': hashlib.sha256(memcheck.read_bytes()).hexdigest(),
                     'scope': 'B1 and B3, actual train1 snapshot; instrumented timing excluded'},
        'scope': 'Two actual AR captures, each initial and with synthetic private KV extension. '
                 'B1..5 are snapshot subsets, not actual future survivor states. All force candidate; '
                 'old copied-control falls back natively for B1..4.',
        'decision': 'Keep production B6..21 unchanged. B1..3 lose all hot cells; B4 has small '
                    'initial gains and loses both extended-tail cells. B5 is positive but marginal '
                    'in one extended-tail case; no automatic scope expansion from two snapshots.',
        'limitations': ['Within-run paired intervals, not cross-document confidence.',
                        'Synthetic tail growth leaves captured Q unchanged.',
                        'Only hot graph timing here; no cold-cache or end-to-end claim.']}
    Path('results/xqa_raw_direct_small_analysis_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cells': len(rows)}))


if __name__ == '__main__':
    main()
