"""Collect corrected observer replays without replacing historical variants."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    names = ['configs/bench_suffix_v2_main_wave0.json','configs/bench_suffix_v2_main_wave1.json','configs/bench_suffix_v2_b8.json']
    rows, pending = [], []
    for name in names:
        for job in read(name):
            argv = job['argv']
            result_path = Path(argv[argv.index('--output')+1])
            if not (ROOT/result_path).exists():
                pending.append(str(result_path)); continue
            result = read(result_path)
            capture = Path(argv[argv.index('--capture')+1])
            meta = read(capture/'metadata.json')
            assert meta['export_semantics_version'] == 2
            assert result['provenance']['capture'] == str(capture)
            assert result['provenance']['document_sha256'] == meta['document_sha256']
            assert result['provenance']['ready_lifetime_measured']
            row = dict(path=str(result_path), sha256=hashlib.sha256((ROOT/result_path).read_bytes()).hexdigest(),
                capture=str(capture), export_metadata_sha256=hashlib.sha256((ROOT/capture/'metadata.json').read_bytes()).hexdigest(),
                B=meta['B'], budget=int(argv[argv.index('--budget')+1]),
                qualified_configurations=sum(t['status']=='ok' for t in result['tuning']),
                failed_configurations=[t for t in result['tuning'] if t['status']!='ok'],
                summary=result['summary'])
            if meta['B'] <= 4:
                assert result['tuning_complete']
                expected = (5,3,85) if meta['B'] == 3 else (15,15,255)
                assert (len(result['partitions']),len(result['hierarchies']),len(result['tuning'])) == expected
                assert len({json.dumps(t['partition']) for t in result['tuning'] if t['status']=='ok' and 'partition' in t}) == expected[0]
                assert len({json.dumps(t['tree']) for t in result['tuning'] if t['status']=='ok' and 'tree' in t}) == expected[1]
                row.update(fixed_partitions=expected[0],exact_hierarchies=expected[1],
                    complete_strong_control_grid=True)
            else:
                assert meta['B'] == 8 and len(result['tuning']) == 34
                row.update(complete_strong_control_grid=False,
                    limitation='Reject-only B8: native independent and union included; all partitions/hierarchies not enumerated on GPU.')
            rows.append(row)
    report = dict(scope='Specified suffix-only observer, corrected fixed-mask native/control replays',
        historical_variant='results/selector_export_semantics_correction.json',
        status='complete' if not pending else 'pending', pending=pending, rows=rows,
        qualified_configurations=sum(r['qualified_configurations'] for r in rows),
        failed_configurations=sum(len(r['failed_configurations']) for r in rows),
        limitations=['All replays use dense-trajectory actual same-forward Q and immutable source KV; not sparse task quality or production cost.',
                    'Operator tuning is per-capture development search, not a deployed predictor.',
                    'B8 extends the original scope and lacks exhaustive GPU partition/hierarchy baselines; losses to included controls only reject this implementation.'])
    (ROOT/'results/suffix_v2_replay/summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'},indent=2))


if __name__ == '__main__':
    main()
