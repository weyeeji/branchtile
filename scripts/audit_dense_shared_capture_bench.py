"""CPU-only audit of the fixed-real-snapshot dense cascade operator experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    assert not output.exists()
    report = {'status': 'audited', 'cells': [], 'scope': 'Known shared-prefix cascade operator control, not a novel mechanism or an end-to-end result.',
        'limitations': ['Snapshots cover two documents and selected layers/ranks; no population confidence claim.',
            'Paired bootstrap intervals concern timing noise only. Cold event quantization can produce degenerate intervals.',
            'Static operator timing excludes host preparation, first graph capture and initialization; full-model runs must charge these.',
            'KV data and original address strides are restored offline; active physical page identifiers are compacted.',
            'B1 is slower and supports native fallback; surviving-batch transition costs require full-model measurement.']}
    for path in sorted(Path(args.results).glob('*.json')):
        d = json.loads(path.read_text())
        assert d['status'] == 'completed', path
        assert all(c['passed'] for c in d['checks'].values()), path
        for name, sha in d['source_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
        assert d['candidate_prepare']['actual_shared_physical_prefix_checked']
        assert not d['candidate_prepare']['source_payload_copy']
        assert d['candidate_prepare']['prefix_plan_calls_this_prepare'] == 0
        assert d['candidate_prepare']['tail_plan_calls_this_prepare'] == 1
        blocks = d['args']['blocks']
        assert len(d['measurements']) == 4 * blocks
        for cache in ('hot', 'cold'):
            for block in range(blocks):
                rows = [x for x in d['measurements'] if x['cache'] == cache and x['block'] == block]
                assert {x['direction'] for x in rows} == {'random', 'reverse'} and len(rows) == 2
            rows = [x for x in d['measurements'] if x['cache'] == cache]
            for method, median in d['summary'][cache]['median_us'].items():
                assert statistics.median(x['us'][method] for x in rows) == median
        report['cells'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'batch': d['args']['batch'], 'capture': d['args']['capture'], 'checks': d['checks'],
            'summary': d['summary'], 'prepare_host_us': d['candidate_prepare']['complete_prepare_host_seconds'] * 1e6,
            'constructor_and_first_prepare_seconds': d['candidate_constructor_and_first_prepare_seconds']})
    assert report['cells']
    b8 = [x for x in report['cells'] if x['batch'] == 8]
    report['b8_cells'] = len(b8)
    for cache in ('hot', 'cold'):
        speeds = [x['summary'][cache]['paired_ratios']['native_xqa_O_only_over_shared_prefix_cascade']['median'] for x in b8]
        if speeds:
            report[cache + '_b8_speedup_range'] = [min(speeds), max(speeds)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cells', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
