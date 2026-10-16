"""Read-only independent-grid evaluation of the frozen dispatch rule; never refit."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vllm_native_raw_pair_dispatch import eligible_from_current_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='results/xqa_raw_dispatch_validation_v1')
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    spec_path = ROOT / 'configs/xqa_raw_dispatch_validation_v1.json'
    spec = json.loads(spec_path.read_text())
    assert sha(ROOT / 'results/xqa_raw_dispatch_policy_v1/policy.json') == spec['frozen_policy_sha256']
    expected = set(itertools.product(spec['prefix_tokens'], spec['additional_private_tokens'], spec['batches']))
    directory = ROOT / args.directory
    rows, seen, receipts = [], set(), []
    for shard in range(spec['shards']):
        path = directory / f'shard{shard}.json'
        report = json.loads(path.read_text())
        assert report['status'] in ('completed', 'completed_with_failed_cells')
        assert report['spec'] == spec and report['spec_sha256'] == sha(spec_path)
        assert report['shard'] == shard
        for source, digest in report['source_sha256'].items():
            assert sha(ROOT / source) == digest, source
        receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        for case in report['cases']:
            key = (case['prefix_tokens'], case['additional_private_tokens'], case['batch'])
            assert key in expected and key not in seen
            seen.add(key)
            lengths = [key[0] + n for n in case['private_lengths']]
            eligible, reason = eligible_from_current_state(key[2], key[0], lengths)
            timing = case.get('timing', {})
            speed = timing.get('native_over_candidate')
            aa = timing.get('AA')
            rows.append({'prefix': key[0], 'additional_private': key[1], 'batch': key[2],
                         'eligible': eligible, 'fallback_reason': reason, 'status': case['status'],
                         'speed': speed, 'native_AA': aa,
                         'AA_within_2percent': aa is not None and .98 <= aa['paired_median_ratio'] <= 1.02})
    assert seen == expected
    eligible_rows = [r for r in rows if r['eligible']]
    result = {'status': 'evaluated_without_refit', 'spec_sha256': sha(spec_path),
              'policy_sha256': spec['frozen_policy_sha256'], 'receipts': receipts,
              'all_cells': len(rows), 'eligible_cells': len(eligible_rows),
              'all_cell_correctness_failures': [r for r in rows if r['status'] != 'completed'],
              'eligible_correctness_failures': [r for r in eligible_rows if r['status'] != 'completed'],
              'eligible_median_regressions': [r for r in eligible_rows if r['speed'] and r['speed']['paired_median_ratio'] < 1],
              'eligible_CI_not_above_one': [r for r in eligible_rows if r['speed'] and r['speed']['bootstrap_95ci'][0] <= 1],
              'AA_outside_2percent': [r for r in rows if not r['AA_within_2percent']],
              'rows': rows,
              'limits': ['Synthetic hot complete-call graphs, not whole-model or natural workload evidence.',
                         'No parameter changes or refitting; all prespecified cells retained.',
                         'Native-fallback cells time the candidate only as a diagnostic; production policy would use native.',
                         'Per-cell bootstrap intervals describe timing blocks, not workload-population confidence.']}
    (directory / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'receipts', 'limits')}))


if __name__ == '__main__':
    main()
