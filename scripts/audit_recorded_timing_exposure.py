"""CPU-only interval audit; absence of recorded overlap is not resource isolation."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guard', default='results/pat_build_visibility_guard_audit_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    guard_path = ROOT / args.guard
    ledger_path = ROOT / 'results/job_ledger.jsonl'
    output = ROOT / args.output
    assert not output.exists(), output
    guard = json.loads(guard_path.read_text())
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    selected = ('native_raw_pair_general_policy_', 'native_raw_pair_production_',
                'native_raw_pair_heldout_', 'gqa4_raw_pair_', 'gqa4_thinking_')
    rows = []
    for start in ledger:
        if start.get('event') != 'start' or not start.get('name', '').startswith(selected):
            continue
        ends = [r for r in ledger if r.get('event') == 'end' and
                r.get('name') == start['name'] and r.get('pid') == start.get('pid')]
        assert len(ends) <= 1, (start['name'], start.get('pid'))
        t0 = datetime.fromisoformat(start['time_utc'])
        t1 = datetime.fromisoformat(ends[0]['time_utc']) if ends else None
        if t1 is not None:
            assert t1 >= t0
        exposures = []
        for build in guard['builds']:
            a = datetime.fromisoformat(build['exposure_start'])
            b = datetime.fromisoformat(build['exposure_end'])
            assert b > a
            # If the job is still running, report only the intersection with the
            # already-ended exposure window, not an invented completion time.
            lo, hi = max(t0, a), min(t1, b) if t1 is not None else b
            if hi > lo:
                exposures.append({'build_receipt': build['receipt'],
                    'build_receipt_sha256': build['sha256'],
                    'overlap_start': lo.isoformat(), 'overlap_end': hi.isoformat(),
                    'overlap_seconds': (hi-lo).total_seconds()})
        rows.append({'name': start['name'], 'pid': start.get('pid'),
            'gpus': start.get('gpus'), 'start': start['time_utc'],
            'end': ends[0]['time_utc'] if ends else None,
            'ledger_exit_code': ends[0].get('exit_code') if ends else None,
            'ledger_timed_out': ends[0].get('timed_out') if ends else None,
            'status': 'recorded_exposure_overlap' if exposures else 'no_overlap_with_recorded_PAT_exposure',
            'exposures': exposures})
    result = {'status': 'recorded_interval_audit',
        'audit_source_sha256': sha(Path(__file__)),
        'guard_path': str(guard_path.relative_to(ROOT)), 'guard_sha256': sha(guard_path),
        'ledger_sha256_at_audit': sha(ledger_path), 'jobs': rows,
        'limits': [
            'Recorded exposure is conservative wall time with all eight GPUs visible; actual GPU activity was not measured.',
            'Overlap does not establish that a benchmark kernel executed, but overlapping performance needs a clean repeat.',
            'No overlap with these two intervals is not proof of exclusive devices or absence of other interference.',
            'Numerical qualification and complete-output equality remain separate from timing-environment validity.',
            'Running jobs have no fabricated terminal receipt; this audit captures only the ledger snapshot.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'jobs': len(rows), 'overlapping_jobs': [r['name'] for r in rows if r['exposures']]}, indent=2))


if __name__ == '__main__':
    main()
