"""Bound what the import-incident receipt can establish; never invent a start time."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    event_path = ROOT / 'results/cpu_visible_cuda_import_incident_20261017_v1.json'
    incident = json.loads(event_path.read_text())
    assert incident['exact_start_end_UTC'] is None
    completed_before = datetime.fromisoformat(incident['recorded_UTC'])
    assert datetime.now(timezone.utc) >= completed_before
    reported_pid = incident['affected_reference_run_from_root']['PID']
    ledger_path = ROOT / 'results/job_ledger.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    rows = []
    plans = [ROOT / 'results/grouped_hint_next8_production_launch_v1.json'] + [
        ROOT / f'results/grouped_hint_next8_repeat{i}_plan_v1.json' for i in (1, 2)]
    for index, plan_path in enumerate(plans):
        for document in json.loads(plan_path.read_text())['documents']:
            row = {'repeat_index': index, 'evaluation_index': document['evaluation_index'],
                'job_name': document.get('job_name'), 'context_feasible': document['context_feasible']}
            if not document['context_feasible']:
                row['temporal_evidence'] = 'context_failure_no_GPU_job'
                rows.append(row)
                continue
            starts = [r for r in ledger if r.get('event') == 'start' and r.get('name') == document['job_name']]
            assert len(starts) <= 1
            if not starts:
                row['temporal_evidence'] = 'not_started_at_audit_after_incident_ended'
                rows.append(row)
                continue
            start = starts[0]
            ends = [r for r in ledger if r.get('event') == 'end' and r.get('name') == start['name'] and r.get('pid') == start.get('pid')]
            assert len(ends) <= 1
            row.update(start_utc=start['time_utc'], end_utc=ends[0]['time_utc'] if ends else None,
                pid=start.get('pid'), allocated_gpus=start['gpus'])
            if start.get('pid') == reported_pid:
                row['temporal_evidence'] = 'possible_overlap_reported_by_root_no_exact_interval'
            elif datetime.fromisoformat(start['time_utc']) >= completed_before:
                row['temporal_evidence'] = 'started_after_this_incident_had_ended'
            else:
                row['temporal_evidence'] = 'not_determinable_without_incident_start_time'
            rows.append(row)
    assert len(rows) == 24
    return {'status': 'incident_uncertainty_retained', 'analysis_sha256': sha(Path(__file__)),
        'incident_path': str(event_path.relative_to(ROOT)), 'incident_sha256': sha(event_path),
        'ledger_sha256': sha(ledger_path), 'incident_completed_before_utc': incident['recorded_UTC'],
        'exact_incident_start_end_utc': None, 'exact_incident_GPU_UUID': None,
        'rows': rows, 'limits': [
            'Receipt creation bounds the end of these three completed commands, not their start.',
            'Elapsed tool times cannot be turned into UTC intervals by subtracting them from receipt creation.',
            'Unknown temporal evidence neither establishes contamination nor proves its absence.',
            'Actual GPU work, memory occupancy and performance impact were not traced; ordinary AA does not prove no impact.',
            'The root-reported PID is a specific possible overlap, not an exhaustive affected-run set.',
            'No planned value is removed or replaced. Additional clean confirmation is supplemental.',
            'Starting after this incident only excludes this incident; it does not establish overall interference-free execution.',
            'Recorded reservation non-overlap is separate evidence and cannot exclude unrecorded CUDA imports.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT / args.output
    assert not output.exists()
    result = audit()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'rows': len(result['rows'])}))


if __name__ == '__main__':
    main()
