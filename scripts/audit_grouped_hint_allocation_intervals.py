"""Audit recorded GPU reservation intersections; not a utilization measurement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/args.output;assert not output.exists()
    ledger_path=ROOT/'results/job_ledger.jsonl';ledger_bytes=ledger_path.read_bytes()
    ledger=[json.loads(line) for line in ledger_bytes.decode().splitlines()]
    now=datetime.now(timezone.utc);intervals=[]
    for start in ledger:
        if start.get('event')!='start':continue
        ends=[r for r in ledger if r.get('event')=='end' and r.get('name')==start['name'] and r.get('pid')==start.get('pid')]
        assert len(ends)<=1,(start['name'],start.get('pid'))
        intervals.append((start,ends[0] if ends else None,datetime.fromisoformat(start['time_utc']),
            datetime.fromisoformat(ends[0]['time_utc']) if ends else now))
    rows=[]
    plan=json.loads((ROOT/'results/grouped_hint_next8_production_launch_v1.json').read_text())
    for document in plan['documents']:
        if not document['context_feasible']:continue
        matches=[v for v in intervals if v[0]['name']==document['job_name']]
        assert len(matches)<=1
        if not matches:
            rows.append({'job_name':document['job_name'],'status':'not_started','overlaps':[]});continue
        start,end,t0,t1=matches[0];overlaps=[]
        for other,oend,u0,u1 in intervals:
            if other['name']==start['name'] and other.get('pid')==start.get('pid'):continue
            common=sorted(set(start['gpus'])&set(other.get('gpus',[])))
            lo,hi=max(t0,u0),min(t1,u1)
            if common and hi>lo:
                overlaps.append({'other_job':other['name'],'other_pid':other.get('pid'),'common_gpus':common,
                    'other_terminal_receipt_present':oend is not None,'start':lo.isoformat(),'end_or_audit_horizon':hi.isoformat(),
                    'seconds':(hi-lo).total_seconds()})
        rows.append({'job_name':start['name'],'status':'terminal' if end else 'running',
            'start_receipt':start,'terminal_receipt':end,'overlaps':overlaps})
    complete=all(r['status']=='terminal' for r in rows)
    assert complete or args.allow_incomplete,'Do not finalize before all six job terminals.'
    report={'status':'complete_recorded_allocation_audit' if complete else 'incomplete_recorded_allocation_audit',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ledger_sha256':hashlib.sha256(ledger_bytes).hexdigest(),'audit_time_utc':now.isoformat(),
        'jobs':rows,'limits':['Only recorded allocations are checked; unrecorded CUDA activity and CPU contention are not excluded.',
            'No reservation intersection is not a claim of stable clocks, power, thermal conditions, or interference-free timing.',
            'A running job without a terminal receipt is bounded at audit time and remains explicitly incomplete.',
            'PAT retrospective visibility exposure has a separate interval audit; this report does not replace that evidence.']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'jobs':len(rows),'overlaps':sum(len(r['overlaps']) for r in rows)},indent=2))


if __name__=='__main__':main()
