"""Cross-manifest GPU exclusion around the immutable allocation ledger runner.

Locks are nonblocking. Existing legacy jobs are recognized from their live
ledger PID before any new process starts. No CUDA initialization is required.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def active_conflicts(gpus, ledger):
    active={}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            event=json.loads(line);key=(event['name'],event.get('pid'))
            if event['event']=='start':active[key]=event
            elif event['event']=='end':active.pop(key,None)
    return [event for event in active.values()
            if set(event['gpus'])&set(gpus) and event.get('pid') is not None
            and Path('/proc',str(event['pid'])).exists()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');a=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    jobs=json.loads(Path(a.manifest).read_text());gpus=[g for job in jobs for g in job['gpus']]
    assert len(gpus)==len(set(gpus)), 'One manifest represents concurrent jobs; use separate manifests for a sequence'
    folder=ROOT/'build/gpu_reservations';folder.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            handle=stack.enter_context((folder/f'gpu{gpu}.lock').open('a'))
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit(f'GPU {gpu} is reserved by another task; nothing launched')
        conflicts=active_conflicts(gpus,ROOT/'results/job_ledger.jsonl')
        if conflicts:
            raise SystemExit('Live GPU allocation conflict; nothing launched: '+json.dumps([
                {'name':e['name'],'pid':e['pid'],'gpus':e['gpus']} for e in conflicts]))
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_jobs.py'),a.manifest],cwd=ROOT)
        raise SystemExit(result.returncode)


if __name__=='__main__':main()
