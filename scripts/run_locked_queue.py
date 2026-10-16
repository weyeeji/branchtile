"""Run an explicit immutable job sequence after a ledgered allocation completes.

This CPU-only coordinator never reserves GPUs while waiting. Each actual job
uses the normal exclusion wrapper and accounting ledger. A failure stops the
sequence for inspection; unstarted experiments are not silently skipped.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after-name', required=True)
    p.add_argument('--after-pid', type=int, required=True)
    p.add_argument('--wait-seconds', type=int, default=3600)
    p.add_argument('manifests', nargs='+')
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    root = Path(__file__).resolve().parents[1]
    frozen = {name: Path(name).read_bytes() for name in a.manifests}
    deadline = time.monotonic() + a.wait_seconds
    while True:
        events = [json.loads(line) for line in (root/'results/job_ledger.jsonl').read_text().splitlines()]
        ends = [e for e in events if e.get('event') == 'end' and e.get('name') == a.after_name and e.get('pid') == a.after_pid]
        if ends:
            assert len(ends) == 1 and ends[0]['exit_code'] == 0, ends
            break
        assert time.monotonic() < deadline, 'Dependency has no terminal ledger entry; nothing launched'
        time.sleep(5)
    # Let the dependency's parent finish releasing its advisory locks.
    time.sleep(1)
    for name, content in frozen.items():
        assert Path(name).read_bytes() == content, f'Queued manifest changed: {name}'
        subprocess.run([sys.executable, str(root/'scripts/run_jobs_locked.py'), name], cwd=root, check=True)


if __name__ == '__main__':
    main()
