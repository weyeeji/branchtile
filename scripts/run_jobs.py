"""Bounded independent GPU jobs with allocation-wall-time accounting.

Each job supplies argv (no shell), GPU IDs, category P1/P2 and timeout_seconds.
The parent Python and all child Python interpreters must be project venvs.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--ledger", default="results/job_ledger.jsonl")
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    jobs = json.loads(Path(a.manifest).read_text())
    seen = set()
    for j in jobs:
        assert j["category"] in ("P1", "P2")
        assert j["gpus"] and not seen.intersection(j["gpus"]), "Concurrent GPU overlap"
        seen.update(j["gpus"])
        exe = (ROOT / j["argv"][0]).absolute()
        assert str(exe).startswith(str(ROOT / ".venv")), "Use project virtualenv"
        assert j["timeout_seconds"] <= 3600
    ledger = ROOT / a.ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    def emit(record):
        with LOCK, ledger.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    def run(j):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, j["gpus"])),
                   OMP_NUM_THREADS="4", MAX_JOBS="4",
                   PATH=f"{ROOT}/.venv/bin:/home/qcjz/miniconda3/envs/sglang-vllm-m120/bin:/usr/local/cuda-13.2/bin:"+env["PATH"])
        env.update(j.get("env", {}))
        log = ROOT / "logs" / (j["name"] + ".log")
        log.parent.mkdir(exist_ok=True)
        base = {"name": j["name"], "category": j["category"], "gpus": j["gpus"],
                "argv": j["argv"], "uid": os.geteuid(), "log": str(log)}
        start = time.monotonic()
        with log.open("w") as stream:
            child = subprocess.Popen(j["argv"], cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            emit(dict(base, event="start", time_utc=now(), pid=child.pid))
            timed_out = False
            try:
                code = child.wait(timeout=j["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        seconds = time.monotonic() - start
        emit(dict(base, event="end", time_utc=now(), pid=child.pid, exit_code=code,
                  timed_out=timed_out, allocated_seconds=seconds,
                  gpu_hours=seconds*len(j["gpus"])/3600))
        return code
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        codes = list(pool.map(run, jobs))
    raise SystemExit(any(c != 0 for c in codes))


if __name__ == "__main__":
    main()
