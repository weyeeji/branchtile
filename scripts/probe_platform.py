"""User-only GPU and optional two-rank NCCL qualification; run in project venv."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() != 0, "Root execution forbidden"
    assert sys.prefix != sys.base_prefix, "Use a virtual environment"
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    p = torch.cuda.get_device_properties(rank)
    result = {
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "uid": os.geteuid(), "python": sys.executable,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rank": rank, "name": p.name, "uuid": str(p.uuid),
        "memory_bytes": p.total_memory, "sm_count": p.multi_processor_count,
        "capability": [p.major, p.minor], "l2_bytes": p.L2_cache_size,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "peer_access": [[torch.cuda.can_device_access_peer(i, j) if i != j else True
                         for j in range(torch.cuda.device_count())]
                        for i in range(torch.cuda.device_count())],
    }
    # Over-L2 streaming copy measures a platform bound, not attention performance.
    x = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(10):
        a.record()
        for _ in range(10):
            y.copy_(x)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 10)
    result["copy_ms"] = times
    result["copy_read_write_GB_s"] = 2 * x.numel() * x.element_size() / (statistics.median(times) * 1e6)
    result["copy_correct"] = bool((y == 1).all())
    del x, y
    if args.nccl:
        import torch.distributed as dist
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
        result["nccl_version"] = torch.cuda.nccl.version()
        result["all_reduce"] = []
        for n in [4096, 1048576, 16777216]:
            t = torch.full((n,), float(rank + 1), device="cuda")
            dist.all_reduce(t)
            assert bool((t == 3).all()), "NCCL numerical failure"
            for _ in range(5):
                t.fill_(rank + 1)
                dist.all_reduce(t)
            torch.cuda.synchronize()
            elapsed = []
            for _ in range(30):
                t.fill_(rank + 1)
                a.record()
                dist.all_reduce(t)
                b.record()
                b.synchronize()
                elapsed.append(a.elapsed_time(b))
            result["all_reduce"].append({"bytes": n * 4, "ms": elapsed})
        dist.destroy_process_group()
    out = Path(args.output)
    if args.nccl:
        out = out.with_name(out.stem + f"_rank{rank}" + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "uuid": result["uuid"],
                      "copy_GB_s": result["copy_read_write_GB_s"]}), flush=True)


if __name__ == "__main__":
    main()
