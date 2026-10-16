import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from flashinfer_baseline import NativeSparseDecode

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--n", type=int, default=16384)
p.add_argument("--gqa", type=int, default=8)
a = p.parse_args()
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
torch.set_num_threads(4)
torch.manual_seed(13)
h, b, g, d, n = 2, 4, a.gqa, 128, a.n
q = torch.randn((h, b * g, d), device="cuda", dtype=torch.bfloat16)
k = torch.randn((h, n, d), device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
supports = [[torch.randperm(n)[:n // 4].sort().values for _ in range(b)] for _ in range(h)]
records = []
for tc in [False, True]:
    for split in [None, 256, 1024]:
        # fixed split size is supported by the tensor-core path only.
        if not tc and split is not None:
            continue
        start = time.perf_counter()
        runner = NativeSparseDecode(q, k, v, supports, use_tensor_cores=tc,
                                   fixed_split_size=split)
        out, lse = runner.run()
        err, lse_err = 0.0, 0.0
        for head in range(h):
            for req in range(b):
                ids = supports[head][req].to("cuda")
                score = q[head, req * g:(req + 1) * g].double() @ k[head, ids].double().T / d**0.5
                ref = score.softmax(-1) @ v[head, ids].double()
                err = max(err, (out[head, req * g:(req + 1) * g].double() - ref).abs().max().item())
                lse_err = max(lse_err, (lse[head, req * g:(req + 1) * g] * math.log(2) - score.logsumexp(-1)).abs().max().item())
        assert err < 0.003 and lse_err < 0.001, (err, lse_err)
        for _ in range(5):
            runner.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                runner.run()
        times = []
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(30):
            t0.record()
            graph.replay()
            t1.record()
            t1.synchronize()
            times.append(t0.elapsed_time(t1) * 1000 / 20)
        records.append({"tensor_cores": tc, "fixed_split_size": split,
                        "plan_ms_including_first_JIT": runner.plan_ms,
                        "max_abs_error_fp64": err, "max_lse_error_fp64": lse_err,
                        "call_us": times, "median_us": statistics.median(times)})
        print(json.dumps(records[-1]), flush=True)
result = {"kind": "native_flashinfer_qualification_synthetic_hot_cache",
          "uid": os.geteuid(), "python": sys.executable,
          "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
          "n": n, "b": b, "h": h, "gqa": g, "d": d,
          "native_version": __import__("flashinfer").__version__, "records": records}
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
