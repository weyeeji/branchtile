#!/usr/bin/env python3
"""Qualify current FlashInfer native BF16 decode without changing Torch."""
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    started = time.time()
    output = ROOT / "results/platform/latest_flashinfer_probe.json"
    output.parent.mkdir(exist_ok=True, parents=True)
    report = {"status": "running", "pid": os.getpid(), "uid": os.geteuid(), "venv": sys.prefix,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "release": "https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2",
              "installation": "https://docs.flashinfer.ai/installation.html",
              "scope": "compatibility and numerical qualification only; no performance or application claim",
              "torch_upgrade_forbidden": True, "FROST_used": False,
              "package_installation": "isolated system-site-packages .venv-latest; core from PyPI, cubin from official flashinfer.ai/whl"}
    try:
        import torch
        import tilelang
        import flashinfer
        from flashinfer_baseline import NativeSparseDecode
        from branchtile.attention import reference, synthetic_owners
        assert flashinfer.__version__ == "0.7.2"
        assert torch.__version__.startswith("2.11.")
        assert torch.cuda.device_count() == 1
        report.update(torch=torch.__version__, tilelang=tilelang.__version__, flashinfer=flashinfer.__version__,
                      torch_module=torch.__file__, tilelang_module=tilelang.__file__, flashinfer_module=flashinfer.__file__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ("flashinfer-python", "flashinfer-cubin", "nvidia-cutlass-dsl", "apache-tvm-ffi", "nvidia-cudnn-frontend")}
        assert report["packages"]["flashinfer-cubin"] == "0.7.2"
        torch.set_num_threads(4)
        torch.manual_seed(20261016)
        h, b, g, n, d = 2, 4, 4, 257, 128
        owners = synthetic_owners(h, n, "cross", b, 20261016)
        supports = [[torch.where((owners[head] & (1 << request)) != 0)[0] for request in range(b)] for head in range(h)]
        q = torch.randn(h, b * g, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(h, n, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        ref, ref_lse = reference(q, k, v, owners, b, g)
        report["shape"] = {"kv_heads": h, "B": b, "GQA": g, "N": n, "D": d, "pattern": "cross"}
        checks = []
        for tensor_cores, split in ((False, None), (True, None), (True, 128)):
            native = NativeSparseDecode(q, k, v, supports, use_tensor_cores=tensor_cores,
                                        fixed_split_size=split, normalize_lse=True)
            out, lse = native.run()
            torch.cuda.synchronize()
            backend = native.wrapper.resolved_backend
            assert backend == "fa2", backend
            check = {"use_tensor_cores": tensor_cores, "fixed_split_size": split, "resolved_backend": backend,
                     "max_output_abs_error_fp64": float((out.double() - ref).abs().max()),
                     "max_ln_lse_abs_error_fp64": float((lse.double() - ref_lse).abs().max())}
            torch.testing.assert_close(out.float(), ref.float(), atol=0.006, rtol=0.015)
            torch.testing.assert_close(lse.float(), ref_lse.float(), atol=0.0003, rtol=0.0001)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                native.run()
            graph.replay()
            torch.cuda.synchronize()
            check["cuda_graph_replay_passed"] = True
            checks.append(check)
        report.update(status="passed", checks=checks, frozen_torch_preserved=True)
    except Exception as error:
        report.update(status="failed", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      instrument_seconds=time.time() - started,
                      allocation_timing_source="results/job_ledger.jsonl exact name+pid process interval; do not sum instrument_seconds again")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
