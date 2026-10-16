#!/usr/bin/env python3
"""Bounded FP64/graph/mutable-input qualification of the static-owner v4 kernel.

Use run_jobs.py to allocate GPUs. This script does no timing or automatic retry.
--cpu-check validates fixtures and arguments without importing Torch/TileLang.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUTPUT_ATOL, OUTPUT_RTOL = 0.006, 0.015
LSE_ATOL, LSE_RTOL = 0.0003, 0.0001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_lists(batch: int, name: str, tokens: int) -> list[list[int]]:
    labels = list(range(1, 1 << batch))
    if name == "all_masks":
        first = [i % (1 << batch) for i in range(tokens)]
        second = [(i * 3 + 1) % (1 << batch) for i in range(tokens)]
    elif name == "unequal":
        # Long owner1, very short remaining owners; head1 has empty regions.
        first = [1] * tokens
        for i, label in enumerate(labels):
            first[i] = label
        second = [0] * tokens
        for i in range(min(19, tokens)):
            second[i] = 1 if i % 3 else (1 << batch) - 1
    elif name == "empty":
        first = second = [0] * tokens
    elif name == "one_token":
        first = [1] + [0] * (tokens - 1)
        second = [1 << (batch - 1)] + [0] * (tokens - 1)
    else:
        raise ValueError(name)
    assert len(first) == len(second) == tokens
    assert all(0 <= value < (1 << batch) for row in (first, second) for value in row)
    return [first, second]


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, choices=(3, 4), default=3)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--block-n", type=int, default=64)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--tokens", type=int, default=259)
    p.add_argument("--fixtures", default="all_masks,unequal,empty,one_token")
    p.add_argument("--module", default="branchtile.attention_v4")
    p.add_argument("--capture")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--budget", type=int, default=32768)
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--output", required=True)
    p.add_argument("--cpu-check", action="store_true")
    return p


def main():
    a = parser().parse_args()
    assert os.geteuid() != 0, "No root execution"
    assert sys.prefix != sys.base_prefix, "Project virtual environment required"
    assert a.splits > 0 and a.tokens >= (1 << a.batch)
    assert a.block_n in (32, 64) and a.threads == 128
    names = [name for name in a.fixtures.split(",") if name]
    for name in names:
        rows = fixture_lists(a.batch, name, a.tokens)
        if name == "all_masks":
            assert set(rows[0]) == set(range(1 << a.batch))
        elif name == "unequal":
            assert set(rows[0]) == set(range(1, 1 << a.batch))
            assert len(set(rows[1]) - {0}) < (1 << a.batch) - 1
        elif name == "empty":
            assert all(value == 0 for row in rows for value in row)
        elif name == "one_token":
            assert all(sum(value != 0 for value in row) == 1 for row in rows)
    if a.cpu_check:
        print(json.dumps({"status": "cpu_contract_passed", "fixtures": names,
                          "cuda_imported": "torch" in sys.modules, "batch": a.batch,
                          "splits": a.splits, "tokens": a.tokens}))
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Allocate GPU with run_jobs.py"
    assert "," not in os.environ["CUDA_VISIBLE_DEVICES"], "Single GPU qualification job"
    import torch
    from branchtile.attention import make_plan, reference
    from branchtile.resources import kernel_resources

    Attention = importlib.import_module(a.module).Attention
    torch.set_num_threads(1)
    torch.manual_seed(20261016)
    output = ROOT / a.output
    assert not output.exists(), "Retain previous attempts; no overwrite"
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler_dir = output.parent / (output.stem + "_compiler")
    assert not compiler_dir.exists(), "Retain previous compiler artifacts"
    compiler_dir.mkdir()
    paths = [Path(__file__), ROOT / "src/branchtile/attention.py",
             ROOT / "src/branchtile/resources.py", ROOT / "scripts/compare_attention.py"]
    for relative in ("attention_v4.py", "tilelang_kernel_v4.py", "attention_fullcol_guard.py",
                     "tilelang_kernel_fullcol_guard.py", "attention_emptyguard.py",
                     "tilelang_kernel_emptyguard.py", "tilelang_kernel_v2.py"):
        candidate = ROOT / "src/branchtile" / relative
        if candidate.exists():
            paths.append(candidate)
    result = {
        "schema": "static-bank-qualification-v1", "status": "running",
        "uid": os.geteuid(), "python": sys.executable,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arguments": vars(a), "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "contracts": {"O": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL},
                      "natural_log_LSE": {"atol": LSE_ATOL, "rtol": LSE_RTOL}},
        "scope": "Numerical and graph qualification only; no performance result.",
        "cases": [],
    }

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def terminate(*_):
        raise RuntimeError("Bounded qualification terminated")

    signal.signal(signal.SIGTERM, terminate)
    save()
    start = time.perf_counter()

    def check(actual, got_lse, q, k, v, owners, batch):
        expected, expected_lse = reference(q, k, v, owners, batch, 8)
        assert torch.equal(torch.isneginf(got_lse), torch.isneginf(expected_lse))
        assert not torch.isnan(actual).any() and not torch.isnan(got_lse).any()
        torch.testing.assert_close(actual.float(), expected.float(), atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
        torch.testing.assert_close(got_lse.float(), expected_lse.float(), atol=LSE_ATOL, rtol=LSE_RTOL)
        finite = torch.isfinite(expected_lse)
        return {
            "status": "passed", "O_max_abs_fp64": float((actual.double() - expected).abs().max()),
            "O_relative_l2_fp64": float(torch.linalg.vector_norm(actual.double() - expected)
                                       / torch.linalg.vector_norm(expected).clamp_min(1e-12)),
            "LSE_max_abs_fp64_finite_rows": float((got_lse.double()[finite] - expected_lse[finite]).abs().max()) if finite.any() else 0.0,
            "empty_rows": int((~finite).sum()), "O_sha256": tensor_sha(actual),
            "Q_sha256": tensor_sha(q), "K_sha256": tensor_sha(k), "V_sha256": tensor_sha(v),
            "owner_sha256": tensor_sha(owners),
        }

    def run_case(name, q, k, v, owners, batch, provenance):
        row = {"name": name, "status": "initializing", "q_shape": list(q.shape),
               "k_shape": list(k.shape), "batch": batch, "provenance": provenance,
               "checks": []}
        result["cases"].append(row)
        save()
        plan = make_plan(owners, batch, 8, "cuda")
        row["owner_labels"] = plan.labels
        row["region_lengths_by_head"] = (plan.offsets[:, 1:] - plan.offsets[:, :-1]).cpu().tolist()
        init_start = time.perf_counter()
        op = Attention(plan, splits=a.splits, block_n=a.block_n, threads=a.threads)
        got, lse = op(q, k, v)
        torch.cuda.synchronize()
        row["initialization_and_first_call_seconds"] = time.perf_counter() - init_start
        row["checks"].append({"phase": "eager", **check(got, lse, q, k, v, owners, batch)})
        prefix = compiler_dir / name
        row["resources"] = kernel_resources(op.kernel, prefix)
        source = prefix.with_suffix(".cu").read_text()
        # Save evidence to review, rather than claim that the word "fragment"
        # proves generated code has no state round trip.
        row["generated_source_audit"] = {
            "canonical_u_lines": [{"line": i, "text": line.strip()} for i, line in enumerate(source.splitlines(), 1)
                                  if "canonical_u" in line or re.search(r"\bcu\[", line)],
            "barrier_text_count": source.count("__syncthreads"),
            "U_shared_traffic_proof": "requires_manual_generated_source_review",
            "compiled_array_copy_evidence_only": True,
        }
        sass = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-sass", str(prefix.with_suffix(".cubin"))],
                              capture_output=True, text=True, check=True).stdout
        prefix.with_suffix(".sass").write_text(sass)
        row["resources"]["sass_sha256"] = sha(prefix.with_suffix(".sass"))
        row["resources"]["sass_local_load_store_count"] = len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", sass))
        row["resources"]["resource_spill_free"] = (row["resources"]["cubin_resources"].get("local", 0) == 0
            and row["resources"]["cubin_resources"].get("stack", 0) == 0
            and row["resources"]["sass_local_load_store_count"] == 0)
        for _ in range(2):
            op(q, k, v)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_lse = op(q, k, v)
        graph.replay()
        torch.cuda.synchronize()
        row["checks"].append({"phase": "graph_baseline", **check(captured, captured_lse, q, k, v, owners, batch)})
        q.add_(0.25)
        graph.replay()
        torch.cuda.synchronize()
        row["checks"].append({"phase": "graph_changed_Q", **check(captured, captured_lse, q, k, v, owners, batch)})
        k.mul_(0.7)
        v.mul_(1.1).add_(0.125)
        graph.replay()
        torch.cuda.synchronize()
        row["checks"].append({"phase": "graph_changed_KV", **check(captured, captured_lse, q, k, v, owners, batch)})
        plan.indices.copy_(k.shape[1] - 1 - plan.indices)
        owners = owners.flip(1).contiguous()
        graph.replay()
        torch.cuda.synchronize()
        row["checks"].append({"phase": "graph_changed_indices", **check(captured, captured_lse, q, k, v, owners, batch)})
        # Rotate owner assignments while preserving static labels, then update
        # the original graph's index/offset buffers in place. Counts can change.
        changed = torch.zeros_like(owners)
        for i, label in enumerate(plan.labels):
            changed[owners == label] = plan.labels[(i + 1) % len(plan.labels)]
        updated = make_plan(changed, batch, 8, "cuda")
        assert updated.labels == plan.labels
        old_offsets = tensor_sha(plan.offsets)
        plan.indices.copy_(updated.indices)
        plan.offsets.copy_(updated.offsets)
        graph.replay()
        torch.cuda.synchronize()
        row["checks"].append({"phase": "graph_changed_offsets_and_indices", "offset_values_changed": old_offsets != tensor_sha(plan.offsets),
                               **check(captured, captured_lse, q, k, v, changed, batch)})
        row["status"] = "passed_numerical_and_mutable_graph"
        save()

    try:
        device = torch.cuda.get_device_properties(0)
        result.update(gpu_uuid=str(device.uuid), gpu_name=device.name,
                      packages={name: importlib.metadata.version(name) for name in ("torch", "tilelang", "triton")})
        for name in names:
            owners = torch.tensor(fixture_lists(a.batch, name, a.tokens), dtype=torch.int64)
            q = torch.randn(2, a.batch * 8, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(2, a.tokens, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            run_case(name, q, k, v, owners, a.batch, {"kind": "synthetic", "seed": 20261016})
        if a.capture:
            from compare_attention import load_capture
            directory = ROOT / a.capture
            assert "suffix_v2" in str(directory), "Require corrected suffix capture"
            q, k, v, owners, batch, gqa, provenance = load_capture(directory, a.layer, a.budget, a.step, a.heads)
            assert gqa == 8 and batch in (3, 4)
            files = [directory / "metadata.json", directory / f"source_l{a.layer}.pt",
                     directory / f"masks_l{a.layer}_b{a.budget}.pt"]
            files.extend(directory / f"decode_q{qi}_l{a.layer}_s{a.step}.pt" for qi in provenance["original_active_requests"])
            provenance["files_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in files}
            run_case("real_capture", q, k, v, owners, batch, provenance)
        assert result["cases"], "No qualification cases requested"
        result["status"] = "passed_numerical_and_mutable_graph_resource_review_pending"
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        save()
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - start
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "output": str(output)}))


if __name__ == "__main__":
    main()
