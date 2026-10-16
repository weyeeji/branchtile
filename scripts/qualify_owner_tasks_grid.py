# Expanded chunk grid: separate frozen script; original wave17 qualification is unchanged.
#!/usr/bin/env python3
"""Owner-task numerical qualification with a distinct graph-topology contract.

Q/K/V/Indices values may change in place under one captured graph. Changes to
offsets/counts/row_map/labels require rebuilding Attention and recapturing.
graph.replay() bypasses host validate_plan(); this script never replays an old
graph after changing topology. Lifecycle wall costs are diagnostic, not speed
benchmarks. Allocate the single GPU through run_jobs.py; no automatic retries.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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

from qualify_static_bank_v2 import fixture_lists

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODULE = "branchtile.attention_owner_tasks"
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
O_ATOL, O_RTOL, LSE_ATOL, LSE_RTOL = .006, .015, .0003, .0001
FIXTURES = ("all_masks", "unequal", "empty", "one_token")
TOPOLOGY_CONTRACT = (
    "Q/K/V/Indices values may change in place. Offsets/counts/row_map/labels "
    "changes require Attention rebuild and CUDA graph recapture. graph.replay "
    "bypasses host version checks; callers must enforce this contract.")
SOURCE_PATHS = ("scripts/qualify_owner_tasks_grid.py", "scripts/qualify_static_bank_v2.py",
                "scripts/compare_attention.py", "src/branchtile/attention.py",
                "src/branchtile/attention_owner_tasks.py",
                "src/branchtile/owner_tasks_plan.py", "src/branchtile/triton_kernel_owner_tasks.py")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, choices=(3, 4), default=3)
    p.add_argument("--block-n", type=int, choices=(32, 64), default=64)
    p.add_argument("--chunk-tokens", type=int, choices=(64, 128, 256, 512, 1024, 2048, 4096), default=512)
    p.add_argument("--threads", type=int, choices=(128,), default=128)
    p.add_argument("--splits", type=int, default=64,
                   help="Compatibility parameter only; chunk-tokens defines tasks")
    p.add_argument("--tokens", type=int, default=259)
    p.add_argument("--module", choices=(MODULE,), default=MODULE)
    p.add_argument("--capture")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--budget", type=int, default=32768)
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--heads", type=int, choices=(2,), default=2)
    p.add_argument("--output", required=True)
    p.add_argument("--cpu-check", action="store_true")
    return p


def cpu_contract(a):
    from branchtile.owner_tasks_plan import build_task_layout
    rows = []
    for name in FIXTURES:
        owners = fixture_lists(a.batch, name, a.tokens)
        labels = sorted(set(value for head in owners for value in head) - {0}) or [1]
        offsets, counts, row_map = [], [], []
        for head in owners:
            extent, head_offsets, head_counts, head_rows = 0, [0], [], []
            for label in labels:
                extent += head.count(label)
                head_offsets.append(extent)
                active = [request*8 + group_head for request in range(a.batch)
                          if label & (1 << request) for group_head in range(8)]
                head_counts.append(len(active))
                head_rows.append(active + [0] * (32-len(active)))
            offsets.append(head_offsets)
            counts.append(head_counts)
            row_map.append(head_rows)
        layout = build_task_layout(offsets, row_map, counts, labels, a.batch, 8,
                                   a.tokens, a.chunk_tokens)
        assert bool(layout["tasks"]) == (name != "empty")
        assert sum(task[3]-task[2] for task in layout["tasks"]) == sum(value != 0 for head in owners for value in head)
        assert all(0 < task[3]-task[2] <= a.chunk_tokens for task in layout["tasks"])
        rows.append({"fixture": name, "task_count": len(layout["tasks"]),
                     "packed_rows": layout["packed_rows"],
                     "structure_sha256": layout["structure_sha256"]})
    assert "torch" not in sys.modules and "triton" not in sys.modules
    return {"status": "owner_task_CPU_contract_passed", "batch": a.batch,
            "chunk_tokens": a.chunk_tokens, "block_n": a.block_n,
            "fixtures": rows, "topology_contract": TOPOLOGY_CONTRACT,
            "splits_partitions_tasks": False, "CUDA_imported": False,
            "source_sha256": {path: sha(ROOT / path) for path in SOURCE_PATHS}}


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return repr(value)


def compiled_artifacts(compiled, prefix, launched):
    if not launched:
        assert compiled is None
        return {"status": "not_launched", "reason": "Zero owner tasks; no producer launch or fabricated artifact."}
    assert compiled is not None, "A launched kernel must retain its Triton CompiledKernel"
    asm = compiled.asm
    assert isinstance(asm["ptx"], str) and isinstance(asm["cubin"], (bytes, bytearray))
    files = {"ptx": prefix.with_suffix(".ptx"), "cubin": prefix.with_suffix(".cubin"),
             "resources": prefix.with_suffix(".resources.txt"), "sass": prefix.with_suffix(".sass")}
    assert not any(path.exists() for path in files.values()), "Retain previous artifacts"
    files["ptx"].write_text(asm["ptx"])
    files["cubin"].write_bytes(asm["cubin"])
    resources = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(files["cubin"])],
                               check=True, capture_output=True, text=True).stdout
    sass = subprocess.run([str(CUOBJDUMP), "--dump-sass", str(files["cubin"])],
                          check=True, capture_output=True, text=True).stdout
    files["resources"].write_text(resources)
    files["sass"].write_text(sass)
    fields = {key.lower(): int(value) for key, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", resources)}
    assert {"reg", "stack", "shared", "local"} <= fields.keys()
    local_loads = len(re.findall(r"\bLDL(?:\.|\s)", sass))
    local_stores = len(re.findall(r"\bSTL(?:\.|\s)", sass))
    metadata = compiled.metadata
    metadata_dict = metadata._asdict() if hasattr(metadata, "_asdict") else vars(metadata)
    return {"status": "saved_for_independent_resource_review",
            "name": getattr(compiled, "name", None),
            "n_regs": json_value(getattr(compiled, "n_regs", None)),
            "n_spills": json_value(getattr(compiled, "n_spills", None)),
            "Triton_metadata": json_value(metadata_dict), "cubin_resources": fields,
            "SASS_LDL_count": local_loads, "SASS_STL_count": local_stores,
            "explicit_local_traffic_absent": fields["local"] == fields["stack"] == local_loads == local_stores == 0,
            "artifacts": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                          for name, path in files.items()}}


def main():
    a = parser().parse_args()
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert a.splits > 0 and a.tokens >= (1 << a.batch)
    if a.cpu_check:
        print(json.dumps(cpu_contract(a), sort_keys=True))
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible and visible != "-1" and "," not in visible, "Allocate one GPU with run_jobs.py"
    output = ROOT / a.output
    compiler_dir = output.parent / (output.stem + "_compiler")
    assert not output.exists() and not compiler_dir.exists(), "Use new paths; retain all failed/timeout attempts"
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler_dir.mkdir()
    result = {"schema": "owner-task-qualification-v1", "status": "running",
              "uid": os.getuid(), "python": sys.executable, "CUDA_VISIBLE_DEVICES": visible,
              "arguments": vars(a), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "code_sha256": {path: sha(ROOT / path) for path in SOURCE_PATHS},
              "contracts": {"O": {"atol": O_ATOL, "rtol": O_RTOL},
                            "natural_log_LSE": {"atol": LSE_ATOL, "rtol": LSE_RTOL}},
              "topology_contract": TOPOLOGY_CONTRACT,
              "fixed_topology_phases": ["eager", "graph_baseline", "graph_changed_Q",
                                        "graph_changed_KV", "graph_changed_indices"],
              "topology_change_phase": "rebuild_and_new_graph_changed_owners",
              "scope": "Numerical and fixed-topology graph + rebuild/recapture qualification; no performance timing or application claim.",
              "stage": "imports", "cases": []}

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def terminate(signum, _frame):
        result["termination_signal"] = signum
        raise RuntimeError("Bounded owner-task qualification terminated")

    signal.signal(signal.SIGTERM, terminate)
    started = time.perf_counter()
    save()
    try:
        import torch
        from branchtile.attention import make_plan, reference
        from branchtile.attention_owner_tasks import Attention
        assert torch.cuda.device_count() == 1
        torch.set_num_threads(1)
        torch.manual_seed(20261016)
        device = torch.cuda.get_device_properties(0)
        result.update(gpu_uuid=str(device.uuid), gpu_name=device.name,
                      packages={name: importlib.metadata.version(name) for name in ("torch", "triton")})

        def check(actual, lse, q, k, v, owners, batch):
            expected, expected_lse = reference(q, k, v, owners, batch, 8)
            assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse))
            assert not torch.isnan(actual).any() and not torch.isnan(lse).any()
            torch.testing.assert_close(actual.float(), expected.float(), atol=O_ATOL, rtol=O_RTOL)
            torch.testing.assert_close(lse.float(), expected_lse.float(), atol=LSE_ATOL, rtol=LSE_RTOL)
            finite = torch.isfinite(expected_lse)
            return {"status": "passed", "O_max_abs_fp64": float((actual.double()-expected).abs().max()),
                    "O_relative_l2_fp64": float(torch.linalg.vector_norm(actual.double()-expected) / torch.linalg.vector_norm(expected).clamp_min(1e-12)),
                    "LSE_max_abs_fp64_finite_rows": float((lse.double()[finite]-expected_lse[finite]).abs().max()) if finite.any() else 0.,
                    "empty_rows": int((~finite).sum()),
                    **{name+"_sha256": tensor_sha(value) for name, value in
                       (("O", actual), ("Q", q), ("K", k), ("V", v), ("owner", owners))}}

        def run_case(name, q, k, v, owners, batch, provenance):
            row = {"name": name, "status": "initializing", "batch": batch,
                   "q_shape": list(q.shape), "k_shape": list(k.shape),
                   "provenance": provenance, "checks": []}
            result["cases"].append(row)
            result["stage"] = name+":initial_plan_and_eager"
            save()
            plan = make_plan(owners, batch, 8, "cuda")
            row["initial_plan_prepare_ms"] = plan.prepare_ms
            op = Attention(plan, splits=a.splits, block_n=a.block_n, threads=128,
                           chunk_tokens=a.chunk_tokens)
            got, lse = op(q, k, v)
            torch.cuda.synchronize()
            row.update(initial_metadata=op.metadata, initial_prepare_ms=op.prepare_ms,
                       initial_task_count=op.num_tasks)
            initial_compiled = (op.last_producer_compiled, op.last_merge_compiled)
            row["checks"].append({"phase": "eager", **check(got, lse, q, k, v, owners, batch)})
            for _ in range(2):
                op(q, k, v)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, captured_lse = op(q, k, v)
            for phase in ("graph_baseline", "graph_changed_Q", "graph_changed_KV", "graph_changed_indices"):
                result["stage"] = name+":"+phase
                if phase == "graph_changed_Q":
                    q.add_(.25)
                elif phase == "graph_changed_KV":
                    k.mul_(.7)
                    v.mul_(1.1).add_(.125)
                elif phase == "graph_changed_indices":
                    plan.indices.copy_(k.shape[1]-1-plan.indices)
                    owners = owners.flip(1).contiguous()
                op.validate_plan()
                graph.replay()
                torch.cuda.synchronize()
                row["checks"].append({"phase": phase, **check(captured, captured_lse, q, k, v, owners, batch)})
                save()
            result["stage"] = name+":topology_invalidation_and_rebuild"
            changed = torch.zeros_like(owners)
            for i, label in enumerate(plan.labels):
                changed[owners == label] = plan.labels[(i+1) % len(plan.labels)]
            old_offset_sha = tensor_sha(plan.offsets)
            reconfigure_start = time.perf_counter()
            updated = make_plan(changed, batch, 8, "cuda")
            plan_end = time.perf_counter()
            plan.offsets.copy_(updated.offsets)
            rejection = None
            try:
                op.validate_plan()
            except RuntimeError as error:
                rejection = str(error)
            assert rejection is not None and "rebuild" in rejection.lower(), "Changed topology must explicitly reject old op"
            reject_end = time.perf_counter()
            # Never replay graph here. Host rejection does not intercept graph.replay.
            new_op = Attention(updated, splits=a.splits, block_n=a.block_n, threads=128,
                               chunk_tokens=a.chunk_tokens)
            rebuild_end = time.perf_counter()
            for _ in range(2):
                new_op(q, k, v)
            torch.cuda.synchronize()
            new_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(new_graph):
                new_o, new_lse = new_op(q, k, v)
            new_op.validate_plan()
            new_graph.replay()
            torch.cuda.synchronize()
            recapture_end = time.perf_counter()
            row["topology_change"] = {"old_graph_replayed_after_topology_mutation": False,
                                      "old_op_validate_plan_rejected": True, "rejection": rejection,
                                      "graph_replay_bypasses_host_check": True,
                                      "offset_values_changed": old_offset_sha != tensor_sha(updated.offsets),
                                      "offset_version_changed_even_if_values_equal": True,
                                      "updated_plan_build_wall_seconds": plan_end-reconfigure_start,
                                      "old_plan_mutation_and_rejection_wall_seconds": reject_end-plan_end,
                                      "Attention_rebuild_wall_seconds": rebuild_end-reject_end,
                                      "warmup_JIT_recapture_first_replay_wall_seconds": recapture_end-rebuild_end,
                                      "total_rebuild_sequence_wall_seconds": recapture_end-reconfigure_start,
                                      "updated_plan_prepare_ms": updated.prepare_ms,
                                      "Attention_prepare_ms": new_op.prepare_ms,
                                      "cost_scope": "Diagnostic host wall includes plan construction/D2H/H2D/allocations, old-plan rejection, Attention rebuild, JIT warmup, new graph capture and first replay; excludes artifact export and FP64/hash audits."}
            row["checks"].append({"phase": "rebuild_and_new_graph_changed_owners",
                                   **check(new_o, new_lse, q, k, v, changed, batch)})
            row.update(rebuilt_metadata=new_op.metadata, rebuilt_task_count=new_op.num_tasks,
                       status="passed_numerical_fixed_topology_graph_and_rebuilt_graph",
                       resource_review_status="pending_export")
            result["stage"] = name+":resource_export"
            save()
            for label, candidate, handles in (("initial", op, initial_compiled),
                                              ("rebuilt", new_op, (new_op.last_producer_compiled, new_op.last_merge_compiled))):
                row[label+"_resources"] = {
                    "producer": compiled_artifacts(handles[0], compiler_dir / f"{name}_{label}_producer", candidate.num_tasks > 0),
                    "merge": compiled_artifacts(handles[1], compiler_dir / f"{name}_{label}_merge", True)}
                if candidate.num_tasks == 0:
                    assert handles[0] is None
            row["resource_review_status"] = "exported_independent_review_pending"
            save()

        for name in FIXTURES:
            owners = torch.tensor(fixture_lists(a.batch, name, a.tokens), dtype=torch.int64)
            q = torch.randn(2, a.batch*8, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(2, a.tokens, 128, device="cuda", dtype=torch.bfloat16)
            run_case(name, q, k, torch.randn_like(k), owners, a.batch,
                     {"kind": "synthetic", "seed": 20261016})
        if a.capture:
            from compare_attention import load_capture
            directory = ROOT / a.capture
            metadata = json.loads((directory / "metadata.json").read_text())
            assert metadata["export_semantics_version"] == 2
            assert "suffix_v2" in str(directory) or metadata.get("derived_capture_overlay")
            files = [directory / "metadata.json", directory / f"source_l{a.layer}.pt",
                     directory / f"masks_l{a.layer}_b{a.budget}.pt"]
            files.extend(directory / f"decode_q{request}_l{a.layer}_s{a.step}.pt" for request in range(a.batch))
            input_hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
            q, k, v, owners, batch, gqa, provenance = load_capture(directory, a.layer, a.budget, a.step, a.heads)
            assert batch == a.batch == metadata["B"] and gqa == 8
            assert provenance["ready_lifetime_measured"], "Require a genuine same-forward ready capture"
            provenance["files_sha256"] = input_hashes
            run_case("real_capture", q, k, v, owners, batch, provenance)
            assert all(sha(ROOT / path) == expected for path, expected in input_hashes.items())
        assert len(result["cases"]) == 4 + bool(a.capture)
        assert all(len(row["checks"]) == 6 for row in result["cases"])
        assert all(sha(ROOT / path) == expected for path, expected in result["code_sha256"].items())
        result.update(status="passed_numerical_fixed_topology_and_rebuild_recapture_resource_review_pending", stage="completed")
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                      failure_stage=result["stage"])
        save()
        raise
    finally:
        result["wall_seconds"] = time.perf_counter()-started
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "output": str(output)}))


if __name__ == "__main__":
    main()
