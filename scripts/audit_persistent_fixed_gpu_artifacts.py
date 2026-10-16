"""CPU-only independent wave6 cubin/source/receipt audit; no GPU launch."""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from check_persistent_fixed_cpu import structural_checks

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/operator_persistent_fixed_qualification/wave6"
OUTPUT = ROOT / "results/operator_persistent_fixed_gpu_audit_v1"
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
PHASES = {"eager", "graph_baseline", "graph_changed_Q", "graph_changed_KV",
          "graph_changed_indices", "graph_changed_offsets_and_indices"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def generated_state_proof(path):
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if re.search(r"for \(int r = 0;", line)]
    loop_elided = not starts
    if loop_elided:
        # The all-empty fixture has a single plan label; the compiler elides
        # the extent-one region loop, retaining its runtime begin/end guard.
        assert any("int label = Labels[0];" in line for line in lines)
        starts = [i for i, line in enumerate(lines) if "if (begin < end) {" in line]
    assert len(starts) == 1
    start = starts[0]
    depth = 0
    stop = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            stop = i
            break
    assert stop is not None
    assert all(any(re.search(rf"float {name}\[{size}\];", line) for line in lines[:start])
               for name, size in (("out", 32), ("maximum", 4), ("denominator", 4)))
    assert not any(re.search(r"\b(?:RowMap|Count|Owner|CU|CM|CL)\b", line) for line in lines)
    # Current generated names are explicit: no canonical state references are
    # combined with shared addresses, inside or outside the owner-region loop.
    assert not any("buf_dyn_shmem" in line and re.search(r"\b(?:out|maximum|denominator)\b", line)
                   for line in lines)
    state_rows = {name: [{"line": i + 1, "text": line.strip()}
                         for i, line in enumerate(lines) if re.search(rf"\b{name}\b", line)]
                  for name in ("out", "maximum", "denominator")}
    out_in_region = [row for row in state_rows["out"] if start < row["line"] - 1 < stop]
    assert out_in_region
    assert all("reinterpret_cast<float*>(out +" in row["text"] or
               re.search(r"out\[[^]]+\] = \(out\[[^]]+\] \* factor", row["text"])
               for row in out_in_region), "Unexpected in-region output transport"
    final_u = [{"line": i + 1, "text": line.strip()} for i, line in enumerate(lines)
               if "U +" in line and "out +" in line]
    assert len(final_u) == 1 and final_u[0]["line"] - 1 > stop
    assert "buf_dyn_shmem" not in final_u[0]["text"]
    initial = {name: [row for row in state_rows[name] if row["line"] - 1 < start
                      and f"*(float4*)({name} +" in row["text"]]
               for name in state_rows}
    assert all(len(rows) == 1 for rows in initial.values())
    assert not any("Q +" in line for line in lines[start:stop + 1])
    shared_reductions = [{"line": i + 1, "text": line.strip()} for i, line in enumerate(lines)
                         if "AllReduce" in line and "buf_dyn_shmem" in line]
    assert len(shared_reductions) == 2
    return {"region_execution_span_lines": [start + 1, stop + 1],
            "single_region_loop_elided": loop_elided,
            "persistent_per_thread_float_arrays": {"out": 32, "maximum": 4, "denominator": 4},
            "state_reference_lines": state_rows, "state_initialization_before_region": initial,
            "in_region_out_only_rescale_or_PV_MMA": True,
            "Q_only_loaded_before_region": True,
            "no_canonical_state_shared_address_reference": True,
            "final_U_direct_from_out": final_u,
            "shared_reduction_scratch_still_present": shared_reductions}


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert "torch" not in sys.modules and "tilelang" not in sys.modules
    inputs = {}

    def record(path):
        inputs[rel(path)] = sha(path)

    source_check = structural_checks()
    source_paths = [ROOT / "src/branchtile/tilelang_kernel_persistent_fixed.py",
                    ROOT / "src/branchtile/attention_persistent_fixed.py",
                    ROOT / "scripts/check_persistent_fixed_cpu.py"]
    for path in source_paths:
        record(path)
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    config = ROOT / "configs/operator_qualification_wave6_jobs.json"
    record(config)
    expected_jobs = [j for j in json.loads(config.read_text()) if j["name"].startswith("persistent_w6_")]
    assert len(expected_jobs) == 7
    jobs, cases = [], []
    common_hashes = None
    for job in expected_jobs:
        path = SOURCE / (job["name"] + ".json")
        record(path)
        result = json.loads(path.read_text())
        assert result["status"] == "passed_numerical_and_mutable_graph_resource_review_pending"
        assert result["arguments"]["module"] == "branchtile.attention_persistent_fixed"
        assert result["uid"] == os.getuid() and len(result["cases"]) == 5
        assert set(c["name"] for c in result["cases"]) == {"all_masks", "unequal", "empty", "one_token", "real_capture"}
        if common_hashes is None:
            common_hashes = result["code_sha256"]
        assert result["code_sha256"] == common_hashes
        for name, expected in common_hashes.items():
            assert sha(ROOT / name) == expected, "Current source differs from qualification"
            record(ROOT / name)
        terminal = [e for e in ledger if e["event"] == "end" and e["name"] == job["name"]]
        assert len(terminal) == 1 and terminal[0]["exit_code"] == 0 and not terminal[0]["timed_out"]
        jobs.append({"name": job["name"], "result_path": rel(path), "arguments": result["arguments"],
                     "original_pending_status_preserved": result["status"], "terminal": terminal[0],
                     "independent_compiled_resource_review": "passed"})
        folder = SOURCE / (path.stem + "_compiler")
        for case in result["cases"]:
            assert case["status"] == "passed_numerical_and_mutable_graph"
            checks = {c["phase"]: c for c in case["checks"]}
            assert len(checks) == len(case["checks"]) == len(PHASES) and set(checks) == PHASES
            assert all(c["status"] == "passed" for c in checks.values())
            assert checks["eager"]["O_sha256"] == checks["graph_baseline"]["O_sha256"]
            assert checks["graph_baseline"]["Q_sha256"] != checks["graph_changed_Q"]["Q_sha256"]
            assert checks["graph_changed_Q"]["K_sha256"] != checks["graph_changed_KV"]["K_sha256"]
            assert checks["graph_changed_Q"]["V_sha256"] != checks["graph_changed_KV"]["V_sha256"]
            prefix = folder / case["name"]
            artifacts = {s: prefix.with_suffix(s) for s in (".cu", ".cubin", ".resources.txt", ".sass")}
            resource = case["resources"]
            for suffix, artifact in artifacts.items():
                record(artifact)
            for suffix, name in ((".cu", "generated_source_sha256"), (".cubin", "cubin_sha256"), (".sass", "sass_sha256")):
                assert sha(artifacts[suffix]) == resource[name]
            fresh_resource = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(artifacts[".cubin"])],
                                            check=True, capture_output=True, text=True).stdout
            fresh_sass = subprocess.run([str(CUOBJDUMP), "--dump-sass", str(artifacts[".cubin"])],
                                        check=True, capture_output=True, text=True).stdout
            assert fresh_resource == artifacts[".resources.txt"].read_text()
            assert fresh_sass == artifacts[".sass"].read_text()
            fields = {k.lower(): int(v) for k, v in re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", fresh_resource)}
            assert fields == resource["cubin_resources"]
            assert resource["total_smem_bytes"] == resource["dynamic_smem_bytes"] + fields["shared"]
            local_ops = re.findall(r"\b(?:LDL|STL)(?:\.|\s)", fresh_sass)
            assert len(local_ops) == resource["sass_local_load_store_count"] == 0
            assert fields["local"] == fields["stack"] == 0 and resource["resource_spill_free"]
            opcodes = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+[0-9]*\s+)?([A-Z][A-Z0-9_.]*)", fresh_sass)
            counts = collections.Counter(op.split(".")[0] for op in opcodes)
            assert counts["LDS"] > 0 and counts["STS"] > 0
            proof = generated_state_proof(artifacts[".cu"])
            cases.append({"job": job["name"], "case": case["name"], "B": case["batch"],
                          "N": result["arguments"]["block_n"], "splits": result["arguments"]["splits"],
                          "resources": resource, "fresh_cubin_resource_fields": fields,
                          "static_SASS_instruction_site_counts": dict(sorted(counts.items())),
                          "generated_CUDA_state_proof": proof,
                          "numerical_and_graph_checks": 6, "eager_graph_O_bitwise_equal": True,
                          "offset_values_changed": checks["graph_changed_offsets_and_indices"]["offset_values_changed"],
                          "maximum_recorded_O_absolute_error": max(c["O_max_abs_fp64"] for c in checks.values()),
                          "maximum_recorded_finite_LSE_absolute_error": max(c["LSE_max_abs_fp64_finite_rows"] for c in checks.values())})
    assert len(cases) == 35
    report = {"schema": 1, "status": "independent_persistent_fixed_artifact_audit_passed",
              "GPU_hours": 0, "torch_imported": "torch" in sys.modules, "tilelang_imported": "tilelang" in sys.modules,
              "auditor_sha256": sha(__file__), "cuobjdump_sha256": sha(CUOBJDUMP), "input_sha256": inputs,
              "CPU_structure_checks": source_check, "all_qualification_code_hashes_identical_and_current": True,
              "qualification_common_code_sha256": common_hashes, "jobs": jobs, "cases": cases,
              "numerical_and_graph_checks": 210,
              "all_fresh_cuobjdump_resource_SASS_match_saved": True,
              "all_LOCAL_STACK_LDL_STL_zero": True,
              "cross_region_out_maximum_denominator_arrays_preserved": True,
              "canonical_shared_U_transport_absent_from_all_generated_sources": True,
              "qualification_GPU_hours_charged": sum(j["terminal"]["gpu_hours"] for j in jobs),
              "limitations": [
                  "Saved numerical receipts are audited; no GPU outputs or timings are recomputed in this CPU review.",
                  "Source shows register-array state references; no local memory in cubin/SASS supports no spilling in these 35 compiled cases, not every future shape.",
                  "Shared Q/K/V/P and cross-warp reduction scratch remain; absence of canonical shared U transport is not absence of shared traffic or synchronization.",
                  "Static SASS counts are instruction sites, not executed instructions, bytes, stalls or critical-path attribution.",
                  "Dynamic SMEM is taken from saved launch receipts, while static resource fields are independently redumped.",
                  "Mutable tests cover Q/KV/indices/offsets, not a changed Labels vector or arbitrary graph topology.",
                  "This fixed M32 control changes masked MMA work relative to compact FullCol, so it is not a pure state-transport-only change.",
                  "No performance, task quality, energy, full-model latency, multi-rank or novel-mechanism claim follows from qualification."]}
    assert not report["torch_imported"] and not report["tilelang_imported"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "jobs": len(jobs), "cases": len(cases),
                      "numerical_checks": 210, "register_range": [min(c["resources"]["cubin_resources"]["reg"] for c in cases),
                                                                   max(c["resources"]["cubin_resources"]["reg"] for c in cases)],
                      "SMEM_values": sorted({c["resources"]["total_smem_bytes"] for c in cases}), "GPU_hours": 0}, indent=2))


if __name__ == "__main__":
    main()
