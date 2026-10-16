"""CPU-only independent audit of frozen lane-select qualification artifacts.

Reads cubins with cuobjdump; imports neither Torch nor TileLang and never
launches a GPU. The earlier wave11 compiler failures remain explicit inputs.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from check_lane_select_split_banks_cpu import structural_checks

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/operator_lane_select_qualification"
OUTPUT = ROOT / "results/operator_lane_select_gpu_audit_v1"
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
PHASES = {"eager", "graph_baseline", "graph_changed_Q", "graph_changed_KV",
          "graph_changed_indices", "graph_changed_offsets_and_indices"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def generated_state_proof(path, batch):
    lines = path.read_text().splitlines()
    region_loops = [i for i, line in enumerate(lines) if re.search(r"for \(int r = 0;", line)]
    assert len(region_loops) <= 1
    # The single-label empty fixture is compiled without a region loop.
    start = region_loops[0] if region_loops else next(
        i for i, line in enumerate(lines) if "if (Count[" in line)
    assert sum("if (begin < end) {" in line or "if (begin_1 < end_1) {" in line
               for line in lines) == 2
    declarations = {}
    for i, line in enumerate(lines):
        match = re.fullmatch(r"\s*float ((?:cu[0-3]|OUT[0-3](?:_1)?|out(?:_1)?))\[([0-9]+)\];", line)
        if match:
            declarations[match.group(1)] = {"line": i + 1, "float_slots_per_thread": int(match.group(2))}
    expected = {**{f"cu{i}": 8 for i in range(4)}, "out": 16, "out_1": 32,
                "OUT0": 8, "OUT1": 8, "OUT0_1": 8, "OUT1_1": 8, "OUT2": 8, "OUT3": 8}
    assert {name: value["float_slots_per_thread"] for name, value in declarations.items()} == expected
    assert all(value["line"] - 1 < start for value in declarations.values())
    names = r"\b(?:cu[0-3]|OUT[0-3](?:_1)?|out(?:_1)?)\b"
    assert not any("buf_dyn_shmem" in line and re.search(names, line) for line in lines)
    initial, gather, slot_copies, scatter, final = {}, [], [], [], []
    for bank in range(4):
        rows = [{"line": i + 1, "text": line.strip()} for i, line in enumerate(lines[:start])
                if f"*(float4*)(cu{bank} +" in line]
        assert len(rows) == 1 and "make_float4" in rows[0]["text"]
        initial[f"cu{bank}"] = rows
    for i, line in enumerate(lines):
        row = {"line": i + 1, "text": line.strip()}
        if re.search(r"condval(?:_[0-9]+)? = cu[0-3]\[", line):
            gather.append(row)
        if re.search(r"OUT[0-3](?:_1)?\[[^]]+\] = out(?:_1)?\[", line):
            slot_copies.append(row)
        if re.search(r"cu[0-3]\[[^]]+\] = (?:OUT[0-3](?:_1)?\[|condval)", line):
            scatter.append(row)
        if "U +" in line:
            final.append(row)
            assert "buf_dyn_shmem" not in line and re.search(r"\*\(float2\*\)\(cu[0-3] \+", line)
    assert len(gather) == 24 and len(slot_copies) == 6 and len(scatter) == batch * 2
    assert len(final) == batch and {re.search(r"\(cu([0-3]) \+", row["text"]).group(1)
                                            for row in final} == {str(i) for i in range(batch)}
    assert not any(re.search(r"cu[0-3]\[[^]]*\b(?:request|label|Owner|RowMap)\b", line) for line in lines)
    scalar_shared = [{"line": i + 1, "text": line.strip()} for i, line in enumerate(lines)
                     if "buf_dyn_shmem" in line and (
                         re.search(r"= (?:maximum(?:_1)?|denominator(?:_1)?)\[", line)
                         or re.search(r"condval(?:_[0-9]+)? = \(\(float\*\)buf_dyn_shmem\).*\bsrc_[0-9]+\b", line))]
    assert len(scalar_shared) == 8
    reductions = sum("AllReduce" in line and "buf_dyn_shmem" in line for line in lines)
    assert reductions == 4
    return {"local_float_arrays": declarations, "initialization_before_region": initial,
            "region_loop_elided": not region_loops,
            "canonical_CU_gather_local_array_read_sites": len(gather),
            "MMA_out_to_independent_OUT_local_copy_sites": slot_copies,
            "OUT_to_fixed_CU_local_write_sites": scatter,
            "final_U_direct_from_CU_banks": final,
            "no_CU_OUT_or_MMA_out_reference_with_shared_address": True,
            "no_request_or_label_dependent_CU_array_index": True,
            "CM_CL_shared_reference_sites": scalar_shared,
            "shared_cross_warp_reduction_sites": reductions,
            "source_whole_CTA_barrier_sites": sum("__syncthreads()" in line for line in lines),
            "shared_KV_Q_P_scalar_and_reduction_scratch_remain": True}


def independent_launch_proof(host_path, meta_path):
    """Extract literal launch tail independently of branchtile.resources_v2."""
    host, meta = host_path.read_text(), json.loads(meta_path.read_text())
    assert set(meta["func_info"]) == {"main_kernel"}
    info = meta["func_info"]["main_kernel"]
    tags = info["launch_param_tags"]
    argument_count = len(info["arg_types"])
    call = re.search(r"TVMFFIFunctionCall\(main_kernel_packed, \(TVMFFIAny\*\) ([a-zA-Z_]\w*), (\d+),", host)
    assert call and int(call.group(2)) == argument_count + len(tags)
    stack = re.escape(call.group(1))
    prefix = host[:call.start()]
    assignments = {}
    for match in re.finditer(r"\(\(\(TVMFFIAny\*\)" + stack + r"\)\[(\d+)\]\.v_int64\) = \(\(int64_t\)(\d+)\);", prefix):
        assignments[int(match.group(1))] = int(match.group(2))
    types = {int(m.group(1)): int(m.group(2)) for m in re.finditer(
        r"\(\(\(TVMFFIAny\*\)" + stack + r"\)\[(\d+)\]\.type_index\) = (\d+);", prefix)}
    values = {}
    for offset, tag in enumerate(tags):
        slot = argument_count + offset
        assert types[slot] == 1 and tag not in values
        values[tag] = assignments[slot]
    assert set(values) == {"blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"}
    return {"launch_grid": [values["blockIdx.x"], values["blockIdx.y"], 1],
            "launch_block": [values[f"threadIdx.{axis}"] for axis in "xyz"],
            "dynamic_smem_bytes": values["tir.use_dyn_shared_memory"],
            "launch_param_tags": tags, "kernel_argument_count": argument_count,
            "packed_call_argument_count": int(call.group(2))}


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert "torch" not in sys.modules and "tilelang" not in sys.modules
    inputs = {}

    def record(path):
        inputs[rel(path)] = sha(path)

    structure = structural_checks()[0]
    record(ROOT / "scripts/check_lane_select_split_banks_cpu.py")
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]

    def terminal(name):
        matches = [entry for entry in ledger if entry["event"] == "end" and entry["name"] == name]
        assert len(matches) == 1 and not matches[0]["timed_out"]
        return matches[0]

    paths = sorted((SOURCE / "wave14").glob("*.json"))
    assert len(paths) == 4
    jobs, cases = [], []
    common = None
    for path in paths:
        record(path)
        result = json.loads(path.read_text())
        assert result["status"] == "passed_numerical_and_mutable_graph_resource_review_pending"
        assert result["uid"] == os.getuid() and len(result["cases"]) == 5
        assert result["arguments"]["module"] == "branchtile.attention_lane_select"
        assert result["arguments"]["threads"] == 128
        if common is None:
            common = result["code_sha256"]
        assert result["code_sha256"] == common
        for source, expected in common.items():
            assert sha(ROOT / source) == expected
            record(ROOT / source)
        end = terminal(path.stem)
        assert end["exit_code"] == 0
        jobs.append({"name": path.stem, "result": rel(path), "raw_status_preserved": result["status"],
                     "terminal": end, "independent_resource_audit": "passed", "arguments": result["arguments"],
                     "whole_qualification_wall_seconds": result["wall_seconds"],
                     "initialization_and_first_call_seconds_by_case": {
                         c["name"]: c["initialization_and_first_call_seconds"] for c in result["cases"]}})
        for case in result["cases"]:
            assert case["status"] == "passed_numerical_and_mutable_graph"
            checks = {check["phase"]: check for check in case["checks"]}
            assert len(checks) == len(case["checks"]) == 6 and set(checks) == PHASES
            assert all(check["status"] == "passed" for check in checks.values())
            assert checks["eager"]["O_sha256"] == checks["graph_baseline"]["O_sha256"]
            assert checks["graph_baseline"]["Q_sha256"] != checks["graph_changed_Q"]["Q_sha256"]
            for tensor in ("K", "V"):
                assert checks["graph_changed_Q"][f"{tensor}_sha256"] != checks["graph_changed_KV"][f"{tensor}_sha256"]
            prefix = path.parent / (path.stem + "_compiler") / case["name"]
            artifacts = {suffix: prefix.with_suffix(suffix) for suffix in
                         (".cu", ".cubin", ".resources.txt", ".sass", ".host.c", ".tvm_meta.json")}
            receipt = case["resources"]
            for artifact in artifacts.values():
                record(artifact)
            for suffix, key in ((".cu", "generated_source_sha256"), (".cubin", "cubin_sha256"),
                                (".sass", "sass_sha256"), (".host.c", "generated_host_source_sha256"),
                                (".tvm_meta.json", "cuda_metadata_sha256")):
                assert sha(artifacts[suffix]) == receipt[key]
            fresh = {}
            for flag, suffix in (("--dump-resource-usage", ".resources.txt"), ("--dump-sass", ".sass")):
                fresh[suffix] = subprocess.run([str(CUOBJDUMP), flag, str(artifacts[".cubin"])],
                                               check=True, capture_output=True, text=True).stdout
                assert fresh[suffix] == artifacts[suffix].read_text()
            fields = {key.lower(): int(value) for key, value in
                      re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", fresh[".resources.txt"])}
            assert fields == receipt["cubin_resources"]
            local_sites = re.findall(r"\b(?:LDL|STL)(?:\.|\s)", fresh[".sass"])
            assert not local_sites and receipt["sass_local_load_store_count"] == 0
            assert fields["local"] == fields["stack"] == 0 and receipt["resource_spill_free"]
            launch = independent_launch_proof(artifacts[".host.c"], artifacts[".tvm_meta.json"])
            for key, value in launch.items():
                assert receipt[key] == value
            assert launch["launch_block"] == [128, 1, 1]
            assert launch["launch_grid"] == [case["q_shape"][0], result["arguments"]["splits"], 1]
            assert launch["dynamic_smem_bytes"] + fields["shared"] == receipt["total_smem_bytes"]
            opcodes = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+[0-9]*\s+)?([A-Z][A-Z0-9_.]*)", fresh[".sass"])
            counts = collections.Counter(op.split(".")[0] for op in opcodes)
            assert counts["LDS"] > 0 and counts["STS"] > 0
            cases.append({"job": path.stem, "case": case["name"], "B": case["batch"],
                          "N": result["arguments"]["block_n"], "splits": result["arguments"]["splits"],
                          "fresh_cubin_resources": fields, "launch_independently_reextracted": launch,
                          "total_smem_bytes": receipt["total_smem_bytes"],
                          "static_SASS_instruction_site_counts": dict(sorted(counts.items())),
                          "generated_state_proof": generated_state_proof(artifacts[".cu"], case["batch"]),
                          "saved_numerical_and_mutable_graph_checks": 6,
                          "eager_graph_O_bitwise_equal": True,
                          "offset_values_changed": checks["graph_changed_offsets_and_indices"]["offset_values_changed"],
                          "maximum_recorded_O_absolute_error": max(c["O_max_abs_fp64"] for c in checks.values()),
                          "maximum_recorded_finite_LSE_absolute_error": max(c["LSE_max_abs_fp64_finite_rows"] for c in checks.values())})
    assert len(cases) == 20
    failed = []
    for path in sorted((SOURCE / "wave11").glob("*.json")):
        record(path)
        result = json.loads(path.read_text())
        end = terminal(path.stem)
        assert result["status"] == "failed" and end["exit_code"] == 1
        assert "cu: [i + 8, d] and [i, d]" in result["error"]
        assert all(not case["checks"] and "resources" not in case for case in result["cases"])
        failed.append({"name": path.stem, "raw_status_preserved": result["status"], "terminal": end,
                       "error": result["error"], "source_sha256_at_run": result["code_sha256"],
                       "compiled_cases": 0, "passed_checks": 0})
    assert len(failed) == 4
    resource_groups = []
    for b, n in sorted({(c["B"], c["N"]) for c in cases}):
        selected = [c for c in cases if (c["B"], c["N"]) == (b, n)]
        resource_groups.append({"B": b, "N": n, "cases": len(selected),
                                "REG_min": min(c["fresh_cubin_resources"]["reg"] for c in selected),
                                "REG_max": max(c["fresh_cubin_resources"]["reg"] for c in selected),
                                "total_SMEM_bytes": sorted({c["total_smem_bytes"] for c in selected})})
    report = {"schema": 1, "status": "independent_lane_select_split_bank_artifact_audit_passed",
              "GPU_hours": 0, "torch_imported": "torch" in sys.modules,
              "tilelang_imported": "tilelang" in sys.modules,
              "auditor_sha256": sha(__file__), "cuobjdump_sha256": sha(CUOBJDUMP), "input_sha256": inputs,
              "CPU_source_structure": structure, "qualification_common_code_sha256": common,
              "all_wave14_qualification_code_hashes_identical_and_current": True,
              "jobs": jobs, "cases": cases, "resource_groups": resource_groups,
              "saved_numerical_and_mutable_graph_checks": 120,
              "all_fresh_resource_SASS_match_saved": True,
              "all_launch_host_metadata_match_saved": True,
              "all_LOCAL_STACK_LDL_STL_zero": True,
              "all_generated_sources_have_no_canonical_shared_U_transport": True,
              "earlier_wave11_failed_jobs_retained": failed,
              "wave14_qualification_GPU_hours_charged": sum(j["terminal"]["gpu_hours"] for j in jobs),
              "wave11_failed_GPU_hours_charged": sum(j["terminal"]["gpu_hours"] for j in failed),
              "limitations": [
                  "Saved numerical receipts are checked; no GPU outputs or performance are recomputed in this CPU audit.",
                  "Source local arrays plus zero cubin LOCAL/STACK and no SASS LDL/STL support no observed spills only for these 20 compiled cases.",
                  "Shared CM/CL RowMap transport, Q/K/V/P storage and cross-warp reduction scratch remain; this is not zero shared-memory traffic.",
                  "Static SASS counts are instruction sites, not dynamic traffic, stalls, executed work or critical-path attribution.",
                  "Initialization_and_first_call_seconds include compilation, allocation and first execution; whole wall includes all qualification work. Neither is isolated JIT.",
                  "The unequal fixture reuses an in-process compiled signature, so its submillisecond setup cannot be compared with cold signatures.",
                  "Only N32/S3 and N64/S64 at B3/B4 are qualified here; timing and other split/BN grids need separate qualification.",
                  "Mutable tests cover Q/KV/indices/offsets, not arbitrary changed Labels or graph topology.",
                  "CPU mathematical routing passed for wave11 despite compiler failure; split-bank wave14 is a changed source, not inherited qualification.",
                  "No speedup, full-model latency, energy, task quality or novelty claim follows from this resource audit."]}
    assert not report["torch_imported"] and not report["tilelang_imported"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "jobs": len(jobs), "cases": len(cases),
                      "saved_checks": 120, "resource_groups": resource_groups,
                      "failed_wave11_jobs_retained": len(failed), "GPU_hours": 0}, indent=2))


if __name__ == "__main__":
    main()
