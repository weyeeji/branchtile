#!/usr/bin/env python3
"""CPU-only review of saved 256-thread FullCol qualification artifacts.

No package with a CUDA runtime is imported, no device is queried, and no kernel
is compiled or launched. An unsuccessful process never inherits another job's
numerical qualification, even when its own partial cubins are spill-free.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/operator_wide_cta_qualification"
OUTPUT = ROOT / "results/operator_wide_cta_gpu_audit_v1"
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
PHASES = {"eager", "graph_baseline", "graph_changed_Q", "graph_changed_KV",
          "graph_changed_indices", "graph_changed_offsets_and_indices"}
FAILED = "fullcol_w13_b3_n64_s3_t256"
PASSED = "passed_numerical_and_mutable_graph_resource_review_pending"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def extract_launch(host, metadata):
    """Independently read the seven trailing slots of the actual packed call.

    This intentionally does not import the receipt-producing resources_v2.
    Metadata establishes slot order; host assignments establish values. The
    kernel declaration independently establishes __launch_bounds__(256, 1).
    """
    assert set(metadata["func_info"]) == {"main_kernel"}
    info = metadata["func_info"]["main_kernel"]
    tags = info["launch_param_tags"]
    expected_tags = ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x",
                     "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"]
    assert tags == expected_tags and info["arg_types"] == ["handle"] * 10
    calls = list(re.finditer(
        r"TVMFFIFunctionCall\(main_kernel_packed, \(TVMFFIAny\*\)\s*(\w+),\s*(\d+),",
        host))
    assert len(calls) == 1 and int(calls[0][2]) == 17
    call = calls[0]
    prefix, stack = host[:call.start()], re.escape(call[1])
    assignments = {}
    for field in ("v_int64", "type_index"):
        pattern = (r"\(\(\(TVMFFIAny\*\)" + stack
                   + r"\)\[(\d+)\]\." + field + r"\)\s*=\s*([^;]+);")
        assignments[field] = {int(m[1]): m[2].strip() for m in re.finditer(pattern, prefix)}
    values, proof = [], []
    for slot, tag in enumerate(tags, 10):
        assert assignments["type_index"][slot] == "1"
        expression = assignments["v_int64"][slot]
        match = re.fullmatch(r"\(\(int64_t\)(\d+)\)", expression)
        assert match, (slot, expression)
        value = int(match[1])
        values.append(value)
        proof.append({"slot": slot, "tag": tag, "value": value, "expression": expression})
    return {"launch_grid": values[:3], "launch_block": values[3:6],
            "dynamic_smem_bytes": values[6], "launch_param_tags": tags,
            "launch_arguments": proof, "kernel_argument_count": 10,
            "packed_call_argument_count": 17, "kernel_name": "main_kernel"}


def cuda_u_proof(source, rows):
    lines = source.splitlines()
    assert len(re.findall(r"__launch_bounds__\(256, 1\)", source)) == 1
    gather = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
              if "= *(float2*)(((float*)buf_dyn_shmem)" in line and "src_" in line]
    scatter = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
               if "buf_dyn_shmem" in line and "= *(float2*)(out" in line]
    rowmap = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
              if re.search(r"int src_\d+ = RowMap", line)]
    final = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
             if "*(float4*)(U +" in line and "buf_dyn_shmem" in line]
    assert len(gather) == len(scatter) == 2 and rowmap and len(final) == 1
    base = rows * 2
    assert all(f"(int64_t){base}))" in item["text"] for item in gather + scatter)
    assert f"+ {base}));" in final[0]["text"]
    return {"RowMap": rowmap, "gather": gather, "scatter": scatter,
            "final_U_from_shared": final, "CU_shared_float_offset": base,
            "CU_shared_byte_offset": base * 4, "canonical_U_payload_bytes": rows * 128 * 4,
            "launch_bounds_threads": 256}


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not ({"torch", "tilelang"} & set(sys.modules))
    inputs = {}

    def record(path):
        inputs[rel(path)] = sha(path)

    candidate = [ROOT / "src/branchtile/attention_fullcol_guard.py",
                 ROOT / "src/branchtile/tilelang_kernel_fullcol_guard.py"]
    for path in candidate:
        record(path)
    kernel_source = candidate[1].read_text()
    assert all(fragment in kernel_source for fragment in (
        'cu = T.alloc_shared((rows, dim), "float32")',
        "out[i, d] = T.if_then_else(valid, CU[src, d], 0)",
        "CU[src, d] = out[i, d]", "threads=threads"))
    assert kernel_source.count("policy=T.GemmWarpPolicy.FullCol") == 2
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    jobs, cases = [], []
    paths = sorted(SOURCE.glob("wave13/*.json")) + sorted(SOURCE.glob("wave15/*.json"))
    assert len(paths) == 8
    for path in paths:
        record(path)
        raw = json.loads(path.read_text())
        args = raw["arguments"]
        assert raw["contracts"] == {"O": {"atol": 0.006, "rtol": 0.015},
                                     "natural_log_LSE": {"atol": 0.0003, "rtol": 0.0001}}
        assert args["module"] == "branchtile.attention_fullcol_guard"
        assert args["threads"] == 256 and args["block_n"] == 64 and raw["uid"] == os.getuid()
        terminals = [row for row in ledger if row["name"] == path.stem and row["event"] == "end"]
        assert len(terminals) == 1 and not terminals[0]["timed_out"]
        terminal = terminals[0]
        failed = path.stem == FAILED
        if failed:
            assert terminal["exit_code"] == -11 and raw["status"] == "running"
            assert [case["name"] for case in raw["cases"]] == ["all_masks", "unequal", "empty", "one_token"]
            effective = "failed_process_SIGSEGV_partial_numerical_receipts_retained"
        else:
            assert terminal["exit_code"] == 0 and raw["status"] == PASSED
            assert len(raw["cases"]) == (5 if path.parent.name == "wave13" else 1)
            effective = "passed_numerical_mutable_graph_and_independent_resource_review"
        source_review = {}
        for source_path, at_run in raw["code_sha256"].items():
            current = ROOT / source_path
            source_review[source_path] = {"at_run": at_run,
                                         "current": sha(current) if current.is_file() else None,
                                         "current_matches_at_run": current.is_file() and sha(current) == at_run}
        assert all(source_review[rel(p)]["current_matches_at_run"] for p in candidate)
        jobs.append({"name": path.stem, "wave": path.parent.name, "result_path": rel(path),
                     "raw_status_preserved": raw["status"], "effective_status": effective,
                     "terminal": terminal, "terminal_sha256": digest_json(terminal),
                     "arguments": args, "completed_cases": len(raw["cases"]),
                     "completed_numerical_checks": len(raw["cases"]) * 6,
                     "complete_real_capture_qualification": not failed,
                     "source_hash_review": source_review,
                     "wall_seconds_if_completed": raw.get("wall_seconds"),
                     "performance_qualification_inherited_from_other_jobs": False})
        compiler = path.parent / (path.stem + "_compiler")
        for case in raw["cases"]:
            assert case["status"] == "passed_numerical_and_mutable_graph"
            assert len(case["checks"]) == 6 and {c["phase"] for c in case["checks"]} == PHASES
            assert all(c["status"] == "passed" for c in case["checks"])
            prefix = compiler / case["name"]
            files = {suffix: prefix.with_suffix(suffix) for suffix in
                     (".cu", ".cubin", ".resources.txt", ".sass", ".host.c", ".tvm_meta.json")}
            for artifact in files.values():
                record(artifact)
            saved = case["resources"]
            for suffix, key in ((".cu", "generated_source_sha256"), (".cubin", "cubin_sha256"),
                                (".sass", "sass_sha256"), (".host.c", "generated_host_source_sha256"),
                                (".tvm_meta.json", "cuda_metadata_sha256")):
                assert sha(files[suffix]) == saved[key]
            fresh_resource = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(files[".cubin"])],
                                            check=True, capture_output=True, text=True).stdout
            fresh_sass = subprocess.run([str(CUOBJDUMP), "--dump-sass", str(files[".cubin"])],
                                        check=True, capture_output=True, text=True).stdout
            assert fresh_resource == files[".resources.txt"].read_text()
            assert fresh_sass == files[".sass"].read_text()
            fields = {key.lower(): int(value) for key, value in
                      re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", fresh_resource)}
            assert fields == saved["cubin_resources"] and fields["local"] == fields["stack"] == 0
            assert fields["shared"] == 1024 and 114 <= fields["reg"] <= 124
            assert len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", fresh_sass)) == saved["sass_local_load_store_count"] == 0
            assert saved["resource_spill_free"]
            opcodes = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+[0-9]*\s+)?([A-Z][A-Z0-9_.]*)", fresh_sass)
            counts = collections.Counter(op.split(".")[0] for op in opcodes)
            assert counts["LDS"] > 0 and counts["STS"] > 0
            launch = extract_launch(files[".host.c"].read_text(), json.loads(files[".tvm_meta.json"].read_text()))
            assert all(saved[key] == value for key, value in launch.items())
            assert launch["launch_block"] == [256, 1, 1]
            assert launch["launch_grid"] == [case["q_shape"][0], args["splits"], 1]
            expected_dynamic = 61632 if case["batch"] == 3 else 65792
            assert launch["dynamic_smem_bytes"] == expected_dynamic
            assert saved["total_smem_bytes"] == expected_dynamic + fields["shared"]
            proof = cuda_u_proof(files[".cu"].read_text(), case["batch"] * 8)
            checks = {check["phase"]: check for check in case["checks"]}
            assert checks["eager"]["O_sha256"] == checks["graph_baseline"]["O_sha256"]
            assert checks["graph_baseline"]["Q_sha256"] != checks["graph_changed_Q"]["Q_sha256"]
            assert checks["graph_changed_Q"]["K_sha256"] != checks["graph_changed_KV"]["K_sha256"]
            assert checks["graph_changed_Q"]["V_sha256"] != checks["graph_changed_KV"]["V_sha256"]
            cases.append({"job": path.stem, "case": case["name"], "B": case["batch"],
                          "N": args["block_n"], "splits": args["splits"], "threads": 256,
                          "job_process_completed_successfully": not failed,
                          "numerical_mutable_graph_checks": 6, "O_eager_graph_bitwise": True,
                          "independently_dumped_cubin_fields": fields, "resources": saved,
                          "independently_extracted_launch": launch,
                          "static_sass_opcode_counts": dict(sorted(counts.items())),
                          "generated_CUDA_shared_U_proof": proof,
                          "graph_offset_values_changed": checks["graph_changed_offsets_and_indices"]["offset_values_changed"],
                          "initialization_and_first_call_seconds": case["initialization_and_first_call_seconds"],
                          "maximum_recorded_O_absolute_error": max(c["O_max_abs_fp64"] for c in checks.values()),
                          "maximum_recorded_finite_LSE_absolute_error": max(c["LSE_max_abs_fp64_finite_rows"] for c in checks.values())})
    assert len(cases) == 23 and len(jobs) == 8
    failed_jobs = [job for job in jobs if job["terminal"]["exit_code"] != 0]
    assert len(failed_jobs) == 1 and failed_jobs[0]["name"] == FAILED
    configurations = [(job["arguments"]["capture"], job["arguments"]["splits"],
                       job["arguments"]["batch"]) for job in jobs if job["wave"] == "wave15"]
    assert len(set(configurations)) == 4
    failed_configuration = tuple(failed_jobs[0]["arguments"][key] for key in ("capture", "splits", "batch"))
    assert failed_configuration not in configurations
    result = {"schema": 1, "status": "saved_artifact_audit_passed_with_one_failed_qualification_job",
              "scope": "CPU-only review of saved numerical receipts and compiled artifacts; no new GPU work or speed measurement.",
              "GPU_hours": 0, "uid": os.getuid(), "python": sys.executable,
              "torch_or_tilelang_imported": bool({"torch", "tilelang"} & set(sys.modules)),
              "auditor_sha256": sha(__file__), "input_sha256": inputs,
              "cuobjdump_path": str(CUOBJDUMP), "cuobjdump_sha256": sha(CUOBJDUMP),
              "jobs": jobs, "cases": cases, "completed_cubin_cases": 23,
              "numerical_checks_saved_and_passed": 138,
              "numerical_checks_in_clean_exit_jobs": 114,
              "partial_numerical_checks_in_SIGSEGV_job": 24,
              "clean_exit_jobs": 7, "failed_exit_jobs": 1,
              "qualification_GPU_hours_including_failure": sum(job["terminal"]["gpu_hours"] for job in jobs),
              "all_saved_cubin_source_host_metadata_SASS_hashes_matched": True,
              "all_fresh_cuobjdump_outputs_matched_saved_artifacts": True,
              "all_LOCAL_STACK_LDL_STL_zero": True,
              "all_actual_host_launch_blocks_256": True,
              "recorded_numerical_contracts": {"O": {"atol": 0.006, "rtol": 0.015},
                                               "natural_log_LSE": {"atol": 0.0003, "rtol": 0.0001}},
              "canonical_U_shared_gather_scatter_confirmed_in_all_generated_sources": True,
              "wave15_four_targets_distinct_from_failed_configuration": True,
              "limitations": [
                  "The failed B3/N64/S3/256 job remains failed (exit -11/SIGSEGV); its four passed fixtures do not qualify its absent real_capture case.",
                  "Wave15 uses four distinct real target/capture/split configurations. None retries or clears the failed B3/S3 configuration.",
                  "Zero LOCAL/STACK/LDL/STL in saved cubins does not establish performance, low register pressure or high occupancy.",
                  "Static SASS sites are not executed instruction counts or physical memory traffic; LDS/STS also include Q/KV/P and reduction data.",
                  "Generated CUDA proves this FullCol control still gathers/scatters canonical U through shared memory. It is not the register-bank mechanism.",
                  "Launch dimensions and dynamic SMEM are reconstructed from saved authoritative metadata and actual host call slots, without a new launch.",
                  "Initialization-and-first-call and completed-job wall times include work beyond JIT, so neither is an isolated compilation measurement.",
                  "The six phases of a fixture are correctness checks, not six independent task-quality samples.",
                  "No speed, application quality, energy or multi-rank critical-path claim is made."]}
    assert not result["torch_or_tilelang_imported"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "completed_cubin_cases",
                     "numerical_checks_saved_and_passed", "clean_exit_jobs", "failed_exit_jobs", "GPU_hours")}))


if __name__ == "__main__":
    main()
