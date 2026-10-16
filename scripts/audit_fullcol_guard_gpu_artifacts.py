#!/usr/bin/env python3
"""CPU-only independent review of saved FullCol GPU qualification artifacts."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/static_bank_v4_qualification/wave3"
OUTPUT = ROOT / "results/operator_fullcol_guard_gpu_audit_v1"
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
PHASES = {"eager", "graph_baseline", "graph_changed_Q", "graph_changed_KV",
          "graph_changed_indices", "graph_changed_offsets_and_indices"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def semantic_ast(path, remove_policy=False, rename_import=False):
    tree = ast.parse(path.read_text())
    if (isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body = tree.body[1:]
    policies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and remove_policy:
            for keyword in node.keywords:
                if keyword.arg == "policy":
                    policies.append(ast.unparse(keyword.value))
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "policy"]
        if isinstance(node, ast.ImportFrom) and rename_import:
            node.module = node.module.replace("tilelang_kernel_fullcol_guard", "tilelang_kernel_emptyguard")
    return ast.dump(tree, include_attributes=False), policies


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert "torch" not in sys.modules
    inputs = {}

    def record(path):
        inputs[rel(path)] = sha(path)

    full_source = ROOT / "src/branchtile/tilelang_kernel_fullcol_guard.py"
    guard_source = ROOT / "src/branchtile/tilelang_kernel_emptyguard.py"
    full_wrapper = ROOT / "src/branchtile/attention_fullcol_guard.py"
    guard_wrapper = ROOT / "src/branchtile/attention_emptyguard.py"
    a, policies = semantic_ast(full_source, remove_policy=True)
    b, original_policies = semantic_ast(guard_source, remove_policy=True)
    assert policies == ["T.GemmWarpPolicy.FullCol"] * 2
    assert original_policies == ["T.GemmWarpPolicy.FullRow"] * 2 and a == b
    assert semantic_ast(full_wrapper, rename_import=True)[0] == semantic_ast(guard_wrapper)[0]
    text = full_source.read_text()
    required_source = ["cu = T.alloc_shared((rows, dim), \"float32\")",
                       "out[i, d] = T.if_then_else(valid, CU[src, d], 0)",
                       "CU[src, d] = out[i, d]"]
    assert all(fragment in text for fragment in required_source)
    for path in (full_source, guard_source, full_wrapper, guard_wrapper):
        record(path)
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    jobs, cases = [], []
    paths = sorted(SOURCE.glob("staticbank_w3_fullcol_guard_*.json"))
    assert len(paths) == 4
    for path in paths:
        record(path)
        result = json.loads(path.read_text())
        assert result["status"] == "passed_numerical_and_mutable_graph_resource_review_pending"
        assert result["arguments"]["module"] == "branchtile.attention_fullcol_guard"
        assert result["uid"] == os.getuid() and len(result["cases"]) == 5
        for source_path in (full_source, full_wrapper, guard_source, guard_wrapper):
            assert result["code_sha256"][rel(source_path)] == sha(source_path)
        terminal = [entry for entry in ledger
                    if entry["name"] == path.stem and entry["event"] == "end"]
        assert len(terminal) == 1 and terminal[0]["exit_code"] == 0 and not terminal[0]["timed_out"]
        job = {"name": path.stem, "result_path": rel(path),
               "original_status_preserved": result["status"],
               "independent_resource_review_status": "passed",
               "source_sha256_at_run": result["code_sha256"],
               "terminal": terminal[0], "arguments": result["arguments"]}
        jobs.append(job)
        compiler = path.parent / (path.stem + "_compiler")
        for case in result["cases"]:
            assert case["status"] == "passed_numerical_and_mutable_graph"
            assert {check["phase"] for check in case["checks"]} == PHASES
            assert all(check["status"] == "passed" for check in case["checks"])
            prefix = compiler / case["name"]
            files = {suffix: prefix.with_suffix(suffix)
                     for suffix in (".cu", ".cubin", ".resources.txt", ".sass")}
            for artifact in files.values():
                record(artifact)
            resource = case["resources"]
            for suffix, key in ((".cu", "generated_source_sha256"), (".cubin", "cubin_sha256"),
                                (".sass", "sass_sha256")):
                assert sha(files[suffix]) == resource[key]
            fresh_resource = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(files[".cubin"])],
                                            check=True, capture_output=True, text=True).stdout
            fresh_sass = subprocess.run([str(CUOBJDUMP), "--dump-sass", str(files[".cubin"])],
                                        check=True, capture_output=True, text=True).stdout
            assert fresh_resource == files[".resources.txt"].read_text()
            assert fresh_sass == files[".sass"].read_text()
            fields = {key.lower(): int(value) for key, value in
                      re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", fresh_resource)}
            assert fields == resource["cubin_resources"]
            assert resource["total_smem_bytes"] == resource["dynamic_smem_bytes"] + fields["shared"]
            local_ops = re.findall(r"\b(?:LDL|STL)(?:\.|\s)", fresh_sass)
            assert len(local_ops) == resource["sass_local_load_store_count"] == 0
            assert fields["local"] == fields["stack"] == 0 and resource["resource_spill_free"]
            opcodes = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+[0-9]*\s+)?([A-Z][A-Z0-9_.]*)", fresh_sass)
            counts = collections.Counter(opcode.split(".")[0] for opcode in opcodes)
            assert counts["LDS"] > 0 and counts["STS"] > 0
            lines = files[".cu"].read_text().splitlines()
            gather = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
                      if "= *(float2*)(((float*)buf_dyn_shmem)" in line and "src_" in line]
            scatter = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
                       if "buf_dyn_shmem" in line and "= *(float2*)(out" in line]
            assert len(gather) == len(scatter) == 2
            rows = case["batch"] * 8
            base = rows * 2
            assert all(f"(int64_t){base})" in item["text"] for item in (*gather, *scatter))
            rowmap = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
                      if re.search(r"int src_\d+ = RowMap", line)]
            final_u = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
                       if "*(float4*)(U +" in line and "buf_dyn_shmem" in line]
            assert rowmap and len(final_u) == 1
            checks = {check["phase"]: check for check in case["checks"]}
            assert checks["eager"]["O_sha256"] == checks["graph_baseline"]["O_sha256"]
            assert checks["graph_baseline"]["Q_sha256"] != checks["graph_changed_Q"]["Q_sha256"]
            assert checks["graph_changed_Q"]["K_sha256"] != checks["graph_changed_KV"]["K_sha256"]
            assert checks["graph_changed_Q"]["V_sha256"] != checks["graph_changed_KV"]["V_sha256"]
            cases.append({"job": path.stem, "case": case["name"], "B": case["batch"],
                          "N": result["arguments"]["block_n"], "splits": result["arguments"]["splits"],
                          "numerical_mutable_graph_checks": 6, "O_eager_graph_bitwise": True,
                          "resources": resource, "independently_dumped_cubin_fields": fields,
                          "static_sass_opcode_counts": dict(sorted(counts.items())),
                          "generated_CUDA_CU_shared_float_offset": base,
                          "generated_CUDA_CU_shared_byte_offset": base * 4,
                          "canonical_U_payload_bytes": rows * 128 * 4,
                          "generated_CUDA_proof": {"RowMap": rowmap, "gather": gather,
                                                    "scatter": scatter, "final_U_from_shared": final_u},
                          "graph_offset_values_changed": checks["graph_changed_offsets_and_indices"]["offset_values_changed"],
                          "maximum_recorded_O_absolute_error": max(check["O_max_abs_fp64"] for check in checks.values()),
                          "maximum_recorded_finite_LSE_absolute_error": max(check["LSE_max_abs_fp64_finite_rows"] for check in checks.values())})
    assert len(cases) == 20
    result = {"schema": 1, "status": "independent_fullcol_control_artifact_audit_passed",
              "scope": "Saved GPU numerical receipts and compiled artifact review only; no new GPU work or speed measurement.",
              "GPU_hours": 0, "torch_imported": "torch" in sys.modules, "uid": os.getuid(),
              "auditor_sha256": sha(__file__), "input_sha256": inputs,
              "cuobjdump_path": str(CUOBJDUMP), "cuobjdump_sha256": sha(CUOBJDUMP),
              "FullCol_vs_emptyguard_AST_equal_after_two_policy_removals": True,
              "wrapper_AST_equal_after_import_rename": True,
              "two_GEMM_policies": policies,
              "emptyguard_original_two_GEMM_policies": original_policies,
              "original_pending_status_interpretation": "Numerical and mutable-graph qualification passed; pending indicated manual resource review, not numerical failure. Original files remain unchanged.",
              "jobs": jobs, "cases": cases, "numerical_checks": 120,
              "all_saved_cubin_source_SASS_hashes_matched": True,
              "all_fresh_cuobjdump_outputs_matched_saved_artifacts": True,
              "all_LOCAL_STACK_LDL_STL_zero": True,
              "canonical_U_shared_gather_scatter_confirmed_in_all_generated_sources": True,
              "limitations": [
                  "Static SASS opcode counts are instruction sites, not executed instruction counts or memory-traffic volumes.",
                  "LDS/STS include KV, Q, P and reduction traffic; this audit does not attribute each SASS site to canonical U.",
                  "Zero LOCAL/STACK/LDL/STL supports absence of explicit local-memory traffic in these cubins; it does not imply low register pressure or high occupancy.",
                  "Dynamic SMEM size is retained from saved launch receipts; it was not reconstructed from a new kernel launch.",
                  "Five fixture cases per shape and repeated phases are not twenty independent task-quality samples.",
                  "This FullCol control retains shared canonical U and runtime RowMap gather/scatter. It is not the proposed register-bank v4 mechanism.",
                  "No timing, application-quality, energy or multi-rank critical-path result is produced."]}
    assert "torch" not in sys.modules
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "jobs": len(jobs), "cases": len(cases),
                      "numerical_checks": 120, "GPU_hours": 0}))


if __name__ == "__main__":
    main()
