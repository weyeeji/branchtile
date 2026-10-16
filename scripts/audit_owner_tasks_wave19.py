#!/usr/bin/env python3
"""Independently audit frozen wave19 receipts and cubins without importing CUDA."""
from __future__ import annotations

import argparse
import ast
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "configs/operator_qualification_wave19_jobs.json"
PROTOCOL = "configs/operator_qualification_wave19_protocol.json"
SOURCE_COMMIT, PROTOCOL_COMMIT = "b7927fe", "b7927fe"
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
EXPECTED_CASES = ("all_masks", "unequal", "empty", "one_token", "real_capture")
PHASES = ("eager", "graph_baseline", "graph_changed_Q", "graph_changed_KV",
          "graph_changed_indices", "rebuild_and_new_graph_changed_owners")
PASSED = "passed_numerical_fixed_topology_and_rebuild_recapture_resource_review_pending"
CONTRACTS = {"O": {"atol": .006, "rtol": .015},
             "natural_log_LSE": {"atol": .0003, "rtol": .0001}}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def git_blob(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                          check=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/operator_owner_tasks_wave19_audit_v2")
    args = parser.parse_args()
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    output = ROOT / args.output
    assert not output.exists(), "Use a new output path; preserve prior audits"
    errors, inputs, hashes, dump_cache = [], {}, {}, {}

    def check(condition, message):
        if not condition:
            errors.append(message)

    def sha(path):
        path = Path(path)
        if path not in hashes:
            with path.open("rb") as handle:
                hashes[path] = hashlib.file_digest(handle, "sha256").hexdigest()
        return hashes[path]

    def record(path):
        relative = str(path.relative_to(ROOT))
        inputs[relative] = sha(path)
        return inputs[relative]

    protocol = json.loads((ROOT / PROTOCOL).read_text())
    manifest = json.loads((ROOT / MANIFEST).read_text())
    for path in (MANIFEST, PROTOCOL):
        check(record(ROOT / path) == digest(git_blob(PROTOCOL_COMMIT, path)),
              f"Protocol commit mismatch: {path}")
    check(len(manifest) == 8, "Expected eight frozen jobs")
    check(protocol["expected_complete_cases"] == 40 and protocol["expected_complete_FP64_phases"] == 240, "Frozen expected counts differ")
    frozen_sources = {}
    for path, expected in protocol["code_sha256"].items():
        commit = PROTOCOL_COMMIT if path == MANIFEST else SOURCE_COMMIT
        frozen_sources[path] = digest(git_blob(commit, path))
        check(expected == frozen_sources[path], f"Protocol source hash mismatch: {path}")
        # Current changes are reported but do not rewrite run-time provenance.
        record(ROOT / path)
    source_current_matches = {p: inputs[p] == h for p, h in frozen_sources.items()}

    # Independent AST equivalence: only the accepted CLI choices and the
    # qualifier's own source filename differ. Kernel, planner and lifecycle
    # qualification logic remain the original wave17 mechanism/contract.
    old_qualifier = ast.parse(git_blob("aac3e85", "scripts/qualify_owner_tasks.py"))
    expanded_qualifier = ast.parse(git_blob(SOURCE_COMMIT, "scripts/qualify_owner_tasks_grid.py"))
    choices_replaced = 0
    for node in ast.walk(expanded_qualifier):
        if isinstance(node, ast.Constant) and node.value == "scripts/qualify_owner_tasks_grid.py":
            node.value = "scripts/qualify_owner_tasks.py"
        if isinstance(node, ast.keyword) and node.arg == "choices" and isinstance(node.value, ast.Tuple):
            values = [part.value if isinstance(part, ast.Constant) else None for part in node.value.elts]
            if values == [64, 128, 256, 512, 1024, 2048, 4096]:
                node.value = ast.Tuple(elts=[ast.Constant(value=value) for value in (256, 512, 1024)], ctx=ast.Load())
                choices_replaced += 1
    check(choices_replaced == 1 and ast.dump(old_qualifier, include_attributes=False)
          == ast.dump(expanded_qualifier, include_attributes=False), "Expanded qualifier changes more than CLI chunk choices/source filename")
    wave17 = json.loads(git_blob("d7ace33", "configs/operator_qualification_wave17_protocol.json"))
    for path in ("src/branchtile/attention_owner_tasks.py", "src/branchtile/owner_tasks_plan.py",
                 "src/branchtile/triton_kernel_owner_tasks.py"):
        check(frozen_sources[path] == wave17["code_sha256"][path], f"Runtime changed since wave17: {path}")
    expected_grid = {(3, 64, 32), (3, 128, 64), (3, 2048, 32), (3, 4096, 64),
                     (4, 64, 64), (4, 128, 32), (4, 2048, 64), (4, 4096, 32)}
    manifest_grid = set()
    for job in manifest:
        argv = job["argv"]
        manifest_grid.add(tuple(int(argv[argv.index(flag) + 1])
                                for flag in ("--batch", "--chunk-tokens", "--block-n")))
        check(argv[1] == "scripts/qualify_owner_tasks_grid.py" and argv[argv.index("--tokens") + 1] == "8197",
              f"{job['name']}: expanded fixture size/qualifier mismatch")
    check(manifest_grid == expected_grid, "Expanded qualification point set differs")

    ledger_bytes = (ROOT / "results/job_ledger.jsonl").read_bytes()
    ledger = [json.loads(line) for line in ledger_bytes.splitlines() if line.strip()]
    names = {job["name"] for job in manifest}
    selected_ledger = [entry for entry in ledger if entry.get("name") in names]
    prior_path = ROOT / "results/natural_capacity_boundary_v1/audit.json"
    record(prior_path)
    prior_hashes = json.loads(prior_path.read_text())["input_sha256"]
    jobs, cases, resources, topology, capture_checks = [], [], [], [], []
    statuses, numerical_statuses, resource_statuses = (collections.Counter() for _ in range(3))
    phases = collections.Counter()
    seen_gpus, gpu_hours, allocated_seconds = [], 0., 0.

    for job in manifest:
        name, argv = job["name"], job["argv"]
        terminal = [entry for entry in selected_ledger if entry["name"] == name and entry["event"] == "end"]
        starts = [entry for entry in selected_ledger if entry["name"] == name and entry["event"] == "start"]
        check(len(starts) == 1 and len(terminal) == 1, f"{name}: start/end count")
        for entry in starts + terminal:
            check(entry["uid"] == 1000 and entry["category"] == "P1"
                  and entry["gpus"] == job["gpus"] and entry["argv"] == argv,
                  f"{name}: ledger authorization/argv mismatch")
        for entry in terminal:
            gpu_hours += entry["gpu_hours"]
            allocated_seconds += entry["allocated_seconds"]
            check(math.isclose(entry["gpu_hours"], entry["allocated_seconds"] * len(job["gpus"]) / 3600,
                               abs_tol=1e-12), f"{name}: ledger cost mismatch")
        path = ROOT / argv[argv.index("--output") + 1]
        row = {"name": name, "result_path": str(path.relative_to(ROOT)),
               "manifest": job, "ledger_start": starts, "ledger_end": terminal}
        jobs.append(row)
        if not path.exists():
            statuses["missing_receipt"] += 1
            check(False, f"{name}: missing receipt")
            continue
        row["receipt_sha256"] = record(path)
        result = json.loads(path.read_text())
        row.update(original_status_preserved=result["status"],
                   failure_stage=result.get("failure_stage"), error=result.get("error"),
                   arguments=result["arguments"], source_sha256_at_run=result["code_sha256"],
                   gpu_uuid=result.get("gpu_uuid"), wall_seconds=result.get("wall_seconds"))
        statuses[result["status"]] += 1
        check(result["status"] == PASSED and result["stage"] == "completed", f"{name}: nonpassing/incomplete receipt")
        check(all(entry["exit_code"] == 0 and not entry["timed_out"] for entry in terminal),
              f"{name}: nonzero or timeout ledger terminal")
        check(result["uid"] == 1000 and result["python"].endswith("/.venv-latest/bin/python"),
              f"{name}: uid/virtualenv")
        check(len(job["gpus"]) == 1 and result["CUDA_VISIBLE_DEVICES"] == str(job["gpus"][0]),
              f"{name}: single physical GPU mapping")
        seen_gpus.extend(job["gpus"])
        for source, expected in result["code_sha256"].items():
            check(expected == frozen_sources.get(source), f"{name}: frozen source mismatch: {source}")
        check(len(result["code_sha256"]) == 7, f"{name}: missing source hashes")
        a = result["arguments"]
        for flag in ("batch", "chunk-tokens", "block-n", "tokens", "capture", "budget", "output"):
            value = a[flag.replace("-", "_")]
            check(str(value) == argv[argv.index("--" + flag) + 1], f"{name}: argument mismatch: {flag}")
        check(a["threads"] == 128 and a["splits"] == 64 and a["heads"] == 2
              and a["layer"] == 15 and a["step"] == 64 and a["module"] == "branchtile.attention_owner_tasks",
              f"{name}: unexpected fixed configuration")
        check(result["contracts"] == CONTRACTS, f"{name}: numerical contract")
        check(result["fixed_topology_phases"] == list(PHASES[:5])
              and result["topology_change_phase"] == PHASES[5], f"{name}: graph phase contract")
        check(tuple(c["name"] for c in result["cases"]) == EXPECTED_CASES, f"{name}: case identity/count")
        for case in result["cases"]:
            ident = name + "/" + case["name"]
            check(case["status"] == "passed_numerical_fixed_topology_graph_and_rebuilt_graph"
                  and case["resource_review_status"] == "exported_independent_review_pending",
                  f"{ident}: case numerical/resource export status")
            checks = case["checks"]
            check(tuple(item["phase"] for item in checks) == PHASES, f"{ident}: phase identity/count")
            mapped = {item["phase"]: item for item in checks}
            for item in checks:
                numerical_statuses[item["status"]] += 1
                phases[item["phase"]] += 1
                check(item["status"] == "passed", f"{ident}/{item['phase']}: numerical failure")
                check(all(math.isfinite(item[field]) and item[field] >= 0 for field in
                          ("O_max_abs_fp64", "O_relative_l2_fp64", "LSE_max_abs_fp64_finite_rows")),
                      f"{ident}/{item['phase']}: invalid recorded error")
            if all(phase in mapped for phase in PHASES):
                check(mapped["eager"]["O_sha256"] == mapped["graph_baseline"]["O_sha256"],
                      f"{ident}: eager/graph O not bitwise equal")
                for tensor, before, after in (("Q", "graph_baseline", "graph_changed_Q"),
                                              ("K", "graph_changed_Q", "graph_changed_KV"),
                                              ("V", "graph_changed_Q", "graph_changed_KV")):
                    check(mapped[before][tensor+"_sha256"] != mapped[after][tensor+"_sha256"],
                          f"{ident}: {tensor} mutation not evidenced")
                if case["name"] != "empty":
                    check(mapped["graph_changed_KV"]["owner_sha256"] != mapped["graph_changed_indices"]["owner_sha256"],
                          f"{ident}: indices mutation not evidenced")
                    check(mapped["graph_changed_indices"]["owner_sha256"] != mapped[PHASES[5]]["owner_sha256"],
                          f"{ident}: owner reassignment not evidenced")
            change = case.get("topology_change", {})
            check(change.get("old_op_validate_plan_rejected") is True
                  and change.get("old_graph_replayed_after_topology_mutation") is False
                  and change.get("graph_replay_bypasses_host_check") is True
                  and change.get("offset_version_changed_even_if_values_equal") is True
                  and "rebuild" in change.get("rejection", "").lower(), f"{ident}: topology invalidation contract")
            duration_fields = ("updated_plan_build_wall_seconds", "old_plan_mutation_and_rejection_wall_seconds",
                               "Attention_rebuild_wall_seconds", "warmup_JIT_recapture_first_replay_wall_seconds")
            durations = [change.get(field, -1) for field in duration_fields]
            check(all(math.isfinite(value) and value >= 0 for value in durations), f"{ident}: lifecycle costs missing")
            check(math.isclose(sum(durations), change.get("total_rebuild_sequence_wall_seconds", -1), abs_tol=1e-12),
                  f"{ident}: lifecycle accounting does not sum")
            topology.append({"job": name, "case": case["name"], **change})
            for stage in ("initial", "rebuilt"):
                meta = case[stage+"_metadata"]
                tasks = case[stage+"_task_count"]
                check(meta["task_count"] == tasks and meta["chunk_tokens"] == a["chunk_tokens"]
                      and meta["requested_splits_compatibility_parameter"] == a["splits"], f"{ident}/{stage}: task metadata")
                check(meta["workspace_bytes"] == sum(meta["workspace_bytes_by_buffer"].values()),
                      f"{ident}/{stage}: workspace accounting")
                check(meta["workspace_bytes_by_buffer"]["packed_U"] == meta["packed_partial_rows"] * 128 * 4,
                      f"{ident}/{stage}: packed U accounting")
                check((tasks == 0) == (case["name"] == "empty"), f"{ident}/{stage}: empty task count")
                for role, resource in case[stage+"_resources"].items():
                    resource_statuses[resource["status"]] += 1
                    record_row = {"job": name, "case": case["name"], "stage": stage,
                                  "role": role, "status": resource["status"]}
                    resources.append(record_row)
                    if resource["status"] == "not_launched":
                        check(role == "producer" and tasks == 0 and "artifacts" not in resource,
                              f"{ident}/{stage}/{role}: fabricated or unexpected nonlaunch")
                        record_row["reason"] = resource.get("reason")
                        continue
                    check(resource["status"] == "saved_for_independent_resource_review",
                          f"{ident}/{stage}/{role}: unknown resource status")
                    files = resource["artifacts"]
                    check(set(files) == {"ptx", "cubin", "resources", "sass"}, f"{ident}/{stage}/{role}: artifact set")
                    for kind, artifact in files.items():
                        artifact_path = ROOT / artifact["path"]
                        check(record(artifact_path) == artifact["sha256"], f"{ident}/{stage}/{role}: {kind} SHA")
                    cubin_sha = files["cubin"]["sha256"]
                    if cubin_sha not in dump_cache:
                        fresh = {}
                        for kind, flag in (("resources", "--dump-resource-usage"), ("sass", "--dump-sass")):
                            fresh[kind] = subprocess.run([str(CUOBJDUMP), flag, str(ROOT / files["cubin"]["path"])],
                                                         check=True, capture_output=True, text=True).stdout
                        dump_cache[cubin_sha] = fresh
                    fresh = dump_cache[cubin_sha]
                    for kind in ("resources", "sass"):
                        check(fresh[kind] == (ROOT / files[kind]["path"]).read_text(),
                              f"{ident}/{stage}/{role}: fresh cuobjdump {kind} mismatch")
                    fields = {key.lower(): int(value) for key, value in
                              re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", fresh["resources"])}
                    loads = len(re.findall(r"\bLDL(?:\.|\s)", fresh["sass"]))
                    stores = len(re.findall(r"\bSTL(?:\.|\s)", fresh["sass"]))
                    check(fields == resource["cubin_resources"], f"{ident}/{stage}/{role}: resource fields")
                    check(loads == resource["SASS_LDL_count"] and stores == resource["SASS_STL_count"],
                          f"{ident}/{stage}/{role}: local instruction counts")
                    check(fields.get("reg") == resource["n_regs"] and resource["n_spills"] == 0,
                          f"{ident}/{stage}/{role}: register/spill evidence")
                    check(fields.get("local") == fields.get("stack") == loads == stores == 0
                          and resource["explicit_local_traffic_absent"] is True, f"{ident}/{stage}/{role}: explicit local traffic")
                    meta = resource["Triton_metadata"]
                    check(meta["num_warps"] * meta["warp_size"] == a["threads"] and meta["global_scratch_size"] == 0,
                          f"{ident}/{stage}/{role}: launch/scratch metadata")
                    record_row.update(n_regs=resource["n_regs"], n_spills=resource["n_spills"],
                                      static_cubin_resources=fields, Triton_dynamic_shared_bytes=meta["shared"],
                                      total_shared_bytes=fields["shared"]+meta["shared"],
                                      SASS_LDL_count=loads, SASS_STL_count=stores,
                                      artifacts=files, fresh_cuobjdump_matched_saved=True)
            cases.append({"job": name, "case": case["name"], "B": case["batch"],
                          "chunk_tokens": a["chunk_tokens"], "block_n": a["block_n"],
                          "case_status": case["status"], "checks": checks,
                          "initial_metadata": case["initial_metadata"], "rebuilt_metadata": case["rebuilt_metadata"],
                          "initial_plan_prepare_ms": case["initial_plan_prepare_ms"],
                          "initial_Attention_prepare_ms": case["initial_prepare_ms"]})
            if case["name"] == "empty":
                check(all(c["empty_rows"] == 2 * a["batch"] * 8 and c["O_max_abs_fp64"] == 0
                          and c["LSE_max_abs_fp64_finite_rows"] == 0 for c in checks), f"{ident}: empty semantics")
            if case["name"] == "real_capture":
                provenance = case["provenance"]
                check(provenance["ready_lifetime_measured"] is True
                      and provenance["includes_system_prefix_and_complete_private_suffix"] is True,
                      f"{ident}: corrected capture semantics")
                check(len(provenance["files_sha256"]) == a["batch"] + 3, f"{ident}: input provenance file count")
                for capture_path, expected in provenance["files_sha256"].items():
                    check(record(ROOT / capture_path) == expected == prior_hashes.get(capture_path),
                          f"{ident}: immutable capture/current/prior audit hash differs: {capture_path}")
                    capture_checks.append({"job": name, "path": capture_path, "sha256": expected,
                                           "current_and_prior_capture_audit_match": True})
                row["capture_provenance"] = provenance

    check(sorted(seen_gpus) == list(range(8)), "Physical GPUs 0–7 not represented exactly once")
    check(len(cases) == 40 and numerical_statuses == {"passed": 240}, "Expected 40 cases and 240 passes")
    check(phases == {phase: 40 for phase in PHASES}, "Expected forty checks per phase")
    check(resource_statuses == {"not_launched": 16, "saved_for_independent_resource_review": 144},
          "Expected 144 compiled roles and sixteen empty nonlaunches")
    check(len(topology) == 40 and len(capture_checks) == 52, "Topology/provenance check count")
    check("torch" not in sys.modules and "triton" not in sys.modules, "Audit imported GPU libraries")
    launched = [resource for resource in resources if resource["status"] != "not_launched"]
    grouped = {}
    for role in ("producer", "merge"):
        rows = [resource for resource in launched if resource["role"] == role]
        grouped[role] = {"launched_roles": len(rows),
                         "registers": sorted({r["n_regs"] for r in rows}),
                         "n_spills": sorted({r["n_spills"] for r in rows}),
                         "cubin_static_shared_bytes": sorted({r["static_cubin_resources"]["shared"] for r in rows}),
                         "Triton_dynamic_shared_bytes": sorted({r["Triton_dynamic_shared_bytes"] for r in rows}),
                         "total_shared_bytes": sorted({r["total_shared_bytes"] for r in rows})}
    result = {"schema": "owner-task-wave19-independent-audit-v1",
              "status": "passed_saved_numerical_and_resource_audit" if not errors else "audit_invariant_failure",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Read-only GPU receipts, frozen source/provenance, CPU cubin inspection and ledger accounting; no new GPU execution or timing result.",
              "uid": os.getuid(), "python": sys.executable, "audit_GPU_hours": 0,
              "torch_or_triton_imported": False, "auditor_sha256": sha(Path(__file__)),
              "source_commit": SOURCE_COMMIT, "protocol_commit": PROTOCOL_COMMIT,
              "qualifier_AST_equals_wave17_except_chunk_choices_and_source_filename": choices_replaced == 1,
              "owner_runtime_planner_kernel_unchanged_since_wave17": True,
              "expanded_qualification_grid_B_chunk_N": sorted(manifest_grid),
              "frozen_source_sha256": frozen_sources, "current_sources_match_frozen": source_current_matches,
              "input_sha256": inputs, "cuobjdump_path": str(CUOBJDUMP), "cuobjdump_sha256": sha(CUOBJDUMP),
              "ledger_full_file_snapshot_sha256": digest(ledger_bytes),
              "ledger_snapshot_note": "The global ledger may append later jobs. The selected wave19 entries below are a frozen full start/end snapshot.",
              "selected_ledger_sha256": digest(canonical(selected_ledger)), "selected_ledger": selected_ledger,
              "job_status_counts": dict(statuses), "numerical_status_counts": dict(numerical_statuses),
              "phase_counts": dict(phases), "cases": cases, "case_count": len(cases),
              "resource_status_counts": dict(resource_statuses), "resources": resources,
              "unique_cubins_independently_dumped": len(dump_cache), "resource_summary": grouped,
              "topology_checks": topology, "immutable_capture_checks": capture_checks,
              "unique_capture_files": len({r["path"] for r in capture_checks}), "jobs": jobs,
              "ledger_accounting": {"terminal_jobs": sum(e["event"] == "end" for e in selected_ledger),
                                    "failed_jobs": sum(e["event"] == "end" and e["exit_code"] != 0 for e in selected_ledger),
                                    "timed_out_jobs": sum(e["event"] == "end" and e["timed_out"] for e in selected_ledger),
                                    "allocated_GPU_seconds": allocated_seconds, "P1_GPU_hours": gpu_hours,
                                    "all_attempts_retained": True},
              "errors": errors,
              "limitations": [
                  "FP64 comparisons are the preserved frozen qualification code's GPU results; this CPU audit verifies receipt completeness, contracts and provenance without recomputing tensors.",
                  "resource_review_pending describes the original manual artifact-review state, not a numerical failure. Original receipts remain unchanged.",
                  "No spill or explicit local-memory instructions does not establish occupancy, memory bandwidth, energy use or speed.",
                  "Triton metadata shared is dynamic launch shared memory; cubin SHARED is static. Both are reported and summed, not substituted for one another.",
                  "Q/K/V/Indices in-place mutations use the original graph. Changed owner assignments require a new Attention and new graph; graph.replay bypasses the host guard.",
                  "The topology guard checks identities/layouts/versions, not checksums of arbitrary external mutations. This qualification is not old v4 same-graph offset mutation support.",
                  "chunk_tokens defines task partitioning. splits=64 is a compatibility parameter and is not a sixty-four-way split result.",
                  "Eight expanded qualification points alternate N32/N64 across chunk extremes; they are not the complete B3/B4 by four-new-chunk by two-N factorial. Every later performance configuration requires its own FP64 check.",
                  "Forty repeated fixture/capture cases do not represent forty independent documents or application-quality samples.",
                  "Rebuild/recapture wall costs include preparation and first replay but are diagnostic lifecycle receipts, not calibrated performance benchmarks.",
                  "This audit makes no application quality, natural-EOS latency or strong-baseline speed claim."]}
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "case_count", "numerical_status_counts", "resource_summary", "ledger_accounting", "errors")}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
