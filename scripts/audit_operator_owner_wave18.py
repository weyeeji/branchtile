#!/usr/bin/env python3
"""Independent CPU audit of complete owner producer-plus-merge wave18 calls."""
from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np

from audit_operator_overhead_wave7 import (CONTRACT, METHODS as PRIOR_METHODS,
                                         PAIRS as PRIOR_PAIRS, argument, close,
                                         grid as prior_grid, key as old_key, partitions,
                                         sha, trees)

ROOT = Path(__file__).resolve().parents[1]
uncached_sha = sha


@functools.lru_cache(maxsize=None)
def sha(path):
    return uncached_sha(path)


METHODS = PRIOR_METHODS | {"dynamic_v4", "best_bank_control", "dynamic_lane_select",
                           "dynamic_fullcol_wide", "best_lane_control", "dynamic_owner_tasks", "best_owner_control"}
PAIRS = PRIOR_PAIRS | {"best_bank_control_over_dynamic_v4",
                       "dynamic_fullcol_guard_over_dynamic_v4",
                       "dynamic_persistent_fixed_over_dynamic_v4",
                       "best_lane_control_over_dynamic_lane_select",
                       "best_control_over_dynamic_lane_select",
                       "dynamic_fullcol_guard_over_dynamic_lane_select",
                       "dynamic_persistent_fixed_over_dynamic_lane_select",
                       "dynamic_v4_over_dynamic_lane_select",
                       "dynamic_fullcol_wide_over_dynamic_lane_select",
                       "dynamic_fullcol_guard_over_dynamic_fullcol_wide"} | {
    f"{control}_over_dynamic_owner_tasks" for control in (
        "best_owner_control", "dynamic_lane_select", "dynamic_persistent_fixed",
        "dynamic_v4", "dynamic_fullcol_guard", "dynamic_fullcol_wide")}


def rel(path):
    return str(Path(path).relative_to(ROOT))


def key(config):
    identity = json.loads(old_key(config))
    if config["method"].startswith("dynamic_"):
        identity["threads"] = config.get("threads", 128)
    if "chunk_tokens" in config:
        identity["chunk_tokens"] = config["chunk_tokens"]
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def candidate_name(config):
    name = config["method"]
    if "partition" in config:
        name += "__p-" + "-".join("_".join(str(i) for i in group) for group in config["partition"])
    if "tree" in config:
        name += "__tree-" + json.dumps(config["tree"], separators=(",", ":"))
    if "fixed_split_size" in config:
        name += "__fixedsplit-" + str(config["fixed_split_size"])
    if "tensor_cores" in config:
        name += "__tc-" + str(int(config["tensor_cores"]))
    if "splits" in config:
        name += f"__splits-{config['splits']}__n-{config['block_n']}"
    if config["method"].startswith("dynamic_"):
        name += f"__threads-{config.get('threads', 128)}"
    if "chunk_tokens" in config:
        name += f"__chunk-{config['chunk_tokens']}"
    return name


def expected_grid(batch):
    configs = prior_grid(batch)
    static_points = [(64, 32), (64, 64)] + ([(128, 32), (128, 64)] if batch == 3
                                           else [(32, 32), (128, 64)])
    configs += [{"method": "dynamic_v4", "version": "v4", "splits": sp, "block_n": n}
                for sp, n in static_points]
    configs += [{"method": "dynamic_lane_select", "version": "lane_select", "splits": sp,
                 "block_n": n, "threads": 128} for sp, n in itertools.product((32, 64, 128), (32, 64))]
    configs += [{"method": "dynamic_fullcol_wide", "version": "fullcol_guard", "splits": sp,
                 "block_n": 64, "threads": 256} for sp in (32, 64, 128)]
    assert len(configs) == (55 if batch == 3 else 87)
    configs += [{"method": "dynamic_owner_tasks", "version": "owner_tasks",
                 "splits": 64, "block_n": n, "threads": 128, "chunk_tokens": chunk}
                for chunk, n in itertools.product((256, 512, 1024), (32, 64))]
    assert len(configs) == (61 if batch == 3 else 93)
    assert len({key(config) for config in configs}) == len(configs)
    assert candidate_name(dict(configs[-1], threads=256)) != candidate_name(configs[-1])
    assert candidate_name(dict(configs[-1], chunk_tokens=256)) != candidate_name(configs[-1])
    return configs


def main():
    output = ROOT / "results/operator_owner_wave18_audit_v1/audit.json"
    assert not output.exists(), "Retain prior audit snapshots"
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules
    protocol_path = ROOT / "configs/operator_overhead_wave18_protocol.json"
    jobs_path = ROOT / "configs/operator_overhead_wave18_jobs.json"
    protocol = json.loads(protocol_path.read_text())
    manifest_paths = [jobs_path]
    jobs = json.loads(jobs_path.read_text())
    expected_jobs, expected_total_configs = 8, 616
    assert protocol["expected_total_configs"] == expected_total_configs
    assert len(jobs) == expected_jobs and sorted(job["gpus"][0] for job in jobs) == list(range(8))
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    job_names = {job["name"] for job in jobs}
    completed = [entry for entry in ledger if entry["name"] in job_names and entry["event"] == "end"]
    assert len(completed) == expected_jobs
    cutoff = max(entry["time_utc"] for entry in completed)
    ledger = [entry for entry in ledger if entry["time_utc"] <= cutoff]
    frozen = json.loads((ROOT / "results/natural_capacity_boundary_v1/audit.json").read_text())["input_sha256"]
    hashes = {rel(path): sha(path) for path in [protocol_path, *manifest_paths,
                                              ROOT / "scripts/audit_operator_overhead_wave7.py"]}
    for path, expected in protocol["code_sha256"].items():
        assert sha(ROOT / path) == expected
        hashes[path] = expected
    for path, expected in protocol["qualification_sha256"].items():
        assert sha(ROOT / path) == expected
        hashes[path] = expected
    source_proof = {}
    proof_fragments = {
        "scripts/bench_operator_overhead_owner.py": (
            "return self.op(q,k,v)",
            'graphs = {method: graph_of(best[cache][method]["op"].run, repetitions)',
            "for cache, repetitions in ((\"hot\", 20), (\"cold\", 1)):",
        ),
        "src/branchtile/attention_owner_tasks.py": (
            "self.last_producer_compiled = producer[(self.num_tasks,)](",
            "self.last_merge_compiled = merge[(heads, rows)](",
            "return self.output, self.lse",
        ),
        "scripts/compare_attention.py": ("with torch.cuda.graph(graph):", "graph.replay()",
                                          "t0.elapsed_time(t1) * 1000 / repetitions"),
    }
    for source, fragments in proof_fragments.items():
        lines = (ROOT / source).read_text().splitlines()
        source_proof[source] = []
        for fragment in fragments:
            matches = [{"line": line, "text": text.strip()} for line, text in enumerate(lines, 1)
                       if fragment in text]
            assert matches, (source, fragment)
            source_proof[source].extend(matches)
    rows, pending, timing_cells, matched, failures = [], [], [], [], []
    source_checks, qualified = 0, 0
    figure_data, configuration_receipts, owner_metadata, planning_receipts = [], [], [], []
    for job in jobs:
        starts = [entry for entry in ledger if entry["name"] == job["name"] and entry["event"] == "start"]
        ends = [entry for entry in ledger if entry["name"] == job["name"] and entry["event"] == "end"]
        assert len(starts) <= 1 and len(ends) <= 1
        assert len(starts) == len(ends) == 1
        terminal = ends[0]
        assert terminal["exit_code"] == 0 and not terminal["timed_out"]
        assert terminal["argv"] == job["argv"] and terminal["gpus"] == job["gpus"]
        assert job["category"] == terminal["category"] == "P2" and len(job["gpus"]) == 1
        close(terminal["gpu_hours"], terminal["allocated_seconds"] / 3600)
        argv = job["argv"]
        assert argument(argv, "--candidate-set") == "owner"
        assert argument(argv, "--blocks") == "20" and argument(argv, "--step") == "64"
        assert argument(argv, "--heads") == "2" and "--require-ready" in argv
        path = ROOT / argument(argv, "--output")
        result = json.loads(path.read_text())
        progress = path.with_suffix(".tuning.jsonl")
        hashes[rel(path)], hashes[rel(progress)] = sha(path), sha(progress)
        assert [json.loads(line) for line in progress.read_text().splitlines()] == result["tuning"]
        batch = result["batch"]
        assert batch in (3, 4)
        assert result["gqa"] == 8 and result["uid"] == os.getuid()
        assert result["python"].endswith("/.venv-latest/bin/python")
        assert result["protocol_version"] == "owner-v1" and result["candidate_set"] == "owner"
        assert result["numerical_contract"] == CONTRACT
        assert result["bankpilot_v4_grid"] == [{"splits": 64, "block_n": n} for n in (32, 64)]
        expected = expected_grid(batch)
        expected_count = len(expected)
        for field, method in (("lane_grid", "dynamic_lane_select"), ("wide_control_grid", "dynamic_fullcol_wide"), ("lane_v4_control_grid", "dynamic_v4"), ("owner_grid", "dynamic_owner_tasks")):
            assert {key(config) for config in result[field]} == {key(config) for config in expected if config["method"] == method}
        assert len(result["tuning"]) == result["tuning_planned"] == result["tuning_completed"] == expected_count
        assert result["tuning_complete"] and all(not values for values in result["missing_qualified_methods"].values())
        assert len({key(row) for row in result["tuning"]}) == expected_count
        assert {key(row) for row in result["tuning"]} == {key(row) for row in expected}
        assert result["partitions"] == partitions(batch)
        assert {json.dumps(tree) for tree in result["hierarchies"]} == {json.dumps(tree) for tree in trees(tuple(range(batch)))}
        for source, expected_sha in result["code_sha256"].items():
            assert sha(ROOT / source) == expected_sha
            hashes[source] = expected_sha
        for source, expected_sha in protocol["code_sha256"].items():
            if source in result["code_sha256"]:
                assert result["code_sha256"][source] == expected_sha
            else:
                assert source == "configs/operator_overhead_wave18_jobs.json"
        for row in result["tuning"]:
            assert row["candidate_name"] == candidate_name(row)
            assert row["source_paths"] == result["candidate_sources"][row["method"]]
            assert row["source_sha256"] == {source: result["code_sha256"][source] for source in row["source_paths"]}
            if row["status"] == "ok":
                qualified += 1
                assert math.isfinite(row["max_abs_error_fp64"]) and math.isfinite(row["max_lse_error_fp64"])
            else:
                failures.append({"job": job["name"], "configuration": row})
            configuration_receipts.append({"job": job["name"],
                "identity": json.loads(key(row)), "candidate_name": row["candidate_name"],
                "status": row["status"], "source_sha256": row["source_sha256"],
                "max_abs_error_fp64": row.get("max_abs_error_fp64"),
                "max_lse_error_fp64": row.get("max_lse_error_fp64"),
                "configuration_wall_seconds": row.get("configuration_wall_seconds"),
                "initialization_ms_including_jit": row.get("initialization_ms_including_jit")})
            if row["method"] == "dynamic_owner_tasks" and row["status"] == "ok":
                meta = row["operator_metadata"]
                assert row["task_count"] == meta["task_count"] > 0
                assert row["packed_rows"] == meta["packed_partial_rows"]
                assert meta["chunk_tokens"] == row["chunk_tokens"]
                assert meta["requested_splits_compatibility_parameter"] == 64
                assert row["workspace_bytes"] == meta["workspace_bytes"] == sum(row["workspace_bytes_by_buffer"].values())
                assert row["workspace_bytes_by_buffer"]["packed_U"] == row["packed_rows"] * 128 * 4
                close(row["metadata_ms"], row["region_plan_prepare_ms"] + row["operator_metadata_ms"])
                assert row["prepare_ms"] >= row["region_plan_prepare_ms"] + row["operator_prepare_ms"]
                assert row["region_plan_bytes"] > 0
                assert "rebuild" in row["graph_topology_contract"] and "bypasses" in row["graph_topology_contract"]
                owner_metadata.append({"job": job["name"], **row})
        capture = ROOT / argument(argv, "--capture")
        metadata = json.loads((capture / "metadata.json").read_text())
        layer, budget = int(argument(argv, "--layer")), int(argument(argv, "--budget"))
        assert metadata["B"] == batch and metadata["ready_lifetime_measured_by_step"]["64"]
        assert result["provenance"]["ready_lifetime_measured"]
        assert result["provenance"]["includes_system_prefix_and_complete_private_suffix"]
        for field in ("model_revision", "document_sha256", "actual_tp_rank", "document_token_range",
                      "actual_shared_document_token_range", "private_protected_document_tail_tokens"):
            assert result["provenance"][field] == metadata[field]
        assert metadata["actual_tp_rank"] == 0
        files = [capture / "metadata.json", capture / f"source_l{layer}.pt",
                 capture / f"masks_l{layer}_b{budget}.pt"]
        files.extend(capture / f"decode_q{request}_l{layer}_s64.pt" for request in range(batch))
        if (capture / "overlay_receipt.json").exists():
            overlay = json.loads((capture / "overlay_receipt.json").read_text())
            capture_frozen = {entry["path"]: entry["sha256"] for entry in overlay["input_links"] + overlay["generated_masks"]}
            capture_frozen[rel(capture / "metadata.json")] = overlay["metadata_sha256"]
        else:
            capture_frozen = frozen
        for file in files:
            expected_sha = capture_frozen.get(rel(file), frozen.get(rel(file.resolve())))
            assert sha(file) == expected_sha
            hashes[rel(file)] = sha(file)
            source_checks += 1
        rows.append({"name": job["name"], "start": starts[0], "terminal": terminal,
                     "B": batch, "configurations": expected_count,
                     "baseline_policy_persistent_bankpilot_controls_exactly_retained": True,
                     "static_v4_grid": [config for config in expected if config["method"] == "dynamic_v4"],
                     "static_v4_grid_is_four_of_six": True,
                     "candidate_identity_includes_dynamic_threads": True,
                     "family_counts": dict(collections.Counter(row["method"] for row in result["tuning"]))})
        index = {(row["method"], row.get("splits"), row.get("block_n")): row for row in result["tuning"]}
        for sp, block_n in itertools.product((64,), (32, 64)):
            v4 = index["dynamic_v4", sp, block_n]
            fullcol = index["dynamic_fullcol_guard", sp, block_n]
            persistent = index["dynamic_persistent_fixed", sp, block_n]
            assert v4["status"] == fullcol["status"] == persistent["status"] == "ok"
            matched.append({"job": job["name"], "splits": sp, "block_n": block_n,
                            "descriptive_tuning_FullCol_over_v4": {cache: fullcol["tuning_us_by_cache"][cache] / v4["tuning_us_by_cache"][cache] for cache in ("hot", "cold")},
                            "descriptive_tuning_persistent_over_v4": {cache: persistent["tuning_us_by_cache"][cache] / v4["tuning_us_by_cache"][cache] for cache in ("hot", "cold")},
                            "scope": "Five search readings per cache at equal split/N; descriptive, not final paired blocks. Persistent still changes M32 MMA work."})
        for cache in ("hot", "cold"):
            selected = result["selected"][cache]
            for method, choice in selected.items():
                record = [row for row in result["tuning"] if key(row) == key(choice["config"])]
                assert len(record) == 1 and record[0]["status"] == "ok"
                assert choice["record"]["candidate_name"] == record[0]["candidate_name"]
                if method == "full_union":
                    eligible = [row for row in result["tuning"] if row.get("partition") == [list(range(batch))]]
                elif method == "best_e1":
                    eligible = [row for row in result["tuning"] if row["method"] in ("e0_native", "e1_native", "e1_own")]
                elif method == "best_control":
                    eligible = [row for row in result["tuning"] if row["method"] in ("e0_native", "e1_native", "e1_own", "e4_native")]
                elif method == "best_dynamic":
                    eligible = [row for row in result["tuning"] if row["method"].startswith("dynamic_")]
                elif method == "best_bank_control":
                    eligible = [row for row in result["tuning"] if row["method"] not in ("dynamic_v4", "dynamic_lane_select", "dynamic_owner_tasks")]
                elif method == "best_lane_control":
                    eligible = [row for row in result["tuning"] if row["method"] not in ("dynamic_lane_select", "dynamic_owner_tasks")]
                elif method == "best_owner_control":
                    eligible = [row for row in result["tuning"] if row["method"] != "dynamic_owner_tasks"]
                else:
                    eligible = [row for row in result["tuning"] if row["method"] == method]
                close(record[0]["tuning_us_by_cache"][cache], min(row["tuning_us_by_cache"][cache] for row in eligible if row["status"] == "ok"))
            measurements = [row for row in result["measurements"] if row["cache"] == cache]
            lookup = {(row["block"], row["direction"]): row for row in measurements}
            assert len(lookup) == len(measurements) == 40
            assert set(lookup) == set(itertools.product(range(20), ("random", "reverse")))
            for block in range(20):
                assert lookup[block, "random"]["order"][::-1] == lookup[block, "reverse"]["order"]
            for row in measurements:
                assert set(row["us"]) == set(row["order"]) == METHODS
            arrays = {method: np.array([[lookup[block, direction]["us"][method]
                                        for direction in ("random", "reverse")] for block in range(20)]) for method in METHODS}
            for method, values in arrays.items():
                assert np.isfinite(values).all() and (values > 0).all()
                close(float(np.median(values)), result["summary"][cache]["median_us"][method])
            assert set(result["summary"][cache]["paired_ratios"]) == PAIRS
            draws = np.random.default_rng(71).integers(0, 20, (10000, 20))
            for name, pair in result["summary"][cache]["paired_ratios"].items():
                numerator, denominator = name.split("_over_")
                ratios = arrays[numerator] / arrays[denominator]
                close(pair["median"], float(np.median(ratios)))
                for actual, expected in zip(pair["paired_block_bootstrap95"], np.quantile(np.median(ratios[draws], axis=(1, 2)), [.025, .975])):
                    close(actual, float(expected))
            standardized = {}
            for method in ("dynamic_v2", "dynamic_persistent_fixed", "dynamic_v4",
                           "dynamic_lane_select", "dynamic_fullcol_wide", "dynamic_owner_tasks"):
                ratios = arrays[method] / arrays["best_owner_control"]
                interval = np.quantile(np.median(ratios[draws], axis=(1, 2)), [.025, .975])
                standardized[method] = {
                    "paired_time_ratio_to_best_owner_control": float(np.median(ratios)),
                    "paired_block_bootstrap95": interval.tolist(),
                    "selected_config": selected[method]["config"]}
            figure_data.append({"job": job["name"], "cache": cache, "B": batch,
                                "layer": layer, "budget": budget,
                                "document_sha256": metadata["document_sha256"],
                                "best_owner_control_config": selected["best_owner_control"]["config"],
                                "methods": standardized, "paired_blocks": 20,
                                "bootstrap_draws": 10000, "source_result_sha256": sha(path)})
            for method, plan in result["planning"][cache].items():
                assert len(plan["samples"]) == 3
                prepare = statistics.median(row["warm_prepare_ms"] for row in plan["samples"]) * 1000
                close(prepare, plan["warm_prepare_median_us"])
                for reuse, actual in plan["amortized_prepare_plus_gpu_call_us"].items():
                    close(actual, prepare / int(reuse) + result["summary"][cache]["median_us"][method])
                if method == "dynamic_owner_tasks":
                    for sample in plan["samples"]:
                        assert sample["warm_prepare_ms"] >= sample["region_plan_prepare_ms"] + sample["operator_prepare_ms"]
                        assert sample["prepare_plus_one_call_host_wall_us"] >= sample["warm_prepare_ms"] * 1000
                        assert sample["task_count"] == sample["operator_metadata"]["task_count"]
                        assert sample["workspace_bytes"] == sample["operator_metadata"]["workspace_bytes"]
                planning_receipts.append({"job": job["name"], "cache": cache, "method": method, **plan})
            v4, fullcol, persistent = (selected[method]["config"] for method in ("dynamic_v4", "dynamic_fullcol_guard", "dynamic_persistent_fixed"))
            timing_cells.append({"job": job["name"], "cache": cache, "layer": layer, "budget": budget,
                                 "paired_blocks": 20, "bootstrap_draws": 10000,
                                 "v4_config": v4, "FullCol_config": fullcol,
                                 "persistent_config": persistent,
                                 "bank_control_config": selected["best_bank_control"]["config"],
                                 "FullCol_v4_same_split_N": all(v4[field] == fullcol[field] for field in ("splits", "block_n")),
                                 "persistent_v4_same_split_N": all(v4[field] == persistent[field] for field in ("splits", "block_n")),
                                 "all_median_us": result["summary"][cache]["median_us"],
                                 "all_paired_ratios": result["summary"][cache]["paired_ratios"],
                                 "lane_config": selected["dynamic_lane_select"]["config"],
                                 "wide_config": selected["dynamic_fullcol_wide"]["config"],
                                 "lane_control_config": selected["best_lane_control"]["config"],
                                 "owner_config": selected["dynamic_owner_tasks"]["config"],
                                 "owner_control_config": selected["best_owner_control"]["config"],
                                 "scope": "Independent family winners with exact retained grids, thread-aware identity. No full-model claim."})
    assert len(rows) == expected_jobs and not pending and qualified == expected_total_configs and not failures
    assert len(owner_metadata) == 48 and len(configuration_receipts) == expected_total_configs
    assert len(timing_cells) == 2 * expected_jobs and len(matched) == 2 * expected_jobs
    result = {"schema": 1, "status": "complete_wave18_independent_CPU_audit_passed",
              "scope": "Complete eight-state wave18 owner-task comparison only; no cross-wave winner assembly.",
              "wave": 18, "threads_are_part_of_candidate_identity": True, "chunk_tokens_are_part_of_candidate_identity": True,
              "auditor_sha256": sha(__file__), "input_sha256": hashes,
              "snapshot_cutoff_utc": cutoff,
              "expected_cells": expected_jobs, "audited_complete_cells": rows,
              "pending_cells": pending, "qualified_configurations": qualified,
              "all_configuration_failures": failures, "frozen_capture_SHA_checks": source_checks,
              "parameter_matched_search_only": matched, "final_paired_cache_cells": timing_cells,
              "verified_final_paired_ratio_statistics": 56 * expected_jobs,
              "complete_call_source_proof": source_proof,
              "configuration_receipts": configuration_receipts,
              "owner_candidate_metadata": owner_metadata, "all_planning_receipts": planning_receipts,
              "CI_scope": "Paired random/reverse timing blocks within each fixed capture. Not cross-document or end-to-end confidence intervals.",
              "within_wave_figure_data": figure_data,
              "source_P2_GPU_hours": sum(row["terminal"]["gpu_hours"] for row in rows),
              "audit_GPU_hours": 0, "torch_imported": "torch" in sys.modules,
              "limitations": [
                  "This snapshot uses wave18 only; all prior snapshots remain separate.",
                  "Recorded FP64 checks passed; this CPU audit does not recompute attention.",
                  "Static v4 has four qualified of six split/N points: B3 S64/S128 both N; B4 S64 both N plus S32/N32 and S128/N64. Lane has six points, wide has three N64/256-thread points.",
                  "Winner comparisons may differ in split/N or chunk size. Search-only readings have five samples and no paired inference.",
                  "Paired bootstrap resamples twenty timing blocks within one fixed capture; intervals do not represent uncertainty across documents, queries, layers or GPUs.",
                  "splits=64 is a compatibility parameter for owner tasks; chunk_tokens defines actual partitioning. Topology changes require rebuilding and recapturing, while graph.replay bypasses host guards.",
                  "Fixed-M32 persistent changes masked MMA work relative to compact FullCol/v4; comparisons are not pure state-transport ablations.",
                  "Owner call timing includes the complete producer and merge and is measured on immutable topology. The uniform/chunk by rectangular/packed 2x2 ablation is incomplete; owner changes partitioning and storage together.",
                  "CUDA-event steady-state timings exclude plan preparation, topology rebuilding, recapture and host synchronized wall. Warm preparation and prepare-plus-one-call wall are preserved separately; amortization is hypothetical. No whole-model, quality, energy or hardware-traffic claim."]}
    assert not result["torch_imported"]
    assert result["source_P2_GPU_hours"] <= protocol["maximum_P2_GPU_hours"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "complete_cells": expected_jobs,
                      "qualified": qualified, "paired_cache_cells": 2 * expected_jobs,
                      "source_P2_GPU_hours": result["source_P2_GPU_hours"],
                      "audit_GPU_hours": 0}))


if __name__ == "__main__":
    main()
