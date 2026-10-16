#!/usr/bin/env python3
"""Independent CPU audit of the frozen eight-cell fixed-M32 persistent screen."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/operator_persistent_wave7_audit_v1/audit.json"
CONTRACT = {"output_atol": .006, "output_rtol": .015, "lse_atol": .0003,
            "lse_rtol": .0001, "lse_base": "natural", "reference": "FP64"}
CONFIG_FIELDS = ("method", "version", "partition", "tree", "splits", "block_n",
                 "fixed_split_size", "tensor_cores")
METHODS = {"e0_native", "e1_native", "e1_own", "e4_native", "full_union",
           "dynamic_v2", "dynamic_v3", "dynamic_emptyguard", "dynamic_fullcol_guard",
           "best_e1", "best_dynamic", "best_control", "full_union_AA", "dynamic_persistent_fixed"}
PAIRS = {"full_union_over_best_e1", "full_union_over_e4_native", "best_e1_over_e4_native",
         "full_union_over_full_union_AA", "best_control_over_best_dynamic",
         "best_control_over_dynamic_v2", "best_control_over_dynamic_v3",
         "best_control_over_dynamic_emptyguard", "best_control_over_dynamic_fullcol_guard",
         "dynamic_emptyguard_over_dynamic_fullcol_guard", "best_control_over_dynamic_persistent_fixed",
         "dynamic_fullcol_guard_over_dynamic_persistent_fixed"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def argument(argv, name):
    return argv[argv.index(name) + 1]


def key(config):
    return json.dumps({field: config[field] for field in CONFIG_FIELDS if field in config},
                      sort_keys=True, separators=(",", ":"))


def partitions(batch):
    # Independent restricted-growth-string enumeration, not benchmark imports.
    result = []
    for labels in itertools.product(range(batch), repeat=batch):
        if labels[0] != 0 or any(labels[i] > 1 + max(labels[:i]) for i in range(1, batch)):
            continue
        result.append([[i for i, label in enumerate(labels) if label == group]
                       for group in range(max(labels) + 1)])
    return sorted(result)


def trees(leaves):
    # Every unordered binary bipartition is anchored by the lowest leaf.
    if len(leaves) == 1:
        return [leaves[0]]
    result = []
    mask_all = (1 << len(leaves)) - 1
    for mask in range(1, mask_all):
        if not mask & 1:
            continue
        left = tuple(leaf for i, leaf in enumerate(leaves) if mask & (1 << i))
        right = tuple(leaf for i, leaf in enumerate(leaves) if not mask & (1 << i))
        result.extend([[a, b] for a in trees(left) for b in trees(right)])
    return result


def grid(batch):
    configs = []
    for partition in partitions(batch):
        configs.extend([{"method": "e1_native", "partition": partition, "fixed_split_size": None},
                        {"method": "e1_own", "partition": partition, "splits": 64, "block_n": 64}])
    configs.extend({"method": "e4_native", "tree": tree, "fixed_split_size": None}
                   for tree in trees(tuple(range(batch))))
    configs.extend({"method": "e0_native", "tensor_cores": tensor, "fixed_split_size": None}
                   for tensor in (False, True))
    for method, version in (("dynamic_v2", "v2"), ("dynamic_v3", "v3"),
                            ("dynamic_emptyguard", "emptyguard"),
                            ("dynamic_fullcol_guard", "fullcol_guard"),
                            ("dynamic_persistent_fixed", "persistent_fixed")):
        configs.extend({"method": method, "version": version, "splits": splits, "block_n": block_n}
                       for splits in (32, 64, 128)
                       for block_n in ((64,) if version == "v3" else (32, 64)))
    assert len(configs) == (42 if batch == 3 else 74)
    return configs


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules
    protocol_path = ROOT / "configs/operator_overhead_wave7_protocol.json"
    jobs_path = ROOT / "configs/operator_overhead_wave7_jobs.json"
    gpu0_path = ROOT / "configs/operator_overhead_wave7_gpu0_jobs.json"
    cases_path = ROOT / "configs/operator_overhead_wave5_protocol.json"
    jobs = sorted(json.loads(jobs_path.read_text()) + json.loads(gpu0_path.read_text()),
                  key=lambda job: job["gpus"][0])
    protocol = json.loads(protocol_path.read_text())
    assert len(jobs) == 8 and sorted(job["gpus"][0] for job in jobs) == list(range(8))
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    prior_inputs = json.loads((ROOT / "results/natural_capacity_boundary_v1/audit.json").read_text())["input_sha256"]
    input_hashes = {rel(protocol_path): sha(protocol_path), rel(jobs_path): sha(jobs_path),
                    rel(gpu0_path): sha(gpu0_path), rel(cases_path): sha(cases_path)}
    for path, expected in protocol["code_sha256"].items():
        assert sha(ROOT / path) == expected, path
        input_hashes[path] = expected
    wave4_jobs = json.loads((ROOT / "configs/operator_qualification_wave4_jobs.json").read_text())
    wave4_end = []
    for job in wave4_jobs:
        ends = [entry for entry in ledger if entry["name"] == job["name"] and entry["event"] == "end"]
        assert len(ends) == 1
        wave4_end.append(ends[0]["time_utc"])
    prior_terminal = max(datetime.datetime.fromisoformat(t) for t in wave4_end)
    gpu0_extended = [entry for entry in ledger if entry["name"] == "staticbank_v4_b3_n32_extended_v1" and entry["event"] == "end"]
    assert len(gpu0_extended) == 1 and gpu0_extended[0]["exit_code"] == 0
    rows, pairs, winners, errors = [], [], [], []
    observed_cases, total_configs, total_source_checks = [], 0, 0
    for job in jobs:
        argv = job["argv"]
        assert job["category"] == "P2" and len(job["gpus"]) == 1
        assert argument(argv, "--candidate-set") == "persistent" and int(argument(argv, "--blocks")) == 20
        assert int(argument(argv, "--step")) == 64 and int(argument(argv, "--heads")) == 2
        assert "--require-ready" in argv and "--include-dynamic" in argv
        layer, budget = int(argument(argv, "--layer")), int(argument(argv, "--budget"))
        capture = ROOT / argument(argv, "--capture")
        document = next(name for name in ("docfinqa256", "publicdev1_256", "publicdev6_256")
                        if name in str(capture))
        observed_cases.append([document, layer, budget])
        starts = [entry for entry in ledger if entry["name"] == job["name"] and entry["event"] == "start"]
        terminal = [entry for entry in ledger if entry["name"] == job["name"] and entry["event"] == "end"]
        assert len(starts) == 1 and len(terminal) <= 1
        assert datetime.datetime.fromisoformat(starts[0]["time_utc"]) >= prior_terminal
        if job["gpus"] == [0]:
            assert datetime.datetime.fromisoformat(starts[0]["time_utc"]) >= datetime.datetime.fromisoformat(gpu0_extended[0]["time_utc"])
        path = ROOT / argument(argv, "--output")
        row = {"name": job["name"], "start": starts[0], "terminal": terminal[0] if terminal else None,
               "final_result_exists": path.exists(), "pending": not terminal,
               "source_gpu_hours": terminal[0]["gpu_hours"] if terminal else None}
        rows.append(row)
        if terminal:
            assert terminal[0]["gpus"] == job["gpus"] and terminal[0]["argv"] == argv
            close(terminal[0]["gpu_hours"], terminal[0]["allocated_seconds"] / 3600)
            row["process_passed"] = terminal[0]["exit_code"] == 0 and not terminal[0]["timed_out"]
        if not path.exists() or not terminal:
            if not args.allow_partial:
                raise RuntimeError("Wait for every job terminal and final result: " + job["name"])
            continue
        result = json.loads(path.read_text())
        input_hashes[rel(path)] = sha(path)
        progress = path.with_suffix(".tuning.jsonl")
        input_hashes[rel(progress)] = sha(progress)
        progress_rows = [json.loads(line) for line in progress.read_text().splitlines()]
        assert progress_rows == result["tuning"]
        assert result["candidate_set"] == "persistent" and result["protocol_version"] == 3
        assert "not a pure state-transport-only change" in result["persistent_fixed_ablation_definition"]
        assert result["numerical_contract"] == CONTRACT and result["uid"] == os.getuid()
        assert result["gqa"] == 8 and result["provenance"]["ready_lifetime_measured"]
        assert result["provenance"]["includes_system_prefix_and_complete_private_suffix"]
        assert result["provenance"]["actual_tp_rank"] == 0
        batch = result["batch"]
        expected = grid(batch)
        expected_keys = {key(config) for config in expected}
        actual_keys = [key(config) for config in result["tuning"]]
        assert len(set(actual_keys)) == len(actual_keys) and set(actual_keys) == expected_keys
        assert result["tuning_complete"] and result["tuning_planned"] == result["tuning_completed"] == len(expected)
        assert result["partitions"] == partitions(batch)
        assert {json.dumps(tree) for tree in result["hierarchies"]} == {json.dumps(tree) for tree in trees(tuple(range(batch)))}
        assert all(not value for value in result["missing_qualified_methods"].values())
        assert "dynamic_v4" not in result["candidate_sources"]
        for source_path, expected_sha in result["code_sha256"].items():
            assert sha(ROOT / source_path) == expected_sha, source_path
            input_hashes[source_path] = expected_sha
        for source_path, expected_sha in protocol["code_sha256"].items():
            assert result["code_sha256"][source_path] == expected_sha
        for record in result["tuning"]:
            assert record["source_paths"] == result["candidate_sources"][record["method"]]
            assert record["source_sha256"] == {p: result["code_sha256"][p] for p in record["source_paths"]}
            if record["status"] != "ok":
                errors.append({"job": job["name"], "record": record})
            else:
                assert math.isfinite(record["max_abs_error_fp64"]) and math.isfinite(record["max_lse_error_fp64"])
                assert all(record["tuning_us_by_cache"][cache] > 0 for cache in ("hot", "cold"))
        total_configs += len(expected)
        metadata = json.loads((capture / "metadata.json").read_text())
        assert metadata["B"] == batch and metadata["ready_lifetime_measured_by_step"]["64"]
        for field in ("model_revision", "document_sha256", "actual_tp_rank", "document_token_range",
                      "actual_shared_document_token_range", "private_protected_document_tail_tokens"):
            assert result["provenance"][field] == metadata[field]
        files = [capture / "metadata.json", capture / f"source_l{layer}.pt", capture / f"masks_l{layer}_b{budget}.pt"]
        files += [capture / f"decode_q{request}_l{layer}_s64.pt" for request in range(batch)]
        if (capture / "overlay_receipt.json").exists():
            overlay = json.loads((capture / "overlay_receipt.json").read_text())
            frozen = {entry["path"]: entry["sha256"] for entry in overlay["input_links"] + overlay["generated_masks"]}
            frozen[rel(capture / "metadata.json")] = overlay["metadata_sha256"]
        else:
            frozen = prior_inputs
        for file in files:
            actual_sha = sha(file)
            expected_sha = frozen.get(rel(file), prior_inputs.get(rel(file.resolve())))
            assert expected_sha == actual_sha, file
            input_hashes[rel(file)] = actual_sha
            total_source_checks += 1
        row.update(batch=batch, configurations=len(expected), qualified_configurations=sum(r["status"] == "ok" for r in result["tuning"]),
                   family_counts=dict(collections.Counter(r["method"] for r in result["tuning"])),
                   all_original_baseline_configs_retained=True, original_baseline_configs=len(expected)-12,
                   all_wave5_policy_configs_retained=True, policy_configs=len(expected)-6)
        index = {(r["method"], r.get("splits"), r.get("block_n")): r for r in result["tuning"]
                 if r["method"] in ("dynamic_fullcol_guard", "dynamic_emptyguard", "dynamic_persistent_fixed")}
        for splits in (32, 64, 128):
            for block_n in (32, 64):
                guard = index["dynamic_emptyguard", splits, block_n]
                fullcol = index["dynamic_fullcol_guard", splits, block_n]
                persistent = index["dynamic_persistent_fixed", splits, block_n]
                assert guard["status"] == fullcol["status"] == persistent["status"] == "ok"
                pairs.append({"job": job["name"], "splits": splits, "block_n": block_n,
                              "all_three_original_FP64_checks_passed": True,
                              "persistent_FullCol_is_pure_transport_ablation": False,
                              "descriptive_tuning_FullCol_over_persistent": {
                                  cache: fullcol["tuning_us_by_cache"][cache] / persistent["tuning_us_by_cache"][cache]
                                  for cache in ("hot", "cold")},
                              "descriptive_tuning_emptyguard_over_FullCol": {
                                  cache: guard["tuning_us_by_cache"][cache] / fullcol["tuning_us_by_cache"][cache]
                                  for cache in ("hot", "cold")},
                              "scope": "Same split/N, five tuning readings per cache; not the randomized paired final experiment."})
        for cache in ("hot", "cold"):
            selected = result["selected"][cache]
            for method, item in selected.items():
                matching = [record for record in result["tuning"] if key(record) == key(item["config"])]
                assert len(matching) == 1 and matching[0]["status"] == "ok"
                assert item["record"]["candidate_name"] == matching[0]["candidate_name"]
                if method == "full_union":
                    eligible = [r for r in result["tuning"]
                                if r.get("partition") == [list(range(batch))]]
                elif method == "best_e1":
                    eligible = [r for r in result["tuning"]
                                if r["method"] in ("e0_native", "e1_native", "e1_own")]
                elif method == "best_control":
                    eligible = [r for r in result["tuning"]
                                if r["method"] in ("e0_native", "e1_native", "e1_own", "e4_native")]
                elif method == "best_dynamic":
                    eligible = [r for r in result["tuning"] if r["method"].startswith("dynamic_")]
                else:
                    eligible = [r for r in result["tuning"] if r["method"] == method]
                close(matching[0]["tuning_us_by_cache"][cache],
                      min(r["tuning_us_by_cache"][cache] for r in eligible if r["status"] == "ok"))
            measurements = [m for m in result["measurements"] if m["cache"] == cache]
            lookup = {(m["block"], m["direction"]): m for m in measurements}
            assert len(measurements) == len(lookup) == 40
            assert set(lookup) == set(itertools.product(range(20), ("random", "reverse")))
            for block in range(20):
                assert lookup[block, "random"]["order"][::-1] == lookup[block, "reverse"]["order"]
            arrays = {}
            for measurement in measurements:
                assert set(measurement["order"]) == set(measurement["us"]) == METHODS
            for method in METHODS:
                values = np.array([[lookup[block, direction]["us"][method]
                                    for direction in ("random", "reverse")] for block in range(20)])
                assert np.isfinite(values).all() and (values > 0).all()
                arrays[method] = values
                close(result["summary"][cache]["median_us"][method], float(np.median(values)))
            draws = np.random.default_rng(71).integers(0, 20, (10000, 20))
            assert set(result["summary"][cache]["paired_ratios"]) == PAIRS
            for pair_name, reported in result["summary"][cache]["paired_ratios"].items():
                numerator, denominator = pair_name.split("_over_")
                values = arrays[numerator] / arrays[denominator]
                close(reported["median"], float(np.median(values)))
                interval = np.quantile(np.median(values[draws], axis=(1, 2)), [.025, .975])
                for actual, expected in zip(reported["paired_block_bootstrap95"], interval):
                    close(actual, float(expected))
            for method, plan in result["planning"][cache].items():
                assert len(plan["samples"]) == 3
                prepare = statistics.median(sample["warm_prepare_ms"] for sample in plan["samples"]) * 1000
                close(prepare, plan["warm_prepare_median_us"])
                for reuse, value in plan["amortized_prepare_plus_gpu_call_us"].items():
                    close(value, prepare / int(reuse) + result["summary"][cache]["median_us"][method])
            fullcol = selected["dynamic_fullcol_guard"]["config"]
            guard = selected["dynamic_emptyguard"]["config"]
            persistent = selected["dynamic_persistent_fixed"]["config"]
            winners.append({"job": job["name"], "cache": cache, "paired_blocks": 20,
                            "bootstrap_draws": 10000,
                            "FullCol_config": fullcol, "emptyguard_config": guard,
                            "persistent_config": persistent,
                            "persistent_FullCol_same_selected_split_and_N": all(persistent[k] == fullcol[k] for k in ("splits", "block_n")),
                            "persistent_FullCol_is_pure_transport_ablation": False,
                            "same_selected_split_and_N": all(fullcol[k] == guard[k] for k in ("splits", "block_n")),
                            "strong_control_config": selected["best_control"]["config"],
                            "strong_control_median_us": float(np.median(arrays["best_control"])),
                            "FullCol_median_us": float(np.median(arrays["dynamic_fullcol_guard"])),
                            "persistent_median_us": float(np.median(arrays["dynamic_persistent_fixed"])),
                            "strong_control_over_persistent": result["summary"][cache]["paired_ratios"]["best_control_over_dynamic_persistent_fixed"],
                            "FullCol_over_persistent": result["summary"][cache]["paired_ratios"]["dynamic_fullcol_guard_over_dynamic_persistent_fixed"],
                            "emptyguard_median_us": float(np.median(arrays["dynamic_emptyguard"])),
                            "strong_control_over_FullCol": result["summary"][cache]["paired_ratios"]["best_control_over_dynamic_fullcol_guard"],
                            "emptyguard_over_FullCol": result["summary"][cache]["paired_ratios"]["dynamic_emptyguard_over_dynamic_fullcol_guard"],
                            "A_A_full_union": result["summary"][cache]["paired_ratios"]["full_union_over_full_union_AA"],
                            "scope": "Independently tuned family winners, then 20 randomized/reverse paired blocks. May differ in split/N; not a pure warp-policy ablation."})
    assert observed_cases == json.loads(cases_path.read_text())["cases"]
    terminal_count = sum(row["terminal"] is not None for row in rows)
    charged = sum(row["source_gpu_hours"] or 0 for row in rows)
    assert charged <= protocol["maximum_P2_GPU_hours_including_gpu0"]
    complete = terminal_count == 8 and all(row["final_result_exists"] for row in rows)
    if not args.allow_partial:
        assert complete and total_configs == 464 and len(pairs) == 48 and len(winners) == 16
    result = {"schema": 1, "status": "terminal_independent_CPU_audit" if complete else "partial_CPU_audit_no_pending_conclusion",
              "auditor_sha256": sha(__file__), "input_sha256": input_hashes,
              "jobs": rows, "terminal_jobs": terminal_count,
              "all_terminal_processes_passed": all(row.get("process_passed", False) for row in rows),
              "all_wave4_jobs_terminal_before_wave7_start": True,
              "gpu0_extended_v4_terminal_before_deferred_gpu0_start": True,
              "gpu0_extended_v4_terminal": gpu0_extended[0],
              "completed_configurations": total_configs, "all_configuration_failures": errors,
              "selected_tuning_winners_recomputed_from_all_eligible_configs": True,
              "frozen_capture_file_SHA_checks": total_source_checks,
              "all_matching_split_N_tuning_pairs": pairs,
              "all_independently_selected_paired_winners": winners,
              "source_GPU_hours_charged": charged, "audit_GPU_hours": 0,
              "torch_imported": "torch" in sys.modules,
              "limitations": [
                  "Configuration status=ok records the benchmark's frozen FP64 checks; this audit does not rerun GPU or FP64 attention.",
                  "Matched split/N comparisons use five unpaired tuning readings per cache and are descriptive.",
                  "Persistent fixed-M32 changes masked MMA work versus compact FullCol, even at identical split/N. It is not a pure state-transport-only ablation.",
                  "Final family winners are selected independently by tuning; 20 random/reverse paired blocks measure those winners, not necessarily equal split/N.",
                  "All fixed partitions and binary trees are retained, but their split grids are bounded. Per-capture selection is a hindsight oracle.",
                  "GPU run() timings include producer/merge and required packing/restoration, not CPU preparation or host synchronized complete-call wall.",
                  "Plan amortization scenarios are not observed request lifetimes; search/JIT/failed allocations remain charged.",
                  "Three development documents, multiple layer/budget states; layers, ranks and repeated blocks do not create independent task samples.",
                  "Artificial hot/cold states are not whole-model interleaving. No task quality, application, energy or hardware traffic conclusion."]}
    assert not result["torch_imported"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "terminal_jobs": terminal_count,
                      "configurations": total_configs, "matching_pairs": len(pairs),
                      "paired_cache_cells": len(winners), "GPU_hours_charged": charged,
                      "audit_GPU_hours": 0, "output": rel(OUTPUT)}))


if __name__ == "__main__":
    main()
