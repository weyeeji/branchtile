#!/usr/bin/env python3
"""Independent complete wave9 / separate B3 wave12 bank-grid snapshots."""
from __future__ import annotations

import argparse
import collections
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
                                         grid as prior_grid, key, partitions,
                                         sha, trees)

ROOT = Path(__file__).resolve().parents[1]
METHODS = PRIOR_METHODS | {"dynamic_v4", "best_bank_control"}
PAIRS = PRIOR_PAIRS | {"best_bank_control_over_dynamic_v4",
                       "dynamic_fullcol_guard_over_dynamic_v4",
                       "dynamic_persistent_fixed_over_dynamic_v4"}


def rel(path):
    return str(Path(path).relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, choices=(9, 12), required=True)
    options = parser.parse_args()
    wave = options.wave
    output = ROOT / ("results/operator_bankpilot_wave9_audit_v2/audit.json" if wave == 9
                     else "results/operator_bankgrid_wave12_audit_v1/audit.json")
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules
    protocol_path = ROOT / f"configs/operator_overhead_wave{wave}_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest_paths = ([ROOT / "configs/operator_overhead_wave9_b3_jobs.json",
                       ROOT / "configs/operator_overhead_wave9_b4_jobs.json"] if wave == 9
                      else [ROOT / "configs/operator_overhead_wave12_jobs.json"])
    jobs = [job for path in manifest_paths for job in json.loads(path.read_text())]
    expected_jobs = 8 if wave == 9 else 4
    expected_total_configs = 480 if wave == 9 else 184
    splits_grid = (64,) if wave == 9 else (64, 128)
    candidate_set, protocol_version = ("bankpilot", 4) if wave == 9 else ("bankgrid", 5)
    assert len(jobs) == expected_jobs
    if wave == 9:
        assert [{"gpu": job["gpus"][0], "name": job["name"]} for job in jobs] == protocol["cases"]
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    job_names = {job["name"] for job in jobs}
    completed = [entry for entry in ledger if entry["name"] in job_names and entry["event"] == "end"]
    assert len(completed) == expected_jobs
    cutoff = max(entry["time_utc"] for entry in completed)
    ledger = [entry for entry in ledger if entry["time_utc"] <= cutoff]
    frozen = json.loads((ROOT / "results/natural_capacity_boundary_v1/audit.json").read_text())["input_sha256"]
    partial_v1_paths = [ROOT / "results/operator_bankpilot_wave9_audit_v1/audit.json",
                        ROOT / "results/operator_bankpilot_wave9_audit_v1/README.md",
                        ROOT / "scripts/audit_operator_bankpilot_wave9.py"]
    original_partial_hashes = {rel(path): sha(path) for path in partial_v1_paths}
    hashes = {rel(path): sha(path) for path in [protocol_path, *manifest_paths,
                                              ROOT / "scripts/audit_operator_overhead_wave7.py"]}
    for path, expected in protocol["code_sha256"].items():
        assert sha(ROOT / path) == expected
        hashes[path] = expected
    rows, pending, timing_cells, matched, failures = [], [], [], [], []
    source_checks, qualified = 0, 0
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
        assert argument(argv, "--candidate-set") == candidate_set
        assert argument(argv, "--blocks") == "20" and argument(argv, "--step") == "64"
        assert argument(argv, "--heads") == "2" and "--require-ready" in argv
        path = ROOT / argument(argv, "--output")
        result = json.loads(path.read_text())
        progress = path.with_suffix(".tuning.jsonl")
        hashes[rel(path)], hashes[rel(progress)] = sha(path), sha(progress)
        assert [json.loads(line) for line in progress.read_text().splitlines()] == result["tuning"]
        batch = result["batch"]
        assert batch in (3, 4) and (wave == 9 or batch == 3)
        assert result["gqa"] == 8 and result["uid"] == os.getuid()
        assert result["protocol_version"] == protocol_version and result["candidate_set"] == candidate_set
        assert result["numerical_contract"] == CONTRACT
        assert result[("bankpilot_v4_grid" if wave == 9 else "bankgrid_v4_grid")] == [
            {"splits": s, "block_n": n} for s in splits_grid for n in (32, 64)]
        expected = prior_grid(batch) + [{"method": "dynamic_v4", "version": "v4", "splits": s,
                                       "block_n": n} for s in splits_grid for n in (32, 64)]
        expected_count = len(expected)
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
            assert result["code_sha256"][source] == expected_sha
        for row in result["tuning"]:
            assert row["source_paths"] == result["candidate_sources"][row["method"]]
            assert row["source_sha256"] == {source: result["code_sha256"][source] for source in row["source_paths"]}
            if row["status"] == "ok":
                qualified += 1
                assert math.isfinite(row["max_abs_error_fp64"]) and math.isfinite(row["max_lse_error_fp64"])
            else:
                failures.append({"job": job["name"], "configuration": row})
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
                     "baseline_policy_persistent_controls_exactly_retained": True,
                     "v4_splits_grid": list(splits_grid), "v4_N_grid": [32, 64], "v4_split32_included": False,
                     "family_counts": dict(collections.Counter(row["method"] for row in result["tuning"]))})
        index = {(row["method"], row.get("splits"), row.get("block_n")): row for row in result["tuning"]}
        for s, block_n in itertools.product(splits_grid, (32, 64)):
            v4 = index["dynamic_v4", s, block_n]
            fullcol = index["dynamic_fullcol_guard", s, block_n]
            persistent = index["dynamic_persistent_fixed", s, block_n]
            assert v4["status"] == fullcol["status"] == persistent["status"] == "ok"
            matched.append({"job": job["name"], "splits": s, "block_n": block_n,
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
                    eligible = [row for row in result["tuning"] if row["method"] != "dynamic_v4"]
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
            for method, plan in result["planning"][cache].items():
                assert len(plan["samples"]) == 3
                prepare = statistics.median(row["warm_prepare_ms"] for row in plan["samples"]) * 1000
                close(prepare, plan["warm_prepare_median_us"])
                for reuse, actual in plan["amortized_prepare_plus_gpu_call_us"].items():
                    close(actual, prepare / int(reuse) + result["summary"][cache]["median_us"][method])
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
                                 "scope": "Independent family winners; v4 bounded S64 or S64/S128 grid, other arms retain S32. No full-grid or pure-transport claim."})
    assert len(rows) == expected_jobs and not pending and qualified == expected_total_configs and not failures
    assert len(timing_cells) == 2 * expected_jobs and len(matched) == 2 * len(splits_grid) * expected_jobs
    if wave == 9:
        original_partial = json.loads(partial_v1_paths[0].read_text())
        for path, expected_sha in original_partial["input_sha256"].items():
            if path in hashes:
                assert hashes[path] == expected_sha
    result = {"schema": 1, "status": "complete_wave9_independent_audit_passed" if wave == 9 else "separate_B3_wave12_bankgrid_audit_passed",
              "scope": "Complete original S64-only wave9, separately from expanded wave12." if wave == 9 else "B3 four-point S64/S128 grid only; no S32 or B4 claim.",
              "wave": wave, "v4_splits_grid": list(splits_grid), "v4_N_grid": [32, 64],
              "preserved_partial_v1_input_SHA": original_partial_hashes,
              "original_partial_B3_shared_inputs_match": True if wave == 9 else None,
              "auditor_sha256": sha(__file__), "input_sha256": hashes,
              "snapshot_cutoff_utc": cutoff,
              "expected_cells": expected_jobs, "audited_complete_cells": rows,
              "pending_cells": pending, "qualified_configurations": qualified,
              "all_configuration_failures": failures, "frozen_capture_SHA_checks": source_checks,
              "parameter_matched_search_only": matched, "final_paired_cache_cells": timing_cells,
              "verified_final_paired_ratio_statistics": 30 * expected_jobs,
              "source_P2_GPU_hours": sum(row["terminal"]["gpu_hours"] for row in rows),
              "audit_GPU_hours": 0, "torch_imported": "torch" in sys.modules,
              "limitations": [
                  "Original wave9 two-point pilot and expanded wave12 four-point results remain separate experiments; neither overwrites the prior partial v1.",
                  "Recorded FP64 checks passed; this CPU audit does not recompute attention.",
                  "V4 uses S64 only in wave9, S64/S128 in wave12, always N32/N64. S32 is absent; other dynamic controls retain their grids, with the original v3 three N64 configs.",
                  "Winner comparisons may differ in split/N. Search-only equal-parameter readings have five samples and no paired inference.",
                  "Fixed-M32 persistent changes masked MMA work relative to compact FullCol/v4; comparisons are not pure state-transport ablations.",
                  "No lane-select or wide-CTA performance result is inferred from concurrent qualification work.",
                  "Run timings exclude CPU plan preparation and host synchronized wall; amortization is hypothetical. No whole-model, quality, energy or hardware-traffic claim."]}
    assert not result["torch_imported"]
    assert all(sha(ROOT / path) == expected for path, expected in original_partial_hashes.items())
    assert result["source_P2_GPU_hours"] <= protocol["maximum_P2_GPU_hours"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "complete_cells": expected_jobs,
                      "qualified": qualified, "paired_cache_cells": 2 * expected_jobs,
                      "source_P2_GPU_hours": result["source_P2_GPU_hours"],
                      "audit_GPU_hours": 0}))


if __name__ == "__main__":
    main()
