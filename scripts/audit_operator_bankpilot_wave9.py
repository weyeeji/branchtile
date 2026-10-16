#!/usr/bin/env python3
"""Frozen B3-only partial audit of wave9; no B4 performance inference."""
from __future__ import annotations

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
OUTPUT = ROOT / "results/operator_bankpilot_wave9_audit_v1/audit.json"
METHODS = PRIOR_METHODS | {"dynamic_v4", "best_bank_control"}
PAIRS = PRIOR_PAIRS | {"best_bank_control_over_dynamic_v4",
                       "dynamic_fullcol_guard_over_dynamic_v4",
                       "dynamic_persistent_fixed_over_dynamic_v4"}


def rel(path):
    return str(Path(path).relative_to(ROOT))


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules
    protocol_path = ROOT / "configs/operator_overhead_wave9_protocol.json"
    b3_path = ROOT / "configs/operator_overhead_wave9_b3_jobs.json"
    b4_path = ROOT / "configs/operator_overhead_wave9_b4_jobs.json"
    protocol = json.loads(protocol_path.read_text())
    b3_jobs, b4_jobs = (json.loads(path.read_text()) for path in (b3_path, b4_path))
    assert len(b3_jobs) == len(b4_jobs) == 4
    jobs = b3_jobs + b4_jobs
    assert [{"gpu": job["gpus"][0], "name": job["name"]} for job in jobs] == protocol["cases"]
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    b3_names = {job["name"] for job in b3_jobs}
    b3_ends = [entry for entry in ledger if entry["name"] in b3_names and entry["event"] == "end"]
    assert len(b3_ends) == 4
    cutoff = max(entry["time_utc"] for entry in b3_ends)
    # Freeze the snapshot at the final B3 end, so later B4 starts/results or
    # subsequent waves cannot silently expand this report on reproduction.
    ledger = [entry for entry in ledger if entry["time_utc"] <= cutoff]
    frozen = json.loads((ROOT / "results/natural_capacity_boundary_v1/audit.json").read_text())["input_sha256"]
    hashes = {rel(path): sha(path) for path in (protocol_path, b3_path, b4_path,
                                               ROOT / "scripts/audit_operator_overhead_wave7.py")}
    for path, expected in protocol["code_sha256"].items():
        assert sha(ROOT / path) == expected
        hashes[path] = expected
    rows, pending, timing_cells, matched, failures = [], [], [], [], []
    source_checks, qualified = 0, 0
    audited_names = {job["name"] for job in b3_jobs}
    for job in jobs:
        starts = [entry for entry in ledger if entry["name"] == job["name"] and entry["event"] == "start"]
        ends = [entry for entry in ledger if entry["name"] == job["name"] and entry["event"] == "end"]
        assert len(starts) <= 1 and len(ends) <= 1
        if job["name"] not in audited_names:
            pending.append({"name": job["name"], "gpus": job["gpus"],
                            "scope": "B4 expected cell; not included in this frozen B3 audit.",
                            "observed_start": starts[0] if starts else None,
                            "observed_terminal": ends[0] if ends else None,
                            "no_performance_conclusion": True})
            continue
        assert len(starts) == len(ends) == 1
        terminal = ends[0]
        assert terminal["exit_code"] == 0 and not terminal["timed_out"]
        assert terminal["argv"] == job["argv"] and terminal["gpus"] == job["gpus"]
        assert job["category"] == terminal["category"] == "P2" and len(job["gpus"]) == 1
        close(terminal["gpu_hours"], terminal["allocated_seconds"] / 3600)
        argv = job["argv"]
        assert argument(argv, "--candidate-set") == "bankpilot"
        assert argument(argv, "--blocks") == "20" and argument(argv, "--step") == "64"
        assert argument(argv, "--heads") == "2" and "--require-ready" in argv
        path = ROOT / argument(argv, "--output")
        result = json.loads(path.read_text())
        progress = path.with_suffix(".tuning.jsonl")
        hashes[rel(path)], hashes[rel(progress)] = sha(path), sha(progress)
        assert [json.loads(line) for line in progress.read_text().splitlines()] == result["tuning"]
        assert result["batch"] == 3 and result["gqa"] == 8 and result["uid"] == os.getuid()
        assert result["protocol_version"] == 4 and result["candidate_set"] == "bankpilot"
        assert result["numerical_contract"] == CONTRACT
        assert result["bankpilot_v4_grid"] == [{"splits": 64, "block_n": n} for n in (32, 64)]
        expected = prior_grid(3) + [{"method": "dynamic_v4", "version": "v4", "splits": 64,
                                   "block_n": n} for n in (32, 64)]
        assert len(result["tuning"]) == result["tuning_planned"] == result["tuning_completed"] == 44
        assert result["tuning_complete"] and all(not values for values in result["missing_qualified_methods"].values())
        assert len({key(row) for row in result["tuning"]}) == 44
        assert {key(row) for row in result["tuning"]} == {key(row) for row in expected}
        assert result["partitions"] == partitions(3)
        assert {json.dumps(tree) for tree in result["hierarchies"]} == {json.dumps(tree) for tree in trees((0, 1, 2))}
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
        assert metadata["B"] == 3 and metadata["ready_lifetime_measured_by_step"]["64"]
        assert result["provenance"]["ready_lifetime_measured"]
        assert result["provenance"]["includes_system_prefix_and_complete_private_suffix"]
        for field in ("model_revision", "document_sha256", "actual_tp_rank", "document_token_range",
                      "actual_shared_document_token_range", "private_protected_document_tail_tokens"):
            assert result["provenance"][field] == metadata[field]
        assert metadata["actual_tp_rank"] == 0
        files = [capture / "metadata.json", capture / f"source_l{layer}.pt",
                 capture / f"masks_l{layer}_b{budget}.pt"]
        files.extend(capture / f"decode_q{request}_l{layer}_s64.pt" for request in range(3))
        for file in files:
            assert sha(file) == frozen[rel(file)]
            hashes[rel(file)] = sha(file)
            source_checks += 1
        rows.append({"name": job["name"], "start": starts[0], "terminal": terminal,
                     "B": 3, "configurations": 44,
                     "baseline30_policy36_persistent42_controls_exactly_retained": True,
                     "v4_grid_is_only_split64_N32_N64": True,
                     "family_counts": dict(collections.Counter(row["method"] for row in result["tuning"]))})
        index = {(row["method"], row.get("splits"), row.get("block_n")): row for row in result["tuning"]}
        for block_n in (32, 64):
            v4 = index["dynamic_v4", 64, block_n]
            fullcol = index["dynamic_fullcol_guard", 64, block_n]
            persistent = index["dynamic_persistent_fixed", 64, block_n]
            assert v4["status"] == fullcol["status"] == persistent["status"] == "ok"
            matched.append({"job": job["name"], "splits": 64, "block_n": block_n,
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
                    eligible = [row for row in result["tuning"] if row.get("partition") == [[0, 1, 2]]]
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
                                 "scope": "Independent family winners; v4 only S64 whereas other dynamic arms search S32/S64/S128. No equal-grid or pure-transport claim."})
    assert len(rows) == len(pending) == 4 and qualified == 176 and not failures
    assert len(timing_cells) == 8 and len(matched) == 8
    result = {"schema": 1, "status": "frozen_partial_B3_audit_passed_expected8_complete4_pending4",
              "scope": "Only four completed B3 cells; B4 results and later split128 work excluded from this frozen pilot audit.",
              "auditor_sha256": sha(__file__), "input_sha256": hashes,
              "snapshot_cutoff_utc": cutoff,
              "expected_cells": 8, "audited_complete_B3_cells": rows,
              "pending_unaudited_B4_cells": pending, "qualified_configurations": qualified,
              "all_configuration_failures": failures, "frozen_capture_SHA_checks": source_checks,
              "parameter_matched_search_only": matched, "final_paired_cache_cells": timing_cells,
              "verified_final_paired_ratio_statistics": 120,
              "source_completed_B3_P2_GPU_hours": sum(row["terminal"]["gpu_hours"] for row in rows),
              "audit_GPU_hours": 0, "torch_imported": "torch" in sys.modules,
              "limitations": [
                  "Four of eight expected cells are audited; pending B4 is neither a numerical failure nor a performance observation.",
                  "Recorded FP64 checks passed; this CPU audit does not recompute attention.",
                  "V4 searches S64 x N32/N64 only; the other dynamic families retain the six S32/S64/S128 x N32/N64 controls (v3 retains its original three N64 configs).",
                  "Winner comparisons may differ in split/N. Search-only equal-parameter readings have five samples and no paired inference.",
                  "Fixed-M32 persistent changes masked MMA work relative to compact FullCol/v4; comparisons are not pure state-transport ablations.",
                  "S128 qualification/performance in subsequent waves is a new experiment and does not overwrite this pilot.",
                  "Run timings exclude CPU plan preparation and host synchronized wall; amortization is hypothetical. No whole-model, quality, energy or hardware-traffic claim."]}
    assert not result["torch_imported"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "complete_B3": 4, "pending_B4": 4,
                      "qualified": qualified, "paired_cache_cells": 8,
                      "source_P2_GPU_hours": result["source_completed_B3_P2_GPU_hours"],
                      "audit_GPU_hours": 0}))


if __name__ == "__main__":
    main()
