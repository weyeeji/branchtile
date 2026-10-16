#!/usr/bin/env python3
"""CPU-only independent audit of fixed-owner/native wave25 confirmation."""
from __future__ import annotations

import argparse
import collections
import datetime
import functools
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

from audit_operator_overhead_wave7 import (CONTRACT, argument, close, grid as old_grid,
                                         partitions, sha as file_sha, trees)
from audit_operator_owner_wave18 import candidate_name, key

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/bench_operator_owner_confirm_native.py"
SOURCE_COMMIT = "f9b4608"
FROZEN = "owner_frozen_128_64"
OWNER_CONFIG = {"method": "dynamic_owner_tasks", "version": "owner_tasks",
                "splits": 64, "block_n": 64, "threads": 128, "chunk_tokens": 128}
NATIVE_FAMILIES = {"e0_native", "e1_native", "e1_own", "e4_native"}
REQUIRED = NATIVE_FAMILIES | {"full_union", "dynamic_owner_tasks"}
SELECTED = REQUIRED | {FROZEN, "best_e1", "best_control", "best_owner_control"}
METHODS = SELECTED | {"full_union_AA", FROZEN + "_AA"}
PAIRS = {"full_union_over_best_e1", "full_union_over_e4_native", "best_e1_over_e4_native",
         "full_union_over_full_union_AA", f"{FROZEN}_over_{FROZEN}_AA"} | {
    f"{control}_over_{FROZEN}" for control in (
        "best_owner_control", "best_control", "e0_native", "e1_native", "e1_own", "best_e1", "e4_native", "full_union")}


@functools.lru_cache(maxsize=None)
def sha(path):
    return file_sha(path)


def rel(path):
    return str(Path(path).relative_to(ROOT))


def blob(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                          check=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/operator_owner_native_wave25_audit_v1")
    args = parser.parse_args()
    assert os.getuid() == 1000 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert "torch" not in sys.modules and "triton" not in sys.modules
    output = ROOT / args.output
    assert not output.exists(), "Use a new output directory; preserve prior audits"
    manifest_path = ROOT / "configs/operator_overhead_wave25_jobs.json"
    protocol_path = ROOT / "configs/operator_overhead_wave25_protocol.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    assert len(manifest) == 8 and protocol["expected_total_configs"] == 384
    assert sorted(job["gpus"][0] for job in manifest) == list(range(8))
    ledger_raw = (ROOT / "results/job_ledger.jsonl").read_bytes()
    ledger = [json.loads(line) for line in ledger_raw.splitlines() if line.strip()]
    job_names = {job["name"] for job in manifest}
    selected_ledger = [entry for entry in ledger if entry["name"] in job_names]
    terminals = [entry for entry in selected_ledger if entry["event"] == "end"]
    # Do not infer completion from partially written tuning files or missing jobs.
    if len(terminals) != 8 or any(not (ROOT / argument(job["argv"], "--output")).exists() for job in manifest):
        print(json.dumps({"status": "not_all_terminal_receipts_available", "terminal_jobs": len(terminals)}))
        raise SystemExit(2)
    cutoff = max(entry["time_utc"] for entry in terminals)
    hashes = {rel(path): sha(path) for path in (
        manifest_path, protocol_path, ROOT / "scripts/audit_operator_overhead_wave7.py",
        ROOT / "scripts/audit_operator_owner_wave18.py")}
    frozen_sources = {}
    for source, expected in protocol["code_sha256"].items():
        assert sha(ROOT / source) == expected
        hashes[source] = expected
        if source != rel(manifest_path):
            frozen_sources[source] = hashlib.sha256(blob(SOURCE_COMMIT, source)).hexdigest()
            assert frozen_sources[source] == expected
    for path, expected in protocol["qualification_sha256"].items():
        assert sha(ROOT / path) == expected
        hashes[path] = expected
    prior_path = ROOT / "results/natural_capacity_boundary_v1/audit.json"
    hashes[rel(prior_path)] = sha(prior_path)
    frozen_capture = json.loads(prior_path.read_text())["input_sha256"]

    # Independently enumerate all original native/own partitions and trees.
    baseline = [{"method": "e1_native", "partition": p, "fixed_split_size": None}
                for p in partitions(4)]
    baseline += [{"method": "e1_own", "partition": p, "splits": 64, "block_n": 64}
                 for p in partitions(4)]
    baseline += [{"method": "e4_native", "tree": tree, "fixed_split_size": None}
                 for tree in trees(tuple(range(4)))]
    baseline += [{"method": "e0_native", "tensor_cores": tc, "fixed_split_size": None}
                 for tc in (False, True)]
    assert len(baseline) == 47
    assert {key(c) for c in baseline} == {key(c) for c in old_grid(4) if c["method"] in NATIVE_FAMILIES}
    expected = baseline + [OWNER_CONFIG]
    expected_keys = {key(c) for c in expected}
    assert len(expected_keys) == 48

    source_proof = {}
    fragments = {
        RUNNER: ("return self.op(q,k,v)", 'graphs = {method: graph_of(best[cache][method]["op"].run, repetitions)',
                 'for cache, repetitions in (("hot", 20), ("cold", 1)):',
                 'assert row["payload_aliases_source"] is True', "retain_frozen_owner(best, config, row, op)"),
        "src/branchtile/attention_owner_tasks.py": (
            "self.last_producer_compiled = producer[(self.num_tasks,)](",
            "self.last_merge_compiled = merge[(heads, rows)](", "return self.output, self.lse"),
        "scripts/compare_attention.py": ("with torch.cuda.graph(graph):", "fn()", "graph.replay()",
                                          "t0.elapsed_time(t1) * 1000 / repetitions"),
    }
    for source, required in fragments.items():
        lines = blob(SOURCE_COMMIT, source).decode().splitlines()
        source_proof[source] = []
        for fragment in required:
            matched = [{"line": line, "text": text.strip()} for line, text in enumerate(lines, 1) if fragment in text]
            assert matched, (source, fragment)
            source_proof[source].extend(matched)

    jobs, configurations, cache_cells, planning, capture_checks, owner_records = [], [], [], [], [], []
    failures, documents, states, gpu_uuids = [], set(), set(), set()
    ratio_count = measurement_rows = 0
    for job in manifest:
        name, argv = job["name"], job["argv"]
        starts = [entry for entry in selected_ledger if entry["name"] == name and entry["event"] == "start"]
        ends = [entry for entry in selected_ledger if entry["name"] == name and entry["event"] == "end"]
        assert len(starts) == len(ends) == 1
        end = ends[0]
        for entry in starts + ends:
            assert entry["uid"] == 1000 and entry["category"] == job["category"] == "P2"
            assert entry["gpus"] == job["gpus"] and entry["argv"] == argv and len(entry["gpus"]) == 1
        assert end["exit_code"] == 0 and not end["timed_out"]
        close(end["gpu_hours"], end["allocated_seconds"] / 3600)
        assert argv[:2] == [".venv-latest/bin/python", RUNNER]
        assert argument(argv, "--blocks") == "40" and argument(argv, "--budget") == "4096"
        assert argument(argv, "--heads") == "2" and "--require-ready" in argv
        layer, step = int(argument(argv, "--layer")), int(argument(argv, "--step"))
        capture = ROOT / argument(argv, "--capture")
        rank = int(capture.name.removeprefix("rank"))
        states.add((rank, layer, step))
        assert step == (16 if layer in (0, 31) else 256)
        path = ROOT / argument(argv, "--output")
        progress = path.with_suffix(".tuning.jsonl")
        hashes[rel(path)], hashes[rel(progress)] = sha(path), sha(progress)
        result = json.loads(path.read_text())
        assert [json.loads(line) for line in progress.read_text().splitlines()] == result["tuning"]
        assert result["protocol_version"] == "owner-confirm-native-v1"
        assert result["candidate_set"] == "native_E0_E1own_E4_plus_fixed_owner"
        assert result["uid"] == 1000 and result["python"].endswith("/.venv-latest/bin/python")
        gpu_uuids.add(result["gpu_uuid"])
        assert result["batch"] == 4 and result["gqa"] == 8 and result["q_shape"] == [2, 32, 128]
        assert result["numerical_contract"] == CONTRACT
        assert result["tuning_planned"] == result["tuning_completed"] == len(result["tuning"]) == 48
        assert result["tuning_complete"] and all(not row for row in result["missing_qualified_methods"].values())
        assert len({key(row) for row in result["tuning"]}) == 48
        assert {key(row) for row in result["tuning"]} == expected_keys
        assert result["partitions"] == partitions(4)
        assert {json.dumps(tree) for tree in result["hierarchies"]} == {json.dumps(tree) for tree in trees(tuple(range(4)))}
        assert result["code_sha256"] == frozen_sources
        for source, expected_sha in result["code_sha256"].items():
            assert sha(ROOT / source) == expected_sha
        confirmation = result["frozen_owner_confirmation"]
        assert confirmation["alias"] == FROZEN and confirmation["config"] == OWNER_CONFIG
        assert confirmation["candidate_name"] == candidate_name(OWNER_CONFIG)
        assert confirmation["required_for_summary"] and confirmation["aa_controls"] == ["full_union_AA", FROZEN+"_AA"]
        for row in result["tuning"]:
            assert row["candidate_name"] == candidate_name(row)
            assert row["source_paths"] == result["candidate_sources"][row["method"]]
            assert row["source_sha256"] == {p: result["code_sha256"][p] for p in row["source_paths"]}
            if row["status"] != "ok":
                failures.append({"job": name, "configuration": row})
            else:
                assert row["payload_aliases_source"] is True
                assert math.isfinite(row["max_abs_error_fp64"]) and math.isfinite(row["max_lse_error_fp64"])
            configurations.append({"job": name, **row})
            if row["method"] == "dynamic_owner_tasks":
                assert key(row) == key(OWNER_CONFIG)
                meta = row["operator_metadata"]
                assert row["task_count"] == meta["task_count"] > 0
                assert row["packed_rows"] == meta["packed_partial_rows"]
                assert meta["chunk_tokens"] == 128 and meta["requested_splits_compatibility_parameter"] == 64
                assert row["workspace_bytes"] == meta["workspace_bytes"] == sum(row["workspace_bytes_by_buffer"].values())
                assert row["workspace_bytes_by_buffer"]["packed_U"] == row["packed_rows"] * 128 * 4
                close(row["metadata_ms"], row["region_plan_prepare_ms"] + row["operator_metadata_ms"])
                assert row["prepare_ms"] >= row["region_plan_prepare_ms"] + row["operator_prepare_ms"]
                owner_records.append({"job": name, **row})

        metadata_path = capture / "metadata.json"
        export_path = capture / "export_summary.json"
        metadata = json.loads(metadata_path.read_text())
        documents.add(metadata["document_sha256"])
        assert metadata["export_semantics_version"] == 2 and metadata["B"] == 4 and metadata["actual_tp_rank"] == rank
        assert metadata["ready_lifetime_measured_by_step"][str(step)] is True
        provenance = result["provenance"]
        assert provenance["original_active_requests"] == list(range(4))
        assert provenance["selected_kv_heads"] == [0, 1]
        assert provenance["ready_lifetime_measured"] is True
        assert provenance["includes_system_prefix_and_complete_private_suffix"] is True
        for field in ("model_revision", "document_sha256", "actual_tp_rank", "document_token_range",
                      "actual_shared_document_token_range", "private_protected_document_tail_tokens"):
            assert provenance[field] == metadata[field]
        export = json.loads(export_path.read_text())
        ready = [item for item in export["ready_groups"] if item["layer"] == layer and item["anchor"] == step]
        assert len(ready) == 1 and provenance["ready_group"] == ready[0] and ready[0]["B"] == 4
        files = [metadata_path, capture/f"source_l{layer}.pt", capture/f"masks_l{layer}_b4096.pt"]
        files += [capture/f"decode_q{request}_l{layer}_s{step}.pt" for request in range(4)]
        for file in files:
            assert sha(file) == frozen_capture[rel(file)]
            hashes[rel(file)] = sha(file)
            capture_checks.append({"job": name, "path": rel(file), "sha256": sha(file)})
        # Export summaries are tracked alongside the immutable capture but are
        # outside the earlier tensor-inventory audit's input map.
        assert sha(export_path) == hashlib.sha256(blob(SOURCE_COMMIT, rel(export_path))).hexdigest()
        hashes[rel(export_path)] = sha(export_path)
        jobs.append({"name": name, "start": starts[0], "terminal": end, "rank": rank,
                     "layer": layer, "step": step, "configurations": 48,
                     "family_counts": dict(collections.Counter(row["method"] for row in result["tuning"])),
                     "provenance": provenance, "arguments": result["arguments"]})

        for cache in ("hot", "cold"):
            selected = result["selected"][cache]
            assert set(selected) == SELECTED
            assert selected[FROZEN]["config"] == selected["dynamic_owner_tasks"]["config"] == OWNER_CONFIG
            for method, choice in selected.items():
                records = [row for row in result["tuning"] if key(row) == key(choice["config"])]
                assert len(records) == 1 and records[0]["status"] == "ok"
                record = records[0]
                assert choice["record"]["candidate_name"] == record["candidate_name"]
                if method == "full_union":
                    eligible = [row for row in result["tuning"] if row.get("partition") == [list(range(4))]]
                elif method == "best_e1":
                    eligible = [row for row in result["tuning"] if row["method"] in ("e0_native", "e1_native", "e1_own")]
                elif method in ("best_control", "best_owner_control"):
                    eligible = [row for row in result["tuning"] if row["method"] in NATIVE_FAMILIES]
                elif method == FROZEN:
                    eligible = [row for row in result["tuning"] if row["method"] == "dynamic_owner_tasks"]
                else:
                    eligible = [row for row in result["tuning"] if row["method"] == method]
                close(record["tuning_us_by_cache"][cache], min(row["tuning_us_by_cache"][cache] for row in eligible if row["status"] == "ok"))
            measurements = [row for row in result["measurements"] if row["cache"] == cache]
            lookup = {(row["block"], row["direction"]): row for row in measurements}
            assert len(lookup) == len(measurements) == 80
            assert set(lookup) == set(itertools.product(range(40), ("random", "reverse")))
            for block in range(40):
                assert lookup[block, "random"]["order"][::-1] == lookup[block, "reverse"]["order"]
            for row in measurements:
                assert set(row["us"]) == set(row["order"]) == METHODS and len(row["order"]) == len(METHODS)
            arrays = {method: np.array([[lookup[block, direction]["us"][method] for direction in ("random", "reverse")]
                                       for block in range(40)]) for method in METHODS}
            assert set(result["summary"][cache]["median_us"]) == METHODS
            for method, values in arrays.items():
                assert np.isfinite(values).all() and (values > 0).all()
                close(float(np.median(values)), result["summary"][cache]["median_us"][method])
            assert set(result["summary"][cache]["paired_ratios"]) == PAIRS
            draws = np.random.default_rng(71).integers(0, 40, (10000, 40))
            for name_pair, receipt in result["summary"][cache]["paired_ratios"].items():
                numerator, denominator = name_pair.split("_over_")
                ratios = arrays[numerator] / arrays[denominator]
                close(receipt["median"], float(np.median(ratios)))
                for actual, expected_ci in zip(receipt["paired_block_bootstrap95"], np.quantile(np.median(ratios[draws], axis=(1, 2)), [.025, .975])):
                    close(actual, float(expected_ci))
                ratio_count += 1
            assert set(result["planning"][cache]) == SELECTED
            for method, plan in result["planning"][cache].items():
                assert len(plan["samples"]) == 3
                prepare = statistics.median(sample["warm_prepare_ms"] for sample in plan["samples"]) * 1000
                close(prepare, plan["warm_prepare_median_us"])
                for reuse, actual in plan["amortized_prepare_plus_gpu_call_us"].items():
                    close(actual, prepare / int(reuse) + result["summary"][cache]["median_us"][method])
                for sample in plan["samples"]:
                    assert sample["prepare_plus_one_call_host_wall_us"] >= sample["warm_prepare_ms"] * 1000
                    if method in (FROZEN, "dynamic_owner_tasks"):
                        assert sample["warm_prepare_ms"] >= sample["region_plan_prepare_ms"] + sample["operator_prepare_ms"]
                        assert sample["task_count"] == sample["operator_metadata"]["task_count"]
                        assert sample["workspace_bytes"] == sample["operator_metadata"]["workspace_bytes"]
                planning.append({"job": job["name"], "cache": cache, "method": method, **plan})
            paired = result["summary"][cache]["paired_ratios"][f"best_owner_control_over_{FROZEN}"]
            lo, hi = paired["paired_block_bootstrap95"]
            verdict = ("within_input_latency_improvement" if lo > 1 + 1e-12 else
                       "within_input_latency_slower" if hi < 1 - 1e-12 else "interval_includes_one")
            owner_prepare = result["planning"][cache][FROZEN]["warm_prepare_median_us"]
            control_prepare = result["planning"][cache]["best_owner_control"]["warm_prepare_median_us"]
            owner_us = result["summary"][cache]["median_us"][FROZEN]
            control_us = result["summary"][cache]["median_us"]["best_owner_control"]
            saved_us = control_us - owner_us
            extra_prepare = owner_prepare - control_prepare
            cache_cells.append({"job": job["name"], "cache": cache, "rank": rank, "layer": layer, "step": step,
                "paired_blocks": 40, "measurements": len(measurements), "bootstrap_draws": 10000,
                "frozen_owner_config": OWNER_CONFIG, "best_native_control": selected["best_owner_control"]["config"],
                "all_selected": selected, "all_median_us": result["summary"][cache]["median_us"],
                "all_paired_ratios": result["summary"][cache]["paired_ratios"], "paired_owner_verdict": verdict,
                "owner_prepare_median_us": owner_prepare, "control_prepare_median_us": control_prepare,
                "extra_owner_prepare_us": extra_prepare, "steady_saved_us_per_reuse": saved_us,
                "conditional_static_break_even_reuses": max(0., extra_prepare)/saved_us if saved_us > 0 else None,
                "preparation_scope": "Median-based illustrative static-topology reuse model, not measured lifetime; excludes search/new JIT/topology recapture.",
                "AA": {pair: result["summary"][cache]["paired_ratios"][pair] for pair in
                       ("full_union_over_full_union_AA", f"{FROZEN}_over_{FROZEN}_AA")}})
            measurement_rows += len(measurements)

    assert states == {(rank, layer, 16 if layer in (0, 31) else 256)
                      for rank, layer in itertools.product((0, 1), (0, 15, 31, 47))}
    assert len(documents) == 1 and len(gpu_uuids) == 8
    assert len(configurations) == 384 and len(owner_records) == 8 and not failures
    assert ratio_count == 208 and measurement_rows == 1280 and len(cache_cells) == 16
    assert len(planning) == 160 and len(capture_checks) == 56
    protocol_notes = []
    if "20random/reverse blocks" in protocol["measurement"]:
        protocol_notes.append("Frozen protocol measurement prose says 20 blocks; manifest --blocks40, frozen runner default40, result arguments and all raw 80-row/cache arrays establish actual forty-block execution. Original protocol is preserved.")
    result = {"schema": "owner-native-fixed-wave25-independent-audit-v1",
        "status": "complete_fixed_owner_native_wave25_CPU_audit_passed", "source_commit": SOURCE_COMMIT,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Eight correlated rank/layer/step inputs from one development document, fixed owner C128/N64, complete original 47-control native/E1-own/E4 grid; no engineering-family or application claim.",
        "uid": os.getuid(), "python": sys.executable, "audit_GPU_hours": 0,
        "torch_or_triton_imported": "torch" in sys.modules or "triton" in sys.modules,
        "auditor_sha256": sha(Path(__file__)), "input_sha256": hashes,
        "source_sha256_at_frozen_commit": frozen_sources, "source_call_and_alias_evidence": source_proof,
        "snapshot_cutoff_utc": cutoff, "selected_ledger": selected_ledger,
        "ledger_snapshot_sha256": hashlib.sha256(ledger_raw).hexdigest(),
        "ledger_note": "Global ledger can append. All eight wave25 start/end entries are preserved separately above.",
        "source_P2_GPU_hours": sum(entry["gpu_hours"] for entry in terminals),
        "allocated_GPU_seconds": sum(entry["allocated_seconds"] for entry in terminals),
        "jobs": jobs, "all_configuration_receipts": configurations, "all_failures": failures,
        "qualified_configurations": len(configurations), "all_source_alias_flags_true": True,
        "native_control_count_per_job": 47, "owner_config_count_per_job": 1,
        "fixed_owner_config_hot_cold_all_cells": OWNER_CONFIG,
        "owner_records": owner_records, "all_planning_receipts": planning,
        "paired_cache_cells": cache_cells, "verified_paired_ratio_statistics": ratio_count,
        "measurement_rows": measurement_rows, "paired_blocks_per_cache": 40,
        "immutable_capture_file_checks": capture_checks, "unique_capture_files_checked": len({c["path"] for c in capture_checks}),
        "document_SHA256s": sorted(documents), "rank_layer_step_states": sorted(states),
        "paired_owner_verdict_counts": dict(collections.Counter(row["paired_owner_verdict"] for row in cache_cells)),
        "protocol_prose_inconsistencies": protocol_notes,
        "limitations": [
            "Owner parameters were fixed before these eight new captured states; all are from one development document and are correlated, not eight independent documents or formal held-out evidence.",
            "Baseline selection remains a same-input hindsight oracle among the full bounded native/E1-own/E4 grid; every partition and binary hierarchy is retained.",
            "This protocol explicitly excludes old v2/v3/guards/persistent/static-v4/lane engineering families. Earlier full-grid results are preserved separately; do not describe this as remeasuring every previous family.",
            "FP64 results and source-alias pointer checks are preserved GPU qualification receipts; this CPU audit does not recompute attention or live device pointers.",
            "Paired bootstrap resamples forty random/reverse timing blocks within each fixed capture, not documents, ranks, layers or natural decoding lifetimes; intervals are not adjusted for sixteen-cell multiplicity.",
            "Hot/cold graph event timings include the complete operator call and exclude CPU preparation, new JIT, topology rebuild and graph recapture. Full preparation/prepare-plus-one-call wall are retained separately.",
            "Source pointers alias original K/V; logical token visits and differently scoped workspace receipts do not establish physical DRAM traffic or comparable peak memory.",
            "The owner topology must remain immutable for graph replay; changing topology requires rebuilding/recapture, and direct replay bypasses host guards.",
            "Conditional preparation break-even and reuse1/16/64/256 are static-plan arithmetic scenarios, not measured lifetime or end-to-end latency.",
            "No pure packing ablation, application quality, natural-EOS JCT, energy or full-model speed claim follows."]}
    assert not result["torch_or_triton_imported"]
    assert result["source_P2_GPU_hours"] <= protocol["maximum_P2_GPU_hours"]
    output.mkdir(parents=True)
    (output/"audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "qualified_configurations", "verified_paired_ratio_statistics",
                                           "measurement_rows", "paired_owner_verdict_counts", "source_P2_GPU_hours")}))


if __name__ == "__main__":
    main()
