"""CPU-only independent audit of frozen wave23 owner-stage diagnostics."""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/operator_owner_stage_diagnostic_v1"
PROTOCOL = "configs/operator_owner_stages_wave23_protocol.json"
JOBS = "configs/operator_owner_stages_wave23_jobs.json"
PREVIOUS = "results/operator_owner_wave20_audit_v1/audit.json"
METHODS = ("complete", "producer_only", "merge_only", "complete_AA")
PHASES = (
    "original_complete_operator",
    "independently_called_same_two_kernels",
    "after_hot_graphs",
    "after_cold_graphs",
)


def read(relative):
    return json.loads((ROOT / relative).read_text())


def digest(relative):
    with (ROOT / relative).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def launch_call(tree, name):
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Subscript)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == name]
    assert len(calls) == 1, (name, len(calls))
    return calls[0]


def bootstrap_aa(rows):
    by_key = {(row["block"], row["direction"]): row for row in rows}
    values = np.array([[by_key[block, direction]["us"]["complete"] /
                        by_key[block, direction]["us"]["complete_AA"]
                        for direction in ("random", "reverse")]
                       for block in range(20)])
    draws = np.random.default_rng(20261016).integers(0, 20, size=(10000, 20))
    boot = np.median(values[draws], axis=(1, 2))
    return {"median_complete_over_complete_AA": float(np.median(values)),
            "paired_block_bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "scope": "Within-cell timing repeatability only; 20 random/reverse pairs, not independent documents."}


def main():
    assert sys.prefix != sys.base_prefix, "Use the project virtual environment"
    assert "torch" not in sys.modules
    protocol, jobs, previous = read(PROTOCOL), read(JOBS), read(PREVIOUS)
    assert len(jobs) == 8
    assert sorted(job["gpus"] for job in jobs) == [[i] for i in range(8)]
    inputs = {relative: digest(relative) for relative in (PROTOCOL, JOBS, PREVIOUS)}
    for relative, expected in protocol["code_sha256"].items():
        inputs[relative] = digest(relative)
        assert inputs[relative] == expected, relative

    bench_tree = ast.parse((ROOT / "scripts/bench_owner_stages.py").read_text())
    wrapper_tree = ast.parse((ROOT / "src/branchtile/attention_owner_tasks.py").read_text())
    launch_receipts = {}
    for name in ("producer", "merge"):
        diagnostic = launch_call(bench_tree, name)
        runtime = launch_call(wrapper_tree, name)
        assert ast.dump(diagnostic) == ast.dump(runtime), name
        launch_receipts[name] = {"AST_equal": True,
                                 "diagnostic_line": diagnostic.lineno,
                                 "runtime_line": runtime.lineno,
                                 "call": ast.unparse(diagnostic)}
    frozen_sources = ast.literal_eval(next(node.value for node in bench_tree.body
                                          if isinstance(node, ast.Assign)
                                          and any(isinstance(target, ast.Name) and target.id == "SOURCES"
                                                  for target in node.targets)))
    assert set(frozen_sources) == set(protocol["code_sha256"]) - {JOBS}

    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()
              if line.strip()]
    ledger_receipts, records, flat, capture_paths = [], [], [], set()
    gpu_hours, checks = 0.0, 0
    for job in jobs:
        argv = job["argv"]
        args = {argv[i][2:].replace("-", "_"): argv[i + 1] for i in range(2, len(argv), 2)}
        output = args["output"]
        inputs[output] = digest(output)
        result = read(output)
        assert result["schema"] == "owner-stage-diagnostic-v1" and result["status"] == "complete"
        assert result["uid"] == 1000 and result["virtual_environment"] == str(ROOT / ".venv-latest")
        assert result["code_sha256"] == {relative: protocol["code_sha256"][relative]
                                          for relative in frozen_sources}
        for key, value in args.items():
            actual = result["arguments"][key]
            assert str(actual) == value, (job["name"], key, actual, value)
        assert args["blocks"] == "20" and args["block_n"] == "64" and args["step"] == "64"
        assert [check["phase"] for check in result["checks"]] == list(PHASES)
        assert all(check["passed"] is True for check in result["checks"])
        assert all(math.isfinite(check[key]) and check[key] >= 0
                   for check in result["checks"] for key in ("max_abs_O", "max_abs_LSE"))
        checks += len(result["checks"])

        starts = [entry for entry in ledger if entry.get("name") == job["name"] and entry["event"] == "start"]
        ends = [entry for entry in ledger if entry.get("name") == job["name"] and entry["event"] == "end"]
        assert len(starts) == len(ends) == 1
        start, end = starts[0], ends[0]
        for entry in (start, end):
            assert entry["argv"] == argv and entry["gpus"] == job["gpus"]
            assert entry["uid"] == 1000 and entry["category"] == "P2"
        assert end["pid"] == start["pid"] and end["exit_code"] == 0 and not end["timed_out"]
        assert start["time_utc"] < result["created_utc"] < end["time_utc"]
        close(end["gpu_hours"], end["allocated_seconds"] / 3600)
        gpu_hours += end["gpu_hours"]
        ledger_receipts.append({"name": job["name"], "start": start, "end": end})
        log = f"results/operator_owner_stages_wave23/supervisor/{job['name']}.txt.gz"
        inputs[log] = digest(log)
        with gzip.open(ROOT / log, "rt") as handle:
            log_rows = [json.loads(line) for line in handle if line.lstrip().startswith("{")]
        assert len(log_rows) == 1
        assert log_rows[0]["checks"] == 4 and log_rows[0]["summary_us"] == result["summary_us"]
        assert log_rows[0]["output"] == str(ROOT / output)

        # The old result is consulted only for frozen configuration/provenance,
        # never as a timing baseline or a cross-wave speedup denominator.
        old_relative = output.replace("operator_owner_stages_wave23/ownerstages_w23_",
                                      "operator_overhead_wave20/overhead_w20_")
        inputs[old_relative] = digest(old_relative)
        old = read(old_relative)
        selected = old["selected"]["hot"]["dynamic_owner_tasks"]["config"]
        assert selected["chunk_tokens"] == result["arguments"]["chunk_tokens"]
        assert selected["block_n"] == result["arguments"]["block_n"]
        assert all(old["provenance"][key] == value for key, value in result["provenance"].items())
        capture = ROOT / args["capture"]
        metadata = read(str(capture.relative_to(ROOT) / "metadata.json"))
        active = [i for i in range(metadata["B"])
                  if (capture / f"decode_q{i}_l{args['layer']}_s{args['step']}.pt").exists()]
        provenance = result["provenance"]
        assert active == provenance["original_active_requests"]
        assert provenance["selected_kv_heads"] == [0, 1]
        assert metadata["gqa"] == 8 and provenance["includes_system_prefix_and_complete_private_suffix"]
        for key in ("model_revision", "document_sha256"):
            assert metadata[key] == provenance[key]
        for leaf in ("metadata.json", f"source_l{args['layer']}.pt",
                     f"masks_l{args['layer']}_b{args['budget']}.pt",
                     *(f"decode_q{i}_l{args['layer']}_s{args['step']}.pt" for i in active)):
            capture_paths.add(str((capture / leaf).relative_to(ROOT)))

        meta, contributions = result["operator_metadata"], result["contribution_counts"]
        assert len(contributions) == 2 and all(len(row) == len(active) for row in contributions)
        contribution_flat = sum(contributions, [])
        assert meta["packed_partial_rows"] == 8 * sum(contribution_flat)
        assert meta["task_count"] == len(result["task_lengths"])
        assert all(0 < length <= result["arguments"]["chunk_tokens"] for length in result["task_lengths"])
        assert meta["max_contributions"] == max(contribution_flat)
        assert meta["payload_aliases_source"] is True
        packed, batch = meta["packed_partial_rows"], len(active)
        expected_bytes = {"packed_U": packed * 128 * 4, "packed_M": packed * 4,
                          "packed_L": packed * 4, "task_descriptors": meta["task_count"] * 6 * 4,
                          "merge_indices": 2 * batch * max(contribution_flat) * 4,
                          "contribution_counts": 2 * batch * 4,
                          "output": 2 * batch * 8 * 128 * 2, "LSE": 2 * batch * 8 * 4}
        assert meta["workspace_bytes_by_buffer"] == expected_bytes
        assert meta["workspace_bytes"] == sum(expected_bytes.values())
        assert sum(result["task_lengths"]) == old["selected"]["hot"]["dynamic_owner_tasks"]["record"]["logical_kv_token_visits"]

        measurements = result["measurements"]
        assert len(measurements) == 80
        rng = random.Random(20261016)
        summaries = {}
        for cache in ("hot", "cold"):
            rows = [row for row in measurements if row["cache"] == cache]
            assert len(rows) == 40
            assert {(row["block"], row["direction"]) for row in rows} == {
                (block, direction) for block in range(20) for direction in ("random", "reverse")}
            for block in range(20):
                order = list(METHODS)
                rng.shuffle(order)
                for direction, expected_order in (("random", order), ("reverse", list(reversed(order)))):
                    row = next(item for item in rows if item["block"] == block and item["direction"] == direction)
                    assert row["order"] == expected_order and list(row["us"]) == expected_order
                    assert set(row["us"]) == set(METHODS)
                    assert all(math.isfinite(value) and value > 0 for value in row["us"].values())
            medians = {method: statistics.median(row["us"][method] for row in rows) for method in METHODS}
            assert medians == result["summary_us"][cache]
            summaries[cache] = {"median_us": medians, "AA": bootstrap_aa(rows),
                                "samples_per_method": len(rows),
                                "sample_range_us": {method: [min(row["us"][method] for row in rows),
                                                            max(row["us"][method] for row in rows)]
                                                    for method in METHODS}}
            flat.append({"job": job["name"], "cache": cache, **medians,
                         "tasks": meta["task_count"], "packed_rows": packed,
                         "contributions_min": min(contribution_flat),
                         "contributions_max": max(contribution_flat),
                         "chunk_tokens": result["arguments"]["chunk_tokens"],
                         "partial_payload_bytes": packed * 130 * 4})
        records.append({"file": output, "arguments": result["arguments"], "provenance": provenance,
                        "complete_output_checks_recorded": result["checks"], "summaries": summaries,
                        "task_count": meta["task_count"], "packed_partial_rows": packed,
                        "contribution_counts": contributions, "workspace_bytes": meta["workspace_bytes"],
                        "partial_payload_bytes": packed * 130 * 4,
                        "wave20_used_only_for_configuration_and_provenance": old_relative})

    # Rehash the 48 actually consumed capture files against the earlier frozen receipt.
    assert len(capture_paths) == 48
    for relative in sorted(capture_paths):
        inputs[relative] = digest(relative)
        assert inputs[relative] == previous["input_sha256"][relative], relative
    for receipt in ledger_receipts:
        a, b = receipt["start"], receipt["end"]
        for other in ledger:
            if other["event"] != "start" or other.get("pid") == a["pid"]:
                continue
            if not set(other["gpus"]) & set(a["gpus"]) or other["time_utc"] >= b["time_utc"]:
                continue
            other_ends = [entry for entry in ledger if entry["event"] == "end"
                          and entry.get("name") == other.get("name") and entry.get("pid") == other.get("pid")]
            assert len(other_ends) == 1
            assert other_ends[0]["time_utc"] <= a["time_utc"], (a["name"], other["name"])
    assert checks == 32 and "torch" not in sys.modules
    assert gpu_hours < protocol["maximum_P2_gpu_hours"]
    selected_ledger_text = "".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in ledger_receipts)
    inputs["results/job_ledger.jsonl:selected_wave23_records_sha256"] = hashlib.sha256(selected_ledger_text.encode()).hexdigest()
    audit = {"schema": "independent-owner-stage-wave23-audit-v1", "status": "passed",
             "scope": "CPU verification of frozen records, not new numerical GPU execution or a competitor comparison.",
             "auditor_sha256": digest("scripts/audit_operator_owner_stages_wave23.py"),
             "input_sha256": inputs, "launch_AST_receipts": launch_receipts,
             "complete_jobs": 8, "ledger_starts": 8, "ledger_clean_ends": 8,
             "recorded_output_checks": checks, "measurement_rows": 640,
             "individual_stage_timings": 2560, "exact_medians_recomputed": 64,
             "frozen_capture_hashes_verified": len(capture_paths), "source_P2_gpu_hours": gpu_hours,
             "audit_GPU_hours": 0, "torch_imported": False,
             "ledger_receipts": ledger_receipts, "cells": records,
             "limitations": [
                 "Producer-only and merge-only graphs change residency, dependencies and scheduling; their sum or difference is not removable fusion overhead.",
                 "Artificial cold merge reads flushed partials rather than freshly produced partials; cold values are quantized and A/A variation is retained.",
                 "No stage-only output is a complete attention result. Four recorded full-output checks per input do not replace the dedicated all-shape/mutation qualifications.",
                 "Wave20 is used only to verify frozen configuration and provenance. No cross-wave producer-only-versus-native speedup is computed.",
                 "Source equality and logical partial bytes are not dynamic DRAM/L2 traffic measurements.",
                 "Eight GPUs execute eight different cells from three documents; no per-cell independent device repetition or application inference follows."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(flat[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    (OUT / "stage_medians.csv").write_text(output.getvalue())
    print(json.dumps({key: audit[key] for key in ("status", "complete_jobs", "measurement_rows",
                                                   "exact_medians_recomputed", "source_P2_gpu_hours",
                                                   "audit_GPU_hours", "torch_imported")}))


if __name__ == "__main__":
    main()
