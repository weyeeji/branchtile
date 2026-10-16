"""Independent CPU-only audit of all wave2 v2/emptyguard cells.

Reads terminal job records, including failures, without changing any benchmark
source or result. Geometry is reconstructed on CPU with CUDA unavailable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def geometry(capture, layer, budget, step, heads):
    import torch
    path = ROOT / capture
    metadata = json.loads((path / "metadata.json").read_text())
    src = torch.load(path / f"source_l{layer}.pt", weights_only=True, mmap=True, map_location="cpu")
    mask_path = path / f"masks_l{layer}_b{budget}.pt"
    masks = torch.load(mask_path, weights_only=True, map_location="cpu")["supports"][:, :heads]
    active = [i for i in range(metadata["B"]) if (path / f"decode_q{i}_l{layer}_s{step}.pt").exists()]
    source_heads = min(heads, src["document_k"].shape[1])
    assert source_heads == heads
    prefix_length = src["prefix_k"].shape[2]
    parts = [torch.full((heads, prefix_length), (1 << len(active)) - 1, dtype=torch.int64),
             sum(masks[qi].long() * (1 << ri) for ri, qi in enumerate(active))]
    suffix_lengths = []
    for ri, qi in enumerate(active):
        dec = torch.load(path / f"decode_q{qi}_l{layer}_s{step}.pt", weights_only=True,
                         mmap=True, map_location="cpu")
        length = dec["private_suffix_k"].shape[2]
        suffix_lengths.append(length)
        parts.append(torch.full((heads, length), 1 << ri, dtype=torch.int64))
    owners = torch.cat(parts, dim=1)
    labels = sorted(set(owners.flatten().tolist()) - {0}) or [1]
    counts = [[int((owners[h] == label).sum()) for label in labels] for h in range(heads)]
    splits_rows = []
    for splits in (32, 64, 128):
        # Counting begin/end directly independently verifies S-min(S,n).
        direct = sum((length * s // splits == length * (s + 1) // splits)
                     for row in counts for length in row for s in range(splits))
        assert direct == sum(max(0, splits - length) for row in counts for length in row)
        total = heads * len(labels) * splits
        splits_rows.append({"splits": splits, "shards": total,
                            "empty_shards": direct, "empty_fraction": direct / total})
    assert not torch.cuda.is_initialized()
    return {"active_requests": active, "prefix_tokens": prefix_length,
            "private_suffix_lengths": suffix_lengths, "labels": labels,
            "region_lengths_by_head": counts, "smallest_region_tokens": min(map(min, counts)),
            "split_geometry": splits_rows,
            "mask_sha256": sha(mask_path), "metadata_sha256": sha(path / "metadata.json"),
            "overlay_receipt_sha256": sha(path / "overlay_receipt.json") if (path / "overlay_receipt.json").exists() else None,
            "CUDA_initialized": False}


def paired(measurements, cache):
    import numpy as np
    rows = [r for r in measurements if r["cache"] == cache]
    blocks = sorted({r["block"] for r in rows})
    lookup = {(r["block"], r["direction"]): r for r in rows}
    assert len(lookup) == len(rows) == 2 * len(blocks)
    values = {method: np.array([[lookup[block, direction]["us"][method]
                                 for direction in ("random", "reverse")] for block in blocks])
              for method in ("dynamic_v2", "dynamic_emptyguard")}
    ratio = values["dynamic_v2"] / values["dynamic_emptyguard"]
    delta = values["dynamic_emptyguard"] - values["dynamic_v2"]
    draws = np.random.default_rng(71).integers(0, len(blocks), (10000, len(blocks)))
    ratio_draws = np.median(ratio[draws], axis=(1, 2))
    return {"cache": cache, "paired_blocks": len(blocks),
            "baseline_v2_median_us": float(np.median(values["dynamic_v2"])),
            "emptyguard_median_us": float(np.median(values["dynamic_emptyguard"])),
            "baseline_over_emptyguard_paired_ratio": float(np.median(ratio)),
            "ratio_bootstrap95": np.quantile(ratio_draws, [.025, .975]).tolist(),
            "ratio_bootstrap_degenerate": bool(np.ptp(ratio_draws) == 0),
            "emptyguard_added_paired_us": float(np.median(delta)),
            "added_us_bootstrap95": np.quantile(np.median(delta[draws], axis=(1, 2)), [.025, .975]).tolist()}


def argument(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="configs/operator_overhead_wave2_jobs.json")
    parser.add_argument("--output", default="results/operator_emptyguard_wave2_audit_v1/audit.json")
    args = parser.parse_args()
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    jobs_path = ROOT / args.jobs
    jobs = [j for j in json.loads(jobs_path.read_text())
            if "scripts/bench_operator_overhead.py" in j["argv"]]
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    all_rows, errors, selected_rows, tuning_pairs, geometry_rows, hashes = [], [], [], [], [], {}
    for job in jobs:
        argv = job["argv"]
        terminal = [r for r in ledger if r["event"] == "end" and r["name"] == job["name"]]
        assert len(terminal) == 1, "Wait for every job terminal before emitting an audit"
        result_path = ROOT / argument(argv, "--output")
        progress_path = result_path.with_suffix(".tuning.jsonl")
        data = json.loads(result_path.read_text()) if result_path.exists() else None
        tuning = data["tuning"] if data else ([json.loads(line) for line in progress_path.read_text().splitlines()]
                                               if progress_path.exists() else [])
        row = {"name": job["name"], "gpus": job["gpus"], "terminal": terminal[0],
               "final_result_exists": data is not None,
               "tuning_completed": len(tuning), "all_configuration_failures": [r for r in tuning if r["status"] != "ok"]}
        if data:
            row.update(tuning_planned=data["tuning_planned"], tuning_complete=data["tuning_complete"],
                       missing_qualified_methods=data["missing_qualified_methods"], numerical_contract=data["numerical_contract"])
            hashes[str(result_path.relative_to(ROOT))] = sha(result_path)
        row["process_passed"] = terminal[0]["exit_code"] == 0 and not terminal[0]["timed_out"]
        if progress_path.exists():
            hashes[str(progress_path.relative_to(ROOT))] = sha(progress_path)
        all_rows.append(row)
        errors.extend({"job": job["name"], **failure} for failure in row["all_configuration_failures"])
        geo = geometry(argument(argv, "--capture"), int(argument(argv, "--layer")),
                       int(argument(argv, "--budget")), int(argument(argv, "--step")),
                       int(argument(argv, "--heads")))
        geometry_rows.append({"job": job["name"], **geo})
        indexed = {(r["method"], r.get("splits"), r.get("block_n")): r for r in tuning
                   if r["method"] in ("dynamic_v2", "dynamic_emptyguard")}
        for splits in (32, 64, 128):
            for block_n in (32, 64):
                baseline = indexed.get(("dynamic_v2", splits, block_n))
                guarded = indexed.get(("dynamic_emptyguard", splits, block_n))
                pair = {"job": job["name"], "splits": splits, "block_n": block_n,
                        "baseline": baseline, "emptyguard": guarded,
                        "both_present": baseline is not None and guarded is not None}
                pair["both_FP64_qualified"] = pair["both_present"] and baseline["status"] == guarded["status"] == "ok"
                if pair["both_FP64_qualified"]:
                    pair["descriptive_tuning_ratios"] = {
                        cache: baseline["tuning_us_by_cache"][cache] / guarded["tuning_us_by_cache"][cache]
                        for cache in ("hot", "cold")}
                tuning_pairs.append(pair)
        if data and row["process_passed"] and data["measurements"]:
            for cache in ("hot", "cold"):
                result = paired(data["measurements"], cache)
                baseline_config = data["selected"][cache]["dynamic_v2"]["config"]
                guard_config = data["selected"][cache]["dynamic_emptyguard"]["config"]
                result.update(job=job["name"], baseline_config=baseline_config,
                              emptyguard_config=guard_config,
                              same_shape_parameters=all(baseline_config[k] == guard_config[k]
                                                        for k in ("splits", "block_n")),
                              baseline_selected_split_geometry=next(g for g in geo["split_geometry"]
                                                                    if g["splits"] == baseline_config["splits"]),
                              guard_selected_split_geometry=next(g for g in geo["split_geometry"]
                                                                 if g["splits"] == guard_config["splits"]))
                selected_rows.append(result)
    import torch
    result = {"schema": 1, "status": "terminal_CPU_audit", "auditor_sha256": sha(__file__),
              "jobs_sha256": sha(jobs_path), "input_result_sha256": hashes,
              "jobs": all_rows, "all_configuration_failures": errors,
              "geometry": geometry_rows, "all_matching_configuration_pairs": tuning_pairs,
              "all_selected_paired_cache_cells": selected_rows,
              "allocation_GPU_hours_charged_by_source_jobs": sum(r["terminal"]["gpu_hours"] for r in all_rows),
              "audit_GPU_hours": 0, "CUDA_initialized": torch.cuda.is_initialized(),
              "limitations": [
                  "Six small-budget layer states from three development documents plus one full-document state; repeated layers are not independent documents.",
                  "This audit compares separately tuned winners; same_shape_parameters flags differing selections, which are not pure single-guard comparisons.",
                  "Per-configuration tuning times are descriptive, not randomized paired final measurements.",
                  "Measured run() includes producer and merge GPU graph execution, not one-time CPU plan preparation or host synchronized complete-call wall.",
                  "Artificial cold flush is not actual whole-model interleaving. No application, quality or energy conclusion.",
                  "Cold event readings are visibly quantized; degenerate bootstrap intervals do not imply arbitrarily precise measurement.",
                  "Geometry is evaluated per state and split; any empty-shard benefit must be distinguished from branch overhead where no shards are empty."
              ]}
    assert result["CUDA_initialized"] is False
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"jobs": len(all_rows), "processes_passed": sum(r["process_passed"] for r in all_rows),
                      "matching_pairs": len(tuning_pairs), "FP64_qualified_pairs": sum(r["both_FP64_qualified"] for r in tuning_pairs),
                      "configuration_failures": len(errors), "paired_cache_cells": len(selected_rows),
                      "empty_shards": sum(g["empty_shards"] for r in geometry_rows for g in r["split_geometry"]),
                      "paired_ratio_range": ([min(r["baseline_over_emptyguard_paired_ratio"] for r in selected_rows),
                                              max(r["baseline_over_emptyguard_paired_ratio"] for r in selected_rows)]
                                             if selected_rows else None),
                      "output": str(out.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
