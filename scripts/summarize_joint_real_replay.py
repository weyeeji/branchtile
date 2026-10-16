"""Summarize paired frozen real-cache chains without calling them model JCT."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def paired_statistics(values, baseline, draws):
    import numpy as np
    ratios = baseline / values
    differences = baseline - values
    return {
        "baseline_median_us": float(np.median(baseline)),
        "method_median_us": float(np.median(values)),
        "paired_speed_ratio": float(np.median(ratios)),
        "paired_speed_ratio_bootstrap95": np.quantile(np.median(ratios[draws], axis=(1, 2)), [.025, .975]).tolist(),
        "paired_saving_us": float(np.median(differences)),
        "paired_saving_us_bootstrap95": np.quantile(np.median(differences[draws], axis=(1, 2)), [.025, .975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    import numpy as np

    manifest = ROOT / args.jobs
    jobs = json.loads(manifest.read_text())
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows, receipts, hashes = [], [], {}
    for job in jobs:
        argv = job["argv"]
        folder = ROOT / argv[argv.index("--output") + 1]
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "completed", (job["name"], summary["status"])
        rank_tables = []
        snapshots = []
        qualifications = []
        for rank in (0, 1):
            path = folder / f"rank{rank}" / "replay.json"
            snapshot_path = folder / f"rank{rank}" / "replay_snapshot.json"
            data = json.loads(path.read_text())
            snapshot = json.loads(snapshot_path.read_text())
            assert data["status"] == "completed" and data["rank"] == rank
            assert snapshot["actual_B3_outputs_bitwise_reproduced"]
            assert snapshot["actual_last_step_indices_bitwise_reproduced"]
            assert data["final_cache_layouts_unchanged"]
            assert all(q["frozen_tolerance_passed"] for q in data["qualification"])
            assert all(q.get("epsilon0_independent_bitwise_equal", True) for q in data["qualification"])
            assert len(data["qualification"]) == 4 * 3 * 48
            qualifications.extend(data["qualification"])
            protocol = data["protocol"]
            blocks = protocol["blocks"]
            table = {}
            for r in data["measurements"]:
                key = (r["block"], r["direction"], r["variant"], r["batch"], r["cache"], r["contract"])
                assert key not in table
                assert r["samples"]
                table[key] = {clock: statistics.median(x[clock] for x in r["samples"])
                              for clock in ("cuda_us", "complete_wall_us")}
                assert all(v > 0 for v in table[key].values())
            assert len(table) == blocks * 2 * 4 * 3 * 2 * 2
            rank_tables.append(table)
            snapshots.append(snapshot)
            for p in (summary_path, path, snapshot_path):
                hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        assert snapshots[0]["positions"] == snapshots[1]["positions"]
        assert snapshots[0]["questions"] == snapshots[1]["questions"]
        draws = np.random.default_rng(20261016).integers(0, blocks, (10000, blocks))
        for rank_scope in ("rank0", "rank1", "rank_max_proxy"):
            for batch in (1, 2, 3):
                for cache in ("hot", "cold"):
                    for contract in ("48_layers_only", "prepare_plus_48_layers"):
                        for clock in ("cuda_us", "complete_wall_us"):
                            arrays = {}
                            for variant in ("independent", "epsilon0", "epsilon001", "epsilon01"):
                                array = []
                                for block in range(blocks):
                                    pair = []
                                    for direction in ("random", "reverse"):
                                        key = (block, direction, variant, batch, cache, contract)
                                        per_rank = [table[key][clock] for table in rank_tables]
                                        pair.append(max(per_rank) if rank_scope == "rank_max_proxy"
                                                    else per_rank[int(rank_scope[-1])])
                                    array.append(pair)
                                arrays[variant] = np.asarray(array)
                            for variant in ("epsilon0", "epsilon001", "epsilon01"):
                                rows.append({"job": job["name"], "document_id": summary["document_id"],
                                             "device_replica": summary["device_replica"], "rank_scope": rank_scope,
                                             "batch": batch, "cache": cache, "contract": contract, "clock": clock,
                                             "variant": variant, "paired_blocks": blocks,
                                             **paired_statistics(arrays[variant], arrays["independent"], draws)})
        preparations = []
        for rank in (0, 1):
            frozen = summary["frozen_variants"][rank]
            assert frozen["rank"] == rank
            variants = {}
            for name, item in frozen["variants"].items():
                if item.get("enabled"):
                    # Preserve the source report: callers must not subtract epsilon0
                    # verification cost from a joint method's deployment cost.
                    variants[name] = item
            preparations.append({"rank": rank,
                                 "score_D2H_bytes": frozen["shared_score_D2H_bytes"],
                                 "shared_score_D2H_and_copy_wall_seconds": frozen["shared_score_transfer_and_copy_wall_seconds"],
                                 "variants": variants})
        receipts.append({"job": job["name"], "document_id": summary["document_id"],
                         "device_replica": summary["device_replica"], "gpus": job["gpus"],
                         "qualification_checks": len(qualifications),
                         "qualification_max_abs": max(q["max_abs"] for q in qualifications),
                         "positions": snapshots[0]["positions"], "questions": snapshots[0]["questions"],
                         "private_generated_tail_lengths": snapshots[0]["generated_tail_lengths"],
                         "actual_B3_outputs_bitwise_reproduced_on_both_ranks": True,
                         "preparation": preparations,
                         "initialization_wall_seconds": summary["initialization_wall_seconds"],
                         "selector_dense_warmup_wall_seconds": summary["selector_dense_warmup_wall_seconds"],
                         "joint_finalize_RPC_wall_seconds": summary["joint_finalize_RPC_wall_seconds"]})
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    names = {j["name"] for j in jobs}
    events = [e for e in ledger if e["name"] in names]
    ends = [e for e in events if e["event"] == "end"]
    assert len(ends) == len(jobs) and all(e["exit_code"] == 0 for e in ends)
    result = {"status": "completed", "source_hashes": hashes, "job_receipts": receipts,
              "cells": rows, "qualification_checks": sum(r["qualification_checks"] for r in receipts),
              "allocation_GPU_hours": sum(e["gpu_hours"] for e in ends), "ledger_events": events,
              "statistical_scope": "Median over per-direction sample medians; 10000 bootstrap draws resample blocks jointly with their random/reverse measurements. Documents, device replicas, batches, caches, contracts and clocks stay separate.",
              "limitations": [
                  "All 48 attention layers retain real original strided KV and production output.copy_, but the chain omits intervening MoE/GEMM/TP communication.",
                  "B1/B2 are prefixes of one actual B3 snapshot, not separate natural survivor states or trajectories.",
                  "rank_max_proxy is max of separately measured rank times; it is not measured TP critical-path model latency.",
                  "CUDA-event complete time brackets CPU prepare submissions and may include GPU waiting; complete_wall_us is the host synchronized interval.",
                  "Repeated blocks and two device replicas are not new independent documents; single-snapshot intervals do not cover trajectory uncertainty.",
                  "Selection and installation are separate costs; pooled-score limits do not certify task quality. No physical source KV is released.",
              ]}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "paired.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"jobs": len(receipts), "cells": len(rows), "qualification_checks": result["qualification_checks"],
                      "allocation_GPU_hours": result["allocation_GPU_hours"], "output": str(output)}))


if __name__ == "__main__":
    main()
