#!/usr/bin/env python3
"""Reproduce document-only tile and owner-tail diagnostics from natural captures."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def partitions(b):
    result = [[]]
    for bit in (1 << i for i in range(b)):
        nxt = []
        for p in result:
            nxt.append(p + [bit])
            for j in range(len(p)):
                q = p.copy()
                q[j] |= bit
                nxt.append(q)
        result = nxt
    return result


def analyze(lengths, b, gqa, n):
    canonical = sum(math.ceil(lengths[o] / n) * math.ceil(o.bit_count() * gqa / 16)
                    for o in range(1, 1 << b))
    relaxed = sum(lengths[o] / n * math.ceil(o.bit_count() * gqa / 16)
                  for o in range(1, 1 << b))
    plans, relaxed_plans = [], []
    for p in partitions(b):
        lens = [sum(lengths[o] for o in range(1, 1 << b) if o & group) for group in p]
        cost = sum(math.ceil(l / n) * math.ceil(group.bit_count() * gqa / 16) for l, group in zip(lens, p))
        ideal = sum(l / n * math.ceil(group.bit_count() * gqa / 16) for l, group in zip(lens, p))
        plans.append((cost, sum(lens), p))
        relaxed_plans.append(ideal)
    best = min(plans)
    union = sum(lengths[1:])
    return {"candidate_tiles": canonical, "best_fixed_tiles": best[0],
            "full_union_tiles": math.ceil(union / n) * math.ceil(b * gqa / 16),
            "candidate_relaxed_tiles": relaxed, "fixed_relaxed_lower_bound": min(relaxed_plans),
            "candidate_logical_kv_token_reads": union, "best_fixed_logical_kv_token_reads": best[1],
            "owner1_or_2_tokens": sum(lengths[o] for o in range(1, 1 << b) if o.bit_count() <= 2),
            "all_owner_tokens": lengths[-1]}


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", default="results/captures/old30b_diagnostic")
    p.add_argument("--manifest", default="data/manifests/qmsum_train_exploratory8.json")
    p.add_argument("--output", default="results/data_natural_tail_opportunity.json")
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    expected_ids = {g["document_id"] for g in manifest["groups"]}
    found = {}
    for path in Path(a.capture_root).glob("*/metadata.json"):
        meta = json.loads(path.read_text())
        if meta.get("status") != "completed" or meta["document_id"] not in expected_ids:
            continue
        # Prefer a complete longer instrumentation window, never a favorable result.
        old = found.get(meta["document_id"])
        priority = (meta["generation"]["max_new_tokens"], meta["ended_epoch_seconds"])
        if old is None or priority > old[0]:
            found[meta["document_id"]] = (priority, path, meta)
    rows, captures = [], []
    for _, path, meta in sorted(found.values(), key=lambda x: x[2]["group_index"]):
        opportunity_path = path.parent / "opportunity.json"
        op = json.loads(opportunity_path.read_text())
        captures.append({"group_index": meta["group_index"], "document_id": meta["document_id"],
                         "metadata": str(path), "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "opportunity_sha256": hashlib.sha256(opportunity_path.read_bytes()).hexdigest(),
                         "generation_cap": meta["generation"]["max_new_tokens"],
                         "natural_eos_count": sum(q["natural_eos"] for q in meta["questions"]),
                         "captured_decode_queries": len(meta["decode_snapshots"]),
                         "gqa": meta["gqa"], "B": meta["B"]})
        for budget in sorted({r["budget"] for r in op}):
            for n in (32, 64):
                items = [analyze(r["owner_token_counts"], meta["B"], meta["gqa"], n)
                         for r in op if r["budget"] == budget]
                totals = {k: sum(r[k] for r in items) for k in items[0]}
                row = {"group": path.parent.name, "group_index": meta["group_index"],
                       "budget": budget, "N_tile": n, **totals}
                row.update({"candidate_vs_best_fixed": totals["candidate_tiles"] / totals["best_fixed_tiles"],
                            "candidate_vs_full_union": totals["candidate_tiles"] / totals["full_union_tiles"],
                            "ideal_without_region_tail_padding_candidate_vs_best_fixed": totals["candidate_relaxed_tiles"] / totals["fixed_relaxed_lower_bound"],
                            "candidate_tail_tiles": totals["candidate_tiles"] - totals["candidate_relaxed_tiles"],
                            "fixed_rounding_gap_to_relaxed_lower_bound": totals["best_fixed_tiles"] - totals["fixed_relaxed_lower_bound"],
                            "union_vs_fixed_token_reads": totals["candidate_logical_kv_token_reads"] / totals["best_fixed_logical_kv_token_reads"],
                            "owner1_or_2_union_fraction": totals["owner1_or_2_tokens"] / totals["candidate_logical_kv_token_reads"],
                            "all_owner_union_fraction": totals["all_owner_tokens"] / totals["candidate_logical_kv_token_reads"]})
                rows.append(row)
    output = {"scope": "natural model supports; logical document-only tile costs, excluding common system prefix, private suffix, actual cache, resources, preparation, and scheduling",
              "capture_root": a.capture_root, "manifest": a.manifest,
              "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
              "groups_complete_at_analysis": len(found), "full_group_denominator": len(expected_ids),
              "missing_document_ids": sorted(expected_ids - found.keys()),
              "fixed_plan_rule": "All set partitions; minimize M16/N tile count, then logical KV reads; this is not the runtime-optimal plan",
              "relaxed_rule": "Replace ceil(owner_length/N) and ceil(group_union_length/N) by exact fractions; preserve ceil(query_rows/16)",
              "captures": captures, "rows": rows}
    Path(a.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": a.output, "complete_groups": len(found), "denominator": len(expected_ids)}))


if __name__ == "__main__":
    main()
