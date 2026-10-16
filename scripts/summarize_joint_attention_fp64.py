#!/usr/bin/env python3
"""CPU validation and descriptive summaries of the frozen attention screen."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/joint_attention_fp64_v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    assert v.size and np.isfinite(v).all()
    return {"count": int(v.size), "min": float(v.min()),
            "median": float(np.median(v)), "max": float(v.max()),
            "mean": float(v.mean())}


def identity(row):
    return tuple(row[k] for k in ("document", "budget", "kv_head", "question", "anchor_step"))


def finite_tree(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, list):
        for v in value:
            finite_tree(v)


def summarize(rows, independent, geometry):
    heads = [(r, h) for r in rows for h in r["per_query_head"]]
    dense_change = [h["dense_relative_l2_change_from_independent"] for _, h in heads]
    mass_change = [h["retained_total_dense_softmax_mass"] - independent[identity(r)]["per_query_head"][j]["retained_total_dense_softmax_mass"]
                   for r in rows for j, h in enumerate(r["per_query_head"])]
    group_changes = [r["dense_relative_l2_change_from_independent"] for r in rows]
    geometric_cells = sorted(set((r["document"], r["budget"], r["kv_head"], r["method"], r["epsilon"], r["request_order"]) for r in rows))
    result = {"request_kvhead_anchor_rows": len(rows), "queryhead_records": len(heads),
              "distinct_document_kvhead_mask_cells": len(geometric_cells),
              "group_relative_l2_vs_dense": stats([r["vs_dense"]["relative_l2"] for r in rows]),
              "group_relative_l2_change_from_independent": stats(group_changes),
              "group_max_abs_change_from_independent": stats([r["dense_max_abs_change_from_independent"] for r in rows]),
              "group_change_counts": {"increased": sum(v > 0 for v in group_changes), "unchanged": sum(v == 0 for v in group_changes), "decreased": sum(v < 0 for v in group_changes)},
              "queryhead_relative_l2_vs_dense": stats([h["vs_dense"]["relative_l2"] for _, h in heads]),
              "queryhead_relative_l2_change_from_independent": stats(dense_change),
              "queryhead_relative_l2_vs_independent": stats([h["vs_independent"]["relative_l2"] for _, h in heads]),
              "queryhead_max_abs_vs_dense": stats([h["vs_dense"]["max_abs"] for _, h in heads]),
              "queryhead_cosine_vs_dense": stats([h["vs_dense"]["cosine"] for _, h in heads if h["vs_dense"]["cosine"] is not None]),
              "queryhead_abs_lse_delta_vs_dense": stats([abs(h["lse_delta_vs_dense"]) for _, h in heads]),
              "queryhead_abs_lse_delta_vs_independent": stats([abs(h["lse_delta_vs_independent"]) for _, h in heads]),
              "queryhead_total_retained_mass": stats([h["retained_total_dense_softmax_mass"] for _, h in heads]),
              "queryhead_total_retained_mass_change_from_independent": stats(mass_change),
              "queryhead_document_conditional_retained_mass": stats([h["retained_document_conditional_mass"] for _, h in heads]),
              "queryhead_change_counts": {"increased": sum(v > 0 for v in dense_change), "unchanged": sum(v == 0 for v in dense_change), "decreased": sum(v < 0 for v in dense_change)},
              "logical_union_reduction_fraction": stats([geometry[k]["actual_common_source_logical"]["union_reduction_fraction"] for k in geometric_cells]),
              "logical_owner_N64_tile_reduction_fraction": stats([geometry[k]["actual_common_source_logical"]["owner_tile_reduction_fraction"] for k in geometric_cells])}
    worst_row, worst_head = max(heads, key=lambda rh: rh[1]["dense_relative_l2_change_from_independent"])
    result["largest_queryhead_relative_l2_increase_case"] = {k: worst_row[k] for k in ("document", "budget", "kv_head", "question", "anchor_step", "actual_prediction_step", "epsilon", "request_order")}
    result["largest_queryhead_relative_l2_increase_case"].update(worst_head)
    return result


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    protocol = json.loads((BASE / "protocol.json").read_text())
    files = [BASE / f"joint_attention_fp64_v1_gpu{gpu}.jsonl" for gpu in (2, 3)]
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    finite_tree(rows)
    assert len(rows) == protocol["expected_rows"]
    unique = {identity(r) + (r["method"], r["epsilon"], r["request_order"]) for r in rows}
    assert len(unique) == len(rows)
    independent = {identity(r): r for r in rows if r["method"] == "independent"}
    assert len(independent) == 120
    previous = [json.loads(line) for line in (ROOT / protocol["cpu_results_path"]).read_text().splitlines()]
    geometry = {tuple(r[k] for k in ("document", "budget", "kv_head", "method", "epsilon", "request_order")): r for r in previous}
    lookup = {(s["id"], f["metadata"]["question_index"], f["metadata"]["ready_group_anchor_step"]): f["metadata"] for s in protocol["sources"] for f in s["decode_files"]}
    exact_rows = []
    for r in rows:
        base = independent[identity(r)]
        md = lookup[r["document"], r["question"], r["anchor_step"]]
        assert r["actual_prediction_step"] == md["prediction_step"]
        assert r["query_position"] == md["query_token_position"]
        assert r["original_total_context_tokens"] == r["query_position"] + 1
        assert r["selected_total_context_tokens"] == r["prefix_tokens"] + r["budget"] + r["private_suffix_tokens"] - r["private_document_tail_tokens"]
        assert len(r["per_query_head"]) == protocol["gqa"]
        assert r["dense_output_fp64_sha256"] == base["dense_output_fp64_sha256"]
        if r["method"] == "exchange":
            assert r["pooled_score_loss_fraction"] <= r["epsilon"]
        if r["method"] == "exchange" and r["epsilon"] == 0:
            assert r["epsilon_zero_output_and_lse_exact"] is True
            assert r["output_fp64_sha256"] == base["output_fp64_sha256"]
            assert r["lse_fp64_sha256"] == base["lse_fp64_sha256"]
            assert r["selected_document_indices_sha256"] == base["selected_document_indices_sha256"]
            assert r["vs_independent"]["relative_l2"] == 0
            exact_rows.append(r)
    assert len(exact_rows) == 240
    buckets = []
    for document in [None] + [s["id"] for s in protocol["sources"]]:
        for budget in protocol["budgets"]:
            variants = [("independent", None, None)] + [("exchange", e, o) for e in protocol["epsilon"] for o in protocol["request_orders"]]
            for method, epsilon, order in variants:
                rr = [r for r in rows if (document is None or r["document"] == document) and (r["budget"],r["method"],r["epsilon"],r["request_order"]) == (budget,method,epsilon,order)]
                buckets.append({"document": document or "all_three_documents", "budget": budget, "method": method, "epsilon": epsilon,
                                "request_order": order, **summarize(rr, independent, geometry)})
    names = {f.stem for f in files}
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    events = [r for r in ledger if r.get("name") in names]
    ends = [r for r in events if r["event"] == "end"]
    starts = {(r["name"],r["pid"]) for r in events if r["event"] == "start"}
    assert len(ends) == len(starts) == 2
    assert all((r["name"],r["pid"]) in starts and r["exit_code"] == 0 and not r["timed_out"] for r in ends)
    validation = {"status": "passed", "rows": len(rows), "unique_rows": len(unique),
                  "queryhead_records": sum(len(r["per_query_head"]) for r in rows),
                  "independent_rows": len(independent), "epsilon_zero_bitwise_equal_rows": len(exact_rows),
                  "complete_private_context_and_actual_steps": True, "all_finite": True,
                  "maximum_mass_vs_lse_identity_absolute_residual": max(h["mass_vs_lse_identity_abs_residual"] for r in rows for h in r["per_query_head"]),
                  "result_files_sha256": {f.name: sha(f) for f in files}}
    (BASE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    summary = {"schema": 1, "protocol_sha256": sha(BASE / "protocol.json"), "scope": protocol["scope"],
               "quality_success_threshold": None, "aggregation": "Descriptive equally weighted rows; requests, heads and steps are correlated and are not independent statistical samples. All orders and epsilon retained; no winner selected.",
               "physical_capacity_released_bytes": 0, "validation": validation,
               "accounting": {"category": "P2", "allocated_gpu_hours": sum(r["gpu_hours"] for r in ends), "events": events},
               "aggregates": buckets}
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status":"validated", "rows":len(rows), "gpu_hours":summary["accounting"]["allocated_gpu_hours"]}))


if __name__ == "__main__":
    main()
