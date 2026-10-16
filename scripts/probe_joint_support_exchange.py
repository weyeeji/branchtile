#!/usr/bin/env python3
"""CPU-only deterministic pooled-score exchange screen, not a quality test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/joint_support_exchange_v1/protocol.json"


def sha(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def score_sum(score, mask):
    return math.fsum(score[mask].tolist())


def protected_mask(n, first=4, last=32):
    mask = np.zeros(n, dtype=bool)
    mask[:first] = True
    mask[max(0, n - last):] = True
    return mask


def stable_order(ids, scores, descending=False):
    return ids[np.lexsort((ids, -scores[ids] if descending else scores[ids]))]


def exchange(scores, baseline, protected, epsilon, order, rounds=2):
    """Sequential requests, prefix-batched swaps; owners refreshed each turn."""
    masks = baseline.copy()
    b, n = masks.shape
    original = np.array([score_sum(scores[q], baseline[q]) for q in range(b)])
    losses = np.zeros(b, dtype=np.float64)
    events = []
    for round_idx in range(rounds):
        round_swaps = 0
        for q in order:
            owners = masks.sum(axis=0)
            remove = np.flatnonzero(masks[q] & (owners == 1) & ~protected)
            add = np.flatnonzero(~masks[q] & (owners > 0) & ~protected)
            remove = stable_order(remove, scores[q])
            add = stable_order(add, scores[q], descending=True)
            removable_count, replacement_count = len(remove), len(add)
            count = min(len(remove), len(add))
            remove, add = remove[:count], add[:count]
            delta = scores[q, remove] - scores[q, add]
            # Removable entries remain original top-k members: shared entries
            # are never removed, so accepting a negative delta signals a bug.
            assert np.all(delta >= 0), "Original support is not protected top-k"
            cumulative = np.cumsum(delta, dtype=np.float64)
            remaining = max(0.0, epsilon * original[q] - losses[q])
            take = int(np.searchsorted(cumulative, remaining, side="right"))
            if epsilon == 0:
                # Exact float32 score equality, with no numerical slack.
                assert np.all(delta[:take] == 0)
            # Final acceptance uses fsum on the actual resulting selected set,
            # not just a potentially rounded cumulative-sum estimate.
            while take:
                candidate = masks[q].copy()
                candidate[remove[:take]] = False
                candidate[add[:take]] = True
                actual_loss = original[q] - score_sum(scores[q], candidate)
                if actual_loss <= epsilon * original[q] and actual_loss / original[q] <= epsilon:
                    break
                take -= 1
            if take:
                masks[q] = candidate
                losses[q] = actual_loss
            assert int(masks[q].sum()) == int(baseline[q].sum())
            assert np.array_equal(masks[q, protected], baseline[q, protected])
            if epsilon == 0:
                assert score_sum(scores[q], masks[q]) == original[q]
            events.append({"round": round_idx + 1, "question": int(q),
                           "removable_candidates": removable_count,
                           "replacement_candidates": replacement_count,
                           "swaps": take, "cumulative_score_loss": float(losses[q])})
            round_swaps += take
        if round_swaps == 0:
            break
    for q in range(b):
        loss = original[q] - score_sum(scores[q], masks[q])
        assert loss <= epsilon * original[q]
        if epsilon == 0:
            assert loss == 0
    return masks, events


def common_control(scores, protected, k, mode):
    aggregate = scores.max(axis=0) if mode == "max" else scores.mean(axis=0)
    ids = stable_order(np.flatnonzero(~protected), aggregate, descending=True)
    one = protected.copy()
    one[ids[:k - int(protected.sum())]] = True
    return np.repeat(one[None, :], len(scores), axis=0)


def geometry(masks, n64=64):
    b = len(masks)
    owner = np.zeros(masks.shape[1], dtype=np.uint16)
    for q in range(b):
        owner |= masks[q].astype(np.uint16) << q
    counts = np.bincount(owner, minlength=1 << b)
    owner_counts = {str(o): int(counts[o]) for o in range(1, 1 << b)}
    union = int(np.count_nonzero(owner))
    histogram = {str(c): int(sum(counts[o] for o in range(1, 1 << b)
                               if o.bit_count() == c)) for c in range(1, b + 1)}
    owner_tiles = sum((int(counts[o]) + n64 - 1) // n64 for o in range(1, 1 << b))
    return {"union_tokens": union, "selected_entries": int(masks.sum()),
            "owner_bitmask_counts": owner_counts, "owner_cardinality_histogram": histogram,
            "active_owner_states": int(sum(c > 0 for c in counts[1:])),
            "fixed_union_N64_tiles": (union + n64 - 1) // n64,
            "owner_partition_N64_tiles": owner_tiles,
            "owner_partition_padded_tokens": owner_tiles * n64,
            "owner_query_N64_slots": sum(o.bit_count() * ((int(counts[o]) + n64 - 1) // n64)
                                         for o in range(1, 1 << b)),
            "independent_N64_tiles": sum((int(row.sum()) + n64 - 1) // n64 for row in masks)}


def describe(scores, masks, baseline, common_n):
    qrows = []
    for q in range(len(scores)):
        initial = score_sum(scores[q], baseline[q])
        retained = score_sum(scores[q], masks[q])
        ids = np.flatnonzero(masks[q]).astype("<i8")
        qrows.append({"question": q, "selected_tokens": len(ids),
                      "original_selected_score": initial, "retained_score": retained,
                      "retained_score_ratio": retained / initial,
                      "score_loss_fraction": (initial - retained) / initial,
                      "changed_positions": int(np.count_nonzero(masks[q] != baseline[q])),
                      "selected_indices_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
    common = geometry(masks[:, :common_n])
    initial_common = geometry(baseline[:, :common_n])
    common["union_reduction_fraction"] = 1 - common["union_tokens"] / initial_common["union_tokens"]
    common["owner_tile_reduction_fraction"] = 1 - common["owner_partition_N64_tiles"] / initial_common["owner_partition_N64_tiles"]
    common["fixed_union_tile_reduction_fraction"] = 1 - common["fixed_union_N64_tiles"] / initial_common["fixed_union_N64_tiles"]
    return {"per_request": qrows, "actual_common_source_logical": common,
            "full_document_logical_only": geometry(masks),
            "private_protected_document_tail_entries": int(masks[:, common_n:].sum()),
            "physical_capacity_released_bytes": 0}


def self_test():
    # Exact ties can change supports at epsilon=0 without sacrificing score.
    scores = np.array([[9, 8, 7, 7], [9, 8, 7, 7]], dtype=np.float64)
    masks = np.array([[1, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)
    protected = np.array([1, 1, 0, 0], dtype=bool)
    out, events = exchange(scores, masks, protected, 0, [0, 1])
    assert geometry(out)["union_tokens"] == 3 and sum(x["swaps"] for x in events) == 1
    # A strict gap disallows epsilon=0; positive loss is explicitly charged.
    scores = np.array([[9, 8, 7, 6], [9, 8, 6, 7]], dtype=np.float64)
    out, _ = exchange(scores, masks, protected, 0, [0, 1])
    assert np.array_equal(out, masks)
    out, _ = exchange(scores, masks, protected, 1 / 24, [0, 1])
    assert geometry(out)["union_tokens"] == 3
    assert score_sum(scores[0], masks[0]) - score_sum(scores[0], out[0]) == 1
    assert geometry(masks)["owner_bitmask_counts"] == {"1": 1, "2": 1, "3": 2}
    assert np.array_equal(common_control(scores, protected, 3, "max")[0], masks[0])
    return {"status": "passed", "checks": ["exact_tie_exchange", "zero_loss_strict_gap",
              "positive_loss_boundary", "owner_counts", "protected_common_control"]}


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    checks = self_test()
    if args.self_test:
        print(json.dumps(checks))
        return
    protocol = json.loads(PROTOCOL.read_text())
    output = PROTOCOL.parent
    assert not (output / "results.jsonl").exists(), "Refuse to overwrite frozen results"
    started = time.time()
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    all_rows, input_provenance = [], []
    for source in protocol["sources"]:
        directory = ROOT / source["directory"]
        metadata_path = directory / "metadata.json"
        assert sha(metadata_path) == source["metadata_sha256"]
        metadata = json.loads(metadata_path.read_text())
        assert metadata["export_semantics_version"] == 2
        assert metadata["actual_tp_rank"] == 0 and metadata["kv_heads"] == 2
        b = metadata["B"]
        start, end = metadata["document_token_range"]
        common_start, common_end = metadata["actual_shared_document_token_range"]
        assert common_start == start
        n, common_n = end - start, common_end - start
        assert 0 <= n - common_n <= protocol["protected_last"]
        protected = protected_mask(n, protocol["protected_first"], protocol["protected_last"])
        loaded = []
        load_start = time.perf_counter()
        for q, item in enumerate(source["selector_files"]):
            path = directory / item["name"]
            assert path.name == f"selector_q{q}_l15.pt"
            assert sha(path) == item["sha256"]
            data = torch.load(path, map_location="cpu", weights_only=False)
            assert data["observation_is_question_suffix_only"] is True
            positions = data["observation_positions"].numpy()
            assert len(positions) <= 32 and np.all(positions >= end)
            # Never inspect observation_q values; no source/ready/decode Q file
            # is opened. Only pooled scores, support and input provenance used.
            loaded.append({"scores": data["pooled_scores"].numpy().astype(np.float64),
                           "supports": {k: data["supports"].get(k, data["supports"].get(str(k))).numpy().copy()
                                        for k in protocol["budgets"]}})
            input_provenance.append({"path": str(path.relative_to(ROOT)), "sha256": item["sha256"],
                                     "document_sha256": metadata["document_sha256"],
                                     "source_request_id": data["source_request_id"],
                                     "suffix_only": True, "observation_count": len(positions),
                                     "observation_position_min": int(positions.min()),
                                     "observation_position_max": int(positions.max())})
            del data
        assert len(loaded) == b
        load_seconds = time.perf_counter() - load_start
        scores_all = np.stack([x["scores"] for x in loaded])
        assert scores_all.shape == (b, 2, n)
        assert np.isfinite(scores_all).all() and (scores_all >= 0).all()
        for budget in protocol["budgets"]:
            all_masks = np.stack([x["supports"][budget] for x in loaded])
            for head in range(2):
                scores = scores_all[:, head, :]
                baseline = all_masks[:, head, :]
                for q in range(b):
                    assert baseline[q].sum() == budget and baseline[q, protected].all()
                    assert scores[q, baseline[q] & ~protected].min() >= scores[q, ~baseline[q]].max()
                    assert score_sum(scores[q], baseline[q]) > 0
                variants = [("independent", None, None), ("common_max", None, None),
                            ("common_mean", None, None)]
                variants += [("exchange", eps, order) for eps in protocol["epsilon"]
                             for order in protocol["request_orders"]]
                for method, epsilon, order_name in variants:
                    tw, tc = time.perf_counter(), time.process_time()
                    events = []
                    if method == "independent":
                        masks = baseline.copy()
                    elif method.startswith("common_"):
                        masks = common_control(scores, protected, budget, method.removeprefix("common_"))
                    else:
                        order = list(range(b)) if order_name == "ascending" else list(reversed(range(b)))
                        masks, events = exchange(scores, baseline, protected, epsilon, order, protocol["max_rounds"])
                    selection_wall, selection_cpu = time.perf_counter() - tw, time.process_time() - tc
                    assert np.all(masks.sum(axis=1) == budget) and masks[:, protected].all()
                    row = {"document": source["id"], "document_sha256": metadata["document_sha256"],
                           "rank": 0, "layer": 15, "kv_head": head, "B": b,
                           "document_tokens": n, "common_source_tokens": common_n,
                           "budget": budget, "method": method, "epsilon": epsilon,
                           "request_order": order_name, "events": events,
                           **describe(scores, masks, baseline, common_n),
                           "selection_wall_seconds": selection_wall, "selection_cpu_seconds": selection_cpu,
                           "selection_and_metrics_wall_seconds": time.perf_counter() - tw,
                           "selection_and_metrics_cpu_seconds": time.process_time() - tc,
                           "source_load_seconds_shared_across_variants": load_seconds,
                           "primary_numpy_state_bytes": int(scores_all.nbytes + all_masks.nbytes + masks.nbytes),
                           "process_peak_rss_kib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    if method == "exchange":
                        assert all(x["score_loss_fraction"] <= epsilon for x in row["per_request"])
                        if epsilon == 0:
                            assert all(x["retained_score"] == x["original_selected_score"] for x in row["per_request"])
                    all_rows.append(row)
        print(json.dumps({"completed_document": source["id"], "result_rows": len(all_rows)}), flush=True)
    assert len(all_rows) == protocol["expected_result_rows"] == 132
    assert not torch.cuda.is_initialized()
    result_path = output / "results.jsonl"
    result_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in all_rows))
    aggregates = []
    for method, epsilon, order in [(r["method"], r["epsilon"], r["request_order"]) for r in all_rows[:11]]:
        group = [r for r in all_rows if (r["method"], r["epsilon"], r["request_order"]) == (method, epsilon, order)]
        u = [r["actual_common_source_logical"]["union_reduction_fraction"] for r in group]
        tiles = [r["actual_common_source_logical"]["owner_tile_reduction_fraction"] for r in group]
        aggregates.append({"method": method, "epsilon": epsilon, "request_order": order,
                           "head_cells": len(group), "union_reduction_min_median_max": [min(u), float(np.median(u)), max(u)],
                           "owner_tile_reduction_min_median_max": [min(tiles), float(np.median(tiles)), max(tiles)],
                           "max_request_score_loss_fraction": max(x["score_loss_fraction"] for r in group for x in r["per_request"]),
                           "cells_union_reduction_ge_15pct": sum(x >= .15 for x in u),
                           "total_selection_cpu_seconds": sum(r["selection_cpu_seconds"] for r in group)})
    summary = {"protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "results_sha256": sha(result_path), "result_rows": len(all_rows),
               "expected_head_cells_per_method": 12, "started_unix_seconds": started,
               "completed_unix_seconds": time.time(), "cpu_seconds": time.process_time() - cpu_started,
               "wall_seconds": time.perf_counter() - wall_started,
               "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "uid": os.getuid(), "venv": sys.prefix, "numpy_version": np.__version__,
               "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
               "cuda_initialized": torch.cuda.is_initialized(), "gpu_hours": 0,
               "self_test": checks, "input_provenance": input_provenance, "aggregates": aggregates,
               "scope": protocol["scope"], "validation": {"budgets_and_protected_positions_unchanged": True,
               "all_independent_masks_are_protected_topk": True, "zero_epsilon_exact_score_assertions_pass": True,
               "all_exchange_losses_within_frozen_per_request_per_head_bound": True,
               "no_decode_query_file_read": True}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "rows": len(all_rows), "gpu_hours": 0,
                      "cpu_seconds": summary["cpu_seconds"], "wall_seconds": summary["wall_seconds"]}))


if __name__ == "__main__":
    main()
