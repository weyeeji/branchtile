"""Audit the two preregistered TP2 capture additions and their four replay cells."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    manifest_path = Path("data/manifests/docfinqa_public_development8_v1.json")
    manifest = read(manifest_path)
    summary = {"scope": "Two frozen development documents, actual official FP8 TP2 dense-trajectory instrumentation; cap256 is not task quality",
               "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256((ROOT / manifest_path).read_bytes()).hexdigest(),
               "prespecified_replay": {"groups": [1, 6], "rank": 0, "layer": 15, "step": 64, "budgets": [4096, 16384]},
               "captures": [], "replays": [], "limitations": [
                   "Cold singleton and APC-warm natural group are different states, not an A/A performance comparison.",
                   "Hooked capture timings include instrumentation and are not application latency estimates.",
                   "The full input reports are retained; dense trajectories do not establish sparse generation quality.",
                   "Replay reports a hindsight per-capture planning oracle, including its actual search costs.",
                   "Exact shared payload uses full APC pages; partial-page document tails stay request-private."]}
    for group_index in (1, 6):
        group = manifest["groups"][group_index]
        platform_path = Path(f"results/platform/capture_main30b_tp2_publicdev{group_index}_256.json")
        platform = read(platform_path)
        row = {"group_index": group_index, "document_id": group["document_id"], "B": group["B"],
               "platform_result": str(platform_path), "status": platform["status"], "ranks": []}
        summary["captures"].append(row)
        if platform["status"] != "completed":
            row["exception"] = platform.get("exception")
            continue
        assert platform["input"]["manifest_sha256"] == summary["manifest_sha256"]
        assert platform["input"]["document_sha256"] == group["document_sha256"]
        assert not platform["input"]["truncated_document"]
        assert platform["arguments"]["max_new_tokens"] == 256
        row.update(prompt_lengths=platform["input"]["prompt_lengths"], document_token_range=platform["input"]["document_token_range"],
                   stage_states=[{"name": s["name"], "instrumented_seconds": s["seconds"],
                                  "generated_tokens": [r["generated_tokens"] for r in s["requests"]],
                                  "hit_caps": [r["hit_token_cap"] for r in s["requests"]]} for s in platform["stages"]],
                   initialized_workers=platform["worker_state_after_initialization"],
                   sampled_peak_board_memory_bytes=platform["sampled_peak_board_memory_bytes"])
        cross_rank_events = {}
        for rank in (0, 1):
            raw_dir = Path(f"results/captures/main30b_tp2_publicdev{group_index}_256/rank{rank}")
            export_dir = Path(f"results/captures/main30b_tp2_publicdev{group_index}_256_export/rank{rank}")
            raw = read(raw_dir / "metadata.json")
            meta, exported = read(export_dir / "metadata.json"), read(export_dir / "export_summary.json")
            assert raw["status"] == meta["status"] == "completed"
            assert meta["B"] == group["B"] and meta["actual_tp_rank"] == rank
            assert meta["budgets"] == [4096, 8192, 16384, 32768]
            assert meta["layers"] == [0, 15, 31, 47]
            assert exported["all_hooked_kv_matches_actual_cache"] and exported["all_shared_full_prefix_pages_identical"]
            warm = sorted((rid for rid, info in raw["requests"].items() if info["phase"] == "apc_warm_natural_group"),
                          key=lambda rid: raw["requests"][rid]["question_index"])
            assert len(warm) == group["B"]
            file_entries = {entry["file"]: entry for entry in raw["files"]}
            ready = []
            for event in exported["ready_groups"]:
                layer, anchor = event["layer"], event["anchor"]
                event_signatures = []
                for qi, rid in enumerate(warm):
                    name = f"layer{layer}_ready_anchor{anchor}_request{rid}.pt"
                    data = (ROOT / raw_dir / name).read_bytes()
                    assert hashlib.sha256(data).hexdigest() == file_entries[name]["sha256"]
                    snapshot = torch.load(ROOT / raw_dir / name, map_location="cpu", weights_only=True)
                    detail = snapshot["metadata"]
                    assert detail["hooked_kv_matches_actual_cache"] and detail["physical_full_prefix_pages_match"]
                    assert set(detail["actual_scheduled_request_ids"]) == set(warm)
                    assert detail["query_start_loc"] == list(range(group["B"] + 1))
                    assert all(len(position) == 1 for position in detail["same_event_query_positions"])
                    scheduled_index = detail["actual_scheduled_request_ids"].index(rid)
                    assert detail["query_token_position"] == detail["same_event_query_positions"][scheduled_index][0]
                    assert snapshot["private_k"].shape[1] == detail["query_token_position"] + 1 - detail["shared_prefix_end"]
                    assert snapshot["private_v"].shape == snapshot["private_k"].shape
                    event_signatures.append({key: detail[key] for key in ("actual_scheduled_request_ids", "query_start_loc", "same_event_query_positions", "sequence_lengths")})
                assert all(signature == event_signatures[0] for signature in event_signatures)
                key = (layer, anchor)
                if rank == 0:
                    cross_rank_events[key] = event_signatures[0]
                else:
                    assert cross_rank_events[key] == event_signatures[0], "TP ranks must capture the same scheduled forward event"
                ready.append({"layer": layer, "anchor": anchor, "actual_prediction_steps": event["actual_prediction_steps"],
                              "request_ids": warm, "same_forward_all_B_verified": True})
            assert any(event["layer"] == 15 and event["anchor"] == 64 for event in ready)
            row["ranks"].append({"rank": rank, "snapshot_checks": exported["snapshot_checks"],
                                 "ready_groups": ready, "budgets": meta["budgets"],
                                 "private_protected_document_tail_tokens": meta["private_protected_document_tail_tokens"],
                                 "actual_shared_document_token_range": meta["actual_shared_document_token_range"],
                                 "selector_seconds": sum(item["seconds"] for item in exported["selector_seconds"]),
                                 "raw_files": len(raw["files"]), "raw_bytes": sum(item["bytes"] for item in raw["files"]),
                                 "all_captured_latest_KV_bitwise_equal_actual_cache": True,
                                 "shared_physical_full_APC_pages_unchanged": True})
        row["all_exported_ready_groups_agree_across_ranks"] = True
        for budget in (4096, 16384):
            result_path = Path(f"results/main_capture_expansion/dev{group_index}_l15_s64_b{budget}.json")
            if not (ROOT / result_path).exists():
                continue
            result = read(result_path)
            assert result["provenance"]["ready_lifetime_measured"]
            expected_partitions, expected_trees = (5, 3) if group["B"] == 3 else (15, 15)
            assert len(result["partitions"]) == expected_partitions and len(result["hierarchies"]) == expected_trees
            qualified_partitions = {json.dumps(t["partition"]) for t in result["tuning"] if t["status"] == "ok" and "partition" in t}
            qualified_trees = {json.dumps(t["tree"]) for t in result["tuning"] if t["status"] == "ok" and "tree" in t}
            if result["tuning_complete"]:
                assert len(qualified_partitions) == expected_partitions
                assert len(qualified_trees) == expected_trees
            summary["replays"].append({"group_index": group_index, "budget": budget, "path": str(result_path),
                "B": result["batch"], "GQA": result["gqa"], "partitions": expected_partitions, "hierarchies": expected_trees,
                "executed_qualified_partitions": len(qualified_partitions), "executed_qualified_hierarchies": len(qualified_trees),
                "tuning_completed": result["tuning_completed"], "tuning_planned": result["tuning_planned"],
                "tuning_complete": result["tuning_complete"], "qualified_candidates": sum(t["status"] == "ok" for t in result["tuning"]),
                "failed_candidates": sum(t["status"] != "ok" for t in result["tuning"]),
                "missing_qualified_methods": result["missing_qualified_methods"], "summary": result["summary"],
                "search_seconds_including_jit_and_failures": result["search_seconds_including_jit_and_failures"],
                "numerical_contract": result["numerical_contract"]})
    jobs = [job for filename in ("configs/capture_main30b_publicdev1_6.json", "configs/export_main30b_publicdev1_6.json", "configs/bench_main30b_publicdev1_6.json") for job in read(filename)]
    job_names = {job["name"] for job in jobs}
    events = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines() if line.strip()]
    relevant = [event for event in events if event.get("name") in job_names]
    ended = {(event["name"], event["pid"]): event for event in relevant if event["event"] == "end"}
    active = [event for event in relevant if event["event"] == "start" and (event["name"], event["pid"]) not in ended]
    summary["budget"] = {"completed_jobs": len(ended), "active_jobs": active,
                         "P1_allocated_GPUh": sum(event["gpu_hours"] for event in ended.values() if event["category"] == "P1"),
                         "P2_allocated_GPUh": sum(event["gpu_hours"] for event in ended.values() if event["category"] == "P2"),
                         "task_caps_GPUh": {"P1": 1.0, "P2": 1.0}, "ended_events": list(ended.values())}
    assert summary["budget"]["P1_allocated_GPUh"] <= 1 and summary["budget"]["P2_allocated_GPUh"] <= 1
    out = ROOT / "results/main_capture_expansion/summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"captures": len(summary["captures"]), "replays": len(summary["replays"]), "budget": {key: value for key, value in summary["budget"].items() if key != "ended_events"}}, indent=2))


if __name__ == "__main__":
    main()
