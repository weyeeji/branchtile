#!/usr/bin/env python3
"""Read-only allocation ledger audit; estimates and reserves stay separate."""
import collections
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    path = ROOT / "results/job_ledger.jsonl"
    raw = path.read_bytes()
    now = dt.datetime.now(dt.timezone.utc)
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for line, content in enumerate(raw.decode().splitlines(), 1):
        event = json.loads(content)
        grouped[(event["name"], event.get("pid"))][event["event"]].append((line, event))
    anomalies, rows, active, absent_without_end = [], [], [], []
    totals = {category: {"exact_timer_gpu_hours": 0.0, "conservative_backfill_gpu_hours": 0.0,
                         "completed_job_count": 0, "failed_job_count": 0} for category in ("P1", "P2")}
    for (name, pid), events in grouped.items():
        starts, ends = events.get("start", []), events.get("end", [])
        if len(starts) != 1 or len(ends) > 1:
            anomalies.append({"name": name, "pid": pid, "start_count": len(starts), "end_count": len(ends),
                              "issue": "non-unique start/end; not silently counted"})
            continue
        line, start = starts[0]
        if not ends:
            record = {"name": name, "pid": pid, "category": start["category"], "gpus": start["gpus"],
                      "start_line": line, "started_utc": start["time_utc"],
                      "excluded_from_completed_totals": True}
            present = pid is not None and Path(f"/proc/{pid}").exists()
            record["pid_present_at_audit"] = present
            record["elapsed_gpu_hours_at_snapshot_not_terminal_charge"] = max(0, (now - dt.datetime.fromisoformat(start["time_utc"])).total_seconds()) * len(start["gpus"]) / 3600
            (active if present else absent_without_end).append(record)
            continue
        end_line, end = ends[0]
        if start["category"] != end["category"] or start["gpus"] != end["gpus"]:
            anomalies.append({"name": name, "pid": pid, "issue": "start/end category or GPU allocation mismatch"})
            continue
        computed = end["allocated_seconds"] * len(end["gpus"]) / 3600
        if end["allocated_seconds"] < 0 or not math.isclose(computed, end["gpu_hours"], rel_tol=1e-12, abs_tol=1e-12):
            anomalies.append({"name": name, "pid": pid, "issue": "allocated GPU-hour arithmetic mismatch"})
            continue
        backfill = "accounting_note" in start or "accounting_note" in end
        category = end["category"]
        subtotal = "conservative_backfill_gpu_hours" if backfill else "exact_timer_gpu_hours"
        totals[category][subtotal] += computed
        totals[category]["completed_job_count"] += 1
        totals[category]["failed_job_count"] += end.get("exit_code", 0) != 0
        wall = (dt.datetime.fromisoformat(end["time_utc"]) - dt.datetime.fromisoformat(start["time_utc"])).total_seconds()
        rows.append({"name": name, "pid": pid, "category": category, "gpus": end["gpus"],
                     "start_line": line, "end_line": end_line, "started_utc": start["time_utc"], "ended_utc": end["time_utc"],
                     "accounting_type": "retrospective_conservative_backfill" if backfill else "run_jobs_monotonic_process_timer",
                     "accounting_note": end.get("accounting_note"), "pid_missing_in_legacy_backfill": pid is None,
                     "allocated_seconds": end["allocated_seconds"], "gpu_hours": computed,
                     "wall_timestamp_interval_seconds": wall,
                     "timer_minus_wall_timestamp_seconds": end["allocated_seconds"] - wall,
                     "exit_code": end.get("exit_code"), "log": end.get("log")})
    for category, value in totals.items():
        value["ledger_completed_gpu_hours"] = value["exact_timer_gpu_hours"] + value["conservative_backfill_gpu_hours"]
        value["proposal_limit_gpu_hours"] = 24 if category == "P1" else 8
        value["additional_unknown_probe_reserve_gpu_hours"] = 1.0
        value["fixed_union_manual_checks_reserve_gpu_hours"] = 0.05 if category == "P2" else 0.0
        value["planning_total_including_reserves_gpu_hours"] = value["ledger_completed_gpu_hours"] + value["additional_unknown_probe_reserve_gpu_hours"] + value["fixed_union_manual_checks_reserve_gpu_hours"]
        value["planning_headroom_after_reserves_gpu_hours"] = value["proposal_limit_gpu_hours"] - value["planning_total_including_reserves_gpu_hours"]
    accounted_pids = {pid for _, pid in grouped if pid is not None}
    metadata_coverage = []
    for folder in ("old30b_diagnostic", "official4b_boundary"):
        for metadata_path in sorted((ROOT / "results/captures" / folder).glob("*/metadata.json")):
            meta = json.loads(metadata_path.read_text())
            pid = meta.get("process_id")
            metadata_coverage.append({"path": str(metadata_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(), "status": meta["status"], "pid": pid,
                "pid_has_ledger_record": pid in accounted_pids if pid is not None else None,
                "note": "allocator-init failure occurred before metadata identity was saved; corresponding data_official4b_pilot PID1174846 has a complete ledger pair" if pid is None else "metadata shares an existing process allocation; never charged separately"})
    manual_evidence = []
    for name in ("results/validation/fp64_tests.txt", "results/validation/compute_sanitizer_memcheck.txt",
                 "results/validation/ncu_permissions.txt", "logs/union_tests.log", "logs/union_memcheck.log"):
        file = ROOT / name
        manual_evidence.append({"path": name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    report = {"schema": 1, "audited_utc": now.isoformat(), "effective_user_id": os.geteuid(),
              "virtual_environment": sys.prefix, "ledger": str(path.relative_to(ROOT)),
              "ledger_sha256": hashlib.sha256(raw).hexdigest(), "ledger_bytes": len(raw),
              "ledger_event_count": len(raw.decode().splitlines()), "unique_name_pid_pairs": len(grouped),
              "matching_rule": "exact (name,pid); two legacy P2 backfill jobs have pid=null and unique names; later visibility-guard backfills retain known process IDs; one end charge per pair",
              "counting_rule": "allocated GPU count times monotonic process wall time; initialization, compilation, CPU portions while allocated, and failures count; no utilization discount; backfill kept separate",
              "anomalies": anomalies, "active_jobs_excluded_from_completed_totals": active,
              "absent_processes_without_end": absent_without_end,
              "all_started_jobs_have_end_or_live_pid": not absent_without_end and not anomalies,
              "all_started_jobs_have_end": not active and not absent_without_end and not anomalies,
              "totals": totals, "completed_jobs": rows,
              "v3_double_count_protection": {"rule": "v3 memcheck includes its child pytest; v3 qualification includes racecheck and both bounded comparisons; child/log times are not added again",
                  "job_keys": [[r["name"], r["pid"]] for r in rows if r["name"].startswith("branchtile_v3_")]},
              "capture_metadata_ledger_coverage": metadata_coverage,
              "unledgered_manual_checks": {"fixed_union": {"pytest_reported_seconds": 42.94,
                  "memcheck_pytest_reported_seconds": 6.51, "exact_allocation_gpu_hours": None,
                  "conservative_planning_reserve_gpu_hours": 0.05, "category": "P2",
                  "reason": "pytest durations exclude some startup/shutdown; full allocation intervals were not logged; reserve is not fabricated exact spend"}},
              "unknown_initial_probe_costs": {"exact_allocation_gpu_hours": None,
                  "known_incomplete_evidence": ["initial FP64 pytest reports 141.90s", "initial memcheck pytest reports 6.14s", "Nsight counter permission probe has no allocation interval"],
                  "other_early_CUDA_import_or_deployment_probes": "not fully enumerated; allocation interval unknown",
                  "additional_reserve_gpu_hours": {"P1": 1.0, "P2": 1.0},
                  "reserve_interpretation": "explicit management contingency for incompletely bounded early probes; not measured spend and not a proven upper bound; replace with exact charges if intervals are recovered"},
              "manual_evidence_files": manual_evidence,
              "budget_interpretation": "24 and 8 GPUh are the original proposal planning budgets, never actual consumption. The user subsequently authorized continued exploration and mechanism/scenario changes until paper-worthy evidence is obtained. Compare each current planning_total_including_reserves_gpu_hours against its original budget explicitly; do not assume it remains below budget. Incomplete early probes prevent claiming a mathematically proven total-cost bound.",
              "ledger_unchanged_during_audit": path.read_bytes() == raw,
              "shared_ledger_was_modified_by_this_script": False}
    output = ROOT / "results/budget_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "totals": totals, "active": len(active), "unmatched": len(absent_without_end), "anomalies": anomalies}, indent=2))


if __name__ == "__main__":
    main()
