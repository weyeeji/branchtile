"""CPU audit of executable E1/E4 coverage, latency, preparation and budget."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for path in sorted((ROOT / "results/partition_cascade").glob("*.json")):
        data = json.loads(path.read_text())
        if "tuning_planned" not in data:
            continue
        tuning = data["tuning"]
        coverage = {}
        for method in ("e1_native", "e1_own", "e4_native"):
            rows = [r for r in tuning if r["method"] == method]
            structures = {json.dumps(r.get("partition", r.get("tree"))) for r in rows}
            qualified = {json.dumps(r.get("partition", r.get("tree"))) for r in rows if r["status"] == "ok"}
            coverage[method] = {"structures_executed": len(structures), "structures_with_qualified_config": len(qualified),
                                "configurations_executed": len(rows), "qualified_configurations": sum(r["status"] == "ok" for r in rows),
                                "failed_reason_counts": dict(collections.Counter(r.get("error", "unknown") for r in rows if r["status"] != "ok"))}
        for cache in ("hot", "cold"):
            if cache not in data["summary"]:
                continue
            rows = [r for r in data["measurements"] if r["cache"] == cache]
            assert len(rows) >= 60
            for block in {r["block"] for r in rows}:
                paired = {r["direction"]: r for r in rows if r["block"] == block}
                assert paired["random"]["order"] == list(reversed(paired["reverse"]["order"]))
        planning_path = path.with_name(path.stem + "_planning.json")
        followup = None
        if planning_path.exists():
            planning = json.loads(planning_path.read_text())
            assert planning["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            followup = {"file": str(planning_path.relative_to(ROOT)),
                        "second_physical_card": planning["gpu_uuid"] != planning["original_gpu_uuid"], "cache": {}}
            for cache, result in planning["cache"].items():
                scenarios = {}
                for reuse, scenario in result["amortization_scenarios"].items():
                    costs = scenario["cost_us_by_method"]
                    control = min((m for m in costs if m != "e4_native"), key=costs.get)
                    scenarios[reuse] = {**scenario, "best_non_e4_control": control,
                                         "best_control_over_e4_cost_ratio": costs[control] / costs["e4_native"]}
                followup["cache"][cache] = {"median_gpu_us": result["median_gpu_us"], "paired_ratios": result["paired_ratios"],
                                            "amortization_scenarios": scenarios}
        records.append({"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "provenance": data["provenance"], "q_shape": data["q_shape"], "k_shape": data["k_shape"],
                        "tuning_complete": data["tuning_complete"], "coverage": coverage,
                        "summary": data["summary"], "planning": data["planning"], "second_card_planning_followup": followup,
                        "search_seconds_including_jit_and_failures": data["search_seconds_including_jit_and_failures"],
                        "selected_configs": {cache: {m: choice["config"] for m, choice in choices.items()} for cache, choices in data["selected"].items()}})
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines() if line]
    jobs = [row for row in ledger if row.get("name", "").startswith("partition_cascade_")]
    ended = [row for row in jobs if row["event"] == "end"]
    finished_pids = {row["pid"] for row in ended}
    pending = [row for row in jobs if row["event"] == "start" and row["pid"] not in finished_pids]
    report = {
        "scope": "First bounded executable E1/E4 follow-up; finite qualified configuration grid, not complete application evaluation.",
        "records": records,
        "distinct_natural_document_sha256": sorted({r["provenance"]["document_sha256"] for r in records if "document_sha256" in r["provenance"]}),
        "budget": {"new_round_limit_gpu_hours": 2, "recorded_allocated_gpu_hours": sum(row["gpu_hours"] for row in ended),
                   "finished_jobs": len(ended), "pending_jobs": pending, "includes_failed_runs": True},
        "comparison_rules": ["Each E1 input executes all 15 B4 partitions using native grouped prefill and own fixed union, plus an optimized independent native decode alternative.",
                             "E4 executes all 15 rooted binary intersection/residual trees with exact pair conservation; all nodes batched in one native call and merged by LSE.",
                             "All methods share Q/K/V and original owner masks. Query gather, output restoration, natural-log LSE conversion and partial merge are timed.",
                             "Each shape/configuration qualifies against FP64 without relaxing output or LSE tolerances.",
                             "Per-capture hindsight oracle search time and warm plan/prepare+run wall time are explicit; neither is a free deployed selector.",
                             "Second-card follow-ups remeasure selected methods and include native full union even when own union wins the first steady-state tuning. Amortization uses the strongest measured selected backend, not just the initially fastest kernel."],
        "research_interpretation": {
            "existing_dynamic_results": "v1/v2/v3 negative evidence is preserved; no new dynamic-row implementation was substituted.",
            "e4_identity": "Exact hierarchical mechanism port using FlashInfer, not original QUILT code or its full planner.",
            "pivot_gate": "A future new mechanism must beat the fastest qualified native full union, E1 and E4 by about 15% on prespecified natural captures after required planning amortization, with paired uncertainty and second-card replication.",
            "remaining_opportunity": "Inspect execution and planning separately. Large Python preparation cost is an engineering target only until measured against mature cached/incremental native planning; it is not evidence of a novel attention dataflow.",
            "stop_condition": "No further unconstrained layout search follows from a logical tile saving or a win over a weaker full-union control alone."},
        "limitations": ["Natural old-30B captures remain diagnostic and official 4B is a GQA4 boundary; the main 2507-FP8 GQA8 study remains separate.",
                        "Hierarchy tree search is binary, with no QUILT-specific planner optimizations or measured multiway-tree policy in this round.",
                        "Cold flush and repeated hot replay do not establish natural cache rotation, shared-ready windows, model quality, JCT or energy.",
                        "The 1/16/64/256 reuse counts are explicit static-plan amortization scenarios, not observed request lifetimes.",
                        "Paired blocks, tree candidates and layers are not independent documents; no population-level conclusion is inferred."]}
    out = ROOT / "results/partition_cascade/summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": len(records), "gpu_hours": report["budget"]["recorded_allocated_gpu_hours"], "pending_jobs": len(pending)}))


if __name__ == "__main__":
    main()
