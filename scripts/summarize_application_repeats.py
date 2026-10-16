"""Separate same-condition A/A, residual JIT, and warm-cache output changes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def summarize(data):
    if data["status"] != "complete":
        raise ValueError("Wait for all configured repeats, including service failures")
    groups = collections.defaultdict(list)
    for run in data["runs"]:
        groups[(run["group_index"], run["arm"])].append(run)
    rows = []
    for (group_index, arm), runs in sorted(groups.items()):
        runs.sort(key=lambda x: x["repeat"])
        if [r["repeat"] for r in runs] != [0, 1, 2, 3]:
            raise ValueError("Missing or duplicated repeat index")
        if [r["cache"] for r in runs] != ["cold", "cold", "cold", "warm"]:
            raise ValueError("Expected exactly three cold and one warm repeats")
        row = {"group_index": group_index, "document_id": runs[0]["document_id"], "arm": arm,
               "statuses": [r["status"] for r in runs],
               "all_repeats_serviced": all(r["status"] == "complete" for r in runs)}
        if not row["all_repeats_serviced"]:
            # No zero-request equality claim or fictitious zero-time service.
            row["token_comparison_eligible"] = False
            rows.append(row)
            continue
        tokens = [[q["output_token_ids"] for q in r["requests"]] for r in runs]
        if not tokens[0] or len({len(t) for t in tokens}) != 1:
            raise ValueError("Mismatched or empty repeated request sets")
        if any(r["question_ids"] != runs[0]["question_ids"] for r in runs):
            raise ValueError("Question identities changed across repeats")
        cold_equal = [a == b == c for a, b, c in zip(*tokens[:3])]
        warm_equal = [a == b for a, b in zip(tokens[1], tokens[3])]
        row.update(token_comparison_eligible=True,
            cold_jct_seconds=[r["group_jct_seconds"] for r in runs[:3]],
            steady_cold_median_seconds=statistics.median(r["group_jct_seconds"] for r in runs[1:3]),
            warm_jct_seconds=runs[3]["group_jct_seconds"], cold_repeats_token_equal_by_request=cold_equal,
            warm_vs_cold_token_equal_by_request=warm_equal,
            token_counts_by_run=[[len(t) for t in row] for row in tokens],
            finish_reasons_by_run=[[q["finish_reason"] for q in r["requests"]] for r in runs],
            cached_tokens_by_run=[[q["num_cached_tokens"] for q in r["requests"]] for r in runs])
        rows.append(row)
    aggregates = {}
    for arm in sorted({r["arm"] for r in rows}):
        all_rows = [r for r in rows if r["arm"] == arm]
        expected = {int(x) for x in data["config"]["groups"].split(",")}
        if {r["group_index"] for r in all_rows} != expected:
            raise ValueError("Incomplete document denominator for arm " + arm)
        subset = [r for r in all_rows if r["token_comparison_eligible"]]
        aggregates[arm] = {"documents": len(all_rows), "all_repeats_serviced_documents": len(subset),
            "ineligible_documents_retained": len(all_rows) - len(subset),
            "total_first_cold_seconds": sum(r["cold_jct_seconds"][0] for r in subset),
            "total_steady_cold_seconds": sum(r["steady_cold_median_seconds"] for r in subset),
            "total_warm_seconds": sum(r["warm_jct_seconds"] for r in subset),
            "requests": sum(len(r["cold_repeats_token_equal_by_request"]) for r in subset),
            "cold_token_equal_requests": sum(sum(r["cold_repeats_token_equal_by_request"]) for r in subset),
            "warm_token_equal_requests": sum(sum(r["warm_vs_cold_token_equal_by_request"]) for r in subset)}
    return {"scope": "Same model/manifest/seed graph-engine repeatability; three cold repeats are A/A, never three independent quality seeds",
        "model": data["model"], "manifest": data["manifest"], "manifest_sha256": data["manifest_sha256"],
        "steady_rule": "Every document and arm uses repeats1/2; repeat0 is the first actual-shape call. Warm is repeat3. JCT totals only include groups served in all repeats; all other groups remain explicit.",
        "limitations": ["Neighboring experiments may be active; not a controlled energy or final production speedup experiment.",
                        "Warm and cold caches can produce different token sequences even with the same seed. Do not call their JCT ratio same-workload acceleration.",
                        "Token identity does not prove identical scheduler paths or sparse full-budget equivalence. Capped request identity is not successful task completion."],
        "initialization_seconds": data["initialization_seconds"], "aggregates": aggregates, "records": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/application_graph_instruct/train8_aa.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    source = ROOT / args.input
    result = summarize(json.loads(source.read_text()))
    result.update(input=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    destination = ROOT / args.output if args.output else source.parent / "repeat_summary.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregates"], indent=2))


if __name__ == "__main__":
    main()
