"""Separate same-condition A/A, residual JIT, and warm-cache output changes."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "results/application_graph_instruct/train8_aa.json"
data = json.loads(source.read_text())
assert data["status"] == "complete"
groups = collections.defaultdict(list)
for run in data["runs"]:
    groups[(run["group_index"], run["arm"])].append(run)
rows = []
for (group_index, arm), runs in groups.items():
    runs.sort(key=lambda x: x["repeat"])
    assert [r["cache"] for r in runs] == ["cold", "cold", "cold", "warm"]
    tokens = [[q["output_token_ids"] for q in r["requests"]] for r in runs]
    cold_equal = [a == b == c for a, b, c in zip(*tokens[:3])]
    warm_equal = [a == b for a, b in zip(tokens[1], tokens[3])]
    rows.append({"group_index": group_index, "document_id": runs[0]["document_id"], "arm": arm,
        "cold_jct_seconds": [r["group_jct_seconds"] for r in runs[:3]],
        "steady_cold_median_seconds": statistics.median(r["group_jct_seconds"] for r in runs[1:3]),
        "warm_jct_seconds": runs[3]["group_jct_seconds"], "cold_repeats_token_equal_by_request": cold_equal,
        "warm_vs_cold_token_equal_by_request": warm_equal,
        "token_counts_by_run": [[len(t) for t in row] for row in tokens],
        "cached_tokens_by_run": [[q["num_cached_tokens"] for q in r["requests"]] for r in runs]})
aggregates = {}
for arm in ["dense", "joint"]:
    subset = [r for r in rows if r["arm"] == arm]
    aggregates[arm] = {"documents": len(subset),
        "total_first_cold_seconds": sum(r["cold_jct_seconds"][0] for r in subset),
        "total_steady_cold_seconds": sum(r["steady_cold_median_seconds"] for r in subset),
        "total_warm_seconds": sum(r["warm_jct_seconds"] for r in subset),
        "requests": sum(len(r["cold_repeats_token_equal_by_request"]) for r in subset),
        "cold_token_equal_requests": sum(sum(r["cold_repeats_token_equal_by_request"]) for r in subset),
        "warm_token_equal_requests": sum(sum(r["warm_vs_cold_token_equal_by_request"]) for r in subset)}
result = {"input": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "scope": "QMSum train8, Instruct4B, graph engine, one seed; three cold repeats are A/A, never three independent quality seeds",
    "steady_rule": "For every document and arm use repeats1/2; repeat0 remains recorded as the first actual-shape call. Warm is repeat3 after those calls.",
    "limitations": ["Parallel neighboring experiments were active; not a controlled energy or final production speedup experiment.",
                    "Warm and cold caches can produce different token sequences even with the same seed. Do not call their JCT ratio same-workload acceleration.",
                    "Only cold-vs-cold repeated token identity is established here; it is not sparse full-budget equivalence."],
    "initialization_seconds": data["initialization_seconds"], "aggregates": aggregates, "records": rows}
(source.parent / "repeat_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(aggregates, indent=2))
