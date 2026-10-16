"""Fit only the predeclared three-threshold family on complete development grid."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_candidate(batch, prefix_tokens, max_private_tokens, parameters):
    if parameters is None or batch <= 4:
        return False
    return (batch >= parameters["min_batch"] and prefix_tokens >= parameters["min_prefix"]
            and max_private_tokens / prefix_tokens <= parameters["max_private_prefix_ratio"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="configs/xqa_raw_dispatch_sweep_v1.json")
    p.add_argument("--output", required=True)
    p.add_argument("inputs", nargs="+")
    a = p.parse_args()
    output = Path(a.output)
    assert not output.exists()
    spec = json.loads(Path(a.spec).read_text())
    assert len(a.inputs) == spec["shards"]
    cells, seen_shards, input_hashes = {}, set(), {}
    for path in a.inputs:
        result = json.loads(Path(path).read_text())
        assert result["status"] in ("completed", "completed_with_failed_cells")
        assert result["spec_sha256"] == sha(a.spec) and result["spec"] == spec
        assert result["shard"] not in seen_shards
        seen_shards.add(result["shard"])
        input_hashes[path] = sha(path)
        assert len(result["planned_cells"]) == len(result["cases"])
        for case in result["cases"]:
            key = (case["prefix_tokens"], case["additional_private_tokens"], case["batch"])
            assert key not in cells
            cells[key] = case
    expected = set(itertools.product(spec["prefix_tokens"], spec["additional_private_tokens"], spec["batches"]))
    assert set(cells) == expected
    family = spec["policy_search"]
    admissible = []
    for batch, prefix, ratio in itertools.product(family["min_batch"], family["min_prefix"], family["max_private_prefix_ratio"]):
        parameters = {"min_batch": batch, "min_prefix": prefix, "max_private_prefix_ratio": ratio}
        covered = [c for c in cells.values() if select_candidate(c["batch"], c["prefix_tokens"], max(c["private_lengths"]), parameters)]
        if len(covered) < 8 or len({c["prefix_tokens"] for c in covered}) < 2 or len({c["batch"] for c in covered}) < 2:
            continue
        if any(c["status"] != "completed" for c in covered):
            continue
        lower = [c["timing"]["native_over_candidate"]["bootstrap_95ci"][0] for c in covered]
        aa = [c["timing"]["AA"]["paired_median_ratio"] for c in covered]
        if min(lower) < 1.03 or min(aa) < .98 or max(aa) > 1.02:
            continue
        admissible.append({"parameters": parameters, "covered_development_cells": len(covered),
                           "worst_development_lower95": min(lower), "covered_cell_ids": sorted(c["cell_id"] for c in covered)})
    admissible.sort(key=lambda c: (c["covered_development_cells"], c["worst_development_lower95"],
                                  c["parameters"]["min_batch"], c["parameters"]["min_prefix"],
                                  -c["parameters"]["max_private_prefix_ratio"]), reverse=True)
    selected = admissible[0] if admissible else None
    result = {"status": "development_policy_frozen", "spec_sha256": sha(a.spec), "inputs_sha256": input_hashes,
              "source_sha256": sha(__file__), "development_cells": len(cells), "admissible_rules": len(admissible),
              "selected": selected, "parameters": selected["parameters"] if selected else None,
              "all_admissible_rules": admissible,
              "heldout_status": "not_validated", "runtime_rule": family["family"],
              "limitations": ["Parameters selected only from synthetic hot-cache development grid.",
                              "No runtime timing or future-output oracle; unseen inputs apply the fixed thresholds.",
                              "New seed/off-grid and cold-cache validation required before broad performance claims."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"parameters": result["parameters"], "covered_cells": selected["covered_development_cells"] if selected else 0}))


if __name__ == "__main__":
    main()
