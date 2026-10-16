"""Plot complete-call evidence and explicit conditional break-even scenarios."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = ROOT / "results/joint_real48_replay_v1/analysis"
    source = folder / "summary.json"
    report = json.loads(source.read_text())
    cells = [r for r in report["cells"] if r["rank_scope"] == "rank_max_proxy"
             and r["contract"] == "prepare_plus_48_layers" and r["clock"] == "complete_wall_us"]
    assert len(cells) == 72
    documents = [("docfinqa-c0f11a67aae850fc", "Original document", "fi_natural_original_b32768"),
                 ("docfinqa-593ec0c33f45ea02", "Public development document 1", "fi_natural_publicdev1_b32768")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for row, (document, title, _) in enumerate(documents):
        for col, cache in enumerate(("hot", "cold")):
            ax = axes[row, col]
            for variant, color, name in (("epsilon0", "#888888", "A/A, epsilon 0"),
                                         ("epsilon001", "#cf7720", "epsilon 0.001"),
                                         ("epsilon01", "#1769a0", "epsilon 0.01")):
                for replica in (0, 1):
                    series = sorted((r for r in cells if r["document_id"] == document
                                     and r["cache"] == cache and r["variant"] == variant
                                     and r["device_replica"] == replica), key=lambda r: r["batch"])
                    assert len(series) == 3
                    center = np.asarray([r["paired_speed_ratio"] for r in series])
                    interval = np.asarray([r["paired_speed_ratio_bootstrap95"] for r in series])
                    ax.errorbar(np.arange(1, 4)+(replica-.5)*.08, center,
                                yerr=[center-interval[:, 0], interval[:, 1]-center],
                                marker="o" if replica == 0 else "s", markersize=4,
                                color=color, linestyle="-" if replica == 0 else "--",
                                capsize=2, label=name if replica == 0 else None)
            ax.axhline(1, color="black", linewidth=.8)
            ax.axhline(1.15, color="#9e7777", linestyle=":", linewidth=1)
            ax.set_ylim(.98, 1.175); ax.set_xticks([1, 2, 3], ["B1 subset", "B2 subset", "Actual B3"])
            ax.set_title(f"{title} / {cache}", fontsize=11)
            ax.grid(alpha=.15)
            if col == 0:
                ax.set_ylabel("Paired independent / method time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.52, .10), frameon=False)
    fig.suptitle("Real 48-layer E0 chain: a small repeatable signal, below the 1.15x target", fontsize=13)
    fig.text(.02, .015, "Includes native prepare, original strided KV, all 48 layers and output.copy_. Solid/circle and dashed/square: device replicas.\n"
             "Max of separately measured ranks is a diagnostic proxy, not TP model JCT. Six paired blocks; intervals do not cover natural trajectories.\n"
             "One snapshot/document; B1/B2 are explicit B3 row subsets. Selection/install costs remain outside these complete-call timings.", fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .95))
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"complete_call.{ext}", dpi=180)
    plt.close(fig)

    natural = {}
    hashes = {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()}
    for doc, _, name in documents:
        path = ROOT / "results/main30b_sparse" / name / "summary.json"
        data = json.loads(path.read_text())
        run = next(r for r in data["runs"] if r["method"] == "e0")
        lengths = [len(r["tokens"]) for r in run["answers"]]
        natural[doc] = {"source": str(path.relative_to(ROOT)), "lengths": lengths,
                        "B3_steps_optimistic_upper_count": min(lengths), "longest_tokens": max(lengths)}
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    economics = []
    for receipt in report["job_receipts"]:
        for variant in ("epsilon001", "epsilon01"):
            prep = max(p["shared_score_D2H_and_copy_wall_seconds"] +
                       p["variants"][variant]["CPU_finalize_wall_seconds"] for p in receipt["preparation"])
            horizon = natural[receipt["document_id"]]
            for cache in ("hot", "cold"):
                cell = next(r for r in cells if r["job"] == receipt["job"] and r["variant"] == variant
                            and r["cache"] == cache and r["batch"] == 3)
                saving = cell["paired_saving_us"] / 1e6
                economics.append({"job": receipt["job"], "document_id": receipt["document_id"],
                                  "variant": variant, "cache": cache,
                                  "selection_only_preparation_proxy_seconds": prep,
                                  "paired_B3_chain_saving_us": cell["paired_saving_us"],
                                  "conditional_break_even_B3_steps": prep/saving if saving > 0 else None,
                                  "baseline_B3_steps_optimistic_upper_count": horizon["B3_steps_optimistic_upper_count"],
                                  "baseline_longest_output_tokens": horizon["longest_tokens"],
                                  "hypothetical_gross_saving_if_B3_gain_persisted_to_longest_tail_seconds": saving * horizon["longest_tokens"],
                                  "selection_cost_recovered_in_that_hypothesis": saving * horizon["longest_tokens"] > prep})
    result = {"source_hashes": hashes, "natural_baseline_horizons": natural,
              "conditional_scenarios": economics,
              "all_scenarios_fail_to_recover_selection_cost": all(not r["selection_cost_recovered_in_that_hypothesis"] for r in economics),
              "limitations": [
                  "Conditional extrapolation from one snapshot, not a measured trajectory or hardware upper bound.",
                  "Preparation proxy excludes incremental installation/RPC costs; epsilon0 qualification cost is not deducted.",
                  "Even the longest-tail scenario incorrectly grants B3 benefit after requests exit, intentionally favoring continuation for this diagnostic.",
                  "Natural lengths belong to independent E0, not new joint outputs. No quality or natural JCT claim follows.",
              ]}
    (folder / "conditional_amortization.json").write_text(json.dumps(result, indent=2)+"\n")
    with (folder / "conditional_amortization.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(economics[0])); writer.writeheader(); writer.writerows(economics)
    print(json.dumps({"plotted_cells": len(cells), "conditional_scenarios": len(economics),
                      "all_scenarios_fail_to_recover_selection_cost": result["all_scenarios_fail_to_recover_selection_cost"]}))


if __name__ == "__main__":
    main()
