"""Render the frozen primary scope; do not pool cache regimes or contracts."""
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

    folder = ROOT / "results/joint_e0_gate_v1"
    source = folder / "summary.json"
    report = json.loads(source.read_text())
    cells = [r for r in report["all_cells"]
             if (r["batching_order"], r["contract"], r["call_scope"]) ==
             ("production_BH", "O_only", "complete_layout")]
    assert len(cells) == 12
    documents = ["docfinqa256", "publicdev1_256", "publicdev6_256"]
    names = ["Original / B3", "Public dev 1 / B3", "Public dev 6 / B4"]
    keys = [(doc, budget) for doc in documents for budget in (4096, 16384)]
    labels = [f"{names[documents.index(doc)]} / {budget // 1024}K"
              for doc, budget in keys]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    methods = [("exchange", "Exchange, 1%", "#1769a0", -0.22),
               ("cover", "Classical cover, 1%", "#c26b18", 0),
               ("independent_AA", "Independent A/A", "#777777", 0.22)]
    for ax, cache in zip(axes, ("hot", "cold")):
        for method, label, color, offset in methods:
            centers, lows, highs = [], [], []
            for doc, budget in keys:
                match = [r for r in cells if
                         (r["document"], r["budget"], r["cache"]) ==
                         (doc, budget, cache)]
                assert len(match) == 1
                r = match[0]
                ratio = r["paired_ratios"][f"independent_over_{method}"]
                center = ratio["median"]
                lo, hi = ratio["paired_block_bootstrap95"]
                assert 0 < lo <= center <= hi
                centers.append(center); lows.append(center-lo); highs.append(hi-center)
                rows.append(dict(document=doc, budget=budget, cache=cache,
                                 method=method, paired_speed_ratio=center,
                                 bootstrap95_low=lo, bootstrap95_high=hi,
                                 independent_median_us=r["median_us"]["independent"],
                                 method_median_us=r["median_us"][method]))
            ax.errorbar(centers, np.arange(6)+offset, xerr=[lows, highs],
                        marker="o", linestyle="none", color=color,
                        markersize=4.5, capsize=3, label=label)
        ax.axvline(1, color="black", linewidth=1)
        ax.axvline(1.15, color="#9e7777", linestyle="--", linewidth=.8)
        ax.set_title("Repeated hot cache" if cache == "hot" else "Flushed cold cache")
        ax.set_xlabel("Paired independent / method latency (>1 is faster)")
        ax.set_xlim(.98, 1.19)
        ax.set_xticks([1, 1.05, 1.10, 1.15])
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(np.arange(6), labels)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.55, .14), frameon=False)
    fig.suptitle("Same native E0, fixed per-request k: hot gains are small; cold gains depend on regime", fontsize=12)
    fig.text(.02, .02,
             "Primary scope: BH order, O only, layout view; contiguous exported KV, FlashInfer 0.7.2.\n"
             "Includes no engine output.copy_ or selector/plan cost. Cold events cluster near 2.048 us bins;\n"
             "within-run block bootstrap intervals do not represent deployment uncertainty. Dashed line: 1.15x screening target.",
             fontsize=9)
    fig.tight_layout(rect=(0, .23, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"primary_scope.{extension}", dpi=180)
    plt.close(fig)
    with (folder / "primary_scope.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {"source": str(source.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "scope": "production_BH / O_only / complete_layout; hot and cold separate",
                "rows": len(rows), "GPU_used": False,
                "limitations": report["engine_scope_limitations"] + [
                    "No across-document, across-regime or across-contract pooled performance estimate.",
                    "Intervals are existing within-run paired-block bootstrap, not new observations."]}
    (folder / "primary_scope_figure.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
