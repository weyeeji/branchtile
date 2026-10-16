#!/usr/bin/env python3
"""Publication PNG/PDF from wave16-only paired timing statistics."""
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/operator_lane_wave16_audit_v1"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules
    source = DIRECTORY / "audit.json"
    audit = json.loads(source.read_text())
    assert audit["status"] == "complete_wave16_independent_CPU_audit_passed"
    assert audit["qualified_configurations"] == 568
    assert sha(ROOT / "scripts/audit_operator_lane_wave16.py") == audit["auditor_sha256"]
    jobs = [row["name"] for row in audit["audited_complete_cells"]]
    assert len(jobs) == 8
    series = [
        ("dynamic_v2", "V2", "#0072B2", "o"),
        ("dynamic_persistent_fixed", "Persistent M32", "#E69F00", "s"),
        ("dynamic_v4", "Static bank", "#009E73", "^"),
        ("dynamic_lane_select", "Lane bank", "#D55E00", "D"),
        ("dynamic_fullcol_wide", "Wide FullCol (256t)", "#CC79A7", "v"),
    ]
    labels = ["D0 B3\n4K / L15", "D0 B3\n32K / L31", "D1 B3\n4K / L15", "D1 B3\n32K / L31",
              "D2 B4\n4K / L15", "D2 B4\n32K / L31", "D2 B4\n64K / L15", "D2 B4\n128K / L31"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True)
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.16, hspace=.20)
    x = np.arange(8)
    offsets = np.linspace(-.27, .27, len(series))
    plotted = []
    for ax, cache, title in zip(axes, ("hot", "cold"), ("Hot cache", "Artificial cold cache")):
        rows = {row["job"]: row for row in audit["within_wave_figure_data"] if row["cache"] == cache}
        assert set(rows) == set(jobs)
        ax.axhline(1, color="#444444", linestyle="--", linewidth=1.0, zorder=1)
        for offset, (method, label, color, marker) in zip(offsets, series):
            values = [rows[job]["methods"][method] for job in jobs]
            median = np.array([value["paired_time_ratio_to_best_lane_control"] for value in values])
            interval = np.array([value["paired_block_bootstrap95"] for value in values])
            assert np.isfinite(median).all() and np.isfinite(interval).all()
            assert (interval[:, 0] <= median).all() and (median <= interval[:, 1]).all()
            ax.errorbar(x + offset, median,
                        yerr=np.stack([median-interval[:, 0], interval[:, 1]-median]),
                        fmt=marker, linestyle="none", color=color, markeredgecolor="white",
                        markeredgewidth=.6, markersize=5.8, elinewidth=1.15,
                        capsize=2.2, capthick=1.0, label=label, zorder=3)
            for job, value in zip(jobs, values):
                plotted.append({"job": job, "cache": cache, "method": method, **value})
        for boundary in (1.5, 3.5):
            ax.axvline(boundary, color="#E3E3E3", linewidth=.9, zorder=0)
        ax.grid(axis="y", color="#DEDEDE", linewidth=.65, zorder=0)
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        ax.set_ylabel("Time / best non-lane control")
        ax.set_ylim(.85, 4.0 if cache == "hot" else 2.85)
        ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5, 4] if cache == "hot" else [1, 1.5, 2, 2.5])
        ax.set_xlim(-.65, 7.65)
    axes[1].set_xticks(x, labels)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.53, .985),
               ncol=5, frameon=False, columnspacing=1.45, handletextpad=.35)
    fig.text(.5, .075, "Wave 16 only; paired medians and 95% block-bootstrap intervals (20 blocks, 10,000 draws).",
             ha="center", fontsize=9.2)
    fig.text(.5, .045, "D0/D1/D2 are frozen development documents. Values above 1 are slower; preparation is excluded.",
             ha="center", fontsize=9.2)
    png = DIRECTORY / "paired_time_ratios.png"
    pdf = DIRECTORY / "paired_time_ratios.pdf"
    fig.savefig(png, dpi=300, facecolor="white", metadata={"Software": "Matplotlib; wave16 paired audit"})
    fig.savefig(pdf, facecolor="white", metadata={"Creator": "plot_operator_lane_wave16.py",
                                               "CreationDate": None, "ModDate": None})
    plt.close(fig)
    receipt = {"schema": 1, "plotter_sha256": sha(__file__), "audit_sha256": sha(source),
               "audit_path": str(source.relative_to(ROOT)), "matplotlib": matplotlib.__version__,
               "PNG_sha256": sha(png), "PDF_sha256": sha(pdf),
               "methods": [method for method, *_ in series], "job_order": jobs,
               "scope": "Wave16-only independently selected method winners; direct method/best_lane_control paired time ratios. No cross-wave assembly.",
               "CI": "Percentile 95% interval from 10,000 draws of 20 random/reverse measurement blocks, recomputed directly for each ratio.",
               "plotted_points": plotted, "audit_GPU_hours": 0,
               "caption": "Same-state complete GPU attention calls, hot and artificial-cold. Dashed line is best non-lane control. Values above one indicate slower execution. One-time preparation is excluded."}
    (DIRECTORY / "figure_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"PNG": str(png), "PDF": str(pdf), "points": len(plotted), "GPU_hours": 0}))


if __name__ == "__main__":
    main()
