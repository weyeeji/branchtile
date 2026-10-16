"""CPU-only scientific figure for the frozen physical-address intervention.

Recomputes all paired block intervals from raw timings before rendering. This
does not run attention, read model outputs, or alter the experiment's analysis.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/native_physical_alias_v1"
OUTPUT = ROOT / "results/native_physical_alias_figures_v1"
PROTOCOL = ROOT / "configs/native_physical_alias_v1.json"
STEM = "native_physical_alias_hot_cold"
DOCS = ("docfinqa256", "publicdev1_256")
BUDGETS = (4096, 8192, 16384, 32768)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect():
    sources = {}

    def read(path):
        payload = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = {"sha256": sha(payload), "bytes": len(payload)}
        return json.loads(payload)

    protocol = read(PROTOCOL)
    analysis = read(SOURCE / "analysis.json")
    assert [s["id"] for s in protocol["sources"]] == list(DOCS)
    assert protocol["budgets"] == list(BUDGETS)
    assert protocol["paired_blocks"] == 30 and protocol["hot_repetitions"] == 32
    assert analysis["protocol_sha256"] == sha(PROTOCOL.read_bytes())
    analysis_rows = {(r["document"], r["budget"], r["cache"], r["rotation"]): r
                     for r in analysis["paired_statistic_rows"]}
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(0, 30, (10000, 30))
    points, cells = [], []
    for doc_i, doc in enumerate(DOCS):
        for budget_i, budget in enumerate(BUDGETS):
            path = SOURCE / f"{doc}_{budget}.json"
            raw = read(path)
            assert raw["status"] == "completed"
            assert raw["protocol_sha256"] == analysis["protocol_sha256"]
            assert sha(path.read_bytes()) == analysis["source_files_sha256"][str(path.relative_to(ROOT))]
            row_i = doc_i * 4 + budget_i
            assert int(raw["CUDA_VISIBLE_DEVICES"]) == row_i
            assert len(raw["qualifications"]) == 6
            assert len(raw["address_mutation_qualification"]) == 48
            assert all(r["status"] == "passed" for r in raw["qualifications"] + raw["address_mutation_qualification"])
            cells.append({"document": doc, "budget": budget, "gpu_slot": row_i,
                          "gpu_uuid": raw["gpu_uuid"], "source": str(path.relative_to(ROOT))})
            for cache in ("hot", "cold"):
                subset = [r for r in raw["measurements"] if r["cache"] == cache]
                lookup = {(r["block"], r["direction"]): r["us"] for r in subset}
                assert len(subset) == len(lookup) == 60
                for rotation in range(3):
                    pairs = np.array([[lookup[block, direction][f"private{rotation}"] /
                                       lookup[block, direction][f"shared{rotation}"]
                                       for direction in ("random", "reverse")] for block in range(30)])
                    median = float(np.median(pairs))
                    ci = np.quantile(np.median(pairs[draws], axis=(1, 2)), [.025, .975]).tolist()
                    a = analysis_rows[doc, budget, cache, rotation]
                    summary = next(s for s in raw["summary"] if s["cache"] == cache)
                    s = summary["paired_ratios"][f"private{rotation}_over_shared{rotation}"]
                    assert np.allclose([median, *ci], [a["private_over_shared"], *a["paired_block_bootstrap95"]], atol=2e-15, rtol=0)
                    assert np.allclose([median, *ci], [s["median"], *s["paired_block_bootstrap95"]], atol=2e-15, rtol=0)
                    points.append({"document": doc, "budget": budget, "cache": cache,
                                   "rotation": rotation, "row": row_i, "private_over_shared": median,
                                   "paired_block_bootstrap95": ci})
    assert len(points) == 48 and len(cells) == 8
    assert len({c["gpu_uuid"] for c in cells}) == 8
    return points, cells, sources, protocol


def render(points):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.185, right=.98, bottom=.275, top=.79, wspace=.12)
    colors = ("#0072B2", "#D55E00", "#009E73")
    markers = ("o", "s", "^")
    offsets = (-.19, 0, .19)
    for ax, cache, label in zip(axes, ("hot", "cold"), ("(a) Hot", "(b) Cold")):
        for p in [p for p in points if p["cache"] == cache]:
            lo, hi = p["paired_block_bootstrap95"]
            x, r = p["private_over_shared"], p["rotation"]
            ax.errorbar(x, p["row"] + offsets[r], xerr=[[x - lo], [hi - x]],
                        fmt=markers[r], markersize=4.8, markeredgewidth=.7,
                        color=colors[r], ecolor=colors[r], elinewidth=1.1,
                        capsize=2.4, capthick=.9, linestyle="none", zorder=3)
        ax.axvline(1, color="#555555", linestyle=(0, (4, 3)), linewidth=1, zorder=1)
        ax.axhline(3.5, color="#AFAFAF", linewidth=.7, zorder=1)
        ax.set_xlim(.955, 1.865)
        ax.set_ylim(7.65, -.65)
        ax.set_xticks([1, 1.2, 1.4, 1.6, 1.8])
        ax.set_xticklabels(["1.0", "1.2", "1.4", "1.6", "1.8"])
        ax.grid(axis="x", color="#ECECEC", linewidth=.6)
        ax.tick_params(axis="y", length=0, pad=8)
        subtitle = "Self-warm; 32-call graph chain" if cache == "hot" else "After flush; one-call graph"
        ax.set_title(label + "\n" + subtitle, fontsize=11, loc="left", pad=12)
        ax.set_xlabel("Paired latency ratio  (private / shared)", labelpad=10)
    axes[0].set_yticks(range(8))
    axes[0].set_yticklabels([f"Doc {'A' if i < 4 else 'B'}  |  {BUDGETS[i % 4] // 1024}K  |  GPU {i}" for i in range(8)])
    axes[0].set_ylabel("Document / support budget / physical GPU slot", labelpad=13)
    axes[1].tick_params(labelleft=False)
    legend = [Line2D([0], [0], color=colors[r], marker=markers[r], linestyle="none",
                     markersize=5, label=f"Address rotation {r}") for r in range(3)]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.57, .895),
               ncol=3, frameon=False, handletextpad=.5, columnspacing=2)
    fig.suptitle("Native E0: physical KV address sharing", x=.185, ha="left", y=.975, fontsize=15, fontweight="bold")
    fig.text(.185, .928, "Same Q, token order, plan, scratch and output buffers; only installed KV page IDs differ.", fontsize=10)
    notes = [
        "Points: median private/shared ratio; whiskers: 95% bootstrap CI over 30 paired blocks (10,000 draws).",
        "Random/reverse measurements stay clustered. Ratio > 1 means the shared-address arm is faster.",
        "Each doc × budget cell ran on a different GPU: inference is within-cell only; budgets are not connected.",
        "Cold timings show 2.048 µs quantization. Intervals omit systematic/quantization uncertainty; some collapse.",
        "BF16 KV, B3 / GQA8 / Hkv2, layer 15, step 64; O + output copy timed; index installation excluded.",
        "Doc A: docfinqa256; Doc B: publicdev1_256. This is not an application gain or a measurement of L2/DRAM traffic.",
    ]
    for i, line in enumerate(notes):
        fig.text(.185, .193 - i * .0255, line, fontsize=8.8, color="#333333")
    fixed_date = datetime(2026, 10, 16, tzinfo=timezone.utc)
    payloads = {}
    for extension in ("png", "pdf"):
        stream = io.BytesIO()
        metadata = {"Software": "plot_native_physical_alias.py; Matplotlib"} if extension == "png" else {
            "Title": "Native E0 physical KV address intervention",
            "Author": "ISCAS experiment audit", "Creator": "plot_native_physical_alias.py",
            "CreationDate": fixed_date, "ModDate": fixed_date}
        fig.savefig(stream, format=extension, dpi=240, metadata=metadata)
        payloads[f"{STEM}.{extension}"] = stream.getvalue()
    plt.close(fig)
    return payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Recompute without rewriting and compare saved data and figures.")
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    points, cells, sources, protocol = collect()
    payloads = render(points)
    receipt = {
        "figure_id": "native-physical-alias-hot-cold-v1-20261016", "gpu_execution": False,
        "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(Path(__file__).read_bytes())},
        "environment": {"python": sys.version, "executable": sys.executable,
                        "matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": matplotlib.get_backend()},
        "source_files": sources,
        "figures": {str((OUTPUT / name).relative_to(ROOT)): {"sha256": sha(data), "bytes": len(data)} for name, data in payloads.items()},
        "statistic": {"point": "Median of 60 within-block/order paired private/shared latency ratios",
                      "interval": "Percentile bootstrap of 30 blocks, retaining both random/reverse observations",
                      "bootstrap_draws": 10000, "bootstrap_seed": protocol["bootstrap_seed"],
                      "recomputed_from_raw_and_matched_raw_summary_and_root_analysis": True},
        "limits": ["Different GPU for each document/budget cell; within-cell intervention only.",
                   "No lines or causal threshold fitted across budgets.",
                   "Hot repeats one working set; cold is a single call following flush.",
                   "Cold event quantization is 2.048 us; CI omits systematic/quantization uncertainty and can collapse.",
                   "Index installation excluded; output copy included; no application, L2-hit or DRAM-byte claim."],
        "cells": cells, "plotted_points": points,
    }
    receipt_path = OUTPUT / "figure_receipt.json"
    if args.check:
        assert json.loads(receipt_path.read_text()) == receipt
        for name, data in payloads.items():
            assert (OUTPUT / name).read_bytes() == data, name
        print("All 48 points/intervals match; source receipt and PNG/PDF are byte-reproducible.")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, data in payloads.items():
            (OUTPUT / name).write_bytes(data)
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved {len(points)} points to {OUTPUT.relative_to(ROOT)} (PNG, PDF, receipt).")


if __name__ == "__main__":
    main()
