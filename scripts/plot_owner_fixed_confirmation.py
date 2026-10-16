#!/usr/bin/env python3
"""CPU-only PNG/PDF of wave25 fixed-owner paired speed ratios, without pooling."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RATIO_KEY = "best_owner_control_over_owner_frozen_128_64"
FROZEN = {"method": "dynamic_owner_tasks", "version": "owner_tasks", "splits": 64,
          "block_n": 64, "threads": 128, "chunk_tokens": 128}
STATES = [(rank, layer, step) for rank in (0, 1)
          for layer, step in ((0, 16), (15, 256), (31, 16), (47, 256))]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", default="results/operator_overhead_wave25")
    parser.add_argument("--output-directory", default="results/operator_owner_confirmation_figure_v1")
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Run with no visible GPU"
    assert "torch" not in sys.modules and "triton" not in sys.modules
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    directory, output = ROOT / args.input_directory, ROOT / args.output_directory
    assert not output.exists(), "Preserve previous figure artifacts; choose a fresh directory"
    inputs = sorted(directory.glob("overhead_w25*.json"))
    assert len(inputs) == 8, "Use exactly the eight completed wave25 inputs"
    audit_path = ROOT / "results/operator_owner_native_wave25_audit_v1/audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "complete_fixed_owner_native_wave25_CPU_audit_passed"
    assert audit["qualified_configurations"] == 384 and audit["native_control_count_per_job"] == 47
    assert sha(ROOT / "scripts/audit_owner_native_wave25.py") == audit["auditor_sha256"]

    states, documents, source_hashes, plotted = {}, set(), None, []
    raw_hashes = {}
    draws = np.random.default_rng(71).integers(0, 40, (10000, 40))
    for path in inputs:
        data = json.loads(path.read_text())
        assert data["protocol_version"] == "owner-confirm-native-v1"
        assert data["batch"] == 4 and data["gqa"] == 8
        assert data["tuning_planned"] == data["tuning_completed"] == len(data["tuning"]) == 48
        assert data["tuning_complete"] and all(row["status"] == "ok" for row in data["tuning"])
        assert all(row["payload_aliases_source"] is True for row in data["tuning"])
        assert all(not values for values in data["missing_qualified_methods"].values())
        assert data["arguments"]["blocks"] == 40 and data["arguments"]["budget"] == 4096
        assert data["frozen_owner_confirmation"]["config"] == FROZEN
        provenance = data["provenance"]
        assert provenance["ready_lifetime_measured"]
        assert provenance["original_active_requests"] == [0, 1, 2, 3]
        state = provenance["actual_tp_rank"], provenance["layer"], provenance["step"]
        assert state in STATES and state not in states
        documents.add(provenance["document_sha256"])
        if source_hashes is None:
            source_hashes = data["code_sha256"]
        assert source_hashes == data["code_sha256"]
        raw_hashes[str(path.relative_to(ROOT))] = sha(path)
        states[state] = data
        for cache in ("hot", "cold"):
            assert data["selected"][cache]["owner_frozen_128_64"]["config"] == FROZEN
            rows = [row for row in data["measurements"] if row["cache"] == cache]
            assert len(rows) == 80
            lookup = {(row["block"], row["direction"]): row for row in rows}
            assert set(lookup) == {(block, direction) for block in range(40) for direction in ("random", "reverse")}
            values = np.array([[lookup[block, direction]["us"]["best_owner_control"] /
                                lookup[block, direction]["us"]["owner_frozen_128_64"]
                                for direction in ("random", "reverse")] for block in range(40)])
            paired_median = float(np.median(values))
            interval = np.quantile(np.median(values[draws], axis=(1, 2)), [.025, .975])
            saved = data["summary"][cache]["paired_ratios"][RATIO_KEY]
            assert np.isclose(paired_median, saved["median"], atol=1e-12, rtol=0)
            assert np.allclose(interval, saved["paired_block_bootstrap95"], atol=1e-12, rtol=0)
            assert interval[0] <= paired_median <= interval[1]
            control = data["selected"][cache]["best_owner_control"]
            assert control["config"]["method"] in {"e0_native", "e1_native", "e1_own", "e4_native"}
            plotted.append({"input": str(path.relative_to(ROOT)), "rank": state[0], "layer": state[1],
                            "natural_anchor": state[2], "cache": cache, "paired_speed_ratio": paired_median,
                            "paired_block_bootstrap95": interval.tolist(),
                            "degenerate_interval": bool(interval[0] == interval[1]),
                            "selected_control_config": control["config"],
                            "selected_control_candidate_name": control["record"]["candidate_name"],
                            "control_marginal_median_us": data["summary"][cache]["median_us"]["best_owner_control"],
                            "owner_marginal_median_us": data["summary"][cache]["median_us"]["owner_frozen_128_64"],
                            "ready_group_actual_prediction_steps": provenance["ready_group"]["actual_prediction_steps"]})
    assert set(states) == set(STATES) and len(documents) == 1
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items())
    assert "torch" not in sys.modules and "triton" not in sys.modules

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10.5, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 6.8), sharex=True, sharey=True)
    figure.subplots_adjust(left=.205, right=.98, top=.79, bottom=.28, wspace=.075)
    colors = {0: "#0072B2", 1: "#D55E00"}
    y = np.arange(len(STATES))
    for ax, cache, title in zip(axes, ("hot", "cold"), ("Hot cache", "Artificial cold cache")):
        ax.axvline(1, color="#444444", linestyle="--", linewidth=1.05, zorder=1)
        ax.axhline(3.5, color="#BBBBBB", linewidth=.9, zorder=0)
        ax.axvline(1.075, color="#EEEEEE", linewidth=.8, zorder=0)
        for position, state in enumerate(STATES):
            point = next(row for row in plotted if (row["rank"], row["layer"], row["natural_anchor"]) == state and row["cache"] == cache)
            median = point["paired_speed_ratio"]
            low, high = point["paired_block_bootstrap95"]
            ax.errorbar(median, position, xerr=[[median-low], [high-median]],
                        fmt="o", color=colors[state[0]], markersize=6.2, markeredgecolor="white",
                        markeredgewidth=.65, elinewidth=1.35, capsize=3, capthick=1.15, zorder=3)
            mark = "†" if point["degenerate_interval"] else ""
            ax.text(1.126, position, f"{median:.3f}{mark}", ha="right", va="center", fontsize=9.4)
        ax.text(1.126, -.75, "Median", ha="right", va="center", color="#555555", fontsize=9)
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_xlim(.84, 1.135)
        ax.set_ylim(7.65, -.9)
        ax.set_xticks([.85, .90, .95, 1.00, 1.05])
        ax.set_xticklabels(["0.85", "0.90", "0.95", "1.00", "1.05"])
        ax.grid(axis="x", color="#E7E7E7", linewidth=.6, zorder=0)
        ax.set_xlabel("Control time / fixed owner time (×)", labelpad=9)
    axes[0].set_yticks(y, [f"Rank {rank}  ·  L{layer}  ·  anchor {step}" for rank, layer, step in STATES])
    axes[0].tick_params(axis="y", length=0, pad=10)
    axes[1].tick_params(axis="y", length=0)
    figure.suptitle("Fixed owner-task confirmation on one development document", y=.965, fontsize=15, fontweight="bold")
    figure.text(.5, .913, "Wave 25 only · B4 / GQA8 · 4K retained-token budget · predeclared C128 / N64", ha="center", fontsize=11)
    figure.legend(handles=[Line2D([0],[0],marker="o",linestyle="none",color=colors[rank],label=f"TP rank {rank}",markersize=6) for rank in (0,1)] +
                          [Line2D([0],[0],color="#444444",linestyle="--",label="Parity = 1")],
                  loc="upper center", bbox_to_anchor=(.6,.877), ncol=3, frameon=False, columnspacing=2)
    caption_lines = [
        "Points: paired speed-ratio medians; bars: 95% block-bootstrap intervals (40 random/reverse blocks, 10,000 draws per cell).",
        "Controls: all 47 bounded native/E1-own/E4 configurations. Ratios >1 favor fixed owner. Complete O + natural-log LSE calls.",
        "Eight correlated cells from one document; CIs describe within-cell timing. † Degenerate cold event CIs, including [1,1], do not prove equivalence.",
        "No pooling or cross-wave best selection. Preparation/search are excluded from GPU events and reported separately in the raw results."]
    for line, height in zip(caption_lines, (.17, .135, .10, .065)):
        figure.text(.5, height, line, ha="center", fontsize=8.8)
    output.mkdir(parents=True)
    png, pdf = output/"fixed_owner_paired_speedup.png", output/"fixed_owner_paired_speedup.pdf"
    figure.savefig(png, dpi=300, facecolor="white", metadata={"Software": "Matplotlib; wave25 fixed-owner paired confirmation"})
    figure.savefig(pdf, facecolor="white", metadata={"Creator": "plot_owner_fixed_confirmation.py", "CreationDate": None, "ModDate": None})
    plt.close(figure)
    degenerate = [row for row in plotted if row["degenerate_interval"]]
    readme = """# Wave25 fixed-owner confirmation figure

`fixed_owner_paired_speedup.png` and `.pdf` show the eight completed wave25 capture cells, with separate hot and artificial-cold panels. Horizontal position is the median of paired `best_owner_control / owner_frozen_128_64` latency ratios. Above 1 favors owner. Bars are the original 95% percentile block-bootstrap intervals, independently recomputed from raw random/reverse timings, not a ratio of marginal timing medians.

The owner C128/N64/threads128 configuration is predeclared and fixed in both caches and every cell. Its comparator is the same-cell, separately hot/cold-selected oracle over all 47 bounded E0 native/E1 native/E1-own/E4 controls. No earlier wave or most favorable cross-wave point is substituted. Other dynamic engineering controls are not part of this reduced native-confirmation protocol.

These eight inputs come from one development document: TP ranks0/1 × layers0/15/31/47; layers0/31 use natural anchor16 and layers15/47 use anchor256. All four requests are present, each retaining its own actual prediction step/position. These are not eight independent documents or a formal held-out evaluation. Forty paired random/reverse timing blocks and 10,000 bootstrap draws give within-cell timing intervals, without multiplicity correction or cross-document inference.

Cold single-call event timings are coarse. A dagger marks a degenerate empirical interval, including [1,1]; its zero width is not evidence of exact equality or equivalence. The raw bounds are neither widened nor replaced. Both A/A controls remain in the raw JSON and independent audit; they are not extra document samples.

Both producer and merge, BF16 output and natural-log FP32 LSE are included in each owner call. Preparation, search/JIT and topology recapture are not included in the GPU event statistic. They cannot be inferred from this figure. No application latency, quality, lifecycle, traffic or energy claim follows.

`metadata.json` retains exact plotted numbers, every input JSON SHA256, the plot script SHA256, frozen benchmark/operator source hashes, audit/protocol-erratum hashes and output hashes. The plotter validates all 384 configuration receipts, the fixed arm, the unique one-document identity, cell coverage and raw block statistics before drawing. Original JSON files are read only.

Reproduce with a fresh output directory, normal uid and the project virtual environment:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/plot_owner_fixed_confirmation.py --output-directory results/operator_owner_confirmation_figure_recheck
```
"""
    (output/"README.md").write_text(readme)
    metadata = {"schema":"owner-fixed-confirmation-figure-v1", "plotter":str(Path(__file__).resolve().relative_to(ROOT)),
                "plotter_sha256":sha(__file__), "input_json_sha256":raw_hashes,
                "benchmark_code_sha256":source_hashes,
                "independent_audit":{"path":str(audit_path.relative_to(ROOT)),"sha256":sha(audit_path),"auditor_sha256":audit["auditor_sha256"]},
                "protocol_erratum":{"path":str((directory/"protocol_erratum.json").relative_to(ROOT)),"sha256":sha(directory/"protocol_erratum.json")},
                "artifacts_sha256":{path.name:sha(path) for path in (png,pdf,output/"README.md")},
                "matplotlib":matplotlib.__version__,"numpy":np.__version__,"python":sys.executable,
                "document_sha256":next(iter(documents)),"document_count":1,"cell_count":8,"cache_cell_count":16,
                "fixed_owner_config":FROZEN,"native_control_configurations_per_cell":47,"validated_configurations":384,
                "ratio_key":RATIO_KEY,"ratio_definition":"median of paired control/fixed-owner times; above1 favors owner",
                "CI_definition":"95% percentile from10,000 resamples of40 within-cell blocks, preserving random/reverse pair; no document pooling or multiplicity adjustment",
                "cell_order":[{"rank":r,"layer":l,"natural_anchor":s} for r,l,s in STATES],
                "degenerate_interval_count":len(degenerate),"degenerate_cold_interval_count":sum(r["cache"]=="cold" for r in degenerate),
                "cold_unit_point_interval_count":sum(r["cache"]=="cold" and r["paired_block_bootstrap95"]==[1.,1.] for r in plotted),
                "caption":caption_lines,"plotted_points":plotted,"GPU_hours":0,"torch_or_triton_imported":False,
                "scope":"Wave25 only, same1development document; complete-call steady-state GPU timing, preparation separate; not eight independent documents or application/heldout validation."}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    assert all(sha(ROOT/path)==digest for path,digest in raw_hashes.items()), "Raw results changed while plotting"
    assert "torch" not in sys.modules and "triton" not in sys.modules
    print(json.dumps({"PNG":str(png),"PDF":str(pdf),"plotted_points":len(plotted),"degenerate_cold_CIs":metadata["degenerate_cold_interval_count"],"GPU_hours":0}))


if __name__ == "__main__":
    main()
