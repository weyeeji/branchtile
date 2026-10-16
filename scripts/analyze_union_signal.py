#!/usr/bin/env python
"""CPU-only, per-capture analysis of the latest fixed-union/native replays.

Run from the repository with .venv/bin/python scripts/analyze_union_signal.py.
Missing or incompletely written inputs remain pending; rerun after jobs finish.
The statistic is the median of paired latency ratios, not a ratio of medians.
Bootstrap units are timing blocks, keeping random/reverse observations together.
"""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "official4b_boundary": (
        "results/latest4b_paired",
        ("latest4b_qmsum0_l0_b4096_s1_allheads",
         "latest4b_qmsum0_l35_b4096_s1_allheads",
         "latest4b_qmsum3_l23_b4096_s256_allheads",
         "latest4b_qmsum7_l35_b4096_s256_allheads")),
    "old30b_diagnostic": (
        "results/latest_old30b_paired",
        ("latest_old30b_qmsum3_l31_b4096_s256",)),
    "synthetic": (
        "results/latest_synthetic_paired",
        ("latest_v2_cross_n16384_g8", "latest_v2_cross_n131072_g8",
         "latest_v2_mixed_n65536_g8")),
}
SCOPES = {
    "official4b_boundary": "Official Qwen3-4B-Thinking-2507 boundary-model operator replay; not the main GQA8 model or an application trial.",
    "old30b_diagnostic": "Old Qwen3-30B-A3B BF16 diagnostic replay; not Qwen3-30B-A3B-Thinking-2507-FP8.",
    "synthetic": "Synthetic operator shapes; no independent natural document or model-quality observation.",
}
PAIRS = {
    "native_raw_over_union": ("native_raw", "union"),
    "native_over_union": ("native", "union"),
    "fixed_over_fixed_AA": ("fixed", "fixed_AA"),
}


class PendingInput(ValueError):
    """A producer has not yet supplied a complete analyzable record."""


def paired_ratio(values, draws):
    """values[block, direction]; identical block draws may be reused per arm."""
    values = np.asarray(values, dtype=np.float64)
    boot = np.median(values[draws], axis=(1, 2))
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "median": float(np.median(values)),
        "paired_block_bootstrap95": [float(low), float(high)],
        "interval_relation_to_one": "above" if low > 1 else "below" if high < 1 else "includes",
    }


def analyze_cache(measurements, cache, resamples, seed):
    rows = [row for row in measurements if row.get("cache") == cache]
    if not rows:
        raise PendingInput(f"missing {cache} measurements")
    blocks = collections.defaultdict(dict)
    for row in rows:
        if "block" not in row or "direction" not in row:
            raise PendingInput(f"incomplete {cache} block metadata")
        block, direction = row["block"], row["direction"]
        if direction not in ("random", "reverse") or direction in blocks[block]:
            raise ValueError(f"duplicate/unknown {cache} block direction: {block}, {direction}")
        blocks[block][direction] = row
    block_ids = sorted(blocks)
    if len(block_ids) < 2:
        raise PendingInput(f"fewer than two complete {cache} blocks")
    methods = sorted({method for pair in PAIRS.values() for method in pair})
    times = {method: [] for method in methods}
    for block in block_ids:
        pair = blocks[block]
        if set(pair) != {"random", "reverse"}:
            raise PendingInput(f"incomplete {cache} random/reverse block {block}")
        orders = [pair[d].get("order") for d in ("random", "reverse")]
        if any(order is None for order in orders):
            raise PendingInput(f"missing method order in {cache} block {block}")
        if list(reversed(orders[0])) != orders[1]:
            raise ValueError(f"method order is not reversed in {cache} block {block}")
        for method in methods:
            try:
                values = [pair[d]["us"][method] for d in ("random", "reverse")]
            except KeyError as exc:
                raise PendingInput(f"missing {method} timing in {cache} block {block}") from exc
            if any(not np.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"nonpositive/nonfinite {method} timing in {cache} block {block}")
            times[method].append(values)
    arrays = {method: np.asarray(values, dtype=np.float64) for method, values in times.items()}
    draws = np.random.default_rng(seed).integers(0, len(block_ids), size=(resamples, len(block_ids)))
    statistics = {name: paired_ratio(arrays[a] / arrays[b], draws) for name, (a, b) in PAIRS.items()}
    aa = arrays["fixed"] / arrays["fixed_AA"]
    statistics["fixed_over_fixed_AA"]["p95_absolute_ratio_deviation_from_one"] = float(np.quantile(np.abs(aa - 1), 0.95))
    return {
        "paired_blocks": len(block_ids), "paired_observations": len(rows), "block_ids": block_ids,
        "median_latency_us": {method: float(np.median(values)) for method, values in arrays.items()},
        "ratios": statistics,
        "warnings": [] if len(block_ids) >= 30 else ["Fewer than the planned 30 paired blocks; exploratory partial protocol."],
    }


def case_label(path, family, provenance, q_shape=None):
    if family == "synthetic":
        pattern = provenance.get("pattern", "mixed" if "mixed" in path else "cross")
        tokens = path.split("_n")[-1].split("_")[0]
        length = f"{int(tokens) // 1024}K" if tokens.isdigit() else "?"
        return f"Synthetic {pattern} {length} GQA8"
    capture = Path(provenance.get("capture", "")).name
    if not capture:
        capture = next((part for part in Path(path).stem.split("_") if part.startswith("qmsum")), "capture")
    layer = provenance.get("layer", next((part[1:] for part in Path(path).stem.split("_") if part.startswith("l") and part[1:].isdigit()), "?"))
    step = provenance.get("step", next((part[1:] for part in Path(path).stem.split("_") if part.startswith("s") and part[1:].isdigit()), "?"))
    prefix = "Official 4B" if family == "official4b_boundary" else "Old 30B"
    return f"{prefix} {capture} L{layer} s{step}"


def analyze_file(root, relative_path, family, resamples, seed):
    entry = {"file": relative_path, "model_scope": family, "scope": SCOPES[family], "status": "pending",
             "label": case_label(relative_path, family, {})}
    path = root / relative_path
    if not path.exists():
        entry["reason"] = "expected raw file not present"
        return entry
    payload = path.read_bytes()
    entry["source_sha256"] = hashlib.sha256(payload).hexdigest()
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        entry["reason"] = f"raw file may still be writing: {exc}"
        return entry
    if not isinstance(data, dict):
        entry.update(status="invalid", reason="raw JSON must be an object")
        return entry
    for key in ("provenance", "packages", "q_shape", "k_shape", "scope", "gpu_uuid",
                "native_lse_contract", "native_raw_lse_contract", "selected", "limitations"):
        entry["raw_scope" if key == "scope" else key] = data.get(key)
    entry["label"] = case_label(relative_path, family, data.get("provenance", {}))
    entry["tuning_audit"] = {}
    tuning = data.get("tuning", [])
    for method in sorted({record.get("method", "unknown") for record in tuning}):
        records = [record for record in tuning if record.get("method", "unknown") == method]
        failed = [record for record in records if record.get("status") != "ok"]
        entry["tuning_audit"][method] = {
            "qualified_configurations": sum(record.get("status") == "ok" for record in records),
            "failed_configurations": len(failed),
            "failed_reason_counts": dict(collections.Counter(record.get("error", "missing failure reason") for record in failed)),
            "failed_records": failed,
        }
    try:
        if not data.get("q_shape") or not data.get("k_shape"):
            raise PendingInput("q_shape or k_shape metadata missing")
        q, k = data["q_shape"], data["k_shape"]
        if len(q) != 3 or len(k) != 3 or q[0] != k[0] or q[2] != k[2] or any(v <= 0 for v in (*q, *k)):
            raise ValueError(f"invalid Q/K shapes: {q}, {k}")
        if not data.get("provenance") or not data.get("packages"):
            raise PendingInput("model/input provenance or package versions missing")
        if data["packages"].get("flashinfer-python") != "0.7.2":
            raise ValueError("expected FlashInfer 0.7.2; preserve this result separately from the latest comparison")
        if data.get("native_lse_contract") != "natural log; conversion timed":
            raise ValueError("native arm does not declare timed natural-log LSE conversion")
        if data.get("native_raw_lse_contract") != "native log2; conversion only in validation":
            raise ValueError("native_raw arm does not declare original log2 LSE interface")
        if not isinstance(data.get("measurements"), list):
            raise PendingInput("measurements missing")
        stable_seed = (seed + int(hashlib.sha256(relative_path.encode()).hexdigest()[:8], 16)) % (2**32)
        entry["cache"] = {cache: analyze_cache(data["measurements"], cache, resamples, stable_seed + i)
                          for i, cache in enumerate(("hot", "cold"))}
        entry["status"] = "complete"
    except PendingInput as exc:
        entry["reason"] = str(exc)
    except (ValueError, TypeError, KeyError) as exc:
        entry["status"] = "invalid"
        entry["reason"] = str(exc)
    return entry


def draw_figure(entries, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(17, max(6.4, len(entries) * .62 + 2.1)), sharey=True)
    styles = (("native_raw_over_union", "#2369A1", "o", "Native raw (log2) / union"),
              ("native_over_union", "#D27822", "s", "Native + ln conversion / union"))
    offsets = (-.14, .14)
    for ax, cache in zip(axes[:2], ("hot", "cold")):
        bounds = [1.0]
        for y, entry in enumerate(entries):
            if entry["status"] != "complete":
                ax.text(1.0, y, entry["status"], color="#777777", ha="center", va="center", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
                continue
            for offset, (metric, color, marker, _) in zip(offsets, styles):
                stat = entry["cache"][cache]["ratios"][metric]
                mid, (low, high) = stat["median"], stat["paired_block_bootstrap95"]
                ax.hlines(y + offset, low, high, color=color, linewidth=1.5)
                ax.plot(mid, y + offset, marker=marker, color=color, markersize=5)
                bounds.extend((low, high, mid))
        spread = max(max(bounds) - min(bounds), .2)
        ax.set_xlim(max(0, min(bounds) - .15 * spread), max(bounds) + .15 * spread)
        ax.set_xlabel("Native latency / fixed-union latency")
        ax.set_title(f"{cache.capitalize()} cache: paired median and 95% CI")
    aa_styles = (("hot", "#30815E", "o"), ("cold", "#8559A5", "s"))
    bounds = [0.995, 1.005]
    for y, entry in enumerate(entries):
        if entry["status"] != "complete":
            axes[2].text(1, y, entry["status"], color="#777777", ha="center", va="center", fontsize=9,
                         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
            continue
        for offset, (cache, color, marker) in zip(offsets, aa_styles):
            stat = entry["cache"][cache]["ratios"]["fixed_over_fixed_AA"]
            mid, (low, high) = stat["median"], stat["paired_block_bootstrap95"]
            axes[2].hlines(y + offset, low, high, color=color, linewidth=1.5)
            axes[2].plot(mid, y + offset, marker=marker, color=color, markersize=5)
            bounds.extend((low, high, mid))
    spread = max(bounds) - min(bounds)
    axes[2].set_xlim(min(bounds) - .15 * spread, max(bounds) + .15 * spread)
    axes[2].ticklabel_format(useOffset=False, axis="x")
    axes[2].set_title("Fixed-row A/A timing control")
    axes[2].set_xlabel("Fixed latency / repeated-fixed latency")
    axes[2].legend(handles=[Line2D([], [], marker=m, color=c, linestyle="none", label=cache.capitalize())
                            for cache, c, m in aa_styles], loc="lower right", fontsize=9)
    for ax in axes:
        ax.axvline(1, color="#333333", linestyle="--", linewidth=1, zorder=0)
        ax.grid(axis="x", color="#eeeeee")
        ax.set_axisbelow(True)
        for i in range(1, len(entries)):
            if entries[i]["model_scope"] != entries[i - 1]["model_scope"]:
                ax.axhline(i - .5, color="#dddddd", linewidth=1)
        ax.set_ylim(len(entries) - .5, -.5)
    axes[0].set_yticks(range(len(entries)), [entry["label"] for entry in entries], fontsize=9)
    fig.suptitle("Existing fixed union vs qualified native decode: engineering signal only", fontsize=13, y=.98)
    fig.legend(handles=[Line2D([], [], marker=m, color=c, linestyle="none", label=label)
                        for _, c, m, label in styles], loc="upper center", bbox_to_anchor=(.55, .94), ncol=2, fontsize=10)
    fig.text(.02, .025, "Ratios > 1 favor fixed union. 95% intervals resample timing blocks, retaining random/reverse pairs; snapshots are not independent documents.\n"
             "Native arms exclude numerically rejected configurations; stronger joint-prefill custom-mask control is pending. No novelty or application claim.\n"
             "Raw native returns log2 LSE; native + ln includes conversion. Cold timestamps may be quantized. Pending files are not successes.", fontsize=9)
    fig.subplots_adjust(left=.22, right=.985, top=.85, bottom=.15, wspace=.25)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(prefix.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", default="results/summary/union_signal_latest.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    root = args.root.resolve()
    entries = []
    for family, (folder, stems) in EXPECTED.items():
        paths = [f"{folder}/{stem}.json" for stem in stems]
        paths += sorted(str(p.relative_to(root)) for p in (root / folder).glob("*.json")
                        if str(p.relative_to(root)) not in paths)
        entries.extend(analyze_file(root, path, family, args.bootstrap, args.seed) for path in paths)
    statuses = collections.Counter(entry["status"] for entry in entries)
    documents = collections.defaultdict(set)
    for entry in entries:
        provenance = entry.get("provenance") or {}
        if entry["status"] == "complete" and provenance.get("document_sha256"):
            documents[(entry["model_scope"], provenance.get("model_revision"))].add(provenance["document_sha256"])
    output = root / args.output
    report = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "complete" if statuses["complete"] == len(entries) else "incomplete",
        "method": {"statistic": "Median of per-observation paired numerator/denominator latency ratios.",
                   "resampling_unit": "Within-file and cache paired timing block; random/reverse remain together.",
                   "bootstrap_resamples": args.bootstrap, "bootstrap_seed": args.seed,
                   "interval": "Percentile 95%; conditional within-run timing uncertainty, not document-population uncertainty.",
                   "numpy_version": np.__version__, "python": sys.executable},
        "interpretation": {"ratio_above_one": "Fixed union is faster than the named native interface.",
                           "native_raw": "Original log2-LSE API; the stricter comparison for a union engineering benefit because it adds no conversion work to native.",
                           "native": "Natural-log LSE with conversion timed; an interface-matched comparison.",
                           "dynamic_v2": "The original dynamic-row proposal remains no-go in the completed feasibility study; this analysis does not reopen it.",
                           "native_comparator_scope": "Best tested configurations qualified under the existing O/LSE tolerances; not an unrestricted claim over all FlashInfer implementations or configurations.",
                           "stronger_joint_paged_prefill_control": "Not included here; a native joint-query custom-mask interface may cover the same fixed-union mechanism and must be evaluated before a superiority claim.",
                           "claim_scope": "Existing masked-union execution engineering signal only; no novelty, application quality or net application benefit claim."},
        "case_counts": {"planned": sum(len(stems) for _, stems in EXPECTED.values()), "discovered_including_planned": len(entries), **statuses},
        "distinct_captured_documents_by_model_scope": [
            {"model_scope": scope, "model_revision": revision, "document_count": len(ids), "document_sha256": sorted(ids)}
            for (scope, revision), ids in documents.items()],
        "independence_note": "Layers, steps, cache regimes and repeated timing blocks are not additional documents. No across-file pooled significance test or quality estimate is computed.",
        "limitations": ["Cold single-call timestamp quantization can produce narrow or degenerate intervals.",
                        "A/A is reported per file and cache, not used to manufacture a cross-document confidence claim.",
                        "Raw provenance, selected tuning configurations and scope are preserved per file.",
                        "Numerically rejected native/native_raw configurations are excluded by the benchmark; their counts and full failure reasons are preserved here.",
                        "There is no separate formal artifact establishing native-calibrated freezing of the O atol=0.006, rtol=0.015 tolerance; do not claim unrestricted numerical equivalence.",
                        "Missing files/shapes/pairs stay pending; malformed completed records are invalid, never silently dropped."],
        "files": entries,
        "figures": [str(output.with_suffix(ext).relative_to(root)) if output.is_relative_to(root) else str(output.with_suffix(ext)) for ext in (".png", ".pdf")],
    }
    draw_figure(entries, output.with_suffix(""))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "analysis_status": report["analysis_status"], "case_counts": report["case_counts"]}))


if __name__ == "__main__":
    main()
