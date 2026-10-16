"""Export all development configurations; no cherry-picked application claim."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
paths = ["qmsum_dense_joint_train8.json", "qmsum_rag_train8.json", "qmsum_rag_instruct_train8.json"]
rows = []
for name in paths:
    report = json.loads((ROOT / "results/application_calibrated" / name).read_text())
    for key, arm in report["arms"].items():
        model, variant = key.split("__")
        rows.append({"model": model, "arm": variant, "documents": arm["documents"], "questions": arm["questions"],
            "rouge1": arm["macro_f1_percent"]["1"], "rouge2": arm["macro_f1_percent"]["2"], "rougeL": arm["macro_f1_percent"]["L"],
            "mean_group_generation_seconds": arm["total_group_jct_seconds"] / arm["documents"],
            "mean_group_composed_seconds": arm["total_measured_cold_phases_seconds"] / arm["documents"],
            "empty_answers": arm["empty_answers"], "groups_hitting_cap": arm["groups_hitting_cap"],
            "cost_scope": "generation plus measured input preparation and serial retrieval; startup/harness separate" if variant.startswith("rag_") else "generation JCT plus measured input preparation; startup/harness separate"})
out = ROOT / "results/application_calibrated"
with (out / "development_grid.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
for ax, suffix, color in zip(axes, ["Instruct", "Thinking"], ["#176c9c", "#ad5125"]):
    subset = [r for r in rows if suffix in r["model"]]
    rag = [r for r in subset if r["arm"].startswith("rag_")]
    ax.scatter([r["mean_group_composed_seconds"] for r in rag], [r["rougeL"] for r in rag],
               marker="o", s=45, color=color, alpha=.65, label="RAG: all 8 development settings")
    for arm, marker in [("dense", "s"), ("joint", "^")]:
        row = next(r for r in subset if r["arm"] == arm)
        x, y = row["mean_group_composed_seconds"], row["rougeL"]
        ax.scatter([x], [y], marker=marker, s=85, color=color, edgecolors="black", linewidths=.5, label=arm)
        ax.annotate(arm, (x, y), xytext=(5, 7), textcoords="offset points")
    ax.set_title("Qwen3-4B-" + suffix + "-2507")
    ax.set_xlabel("Mean measured phases per document (seconds)\nGeneration + input preparation + RAG preparation")
    ax.grid(alpha=.2)
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("Original Perl ROUGE-L, macro F1 (%)")
fig.suptitle("QMSum train development: 8 documents, 32 questions, one seed", fontsize=12)
fig.text(.5, .015, "Eager engine; unoptimized screening. Different model-recommended sampling. ROUGE is not factual correctness or noninferiority.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, .05, 1, .96))
fig.savefig(out / "development_grid.png", dpi=180)
fig.savefig(out / "development_grid.pdf")
print(json.dumps({"configurations": len(rows), "answers": sum(r["questions"] for r in rows), "empty_answers": sum(r["empty_answers"] for r in rows), "cap_groups": sum(r["groups_hitting_cap"] for r in rows)}))
