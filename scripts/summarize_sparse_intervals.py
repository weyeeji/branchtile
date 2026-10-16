"""Development-only paired intervals, retaining a warning on higher groups."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paired_statistics import paired_cluster_interval

source = ROOT / "results/sparse_reference/train8_quality.json"
data = json.loads(source.read_text())
reference = {(r["document_sha256"], r["question_sha256"]): r
             for r in data["cohorts"]["1.0"]["per_question"]}
result = {"source": str(source.relative_to(ROOT)),
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "scope": "Single-seed train8 diagnostic. Document-only clustering pending higher source-group recovery. Never formal noninferiority or factual correctness.",
          "metric": "calibrated original Perl ROUGE-L F1", "comparisons": {}}
for fraction in ["0.1", "0.25", "0.5"]:
    candidate = data["cohorts"][fraction]["per_question"]
    assert {(r["document_sha256"], r["question_sha256"]) for r in candidate} == set(reference)
    rows = []
    for row in candidate:
        ref = reference[(row["document_sha256"], row["question_sha256"])]
        assert row["seed"] == ref["seed"]
        rows.append(dict(document_id=row["document_sha256"], question_id=row["question_sha256"],
                         cluster_id=row["document_sha256"], seed=0,
                         reference=ref["rouge"]["L"]["f1"], candidate=row["rouge"]["L"]["f1"]))
    result["comparisons"][fraction] = paired_cluster_interval(rows, expected_seeds=[0])
    result["comparisons"][fraction]["replicate_label_note"] = "Replicate0 is the sole experiment seed family; resolved sampling seeds vary by question and are checked equal across paired methods."
destination = source.parent / "train8_development_intervals.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v["question_micro"] for k, v in result["comparisons"].items()}, indent=2))
