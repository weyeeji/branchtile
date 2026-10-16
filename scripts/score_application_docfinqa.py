"""Score complete financial-answer runs without dropping service failures."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from financial_answer_metric import score_pairs, METRIC_ID


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source = ROOT / args.input
    data = json.loads(source.read_text())
    assert data["status"] == "complete", "Incomplete generation is not a complete quality denominator"
    manifest_path = ROOT / data["manifest"]
    assert sha(manifest_path) == data["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    grouped = collections.defaultdict(list)
    for run in data["runs"]:
        if run["cache"] != "cold" or run["repeat"] != 0:
            continue
        group = manifest["groups"][run["group_index"]]
        assert group["dataset"] == "DocFinQA"
        assert run["question_ids"] == [q["question_sha256"] for q in group["questions"]]
        answers = run.get("answers", [""] * group["B"])
        assert len(answers) == group["B"]
        for question, answer in zip(group["questions"], answers):
            grouped[run["arm"]].append({"document_id": group["document_id"], "group_index": run["group_index"],
                "question_id": question["question_sha256"], "prediction": answer, "gold": question["answer"],
                "source_answer_variants": question.get("source_answer_variants", [question["answer"]]),
                "conflicting_source_answers": question.get("conflicting_source_answers", False),
                "generation_status": run["status"], "cap_hit": run.get("any_cap_hit"),
                "group_jct_seconds": run.get("group_jct_seconds"),
                "composed_serial_cold_pipeline_seconds": run.get("composed_serial_cold_pipeline_seconds")})
    expected_indices = [int(x) for x in data["config"]["groups"].split(",")]
    expected_count = sum(manifest["groups"][i]["B"] for i in expected_indices)
    result = {"source": args.input, "source_sha256": sha(source), "manifest": data["manifest"],
        "manifest_sha256": sha(manifest_path), "metric_id": METRIC_ID,
        "metric_source_sha256": sha(ROOT / "src/financial_answer_metric.py"),
        "metric_protocol": "results/metric_calibration_docfinqa.md",
        "scope": "Frozen development, one seed. This FINAL-answer metric is not official FinQA/DocFinQA program-execution accuracy, factual audit, or formal noninferiority.",
        "primary": "exact_match", "mandatory_sensitivity": "rounded_match",
        "denominator_rule": "All configured questions including unserviceable context and parse/cap failures. Gold parse failures remain included.",
        "expected_questions_per_arm": expected_count, "arms": {}}
    for arm, rows in grouped.items():
        assert len(rows) == expected_count
        assert len({(r["document_id"], r["question_id"]) for r in rows}) == len(rows)
        scored = score_pairs([r["prediction"] for r in rows], [r["gold"] for r in rows])
        for row, score in zip(rows, scored.pop("per_example")):
            row["score"] = score
        docs = {r["document_id"]: r for r in rows}
        scored.update(documents=len(docs), per_question=rows,
            generation_status_question_counts=dict(collections.Counter(r["generation_status"] for r in rows)),
            cap_hit_documents=sum(bool(r["cap_hit"]) for r in docs.values()),
            serviced_group_jct_seconds=sum(r["group_jct_seconds"] or 0 for r in docs.values()))
        result["arms"][arm] = scored
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({arm: {k: v for k, v in scores.items() if k != "per_question"} for arm, scores in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
