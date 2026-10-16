"""CPU-only scoring of unprofiled natural native-opportunity runs.

Uses the previously frozen metric, including its unit/parse failure rules.
Fixed-length and profiled runs are deliberately ineligible for task scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_application import extract_final
from src.financial_answer_metric import METRIC_ID, score_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    rows = []
    for name in args.inputs:
        path = Path(name)
        data = json.loads(path.read_text())
        assert data["status"] == "completed", (name, data["status"])
        manifest = ROOT / data["manifest"]
        assert sha(manifest) == data["manifest_sha256"]
        group = next(g for g in json.loads(manifest.read_text())["groups"]
                     if g["document_id"] == data["document_id"])
        assert group["document_sha256"] == data["document_sha256"]
        for run in data["runs"]:
            if not run["natural_eos_enabled"] or run["profiled"]:
                continue
            answers = sorted(run["answers"], key=lambda a: a["question_index"])
            assert [a["question_index"] for a in answers] == list(range(len(group["questions"])))
            predictions, details = [], []
            for question, answer in zip(group["questions"], answers):
                final, _, parse = extract_final(answer["text"], thinking=True)
                eligible = answer["finish_reason"] == "stop" and parse == "closed_thinking"
                predictions.append(final if eligible else "")
                details.append({"question_sha256": question["question_sha256"],
                                "gold": question["answer"], "final_text": final,
                                "thinking_parse": parse, "completion_eligible": eligible,
                                "finish_reason": answer["finish_reason"],
                                "output_tokens": len(answer["tokens"]),
                                "raw_text_sha256": hashlib.sha256(answer["text"].encode()).hexdigest()})
            score = score_pairs(predictions, [q["answer"] for q in group["questions"]])
            for detail, result in zip(details, score.pop("per_example")):
                detail["score"] = result
            assert sum(a["output_tokens"] for a in details) == run["total_output_tokens"]
            rows.append({"source": str(path), "source_sha256": sha(path),
                         "manifest": str(manifest.relative_to(ROOT)),
                         "manifest_sha256": sha(manifest),
                         "document_id": data["document_id"], "method": data["method"],
                         "phase": run["phase"], "sampling": run["sampling"],
                         "batch_size_at_admission": data["B"],
                         "generate_wall_seconds": run["complete_generate_wall_seconds"],
                         "anchor_wall_seconds": run["anchor_wall_seconds"],
                         "cold_anchor_plus_generate_seconds": run["cold_shared_document_total_wall_seconds"],
                         "total_output_tokens": run["total_output_tokens"],
                         "score": score, "answers": details})
    assert rows, "No completed, unprofiled natural run"
    report = {"status": "completed", "metric_id": METRIC_ID,
              "metric_source_sha256": sha(ROOT / "src/financial_answer_metric.py"),
              "extraction_source_sha256": sha(ROOT / "scripts/screen_application.py"),
              "analysis_source_sha256": sha(Path(__file__)), "rows": rows,
              "independent_documents": len({r["document_id"] for r in rows}),
              "limitations": ["Development/training pilot only, not held-out evaluation or quality noninferiority.",
                              "Strict and rounded FINAL matching are not official executable-program accuracy.",
                              "Different natural lengths and failures are retained; no token-normalized speedup inference.",
                              "Anchor plus generation is a sum of separately measured phases; model initialization and setup are separate."]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "rows": len(rows),
                      "independent_documents": report["independent_documents"]}))


if __name__ == "__main__":
    main()
