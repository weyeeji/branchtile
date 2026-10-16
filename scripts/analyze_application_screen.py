"""Summarize complete natural-EOS development outputs with explicit scope."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="results/application_screen_v2/*.json")
    parser.add_argument("--output", default="results/application_screen_v2/summary.json")
    args = parser.parse_args()
    scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows, sources = [], []
    for path in sorted(ROOT.glob(args.inputs)):
        data = json.loads(path.read_text())
        if "runs" not in data:
            continue
        assert data["status"] == "complete", f"Incomplete file cannot be silently dropped: {path}"
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest_path = ROOT / data["manifest"]
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == data["manifest_sha256"]
        manifest = json.loads(manifest_path.read_text())
        for run in data["runs"]:
            group = manifest["groups"][run["group_index"]]
            assert run["question_ids"] == [q["question_sha256"] for q in group["questions"]]
            answers = run.get("answers", [""] * group["B"])
            assert len(answers) == group["B"]
            per_question = []
            for question, prediction in zip(group["questions"], answers):
                result = scores.score(question["answer"], prediction)
                per_question.append({"question_id": question["question_sha256"], "empty_final": not bool(prediction),
                                     **{metric: value.fmeasure * 100 for metric, value in result.items()}})
            metrics = [request["metrics"] for request in run.get("requests", []) if isinstance(request.get("metrics"), dict)]
            generated = [r["generated_tokens"] for r in run.get("requests", [])]
            per_request_tpot = [(m["last_token_ts"]-m["first_token_ts"])/(n-1)
                                for m, n in zip(metrics, generated) if n > 1]
            trace_path = ROOT / run["step_observer_trace"] if run.get("step_observer_trace") else None
            observed = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path else []
            rows.append({"file": str(path.relative_to(ROOT)), "group_index": run["group_index"],
                "document_id": run["document_id"], "arm": run["arm"], "cache": run["cache"], "repeat": run["repeat"],
                "status": run["status"], "B": group["B"], "group_jct_seconds": run.get("group_jct_seconds"),
                "any_cap_hit": run.get("any_cap_hit"), "parse_status": run.get("answer_parse_status"),
                "generated_tokens_total": sum(generated), "generated_tokens_by_request": generated,
                "ttft_seconds_by_request": [m.get("first_token_latency") for m in metrics],
                "tpot_seconds_by_request": per_request_tpot,
                "num_observer_steps": len(observed),
                "steps_with_nonfinal_output": sum(any(not r["finished"] for r in x["requests"]) for x in observed),
                "questions": per_question,
                **{metric: statistics.mean(q[metric] for q in per_question) for metric in ["rouge1", "rouge2", "rougeL"]}})
    by_arm = collections.defaultdict(list)
    paired = collections.defaultdict(dict)
    for row in rows:
        if row["cache"] == "cold" and row["repeat"] == 0:
            by_arm[row["arm"]].append(row)
            assert row["arm"] not in paired[row["document_id"]]
            paired[row["document_id"]][row["arm"]] = row
    aggregates = {}
    for arm, group_rows in by_arm.items():
        questions = [question for row in group_rows for question in row["questions"]]
        aggregates[arm] = {"documents": len(group_rows), "questions": len(questions),
                           "empty_answers": sum(q["empty_final"] for q in questions),
                           "groups_hitting_cap": sum(bool(row["any_cap_hit"]) for row in group_rows),
                           "median_group_jct_seconds": statistics.median(row["group_jct_seconds"] for row in group_rows),
                           "total_group_jct_seconds": sum(row["group_jct_seconds"] for row in group_rows),
                           "total_generated_tokens": sum(row["generated_tokens_total"] for row in group_rows),
                           **{metric: statistics.mean(q[metric] for q in questions) for metric in ["rouge1", "rouge2", "rougeL"]}}
    contrasts = []
    for document, arms in paired.items():
        if {"dense", "joint"} <= arms.keys():
            contrasts.append({"document_id": document, "group_index": arms["dense"]["group_index"],
                              "joint_over_dense_jct": arms["joint"]["group_jct_seconds"] / arms["dense"]["group_jct_seconds"],
                              "joint_minus_dense_rougeL": arms["joint"]["rougeL"] - arms["dense"]["rougeL"]})
    result = {"scope": "single-seed frozen QMSum train development; eager engine; not formal noninferiority or optimized production timing",
              "source_files": sources,
              "rouge_implementation": {"package": "rouge-score", "version": "0.1.2", "use_stemmer": True,
                  "metrics": ["rouge1 F1", "rouge2 F1", "rougeL F1"],
                  "official_qmsum_scorer_equivalence_verified": False,
                  "warning": "Diagnostic overlap metrics; exact original QMSum scoring reproduction and blind factual review remain required."},
              "ready_lifetime": "Not measured. LLM.generate supplied only final outputs to this observer; step traces record completions, not active/ready query membership.",
              "aggregates": aggregates, "document_contrasts": contrasts, "records": rows}
    if contrasts:
        rng = np.random.default_rng(20261016)
        delta = np.array([r["joint_minus_dense_rougeL"] for r in contrasts])
        bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
        result["development_paired_rougeL"] = {"mean_difference": float(delta.mean()),
            "document_bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "interpretation": "Descriptive train screening, not a frozen test confidence claim or factual correctness measure."}
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "aggregates": aggregates,
                      "development_paired_rougeL": result.get("development_paired_rougeL")}, indent=2))


if __name__ == "__main__":
    main()
