"""Score frozen development outputs using the calibrated original Perl ROUGE."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

from calibrate_qmsum_rouge import score_pairs

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    grouped, sources = collections.defaultdict(list), []
    for pattern in args.inputs:
        for path in sorted(ROOT.glob(pattern)):
            data = json.loads(path.read_text())
            if "runs" not in data:
                continue
            assert data["status"] == "complete", f"Cannot exclude incomplete report {path}"
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest_path = ROOT / data["manifest"]
            assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == data["manifest_sha256"]
            manifest = json.loads(manifest_path.read_text())
            for run in data["runs"]:
                if run["cache"] != "cold" or run["repeat"] != 0:
                    continue
                group = manifest["groups"][run["group_index"]]
                assert run["question_ids"] == [q["question_sha256"] for q in group["questions"]]
                key = Path(data["model"]).name + "__" + run["arm"]
                answers = run.get("answers", [""] * group["B"])
                assert len(answers) == group["B"]
                input_preparation = run["tokenization_seconds_shared_preparation"] + run.get("arm_tokenization_seconds", 0.)
                measured_phase_cost = (run.get("composed_serial_cold_pipeline_seconds", run["group_jct_seconds"] + run.get("arm_tokenization_seconds", 0.))
                                       + run["tokenization_seconds_shared_preparation"])
                for i, (q, answer) in enumerate(zip(group["questions"], answers)):
                    grouped[key].append({"document_id": group["document_id"], "group_index": run["group_index"],
                        "question_index": i, "question_id": q["question_sha256"],
                        "prediction": answer, "reference": q["answer"], "status": run["status"],
                        "cap_hit": run.get("any_cap_hit"), "group_jct_seconds": run.get("group_jct_seconds"),
                        "composed_serial_cold_pipeline_seconds": run.get("composed_serial_cold_pipeline_seconds"),
                        "composed_serial_cached_index_pipeline_seconds": run.get("composed_serial_cached_index_pipeline_seconds"),
                        "generation_input_preparation_seconds": input_preparation,
                        "measured_cold_phases_seconds": measured_phase_cost,
                        "parse_status": run.get("answer_parse_status")})
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"scope": "QMSum frozen train development screening; single seed; not test noninferiority or factual correctness",
        "sources": sources, "metric": "original ROUGE-1.5.5 with author-linked flags and calibrated sentence splitting",
        "calibration": "results/metric_calibration_qmsum.json",
        "calibration_sha256": hashlib.sha256((ROOT / "results/metric_calibration_qmsum.json").read_bytes()).hexdigest(),
        "calibration_limit": "Published fixture differs by up to 0.011 score points; not claimed bitwise reproduction of unpublished author environment",
        "timing_limit": "Eager unless source says otherwise. measured_cold_phases charges shared full-input tokenization once per document for each standalone arm plus arm preparation, generation, and separately measured RAG preparation. Initialization, warmup, serialization, and harness overhead remain separate; not an integrated pipeline speedup claim. Historical composed_serial fields omitted shared generator-input preparation and are retained with this correction.",
        "arms": {}}
    for key, rows in grouped.items():
        rows.sort(key=lambda x: (x["group_index"], x["question_index"]))
        assert len({(r["document_id"], r["question_id"]) for r in rows}) == len(rows)
        measured = score_pairs([r["prediction"] for r in rows], [r["reference"] for r in rows],
                               output_prefix=output.parent / "perl" / output.stem / key)
        for row, score in zip(rows, measured["per_example"]):
            row["rouge"] = {k: v for k, v in score.items() if k != "index"}
        docs = {}
        for row in rows:
            docs.setdefault(row["document_id"], row)
        aggregate = {"documents": len(docs), "questions": len(rows),
            "empty_answers": sum(not r["prediction"].strip() for r in rows),
            "groups_hitting_cap": sum(bool(r["cap_hit"]) for r in docs.values()),
            "median_group_jct_seconds": statistics.median(r["group_jct_seconds"] for r in docs.values()),
            "total_group_jct_seconds": sum(r["group_jct_seconds"] for r in docs.values()),
            "total_measured_cold_phases_seconds": sum(r["measured_cold_phases_seconds"] for r in docs.values()),
            "official_bootstrap_average_percent": measured["reported_average_percent"],
            "macro_f1_percent": measured["per_case_macro_f1_percent_from_printed_5dp"], "per_question": rows}
        for cost in ["composed_serial_cold_pipeline_seconds", "composed_serial_cached_index_pipeline_seconds"]:
            if all(r[cost] is not None for r in docs.values()):
                aggregate["total_" + cost] = sum(r[cost] for r in docs.values())
        result["arms"][key] = aggregate
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"arm": key, "questions": len(rows), "macro_f1_percent": aggregate["macro_f1_percent"]}), flush=True)


if __name__ == "__main__":
    main()
