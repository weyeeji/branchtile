"""Calibrated original Perl ROUGE for complete frozen A4 development cohorts."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

from calibrate_qmsum_rouge import score_pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    cohorts=collections.defaultdict(list);sources=[];expected={};seen=collections.defaultdict(set)
    for fn in a.inputs:
        data=json.loads(Path(fn).read_text())
        assert data["status"]=="completed" and data["mode"]=="natural", "Do not silently discard unfinished/qualification runs"
        manifest=json.loads(Path(data["manifest"]).read_text())
        assert sha(data["manifest"])==data["manifest_sha256"]
        sources.append({"path":fn,"sha256":sha(fn),"model_revision":data["model_revision"]})
        for g in data["groups"]:
            gi=g["group_index"];gold=manifest["groups"][gi]
            assert g["document_sha256"]==gold["document_sha256"]
            expected[gi]=set(range(g["original_natural_B"]))
            for q in g["questions"]:
                qi=q["question_index"]
                assert q["status"]=="completed" and q["question_sha256"]==gold["questions"][qi]["question_sha256"]
                assert qi not in seen[gi],"Duplicate question source"
                seen[gi].add(qi)
                for variant in q["variants"]:
                    assert variant["status"]=="completed"
                    cohorts[str(variant["fraction"])].append({
                        "group_index":gi,"question_index":qi,"question_sha256":q["question_sha256"],
                        "prediction":variant["generated_text"],"reference":gold["questions"][qi]["answer"],
                        "natural_eos":variant["natural_eos"],"hit_cap":variant["hit_cap"],
                        "generated_tokens":variant["generated_tokens"],"budget_tokens":variant["budget_tokens"],
                        "decode_seconds":variant["decode_seconds"],"gather_seconds":variant["gather_seconds"],
                        "topk_and_support_seconds":variant["topk_and_support_seconds"],
                        "selector_observer_seconds":q["selector_observer_seconds"]})
    assert all(seen[gi]==ids for gi,ids in expected.items()),"Missing frozen natural questions"
    expected_pairs={(gi,qi) for gi,ids in expected.items() for qi in ids}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    result={"scope":"Frozen QMSum train development only, one seed, official 4B Instruct boundary model; not test noninferiority or main-model quality proof",
            "sources":sources,"metric":"calibrated original ROUGE-1.5.5 author-linked flags",
            "calibration":"results/metric_calibration_qmsum.json",
            "calibration_sha256":sha("results/metric_calibration_qmsum.json"),
            "metric_limit":"Published calibration fixture differs by up to 0.011 score points",
            "cost_limit":"HF instrumented reference costs, including physical per-head gather. No serving latency or speedup claim.",
            "cohorts":{}}
    for fraction,rows in sorted(cohorts.items(),key=lambda x:float(x[0])):
        rows.sort(key=lambda r:(r["group_index"],r["question_index"]))
        assert {(r["group_index"],r["question_index"]) for r in rows}==expected_pairs
        measured=score_pairs([r["prediction"] for r in rows],[r["reference"] for r in rows],
                    output_prefix=output.parent/"perl"/(output.stem+"_fraction"+fraction))
        for r,s in zip(rows,measured["per_example"]):
            r["rouge"]={k:v for k,v in s.items() if k!="index"}
        result["cohorts"][fraction]={"documents":len(expected),"questions":len(rows),
            "natural_eos":sum(r["natural_eos"] for r in rows),"cap_hits":sum(r["hit_cap"] for r in rows),
            "total_generated_tokens":sum(r["generated_tokens"] for r in rows),
            "macro_f1_percent":measured["per_case_macro_f1_percent_from_printed_5dp"],
            "official_bootstrap_average_percent":measured["reported_average_percent"],
            "median_reference_decode_seconds":statistics.median(r["decode_seconds"] for r in rows),
            "per_question":rows}
        print(json.dumps({"fraction":fraction,"macro_f1_percent":result["cohorts"][fraction]["macro_f1_percent"]}),flush=True)
    result["delta_vs_full_budget_points"]={f:{m:c["macro_f1_percent"][m]-result["cohorts"]["1.0"]["macro_f1_percent"][m] for m in ["1","2","L"]} for f,c in result["cohorts"].items()}
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
