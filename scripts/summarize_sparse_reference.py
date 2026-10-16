"""Calibrated original Perl ROUGE for complete frozen A4 development cohorts."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

from calibrate_qmsum_rouge import score_pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    return {"count":len(values),"minimum":min(values),"median":statistics.median(values),
            "mean":statistics.mean(values),"maximum":max(values),"sum":sum(values)}


def aggregate(rows):
    cost_names=["decode_seconds","gather_seconds","topk_and_support_seconds",
                "selector_observer_seconds","question_prefill_including_instrumented_selector_seconds",
                "audit_hashing_and_checks_seconds","mask_export_seconds",
                "variant_reference_seconds_excluding_final_checkpoint"]
    return {"questions":len(rows),"natural_eos":sum(r["natural_eos"] for r in rows),
            "cap_hits":sum(r["hit_cap"] for r in rows),
            "output_parsing_counts":dict(collections.Counter(r["output_parsing"]["status"] for r in rows)),
            "thinking_delimiter_outputs":sum(r["output_parsing"]["contains_think_delimiter"] for r in rows),
            "chat_delimiter_outputs":sum(r["output_parsing"]["contains_chat_delimiter"] for r in rows),
            "generated_token_distribution":distribution([r["generated_tokens"] for r in rows]),
            "macro_f1_percent":{m:100*statistics.mean(r["rouge"][m]["f1"] for r in rows) for m in ["1","2","L"]},
            "reference_cost_seconds":{name:distribution([r[name] for r in rows if name in r])
                         for name in cost_names if any(name in r for r in rows)}}


def write_blind_audit(rows,output):
    """Keep arm identity outside the reviewer-facing source queue."""
    shuffled=rows.copy()
    random.Random(20261016).shuffle(shuffled)
    queue=[];mapping={}
    for index,row in enumerate(shuffled):
        item_id=f"A4-{index+1:04d}"
        source_path=Path(row["document_path"]).resolve()
        queue.append({"item_id":item_id,"document_path":str(source_path),
                      "document_line_start":1,"document_line_end":len(source_path.read_text().splitlines()),
                      "document_sha256":row["document_sha256"],"question":row["question"],
                      "gold_reference":row["reference"],"answer":row["prediction"]})
        mapping[item_id]={key:row[key] for key in ["source_path","group_index","question_index",
                      "fraction","seed","model_revision","question_sha256"]}
        mapping[item_id]["answer_sha256"]=hashlib.sha256(row["prediction"].encode()).hexdigest()
    queue_path=output.with_name(output.stem+"_blind_audit.jsonl")
    map_path=output.with_name(output.stem+"_blind_audit_mapping.json")
    queue_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in queue))
    map_path.write_text(json.dumps({"status":"pending; no factual verdicts assigned",
        "shuffle_seed":20261016,"items":mapping},indent=2,ensure_ascii=False)+"\n")
    return {"status":"pending; ROUGE is not a factuality assessment","items":len(queue),
            "queue":str(queue_path),"queue_sha256":sha(queue_path),
            "separate_mapping":str(map_path),"mapping_sha256":sha(map_path),
            "protocol":"Reviewers see the queue with original question, gold reference, candidate answer, and full source file/line range. All answers retained, including duplicates. Do not open arm mapping until verdicts are frozen. Planned independent LLM-assisted review is not a human evaluation."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--vllm-summary")
    p.add_argument("--vllm-raw")
    a=p.parse_args()
    cohorts=collections.defaultdict(list);sources=[];expected={};seen=collections.defaultdict(set)
    source_groups=[];generation=None;model_revision=None
    for fn in a.inputs:
        data=json.loads(Path(fn).read_text())
        assert data["status"]=="completed" and data["mode"]=="natural", "Do not silently discard unfinished/qualification runs"
        manifest=json.loads(Path(data["manifest"]).read_text())
        assert sha(data["manifest"])==data["manifest_sha256"]
        if generation is None:
            generation=data["generation"];model_revision=data["model_revision"]
        assert data["generation"]==generation and data["model_revision"]==model_revision,"Mixed sampling or model identity"
        sources.append({"path":fn,"sha256":sha(fn),"model_revision":data["model_revision"],
                        "model_initialization_seconds":data["model_initialization_seconds"],
                        "script_sha256":data["script_sha256"],"core_sha256":data["core_sha256"]})
        for g in data["groups"]:
            gi=g["group_index"];gold=manifest["groups"][gi]
            assert g["document_sha256"]==gold["document_sha256"]
            source_groups.append({"source_path":fn,"group_index":gi,
                                  "dense_prefix_prefill_seconds":g["dense_prefix_prefill_seconds"]})
            expected[gi]=set(range(g["original_natural_B"]))
            for q in g["questions"]:
                qi=q["question_index"]
                assert q["status"]=="completed" and q["question_sha256"]==gold["questions"][qi]["question_sha256"]
                assert qi not in seen[gi],"Duplicate question source"
                seen[gi].add(qi)
                for variant in q["variants"]:
                    assert variant["status"]=="completed"
                    assert all(variant[k] for k in ["all_layers_equal_physical_length","system_prefix_retained",
                               "entire_private_suffix_retained","sorted_original_positions"])
                    positions=variant["absolute_decode_positions"]
                    assert positions==list(range(q["prompt_tokens"],q["prompt_tokens"]+len(positions)))
                    assert variant["final_physical_cache_length"]==variant["initial_physical_cache_length"]+len(positions)
                    assert variant["generated_tokens"]==len(variant["generated_ids"])==len(positions)+1
                    prediction=variant["generated_text"]
                    parsing={"status":"nonempty_free_text" if prediction.strip() else "empty",
                             "characters":len(prediction),"contains_think_delimiter":any(t in prediction for t in ["<think>","</think>"]),
                             "contains_chat_delimiter":any(t in prediction for t in ["<|im_start|>","<|im_end|>"])}
                    row={
                        "group_index":gi,"question_index":qi,"question_sha256":q["question_sha256"],
                        "source_path":fn,"model_revision":data["model_revision"],"seed":variant["seed"],
                        "fraction":variant["fraction"],"document_path":gold["document_path"],
                        "document_sha256":gold["document_sha256"],"question":gold["questions"][qi]["question"],
                        "prediction":prediction,"reference":gold["questions"][qi]["answer"],"output_parsing":parsing,
                        "natural_eos":variant["natural_eos"],"hit_cap":variant["hit_cap"],
                        "generated_tokens":variant["generated_tokens"],"budget_tokens":variant["budget_tokens"],
                        "decode_seconds":variant["decode_seconds"],"gather_seconds":variant["gather_seconds"],
                        "topk_and_support_seconds":variant["topk_and_support_seconds"],
                        "selector_observer_seconds":q["selector_observer_seconds"],
                        "question_prefill_including_instrumented_selector_seconds":q["question_prefill_including_instrumented_selector_seconds"]}
                    for key in ["audit_hashing_and_checks_seconds","mask_export_seconds","variant_reference_seconds_excluding_final_checkpoint"]:
                        if key in variant:
                            row[key]=variant[key]
                    cohorts[str(variant["fraction"])].append(row)
    assert all(seen[gi]==ids for gi,ids in expected.items()),"Missing frozen natural questions"
    expected_pairs={(gi,qi) for gi,ids in expected.items() for qi in ids}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    result={"scope":"Frozen QMSum train development only, one seed, official 4B Instruct boundary model; not test noninferiority or main-model quality proof",
            "sources":sources,"metric":"calibrated original ROUGE-1.5.5 author-linked flags",
            "calibration":"results/metric_calibration_qmsum.json",
            "calibration_sha256":sha("results/metric_calibration_qmsum.json"),
            "metric_limit":"Published calibration fixture differs by up to 0.011 score points",
            "cost_limit":"HF instrumented reference costs, including physical per-head gather. No serving latency or speedup claim.",
            "cost_accounting":"Model initialization and dense prefix prefill are listed by actual execution; group0 prefix ran twice on separate question shards. Selector is shared across budgets and is a component of instrumented question prefill, not additive to it. Decode includes periodic checkpoint writes. Per-row sums must not double-count shared costs.",
            "missing_cost_fields":"Pilot group0 predates audit/export sub-timers; absent fields are omitted, never estimated. New cost fields have 28/32 observations per budget.",
            "output_parsing":"Official Instruct free-text summaries need no JSON/answer delimiter. Report empty outputs and unexpected thinking/chat delimiters; no postprocessing beyond tokenizer skip_special_tokens.",
            "generation":generation,"source_group_prefill_costs":source_groups,
            "cohorts":{}}
    for fraction,rows in sorted(cohorts.items(),key=lambda x:float(x[0])):
        rows.sort(key=lambda r:(r["group_index"],r["question_index"]))
        assert len(rows)==len(expected_pairs) and {(r["group_index"],r["question_index"]) for r in rows}==expected_pairs
        raw_prefix=output.parent/"perl"/(output.stem+"_fraction"+fraction.replace(".","p"))
        measured=score_pairs([r["prediction"] for r in rows],[r["reference"] for r in rows],
                    output_prefix=raw_prefix)
        for r,s in zip(rows,measured["per_example"]):
            r["rouge"]={k:v for k,v in s.items() if k!="index"}
        result["cohorts"][fraction]={**aggregate(rows),"documents":len(expected),"questions":len(rows),
            "natural_eos":sum(r["natural_eos"] for r in rows),"cap_hits":sum(r["hit_cap"] for r in rows),
            "total_generated_tokens":sum(r["generated_tokens"] for r in rows),
            "macro_f1_percent":measured["per_case_macro_f1_percent_from_printed_5dp"],
            "official_bootstrap_average_percent":measured["reported_average_percent"],
            "raw_perl_path":str(raw_prefix.with_suffix(".perl.txt")),
            "raw_perl_sha256":sha(raw_prefix.with_suffix(".perl.txt")),
            "per_document":{str(gi):aggregate([r for r in rows if r["group_index"]==gi]) for gi in sorted(expected)},
            "median_reference_decode_seconds":statistics.median(r["decode_seconds"] for r in rows),
            "per_question":rows}
        print(json.dumps({"fraction":fraction,"macro_f1_percent":result["cohorts"][fraction]["macro_f1_percent"]}),flush=True)
    result["delta_vs_full_budget_points"]={f:{m:c["macro_f1_percent"][m]-result["cohorts"]["1.0"]["macro_f1_percent"][m] for m in ["1","2","L"]} for f,c in result["cohorts"].items()}
    full=result["cohorts"]["1.0"]
    for fraction,cohort in result["cohorts"].items():
        for gi,doc in cohort["per_document"].items():
            doc["delta_vs_full_budget_points"]={m:doc["macro_f1_percent"][m]-full["per_document"][gi]["macro_f1_percent"][m] for m in ["1","2","L"]}
        for row,baseline in zip(cohort["per_question"],full["per_question"]):
            row["delta_vs_full_budget_points"]={m:100*(row["rouge"][m]["f1"]-baseline["rouge"][m]["f1"]) for m in ["1","2","L"]}
    all_rows=[row for cohort in result["cohorts"].values() for row in cohort["per_question"]]
    result["blind_factual_audit"]=write_blind_audit(all_rows,output)
    csv_path=output.with_suffix(".csv")
    with csv_path.open("w",newline="") as file:
        fields=["fraction","group_index","question_index","natural_eos","hit_cap","generated_tokens","budget_tokens",
                "decode_seconds","gather_seconds","topk_and_support_seconds","selector_observer_seconds",
                "rouge1_f1_percent","rouge2_f1_percent","rougeL_f1_percent"]
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in all_rows:
            flat={k:row[k] for k in fields if k in row}
            flat.update({f"rouge{m}_f1_percent":100*row["rouge"][m]["f1"] for m in ["1","2","L"]})
            writer.writerow(flat)
    result["per_question_csv"]=str(csv_path)
    if a.vllm_summary or a.vllm_raw:
        assert a.vllm_summary and a.vllm_raw
        other=json.loads(Path(a.vllm_summary).read_text())
        raw=json.loads(Path(a.vllm_raw).read_text())
        assert raw["model_identity"]["revision"]==model_revision
        assert raw["manifest_sha256"]==data["manifest_sha256"]
        comparison={"scope":"Descriptive full-budget HF versus same-model vLLM outputs; different sampling parameters, RNG/seed rules and execution stacks. Not a compression effect or backend-equivalence test.",
                    "vllm_summary":a.vllm_summary,"vllm_summary_sha256":sha(a.vllm_summary),
                    "vllm_raw":a.vllm_raw,"vllm_raw_sha256":sha(a.vllm_raw),
                    "hf_sampling":generation,"hf_seed_rule":"seed + question_index; top_k disabled",
                    "vllm_sampling":raw["sampling"],"vllm_engine":raw["engine"],"arms":{}}
        for name,arm in other["arms"].items():
            lookup={(r["group_index"],r["question_index"]):r for r in arm["per_question"]}
            assert set(lookup)==expected_pairs
            pairs=[]
            for row in full["per_question"]:
                baseline=lookup[(row["group_index"],row["question_index"])]
                assert baseline["question_id"]==row["question_sha256"] and baseline["reference"]==row["reference"]
                pairs.append({"group_index":row["group_index"],"question_index":row["question_index"],
                              "text_exact_equal":row["prediction"]==baseline["prediction"],
                              "hf_full_budget_answer":row["prediction"],"vllm_answer":baseline["prediction"],
                              "hf_minus_vllm_rouge_points":{m:100*(row["rouge"][m]["f1"]-baseline["rouge"][m]["f1"]) for m in ["1","2","L"]}})
            comparison["arms"][name]={"questions":len(pairs),"text_exact_equal":sum(r["text_exact_equal"] for r in pairs),
                "hf_minus_vllm_macro_f1_points":{m:full["macro_f1_percent"][m]-arm["macro_f1_percent"][m] for m in ["1","2","L"]},
                "per_question":pairs}
        comparison_path=output.with_name(output.stem+"_cross_stack_descriptive.json")
        comparison_path.write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+"\n")
        result["cross_stack_descriptive_comparison"]={"path":str(comparison_path),"sha256":sha(comparison_path)}
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
