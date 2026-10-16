"""Freeze an explicitly new DocFinQA cohort after the historical split was not found.

Selection uses document hashes only. Preserve the already exposed dev8. Test
contexts cannot enter training, including unselected test contexts. Questions
and answers never enter the ordering rule. Length/B failures are not filtered.
"""
from __future__ import annotations
import collections
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import pyarrow.parquet as pq
from prepare_data import finqa_identity, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/data_qualification"
SEED = "branchtile-replacement-report-cohort-v2-20261016"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-version", type=int, choices=[2, 3], default=3)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    dev_path = ROOT / "data/manifests/docfinqa_exploratory_dev8.json"
    dev = json.loads(dev_path.read_text())
    dev_hashes = {g["document_sha256"] for g in dev["groups"]}
    sources = json.loads((ROOT / "results/data_docfinqa_parquet_validation.json").read_text())["sources"]
    for source in sources:
        path = ROOT / source["path"]
        assert path.stat().st_size == source["bytes"] and sha(path.read_bytes()) == source["sha256"]
    identities = finqa_identity()
    documents = {}
    for split in ("validation", "test", "train"):
        row_index = 0
        for batch in pq.ParquetFile(ROOT / f"data/raw/docfinqa/parquet/{split}.parquet").iter_batches(batch_size=16):
            for row in batch.to_pylist():
                text = row["Context"]
                dh, qh = sha(text), sha(row["Question"])
                item = documents.setdefault(dh, {"text": text, "splits": set(), "questions": {}, "identities": set()})
                item["splits"].add(split)
                item["identities"].update(identities.get(row["Question"], set()))
                q = item["questions"].setdefault(qh, {"question_sha256": qh, "question": row["Question"],
                                                      "answer": row["Answer"], "source_occurrences": []})
                q["source_occurrences"].append({"split": split, "row": row_index, "answer": row["Answer"],
                                                "program_sha256": sha(row["Program"])})
                row_index += 1
        print(json.dumps({"scanned_split": split, "rows": row_index, "unique_contexts_so_far": len(documents)}), flush=True)
    def key(dh):
        return sha(SEED + "\n" + dh)
    all_test = {dh for dh, item in documents.items() if "test" in item["splits"]}
    eval_pool = all_test - dev_hashes
    train_pool = {dh for dh, item in documents.items() if "train" in item["splits"]} - all_test - dev_hashes
    # Contexts can be different versions of the same annual report. Conservatively
    # connect every FinQA company/year candidate; ambiguity may over-exclude but
    # must not silently leak known development reports into evaluation.
    parents = {dh: dh for dh in documents}
    def root(dh):
        while parents[dh] != dh:
            parents[dh] = parents[parents[dh]]
            dh = parents[dh]
        return dh
    year_first = {}
    for dh, item in documents.items():
        for year in item["identities"]:
            previous = year_first.setdefault(year, dh)
            parents[root(dh)] = root(previous)
    dev_components = {root(dh) for dh in dev_hashes}
    test_components = {root(dh) for dh in all_test}
    if args.cohort_version == 3:
        eval_pool = {dh for dh in eval_pool if root(dh) not in dev_components}
        train_pool = {dh for dh in train_pool if root(dh) not in dev_components | test_components}
    selections = {"evaluation100": sorted(eval_pool, key=key)[:100],
                  "training167": sorted(train_pool, key=key)[:167]}
    assert len(selections["evaluation100"]) == 100 and len(selections["training167"]) == 167
    assert not set(selections["evaluation100"]) & set(selections["training167"])
    common = {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "historical_split_restored": False, "cohort_status": "new_replacement_pending_formal_qualification",
              "cohort_version": args.cohort_version,
              "cross_split_identity_policy": "v3 additionally excludes all context components connected by any FinQA company/year candidate to dev (evaluation) or dev/all test (training); unresolved source identity still limits a formal independence claim" if args.cohort_version == 3 else "v2 exact context only; known annual-report aliases can overlap and prevent formal use",
              "selection_rule": "Select first N by SHA256(seed + newline + document SHA256); no answer, length, B or performance filter",
              "selection_seed": SEED, "source_files": sources,
              "preserved_development_manifest": str(dev_path.relative_to(ROOT)), "preserved_development_manifest_sha256": sha(dev_path.read_bytes()),
              "question_rule": "Exact-context natural questions across official splits, deduplicated and sorted by question SHA256, at most four; no historical representative exists",
              "conflicting_answer_rule": "Retain every source answer and flag conflicts; primary label comes from target official split's earliest row, then validation/test/train precedence. No model output or numerical closeness chooses labels; unresolved conflicts require separate scoring audit.",
              "identity_limitations": "Context SHA is exact input identity. Company/year inferred from exact FinQA question matching may be ambiguous; this is not the historical 7/48-company cohort. Company/year aliases can span different context hashes and require cluster audit before inferential statistics.",
              "length_rule": "No context truncation or exclusion at freeze; future capacity failures remain in the selected denominator"}
    summary = {"schema": 1, "selection_seed": SEED, "historical_split_restored": False,
               "cohort_version": args.cohort_version,
               "dev8_unchanged": True, "all_unique_contexts": len(documents),
               "evaluation_pool_contexts": len(eval_pool), "training_pool_contexts": len(train_pool),
               "training_excludes_all_official_test_contexts": True, "manifests": []}
    for split, hashes in selections.items():
        groups = []
        for rank, dh in enumerate(hashes):
            item = documents[dh]
            candidates = sorted(item["identities"])
            companies = sorted({c.rsplit("_", 1)[0] for c in candidates})
            qs = [item["questions"][h] for h in sorted(item["questions"])[:4]]
            preferred = "test" if split == "evaluation100" else "train"
            order = [preferred] + [s for s in ("validation", "test", "train") if s != preferred]
            for q in qs:
                variants = sorted({o["answer"] for o in q["source_occurrences"]})
                primary = min(q["source_occurrences"], key=lambda o: (order.index(o["split"]), o["row"]))
                q.update(answer=primary["answer"], answer_source={"split": primary["split"], "row": primary["row"]},
                         source_answer_variants=variants, conflicting_source_answers=len(variants)>1)
            relative = f"data/documents/docfinqa-{dh[:16]}.txt"
            path = ROOT / relative
            if path.exists():
                assert sha(path.read_bytes()) == dh
            else:
                path.write_text(item["text"])
            groups.append({"dataset": "DocFinQA", "document_id": "docfinqa-" + dh[:16],
                           "document_sha256": dh, "document_path": relative,
                           "selection_rank": rank, "selection_key_sha256": key(dh),
                           "source_splits": sorted(item["splits"]), "company_year_candidates": candidates,
                           "conservative_identity_component": root(dh),
                           "company_year": candidates[0] if len(candidates) == 1 else None,
                           "company": companies[0] if len(companies) == 1 else None,
                           "identity_mapping_status": "unique_exact_question_match" if len(candidates) == 1 else "ambiguous_or_missing",
                           "natural_question_count": len(item["questions"]), "B": len(qs), "questions": qs,
                           "document_characters": len(item["text"])})
        manifest = dict(common, split_name=f"replacement_v{args.cohort_version}_" + split, groups=groups)
        suffix = "" if args.cohort_version == 2 else f"_v{args.cohort_version}"
        dest = OUT / f"docfinqa_replacement_{split}{suffix}.json"
        write_json(dest, manifest)
        summary["manifests"].append({"path": str(dest.relative_to(ROOT)), "sha256": sha(dest.read_bytes()),
                                    "document_groups": len(groups), "selected_questions": sum(g["B"] for g in groups),
                                    "B_histogram": dict(collections.Counter(g["B"] for g in groups)),
                                    "unambiguous_company_count": len({g["company"] for g in groups if g["company"]}),
                                    "ambiguous_company_groups": sum(g["company"] is None for g in groups),
                                    "ambiguous_company_year_groups": sum(g["company_year"] is None for g in groups),
                                    "selected_questions_with_conflicting_source_answers": sum(q["conflicting_source_answers"] for g in groups for q in g["questions"])})
    write_json(OUT / f"docfinqa_replacement_freeze{suffix}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
