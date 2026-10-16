#!/usr/bin/env python3
"""Prepare honest, deterministic exploratory inputs from official source files.

The proposal's historical 8/100/167 manifest is not present in this checkout.
This script never labels the newly selected documents as that historical split.
No answer, B count, model output, or performance result enters selection.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = "branchtile-exploratory-v1"


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_info(path: Path, url: str, revision: str) -> dict:
    return {"path": str(path.relative_to(ROOT)), "url": url, "revision": revision,
            "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def finqa_identity() -> dict:
    mapping = collections.defaultdict(set)
    for split in ("train", "dev", "test"):
        p = ROOT / f"data/raw/finqa/{split}.json"
        if not p.exists():
            continue
        for item in json.loads(p.read_text()):
            parts = item["id"].split("/")
            mapping[item["qa"]["question"]].add(parts[0] + "_" + parts[1])
    return mapping


def docfinqa_groups(path: Path) -> list[dict]:
    identities = finqa_identity()
    groups = {}
    for idx, row in enumerate(json.loads(path.read_text())):
        text = row["Context"]
        dh = sha(text)
        if dh not in groups:
            groups[dh] = {"dataset": "DocFinQA", "document_id": "docfinqa-" + dh[:16],
                          "document_sha256": dh, "document_text": text,
                          "questions_by_sha": {}, "source_rows": [], "identity_candidates": set()}
        g = groups[dh]
        q = row["Question"]
        qh = sha(q)
        g["source_rows"].append(idx)
        g["identity_candidates"].update(identities.get(q, []))
        if qh not in g["questions_by_sha"]:
            g["questions_by_sha"][qh] = {"question_sha256": qh, "question": q,
                                        "answer": row["Answer"], "source_row": idx}
        else:
            assert g["questions_by_sha"][qh]["answer"] == row["Answer"], "Conflicting duplicate QA"
    for g in groups.values():
        candidates = sorted(g.pop("identity_candidates"))
        g["company_year_candidates"] = candidates
        g["identity_mapping_status"] = "unique_exact_question_match" if len(candidates) == 1 else "ambiguous_or_missing"
        g["company_year"] = candidates[0] if len(candidates) == 1 else None
        companies = sorted({c.rsplit("_", 1)[0] for c in candidates})
        g["company"] = companies[0] if len(companies) == 1 else None
        g["company_mapping_status"] = "unique_across_exact_question_matches" if len(companies) == 1 else "ambiguous_or_missing"
        qs = g.pop("questions_by_sha")
        g["natural_question_count"] = len(qs)
        g["questions"] = [qs[k] for k in sorted(qs)[:4]]
        g["B"] = len(g["questions"])
    return list(groups.values())


def qmsum_groups(path: Path) -> list[dict]:
    groups = []
    for idx, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        text = "\n".join(f'{u["speaker"]}: {u["content"]}' for u in row["meeting_transcripts"])
        dh = sha(text)
        qs = {}
        for q in row["specific_query_list"]:
            qh = sha(q["query"])
            qs[qh] = {"question_sha256": qh, "question": q["query"], "answer": q["answer"]}
        groups.append({"dataset": "QMSum", "document_id": "qmsum-" + dh[:16],
                       "document_sha256": dh, "document_text": text, "source_row": idx,
                       "source_meeting_id": row.get("meeting_id"), "natural_question_count": len(qs),
                       "questions": [qs[k] for k in sorted(qs)[:4]], "B": min(4, len(qs))})
    return groups


def freeze_groups(groups: list[dict], out: Path, split: str, count: int | None, tokenizer) -> dict:
    # Hash order is frozen before lengths, questions, and answers are examined.
    groups.sort(key=lambda g: sha(SELECTION + "\n" + g["document_sha256"]))
    selected = groups if count is None else groups[:count]
    manifest = []
    for i, g in enumerate(selected):
        text = g.pop("document_text")
        filename = f'data/documents/{g["document_id"]}.txt'
        p = ROOT / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        g.update({"document_path": filename, "selection_rank": i, "split": split,
                  "selection_key_sha256": sha(SELECTION + "\n" + g["document_sha256"]),
                  "document_characters": len(text),
                  "document_tokens": len(tokenizer.encode(text, add_special_tokens=False))})
        manifest.append(g)
    result = {"schema": 1, "split_name": split, "historical_split_restored": False,
              "selection_rule": "first N document hashes sorted by SHA256('" + SELECTION + "\\n' + document_sha256); no length/B/answer filtering",
              "question_rule": "deduplicate exact question text, sort SHA256, retain up to four; no historical representative available",
              "tokenizer": str(tokenizer.name_or_path), "source_document_groups": len(groups),
              "groups": manifest}
    write_json(out, result)
    return {"path": str(out.relative_to(ROOT)), "sha256": sha(out.read_bytes()),
            "groups": len(manifest), "questions": sum(g["B"] for g in manifest),
            "B_histogram": dict(collections.Counter(g["B"] for g in manifest)),
            "lengths": [g["document_tokens"] for g in manifest]}


def supplement_docfinqa_questions() -> None:
    """Keep frozen documents and one exploratory representative; add exact-context QAs."""
    import pyarrow.parquet as pq
    manifest_path = ROOT / "data/manifests/docfinqa_exploratory_dev8.json"
    original_path = manifest_path.with_name("docfinqa_exploratory_dev8_devquestions_v1.json")
    if not original_path.exists():
        original_path.write_bytes(manifest_path.read_bytes())
    manifest = json.loads(original_path.read_text())
    assert manifest.get("question_scope") != "all_official_splits_exact_context"
    identities = finqa_identity()
    revision = json.loads((ROOT / "results/data_docfinqa_parquet_revision.json").read_text())
    expected = {item["name"].split("/")[1]: item["lfs"] for item in revision["files"] if item.get("lfs")}
    source_records = []
    for split, info in expected.items():
        path = ROOT / f"data/raw/docfinqa/parquet/{split}.parquet"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        assert path.stat().st_size == info["size"] and digest.hexdigest() == info["sha256"], split
        source_records.append({"path": str(path.relative_to(ROOT)), "split": split, "sha256": digest.hexdigest(),
                               "bytes": path.stat().st_size, "revision": revision["revision"]})
    raw_dev = json.loads((ROOT / "data/raw/docfinqa/dev.json").read_text())
    converted_dev = pq.read_table(ROOT / "data/raw/docfinqa/parquet/validation.parquet").to_pylist()
    assert raw_dev == converted_dev, "Converted validation must match raw official dev row-for-row in all fields"
    validation = {"status": "verified", "raw_dev_matches_converted_validation": True,
                  "comparison": "all rows, original order, all four fields, exact strings", "rows": len(raw_dev),
                  "raw_dev_sha256": sha((ROOT / "data/raw/docfinqa/dev.json").read_bytes()),
                  "converted_revision": revision["revision"], "sources": source_records}
    del raw_dev, converted_dev
    targets = {g["document_sha256"]: {} for g in manifest["groups"]}
    split_rows = {}
    for split in ("validation", "train", "test"):
        count = 0
        parquet = pq.ParquetFile(ROOT / f"data/raw/docfinqa/parquet/{split}.parquet")
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                index = count
                count += 1
                dh = sha(row["Context"])
                if dh not in targets:
                    continue
                qh = sha(row["Question"])
                qs = targets[dh]
                occurrence = {"split": split, "row": index, "program_sha256": sha(row["Program"])}
                if qh not in qs:
                    qs[qh] = {"question_sha256": qh, "question": row["Question"], "answer": row["Answer"],
                              "source_occurrences": []}
                assert qs[qh]["answer"] == row["Answer"], "Conflicting answer for exact document/question pair"
                qs[qh]["source_occurrences"].append(occurrence)
        split_rows[split] = count
    for group in manifest["groups"]:
        qs = targets[group["document_sha256"]]
        representative = group["questions"][0]["question_sha256"]
        assert representative in qs
        selected = [representative] + [qh for qh in sorted(qs) if qh != representative][:3]
        group["dev_only_question_count_v1"] = group["natural_question_count"]
        group["exploratory_representative_question_sha256"] = representative
        group["natural_question_count"] = len(qs)
        group["available_questions"] = [qs[qh] for qh in sorted(qs)]
        group["questions"] = [qs[qh] for qh in selected]
        group["B"] = len(selected)
        group["question_scope"] = "all_official_splits_exact_context"
        candidates = sorted({identity for q in qs.values() for identity in identities.get(q["question"], [])})
        group["company_year_candidates"] = candidates
        group["company_year"] = candidates[0] if len(candidates) == 1 else None
        group["identity_mapping_status"] = "unique_exact_question_match" if len(candidates) == 1 else "ambiguous_or_missing"
        companies = sorted({c.rsplit("_", 1)[0] for c in candidates})
        group["company"] = companies[0] if len(companies) == 1 else None
        group["company_mapping_status"] = "unique_across_exact_question_matches" if len(companies) == 1 else "ambiguous_or_missing"
    manifest.update({"schema": 2, "question_scope": "all_official_splits_exact_context",
                     "question_rule": "keep first question of frozen exploratory dev-only v1 as representative, then exact-question SHA order up to four from all official splits with byte-identical Context; no historical representative is claimed",
                     "previous_manifest": str(original_path.relative_to(ROOT)), "previous_manifest_sha256": sha(original_path.read_bytes()),
                     "supplement_source_validation": "results/data_docfinqa_parquet_validation.json",
                     "supplement_before_docfinqa_model_experiment": True})
    write_json(manifest_path, manifest)
    validation.update({"split_rows": split_rows, "frozen_document_ids_unchanged": True,
                       "all_question_counts": [g["natural_question_count"] for g in manifest["groups"]],
                       "result_manifest_sha256": sha(manifest_path.read_bytes())})
    write_json(ROOT / "results/data_docfinqa_parquet_validation.json", validation)
    summary_path = ROOT / "results/data_preparation.json"
    summary = json.loads(summary_path.read_text())
    summary["manifests"]["docfinqa_exploratory_dev8"].update({"sha256": sha(manifest_path.read_bytes()),
        "questions": sum(g["B"] for g in manifest["groups"]),
        "B_histogram": dict(collections.Counter(g["B"] for g in manifest["groups"])),
        "question_scope": manifest["question_scope"], "previous_manifest": manifest["previous_manifest"]})
    summary["docfinqa_supplement_validation"] = "results/data_docfinqa_parquet_validation.json"
    write_json(summary_path, summary)
    print(json.dumps(validation, indent=2))


def main() -> None:
    assert os.geteuid() != 0, "Running data preparation as root is forbidden"
    assert sys.prefix != sys.base_prefix, "Run data preparation using the experiment virtual environment"
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/Qwen3-4B-Thinking-2507"))
    parser.add_argument("--supplement-docfinqa-questions", action="store_true")
    args = parser.parse_args()
    if args.supplement_docfinqa_questions:
        supplement_docfinqa_questions()
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    summaries = {}
    docpath = ROOT / "data/raw/docfinqa/dev.json"
    if docpath.exists():
        summaries["docfinqa_exploratory_dev8"] = freeze_groups(
            docfinqa_groups(docpath), ROOT / "data/manifests/docfinqa_exploratory_dev8.json",
            "new_exploratory_dev8_from_official_dev_not_historical", 8, tokenizer)
    for split, count in (("train", 8), ("test", None)):
        p = ROOT / f"data/raw/qmsum/{split}.jsonl"
        if p.exists():
            name = "qmsum_train_exploratory8" if split == "train" else "qmsum_official_test35"
            summaries[name] = freeze_groups(qmsum_groups(p), ROOT / f"data/manifests/{name}.json",
                                            name, count, tokenizer)
    sources = []
    if docpath.exists():
        sources.append(source_info(docpath, "https://huggingface.co/datasets/kensho/DocFinQA", "64ebaff62f692495bcc182f45cf9a9606251b19b"))
    for repo, folder in (("Yale-LILY/QMSum", "qmsum"), ("czyssrs/FinQA", "finqa")):
        cp = ROOT / f"data/raw/{folder}/source_commit.json"
        if cp.exists():
            rev = json.loads(cp.read_text())["sha"]
            for p in sorted(cp.parent.glob("*json*")):
                if p != cp:
                    sources.append(source_info(p, "https://github.com/" + repo, rev))
    write_json(ROOT / "results/data_preparation.json", {
        "status": "exploratory_data_prepared_historical_split_missing", "sources": sources,
        "manifests": summaries,
        "historical_manifest_audit": {
            "git": "initial repository commit 083384c contains proposal and proxy helper only",
            "searched_locations": ["/home/qcjz/iscas", "/home/qcjz/models", "/home/qcjz/.cache/huggingface", "/home/qcjz/results", "/home/qcjz/artifacts", "/home/qcjz/Downloads", "/tmp"],
            "result": "No prior DocFinQA/QMSum manifests found in bounded targeted filename search; this does not prove they do not exist elsewhere.",
            "limitation": "New exploratory eight groups are not the proposed 8 reports/7 companies; no old 100-eval or 167-train split is claimed restored."}})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
