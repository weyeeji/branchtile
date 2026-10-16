"""Adopt a new public-source protocol without changing any preselected task.

The user explicitly waived historical-manifest recovery on 2026-10-16. This
freeze adopts the existing input-only v3 ordering, preserves exposed dev data,
and labels connected identity clusters as inferred rather than verified firms.
"""
from __future__ import annotations
import collections
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/manifests"


def digest(value):
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    inputs = {
        "development8": "data/manifests/docfinqa_exploratory_dev8.json",
        "evaluation100": "results/data_qualification/docfinqa_replacement_evaluation100_v3.json",
        "training167": "results/data_qualification/docfinqa_replacement_training167_v3.json",
    }
    manifests = {s: json.loads((ROOT / p).read_text()) for s, p in inputs.items()}
    frozen = json.loads((ROOT / "results/data_qualification/docfinqa_replacement_freeze_v3.json").read_text())
    for record in frozen["manifests"]:
        assert digest((ROOT / record["path"]).read_bytes()) == record["sha256"]
    source_validation = ROOT / "results/data_docfinqa_parquet_validation.json"
    sources = json.loads(source_validation.read_text())["sources"]
    for source in sources:
        path = ROOT / source["path"]
        assert path.stat().st_size == source["bytes"]
        assert digest(path.read_bytes()) == source["sha256"]

    all_groups = [g for m in manifests.values() for g in m["groups"]]
    hashes = [g["document_sha256"] for g in all_groups]
    assert len(hashes) == len(set(hashes)) == 275
    for group in all_groups:
        assert digest((ROOT / group["document_path"]).read_bytes()) == group["document_sha256"]
        assert group["B"] == len(group["questions"]) <= 4
        assert len({q["question_sha256"] for q in group["questions"]}) == group["B"]
        for question in group["questions"]:
            assert digest(question["question"]) == question["question_sha256"]

    # Connect across all selected splits, including same-company different years
    # and ambiguous identities. This is conservative conditional on these inferred
    # candidates; it cannot prove the completeness of company identity metadata.
    parents = {dh: dh for dh in hashes}
    def root(dh):
        while parents[dh] != dh:
            parents[dh] = parents[parents[dh]]
            dh = parents[dh]
        return dh
    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)
    seen = {}
    for group in all_groups:
        dh = group["document_sha256"]
        companies = {y.rsplit("_", 1)[0] for y in group["company_year_candidates"]}
        tokens = ["company:" + company for company in sorted(companies)]
        if group.get("conservative_identity_component"):
            tokens.append("report-component:" + group["conservative_identity_component"])
        if not companies:
            tokens.append("all-unresolved-company-identities")
        for token in tokens:
            union(dh, seen.setdefault(token, dh))
    members = collections.defaultdict(list)
    for dh in hashes:
        members[root(dh)].append(dh)
    cluster_ids = {dh: "inferred-component-" + digest("\n".join(sorted(members[root(dh)])))[:16] for dh in hashes}

    sets = {}
    for split, manifest in manifests.items():
        groups = manifest["groups"]
        sets[split] = {
            "documents": {g["document_sha256"] for g in groups},
            "company_year_candidates": {c for g in groups for c in g["company_year_candidates"]},
            "company_candidates": {c.rsplit("_", 1)[0] for g in groups for c in g["company_year_candidates"]},
            "selected_question_hashes": {q["question_sha256"] for g in groups for q in g["questions"]},
        }
    overlap = []
    for a, b in itertools.combinations(sets, 2):
        counts = {key: len(sets[a][key] & sets[b][key]) for key in sets[a]}
        assert counts["documents"] == counts["company_year_candidates"] == 0
        overlap.append({"splits": [a, b], **counts})

    created = datetime.datetime.now(datetime.timezone.utc).isoformat()
    protocol = {
        "schema": 1, "protocol_id": "docfinqa-public-v1-20261016", "frozen_utc": created,
        "status": "frozen_new_public_source_input_protocol",
        "authorization": "User explicitly said there is no old manifest and to disregard the old data on 2026-10-16; historical recovery is no longer a requirement or blocker.",
        "selection": "Adopt v3 input-only hash-ranked evaluation100/training167 and preserve all already-exposed development8 unchanged. The 8/100/167 sizes are new cohort sizes, not a recovery of the old reports or company distribution.",
        "source_dataset": "https://huggingface.co/datasets/kensho/DocFinQA",
        "source_revision": "64ebaff62f692495bcc182f45cf9a9606251b19b",
        "converted_parquet_revision": "7ab8d931022705be37535159d637de7e4b341fb6",
        "source_files": sources,
        "source_validation": {"path": str(source_validation.relative_to(ROOT)), "sha256": digest(source_validation.read_bytes())},
        "question_policy": "Same byte-identical complete report only; deduplicate exact question text, deterministic order and at most four natural questions; retain actual B1/B2/B3/B4. The existing development representative is preserved; evaluation/training use question-SHA order. No answer, length, B, measured mask or model performance enters report selection.",
        "leakage_policy": "Training excludes all official test contexts and their conservative FinQA company/year components, including unselected official test reports. Evaluation excludes development contexts and such components. Same company across different years may remain and is not out-of-company generalization.",
        "inference_cluster_policy": "Connected components of all available inferred company candidates across years, plus conservative report-identity components, across the selected development/evaluation/training cohorts. Missing-company reports are combined into one unresolved component. These are inferred identity components, not verified company clusters; report-level and component-level sensitivity results must both be shown.",
        "identity_uncertainty": "Exact FinQA question matching can produce ambiguous or incomplete company/year identities. Component clustering is conservative relative to the observed candidate graph but does not establish complete statistical independence. Do not claim the historical 48-company distribution or unconditional company-cluster coverage.",
        "label_policy": "Retain every source variant. Evaluation/training primary labels use earliest row of the target official split, then deterministic source precedence fixed before model outputs. Never choose the closest label to a prediction. Report primary-label results and adversarial sensitivity for each unresolved conflicting question; no conflict is silently dropped.",
        "capacity_policy": "Full input, no truncation/replacement/filtering after freeze; all selected reports and questions remain in denominators. Report model-context, allocation/OOM, execution, output-cap, parse, and quality failures separately. The context audit is tokenizer-only, not a GPU capacity guarantee.",
        "statistics_scope": "This freezes inputs, cluster labels and source-label handling. Scorer calibration, three-seed paired execution, frozen model/budget and completed application results remain separately required. No quality or noninferiority conclusion follows from this freeze.",
        "cross_split_overlap": overlap,
        "manifests": [], "conflicting_questions": [],
        "script_sha256": digest(Path(__file__).read_bytes()),
    }
    for split, manifest in manifests.items():
        original = ROOT / inputs[split]
        groups = manifest["groups"]
        for group in groups:
            group["inference_cluster_id"] = cluster_ids[group["document_sha256"]]
            group["inference_cluster_status"] = "inferred_connected_identity_component_not_verified_company"
            for question in group["questions"]:
                variants = question.get("source_answer_variants", sorted({o.get("answer", question["answer"]) for o in question.get("source_occurrences", [])}))
                variants = variants or [question["answer"]]
                question["source_answer_variants"] = variants
                question["conflicting_source_answers"] = len(variants) > 1
                if len(variants) > 1:
                    protocol["conflicting_questions"].append({"split": split, "document_id": group["document_id"], "question_sha256": question["question_sha256"], "primary_answer": question["answer"], "source_answer_variants": variants})
        dest = OUT / f"docfinqa_public_{split}_v1.json"
        value = {"schema": 3, "protocol_id": protocol["protocol_id"], "status": protocol["status"],
                 "split_name": "public_v1_" + split, "protocol": "results/data_qualification/docfinqa_public_protocol_v1.json",
                 "adopted_input_manifest": str(original.relative_to(ROOT)), "adopted_input_manifest_sha256": digest(original.read_bytes()),
                 "selection_rule": manifest["selection_rule"], "question_rule": manifest["question_rule"],
                 "groups": groups}
        write(dest, value)
        protocol["manifests"].append({"split": split, "path": str(dest.relative_to(ROOT)), "sha256": digest(dest.read_bytes()),
            "reports": len(groups), "questions": sum(g["B"] for g in groups),
            "B_histogram": dict(collections.Counter(g["B"] for g in groups)),
            "inferred_connected_components": len({g["inference_cluster_id"] for g in groups}),
            "inferred_unambiguous_companies": len({g["company"] for g in groups if g.get("company")}),
            "groups_without_unique_company": sum(g.get("company") is None for g in groups),
            "groups_without_unique_company_year": sum(g.get("company_year") is None for g in groups)})
    destination = ROOT / "results/data_qualification/docfinqa_public_protocol_v1.json"
    write(destination, protocol)
    status_path = ROOT / "results/data_qualification/docfinqa_split_status.json"
    status = json.loads(status_path.read_text())
    status.update(status=protocol["status"], active_protocol=str(destination.relative_to(ROOT)),
                  historical_manifest_recovery_required=False, historical_manifest_absence_is_blocker=False,
                  user_waiver_date="2026-10-16", active_manifests=protocol["manifests"],
                  identity_limit=protocol["identity_uncertainty"])
    write(status_path, status)
    print(json.dumps({"protocol": str(destination.relative_to(ROOT)), "manifests": protocol["manifests"], "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
