"""Join a completed blind audit to its sealed identity map, without rejudging."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paired_statistics import paired_cluster_interval

queue_path = ROOT / "results/sparse_reference/train8_quality_blind_audit.jsonl"
map_path = ROOT / "results/sparse_reference/train8_quality_blind_audit_mapping.json"
audit_path = ROOT / "results/factual_audit/train8_claims_v1.jsonl"
identity_path = ROOT / "results/qmsum_identity/qmsum_train_exploratory8_identity.json"
queue = {r["item_id"]: r for r in map(json.loads, queue_path.read_text().splitlines())}
mapping = json.loads(map_path.read_text())["items"]
audits = list(map(json.loads, audit_path.read_text().splitlines()))
assert len(audits) == len({r["item_id"] for r in audits}) == len(queue) == 128
assert set(queue) == set(mapping) == {r["item_id"] for r in audits}
higher_groups = {g["document_sha256"]: g["higher_source_cluster"]
                 for g in json.loads(identity_path.read_text())["groups"]}
groups = collections.defaultdict(list)
for audit in audits:
    item = queue[audit["item_id"]]
    meta = mapping[audit["item_id"]]
    assert hashlib.sha256(item["answer"].encode()).hexdigest() == meta["answer_sha256"]
    assert item["document_sha256"] == audit["document_sha256"]
    labels = collections.Counter(c["label"] for c in audit["major_claims"])
    assert labels["supported"] == audit["supported_claims"]
    assert labels["contradicted"] == audit["contradicted_claims"]
    assert labels["unverifiable"] == audit["unverifiable_claims"]
    assert audit["answer_factual_pass"] == (audit["contradicted_claims"] == audit["unverifiable_claims"] == 0 and audit["supported_claims"] > 0)
    groups[str(meta["fraction"])].append(dict(audit, question_id=meta["question_sha256"],
        group_index=meta["group_index"], question_index=meta["question_index"],
        resolved_sampling_seed=meta["seed"], source_path=meta["source_path"],
        fully_covered_and_factual=audit["answer_factual_pass"] and audit["question_coverage"] == 2))
result = {"scope": "Budget-blind LLM-assisted single-judge audit of development only; not human review, inter-judge agreement, formal quality noninferiority or a correctness-defined energy denominator.",
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [queue_path, map_path, audit_path, identity_path]},
          "limitations": ["One judge has general project context; method family is not a separately blinded multi-method experiment.",
                          "Candidate identity mapping was withheld until the judge completed all128 verdicts; this join does not revise verdicts.",
                          "Source evidence, uncertain claims, coverage, and contradictions are separately retained. Gold is not treated as unquestionable truth.",
                          "Claim counts depend on output length and are descriptive. Per-answer factual pass and coverage are reported jointly."],
          "cohorts": {}, "paired_higher_source_diagnostics": {}}
for fraction, rows in sorted(groups.items(), key=lambda x: float(x[0])):
    rows.sort(key=lambda x: (x["group_index"], x["question_index"]))
    assert len(rows) == 32
    result["cohorts"][fraction] = {
        "answers": len(rows), "factual_pass": sum(r["answer_factual_pass"] for r in rows),
        "fully_covered_and_factual": sum(r["fully_covered_and_factual"] for r in rows),
        "uncertain_answers": sum(r["uncertain"] for r in rows),
        "coverage_counts": dict(collections.Counter(r["question_coverage"] for r in rows)),
        **{k: sum(r[k] for r in rows) for k in ["supported_claims", "contradicted_claims", "unverifiable_claims"]},
        "per_answer": rows}
refs = {(r["document_sha256"], r["question_id"]): r for r in groups["1.0"]}
for fraction in ["0.1", "0.25", "0.5"]:
    result["paired_higher_source_diagnostics"][fraction] = {}
    for metric in ["answer_factual_pass", "fully_covered_and_factual", "question_coverage"]:
        rows = []
        for row in groups[fraction]:
            ref = refs[(row["document_sha256"], row["question_id"])]
            assert row["resolved_sampling_seed"] == ref["resolved_sampling_seed"]
            divisor = 2 if metric == "question_coverage" else 1
            rows.append(dict(document_id=row["document_sha256"], question_id=row["question_id"], seed=0,
                             cluster_id=higher_groups[row["document_sha256"]],
                             reference=ref[metric] / divisor, candidate=row[metric] / divisor))
        result["paired_higher_source_diagnostics"][fraction][metric] = paired_cluster_interval(rows, expected_seeds=[0])
out = ROOT / "results/factual_audit/train8_unblinded_v1.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: {name: value for name, value in v.items() if name != "per_answer"} for k, v in result["cohorts"].items()}, indent=2))
