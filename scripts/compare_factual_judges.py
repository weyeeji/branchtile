"""Join frozen development sample only after the second judge's final commit.

Primary old verdict is the first origin in the pre-existing identity mapping.
All other verdicts on identical text remain explicit, never cherry-picked.
"""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINS = {
    'results/sparse_reference/train8_quality_blind_audit.jsonl': 'results/factual_audit/train8_claims_v1.jsonl',
    'results/fastkvzip_reference/train8_quality_blind_audit.jsonl': 'results/factual_audit/fastkvzip_new96_claims_v1.jsonl',
    'results/sparse_reference_absolute/train8_quality_blind_audit_new56.jsonl': 'results/factual_audit/absolute_new56_claims_v1.jsonl',
}
FIELDS = ['answer_factual_pass', 'question_coverage', 'uncertain']


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in (ROOT/path).read_text().splitlines()]
    assert len(rows) == len({r['item_id'] for r in rows})
    return {r['item_id']:r for r in rows}


def agreement(pairs):
    n = len(pairs)
    labels = sorted(set(a for a,b in pairs) | set(b for a,b in pairs))
    counts_a, counts_b = collections.Counter(a for a,b in pairs), collections.Counter(b for a,b in pairs)
    matched = sum(a == b for a,b in pairs)
    expected = sum(counts_a[k]*counts_b[k] for k in labels) / n**2
    return dict(n=n, equal=matched, fraction=matched/n,
        cohen_kappa=(matched/n-expected)/(1-expected) if expected != 1 else None,
        kappa_rule='Unweighted descriptive Cohen kappa; null when both marginals constant. No iid confidence interval.',
        labels=labels, confusion_first_rows_second_columns=[[sum(a == x and b == y for a,b in pairs) for y in labels] for x in labels])


def main():
    folder = Path('results/factual_audit_second')
    protocol = json.loads((ROOT/folder/'protocol_v1.json').read_text())
    queue_path = folder/'development32_blind_queue.jsonl'
    assert sha(queue_path) == protocol['queue_sha256']
    queues = {p:read_rows(p) for p in ORIGINS}
    reviews = {p:read_rows(r) for p,r in ORIGINS.items()}
    for p in ORIGINS:
        assert sha(p) == protocol['source_queues_sha256'][p]
    second = read_rows(folder/'development32_claims_v1.jsonl')
    queue = read_rows(queue_path)
    mapping = json.loads((ROOT/folder/'development32_identity_mapping.json').read_text())
    assert set(second) == set(queue) == set(mapping) and len(second) == 32
    rows = []
    for item_id in sorted(queue):
        candidate, audit = queue[item_id], second[item_id]
        assert audit['document_sha256'] == candidate['document_sha256'] == sha(candidate['document_path'])
        assert hashlib.sha256(candidate['answer'].encode()).hexdigest() == mapping[item_id]['answer_sha256']
        labels = collections.Counter(c['label'] for c in audit['major_claims'])
        assert audit['answer_factual_pass'] == (labels['supported']>0 and labels['contradicted']==labels['unverifiable']==0)
        prior_versions = []
        for origin in mapping[item_id]['origins']:
            source, key = origin['source_queue'], origin['item_id']
            old_candidate = queues[source][key]
            assert all(candidate[f] == old_candidate[f] for f in ['document_sha256','question','gold_reference','answer'])
            if key not in reviews[source]:
                # A2 full answers are exact duplicates from A4; frozen origin
                # order always places their A4 review first. Preserve the alias
                # explicitly instead of creating another purported judgment.
                assert source == 'results/fastkvzip_reference/train8_quality_blind_audit.jsonl'
                prior_versions.append(dict(origin=origin, judgment_reused_alias_without_new_review=True))
                continue
            old = reviews[source][key]
            assert old['document_sha256'] == candidate['document_sha256']
            prior_versions.append(dict(origin=origin, verdict={f:old[f] for f in FIELDS}))
        assert 'verdict' in prior_versions[0], 'Primary origin must have a real frozen judgment'
        primary = prior_versions[0]['verdict']
        new = {f:audit[f] for f in FIELDS}
        rows.append(dict(item_id=item_id, source_sha256=candidate['document_sha256'],
            question=candidate['question'], first_primary=primary, second=new,
            agreed={f:primary[f]==new[f] for f in FIELDS}, all_prior_versions=prior_versions,
            old_versions_consistent={f:len({v['verdict'][f] for v in prior_versions if 'verdict' in v})==1 for f in FIELDS}))
    outputs = {f:agreement([(r['first_primary'][f],r['second'][f]) for r in rows]) for f in FIELDS}
    both = lambda r:r['answer_factual_pass'] and r['question_coverage']==2
    outputs['fully_covered_and_factual'] = agreement([(both(r['first_primary']),both(r['second'])) for r in rows])
    result = dict(scope='Descriptive second LLM-assisted judge agreement on one sampled unique text per question',
        sample_size=32, source_documents=len({r['source_sha256'] for r in rows}), higher_source_clusters=7,
        primary_origin_rule='First origin in the pre-frozen mapping, source queue order A4 then A2 then absolute. Never choose whichever old verdict agrees.',
        limitations=['Both sessions share the same model family and general project context; neither is a human reviewer.',
            'One unique text per question sampled, not balanced over methods/budgets; no per-method quality conclusion.',
            '32 questions are not independent source documents; kappa is descriptive without iid intervals.',
            'No verdict is adjudicated, replaced or retroactively corrected by agreement analysis.'],
        source_sha256={str(p):sha(p) for p in [queue_path,folder/'protocol_v1.json',folder/'development32_identity_mapping.json',folder/'development32_claims_v1.jsonl',*ORIGINS.values()]},
        agreements=outputs, rows=rows)
    (ROOT/folder/'agreement_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(outputs,indent=2))


if __name__ == '__main__':
    main()
