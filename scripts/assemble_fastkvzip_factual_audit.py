"""Join 96 blind judgments with 32 hash-identical prior judgments, without rejudging."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paired_statistics import paired_cluster_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {row['item_id']: row for row in rows}
    assert len(result) == len(rows), 'Duplicate audit ID'
    return result


def main():
    paths = {
        'queue': ROOT / 'results/fastkvzip_reference/train8_quality_blind_audit.jsonl',
        'new_queue': ROOT / 'results/fastkvzip_reference/train8_quality_blind_audit_new96.jsonl',
        'mapping': ROOT / 'results/fastkvzip_reference/train8_quality_blind_audit_mapping.json',
        'reviews': ROOT / 'results/factual_audit/fastkvzip_new96_claims_v1.jsonl',
        'prior_queue': ROOT / 'results/sparse_reference/train8_quality_blind_audit.jsonl',
        'prior_mapping': ROOT / 'results/sparse_reference/train8_quality_blind_audit_mapping.json',
        'prior_reviews': ROOT / 'results/factual_audit/train8_claims_v1.jsonl',
        'links': ROOT / 'results/fastkvzip_reference/fullbudget_prior_audit_reuse_mapping.json',
        'identity': ROOT / 'results/qmsum_identity/qmsum_train_exploratory8_identity.json',
    }
    queue, new_queue, prior_queue = [read_rows(paths[k]) for k in ['queue', 'new_queue', 'prior_queue']]
    reviews, prior_reviews = [read_rows(paths[k]) for k in ['reviews', 'prior_reviews']]
    mapping = json.loads(paths['mapping'].read_text())['items']
    links = json.loads(paths['links'].read_text())
    assert len(queue) == len(mapping) == 128 and len(new_queue) == len(reviews) == 96
    assert set(new_queue) == set(reviews)
    for name, key in [('queue', 'current_queue_sha256'), ('mapping', 'current_map_sha256'),
                      ('prior_queue', 'prior_queue_sha256'), ('prior_mapping', 'prior_map_sha256'),
                      ('new_queue', 'new_review_queue_sha256')]:
        assert sha(paths[name]) == links[key], (name, 'source changed')
    joined = {key: dict(value, audit_origin_item_id=key, judgment_reused=False)
              for key, value in reviews.items()}
    for link in links['links']:
        current, prior = link['new_item_id'], link['prior_item_id']
        assert current not in joined and link['full_source_question_gold_answer_identical']
        for field in ['document_sha256', 'question', 'gold_reference', 'answer']:
            assert queue[current][field] == prior_queue[prior][field], (current, field)
        assert hashlib.sha256(queue[current]['answer'].encode()).hexdigest() == link['answer_sha256']
        joined[current] = dict(prior_reviews[prior], item_id=current,
                               audit_origin_item_id=prior, judgment_reused=True)
    assert len(links['links']) == 32 and set(joined) == set(queue) == set(mapping)
    higher = {g['document_sha256']: g['higher_source_cluster']
              for g in json.loads(paths['identity'].read_text())['groups']}
    verified_sources = set()
    groups = collections.defaultdict(list)
    for key, audit in joined.items():
        item, meta = queue[key], mapping[key]
        assert item['document_sha256'] == audit['document_sha256']
        source = Path(item['document_path'])
        if source not in verified_sources:
            assert sha(source) == item['document_sha256']
            verified_sources.add(source)
        assert hashlib.sha256(item['answer'].encode()).hexdigest() == meta['answer_sha256']
        labels = collections.Counter(c['label'] for c in audit['major_claims'])
        for label in ['supported', 'contradicted', 'unverifiable']:
            assert labels[label] == audit[label + '_claims']
        assert audit['answer_factual_pass'] == (labels['supported'] > 0 and labels['contradicted'] == labels['unverifiable'] == 0)
        assert audit['question_coverage'] in [0, 1, 2]
        groups[str(meta['fraction'])].append(dict(audit,
            question_id=hashlib.sha256(item['question'].encode()).hexdigest(),
            group_index=meta['group_index'], question_index=meta['question_index'],
            resolved_sampling_seed=meta['seed'], source_path=meta['source_path'],
            fully_covered_and_factual=audit['answer_factual_pass'] and audit['question_coverage'] == 2))
    result = {
        'scope': 'Development one-seed LLM-assisted single-judge audit; budget mapping withheld during96 new judgments. Not human review, inter-judge agreement or formal noninferiority.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        'new_judgments': 96, 'reused_prior_judgments': 32,
        'reuse_rule': 'Exact source SHA, question, gold and answer identity verified again at join. Reused judgments are not independent samples or newly reviewed answers.',
        'limitations': ['Method family is known to the judge; one judge has general project context.',
                       'Claim counts are length-dependent; report factual pass jointly with coverage.',
                       'FastKVzip-NVIDIA global shared-prefix mask and QC-SnapKV per-head document retention use different budget allocation policies.'],
        'cohorts': {}, 'paired_higher_source_diagnostics': {}}
    for fraction, rows in sorted(groups.items(), key=lambda x: float(x[0])):
        rows.sort(key=lambda r: (r['group_index'], r['question_index']))
        assert len(rows) == len({(r['document_sha256'], r['question_id']) for r in rows}) == 32
        result['cohorts'][fraction] = {'answers': 32,
            'factual_pass': sum(r['answer_factual_pass'] for r in rows),
            'fully_covered_and_factual': sum(r['fully_covered_and_factual'] for r in rows),
            'uncertain_answers': sum(r['uncertain'] for r in rows),
            'coverage_counts': dict(collections.Counter(r['question_coverage'] for r in rows)),
            **{k: sum(r[k] for r in rows) for k in ['supported_claims', 'contradicted_claims', 'unverifiable_claims']},
            'per_answer': rows}
    refs = {(r['document_sha256'], r['question_id']): r for r in groups['1.0']}
    for fraction in ['0.1', '0.25', '0.5']:
        result['paired_higher_source_diagnostics'][fraction] = {}
        for metric in ['answer_factual_pass', 'fully_covered_and_factual', 'question_coverage']:
            rows = []
            for row in groups[fraction]:
                ref = refs[(row['document_sha256'], row['question_id'])]
                assert row['resolved_sampling_seed'] == ref['resolved_sampling_seed']
                divisor = 2 if metric == 'question_coverage' else 1
                rows.append(dict(document_id=row['document_sha256'], question_id=row['question_id'],
                    seed=0, cluster_id=higher[row['document_sha256']],
                    reference=ref[metric] / divisor, candidate=row[metric] / divisor))
            result['paired_higher_source_diagnostics'][fraction][metric] = paired_cluster_interval(rows, expected_seeds=[0])
    destination = ROOT / 'results/factual_audit/fastkvzip_train8_unblinded_v1.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {name: value for name, value in v.items() if name != 'per_answer'} for k, v in result['cohorts'].items()}, indent=2))


if __name__ == '__main__':
    main()
