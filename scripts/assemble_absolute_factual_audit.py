"""Join absolute-budget blind judgments and explicit saturated-full references."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paired_statistics import paired_cluster_interval
from assemble_fastkvzip_factual_audit import read_rows, sha


def main():
    paths = {
        'quality': ROOT / 'results/sparse_reference_absolute/train8_quality.json',
        'queue': ROOT / 'results/sparse_reference_absolute/train8_quality_blind_audit_new56.jsonl',
        'mapping': ROOT / 'results/sparse_reference_absolute/train8_quality_blind_audit_mapping.json',
        'reviews': ROOT / 'results/factual_audit/absolute_new56_claims_v1.jsonl',
        'links': ROOT / 'results/sparse_reference_absolute/train8_quality_prior_audit_reuse_mapping.json',
        'prior_queue': ROOT / 'results/sparse_reference/train8_quality_blind_audit.jsonl',
        'prior_mapping': ROOT / 'results/sparse_reference/train8_quality_blind_audit_mapping.json',
        'prior_reviews': ROOT / 'results/factual_audit/train8_claims_v1.jsonl',
    }
    queue, reviews, prior_queue, prior_reviews = [read_rows(paths[k]) for k in ['queue', 'reviews', 'prior_queue', 'prior_reviews']]
    mapping = json.loads(paths['mapping'].read_text())['items']
    assert len(queue) == len(reviews) == len(mapping) == 56
    assert set(queue) == set(reviews) == set(mapping)
    links_file = json.loads(paths['links'].read_text())
    assert sha(paths['prior_queue']) == links_file['prior_queue_sha256']
    assert sha(paths['prior_mapping']) == links_file['prior_map_sha256']
    quality = json.loads(paths['quality'].read_text())
    assert sha(ROOT / quality['source']) == quality['source_sha256']
    new_by_key = {(r['group_index'], r['question_index'], str(r['requested_budget'])): key for key, r in mapping.items()}
    links = {(r['group_index'], r['question_index'], str(r['requested_budget'])): r for r in links_file['links']}
    assert len(new_by_key) == 56 and len(links) == 104 and not set(new_by_key).intersection(links)
    prior_map = json.loads(paths['prior_mapping'].read_text())['items']
    prior_full = {(r['group_index'], r['question_index']): key for key, r in prior_map.items() if r['fraction'] == 1.0}
    groups = collections.defaultdict(list)
    comparison_duplicates = []
    checked_sources = set()
    for budget, cohort in quality['cohorts'].items():
        for row in cohort['per_question']:
            key = (row['group_index'], row['question_index'], budget)
            if row['new_generation_performed']:
                item_id = new_by_key[key]
                candidate, audit = queue[item_id], reviews[item_id]
                reused = False
                assert hashlib.sha256(candidate['answer'].encode()).hexdigest() == mapping[item_id]['answer_sha256']
            else:
                link = links[key]
                assert link['source_question_gold_answer_identical']
                assert link['source_generation_id'] == row['source_generation_id']
                assert sha(ROOT / link['source_path']) == link['source_sha256']
                item_id = link['prior_item_id']
                candidate, audit = prior_queue[item_id], prior_reviews[item_id]
                reused = True
            for actual, expected in [(candidate['document_sha256'], row['document_sha256']),
                                     (candidate['question'], row['question']),
                                     (candidate['gold_reference'], row['reference']),
                                     (candidate['answer'], row['prediction'])]:
                assert actual == expected, (key, 'candidate identity changed')
            source = ROOT / row['document_path']
            if source not in checked_sources:
                assert sha(source) == row['document_sha256']
                checked_sources.add(source)
            assert audit['document_sha256'] == row['document_sha256']
            labels = collections.Counter(c['label'] for c in audit['major_claims'])
            for label in ['supported', 'contradicted', 'unverifiable']:
                assert labels[label] == audit[label + '_claims']
            assert audit['answer_factual_pass'] == (labels['supported'] > 0 and labels['contradicted'] == labels['unverifiable'] == 0)
            assert audit['question_coverage'] in [0, 1, 2]
            item = dict(audit, nominal_id=f'g{key[0]}:q{key[1]}:b{budget}',
                audit_origin_item_id=item_id, judgment_reused=reused,
                source_generation_id=row['source_generation_id'], requested_budget=budget,
                actual_document_fraction=row['actual_document_fraction'],
                question_id=hashlib.sha256(row['question'].encode()).hexdigest(),
                group_index=key[0], question_index=key[1], resolved_sampling_seed=row['seed'],
                higher_source_cluster=row['higher_source_cluster'],
                fully_covered_and_factual=audit['answer_factual_pass'] and audit['question_coverage'] == 2)
            groups[budget].append(item)
            if not reused and row['token_identical_to_prior_full']:
                old_id = prior_full[key[:2]]
                old_candidate, old_audit = prior_queue[old_id], prior_reviews[old_id]
                for field in ['document_sha256', 'question', 'gold_reference', 'answer']:
                    assert candidate[field] == old_candidate[field]
                comparison_duplicates.append({'new_item_id': item_id, 'prior_item_id': old_id,
                    'source_generation_id': row['source_generation_id'],
                    'scores': {field: [old_audit[field], audit[field]] for field in ['answer_factual_pass', 'question_coverage', 'uncertain']},
                    'rule': 'Identical outputs judged in two batches by the same assisted judge; retain both versions, not independent inter-rater evidence.'})
    result = {'scope': 'Absolute-budget development, one seed, one LLM-assisted budget-blind judge. No human or formal noninferiority claim.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        'nominal_rows': 160, 'new_judgments': 56, 'reused_nominal_judgments': 104,
        'unique_prior_full_answers': 32, 'reused_rows_are_independent_samples': False,
        'limitations': ['Saturated budgets reuse original32 full generations;104 references are not104 new generations or judgments.',
                       'Claim counts depend on answer length. All metrics are descriptive development evidence.',
                       'Four newly generated answers equal prior full outputs and have separate same-judge verdicts; discrepancies are explicit and never silently adjudicated.'],
        'identical_output_repeat_judgments': comparison_duplicates, 'cohorts': {}, 'paired_higher_source_diagnostics': {}}
    for budget, rows in groups.items():
        assert len(rows) == len({(r['document_sha256'], r['question_id']) for r in rows}) == 32
        result['cohorts'][budget] = {'answers': 32,
            'reused_judgments': sum(r['judgment_reused'] for r in rows),
            'factual_pass': sum(r['answer_factual_pass'] for r in rows),
            'fully_covered_and_factual': sum(r['fully_covered_and_factual'] for r in rows),
            'uncertain_answers': sum(r['uncertain'] for r in rows),
            'coverage_counts': dict(collections.Counter(r['question_coverage'] for r in rows)),
            **{k: sum(r[k] for r in rows) for k in ['supported_claims', 'contradicted_claims', 'unverifiable_claims']},
            'per_answer': rows}
    refs = {(r['document_sha256'], r['question_id']): r for r in groups['full']}
    for budget in ['4096', '8192', '16384', '32768']:
        result['paired_higher_source_diagnostics'][budget] = {}
        for metric in ['answer_factual_pass', 'fully_covered_and_factual', 'question_coverage']:
            paired = []
            for row in groups[budget]:
                ref = refs[(row['document_sha256'], row['question_id'])]
                assert row['resolved_sampling_seed'] == ref['resolved_sampling_seed']
                divisor = 2 if metric == 'question_coverage' else 1
                paired.append(dict(document_id=row['document_sha256'], question_id=row['question_id'], seed=0,
                    cluster_id=row['higher_source_cluster'], reference=ref[metric] / divisor, candidate=row[metric] / divisor))
            result['paired_higher_source_diagnostics'][budget][metric] = paired_cluster_interval(paired, expected_seeds=[0])
    destination = ROOT / 'results/factual_audit/absolute_train8_unblinded_v1.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {name: value for name, value in v.items() if name != 'per_answer'} for k, v in result['cohorts'].items()}, indent=2))
    print(json.dumps(comparison_duplicates, indent=2))


if __name__ == '__main__':
    main()
