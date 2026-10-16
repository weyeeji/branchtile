"""Independent input audit; never loads generated outputs or changes selection."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import pyarrow.parquet as pq
    path = ROOT / 'data/manifests/docfinqa_evaluation_natural_next8_cohort6plus_v1.json'
    manifest = json.loads(path.read_text())
    source = json.loads((ROOT / manifest['source_manifest']).read_text())
    eligible = [g for g in source['groups'] if g['natural_question_count'] >= 6]
    selected = manifest['groups']
    assert [g['document_id'] for g in selected] == [g['document_id'] for g in eligible[8:16]]
    for name, expected in manifest['protected_source_sha256'].items():
        assert file_sha(ROOT / name) == expected, name
    rebuilt = {g['document_sha256']: collections.defaultdict(list) for g in selected}
    splits = ['train', 'validation', 'test']
    raw_count = {}
    for split in splits:
        name = f'data/raw/docfinqa/parquet/{split}.parquet'
        raw = ROOT / name
        assert file_sha(raw) == manifest['source_parquet_sha256'][name]
        i = 0
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                ch = sha_bytes(row['Context'].encode())
                if ch in rebuilt:
                    qh = sha_bytes(row['Question'].encode())
                    rebuilt[ch][qh].append({'split': split, 'row': i, 'answer': row['Answer'],
                        'program_sha256': sha_bytes(row['Program'].encode()), 'question': row['Question']})
                i += 1
        raw_count[split] = i
    finqa = collections.defaultdict(list)
    finqa_sources = {}
    for split in ['train', 'dev', 'test']:
        raw = ROOT / f'data/raw/finqa/{split}.json'
        finqa_sources[str(raw.relative_to(ROOT))] = file_sha(raw)
        for row in json.loads(raw.read_text()):
            finqa[row['qa']['question']].append({'split': split, 'id': row['id'], 'qa': row['qa']})
    records = []
    for group in selected:
        document = ROOT / group['document_path']
        assert file_sha(document) == group['document_sha256']
        actual = rebuilt[group['document_sha256']]
        assert len(actual) == group['natural_question_count'] == len(group['questions'])
        assert sorted(actual) == [q['question_sha256'] for q in group['questions']]
        questions = []
        for question in group['questions']:
            qh = question['question_sha256']
            assert sha_bytes(question['question'].encode()) == qh
            occurrences = []
            for row in actual[qh]:
                assert row['question'] == question['question']
                occurrences.append({k:v for k,v in row.items() if k != 'question'})
            assert occurrences == question['source_occurrences']
            primary = min(occurrences, key=lambda x:(splits.index(x['split']), x['row']))
            assert question['answer'] == primary['answer']
            variants = sorted({x['answer'] for x in occurrences})
            assert variants == question['source_answer_variants']
            assert question['conflicting_source_answers'] == (len(variants) > 1)
            matches = finqa[question['question']]
            questions.append({'question_sha256': qh, 'source_occurrences': len(occurrences),
                'conflicting_source_answers': question['conflicting_source_answers'],
                'exact_question_FinQA_matches': [{'split': m['split'], 'id': m['id']} for m in matches],
                'FinQA_match_scope': 'Question text only; not proof that a particular reference or table is preserved in DocFinQA full context.'})
        records.append({'evaluation_index': group['original_evaluation_group_index'],
            'document_id': group['document_id'], 'B': len(questions),
            'complete_original_context_hash_verified': True,
            'all_natural_questions_and_source_occurrences_verified': True, 'questions': questions})
    output = ROOT / 'results/grouped_hint_next8_source_integrity_v1.json'
    assert not output.exists()
    output.write_text(json.dumps({'status': 'source_integrity_verified', 'uid': os.geteuid(),
        'python': sys.executable, 'CUDA_VISIBLE_DEVICES': '', 'imports_torch_or_CUDA': False,
        'selection_manifest_sha256': file_sha(path), 'source_script_sha256': file_sha(Path(__file__)),
        'selected_documents': len(records), 'selected_questions': sum(r['B'] for r in records),
        'parquet_rows': raw_count, 'parquet_hashes': manifest['source_parquet_sha256'],
        'finqa_hashes': finqa_sources, 'reads_model_outputs': False,
        'scope': 'Independent re-reconstruction of full exact-context natural cohorts. Does not establish semantic reference correctness, source table completeness, or model quality; does not change inputs, scoring, or eligibility.',
        'records': records}, indent=2) + '\n')
    print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
