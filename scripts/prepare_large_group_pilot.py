"""Freeze the first two training reports with at least eight natural questions.

This extends the original B<=4 scope, not the held-out protocol. No model
output, masks, lengths or performance determine report/question selection.
"""
import collections
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def main():
    path = ROOT / 'data/manifests/docfinqa_public_training167_v1.json'
    source = json.loads(path.read_text())
    selected = [(i, g) for i, g in enumerate(source['groups']) if g['natural_question_count'] >= 8][:2]
    assert len(selected) == 2
    questions = {g['document_sha256']: {} for _, g in selected}
    validation = json.loads((ROOT / 'results/data_docfinqa_parquet_validation.json').read_text())
    source_hashes = {}
    for split in ['train', 'validation', 'test']:
        raw = ROOT / f'data/raw/docfinqa/parquet/{split}.parquet'
        actual = sha(raw.read_bytes())
        expected = next(r for r in validation['sources'] if r['path'] == str(raw.relative_to(ROOT)))
        assert actual == expected['sha256']
        source_hashes[str(raw.relative_to(ROOT))] = actual
        row_index = 0
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                dh = sha(row['Context'])
                if dh in questions:
                    qh = sha(row['Question'])
                    item = questions[dh].setdefault(qh, dict(question=row['Question'], question_sha256=qh,
                        answer=row['Answer'], source_occurrences=[]))
                    item['source_occurrences'].append(dict(split=split, row=row_index, answer=row['Answer'],
                        program_sha256=sha(row['Program'])))
                row_index += 1
    groups = []
    for index, original in selected:
        qs = questions[original['document_sha256']]
        assert len(qs) == original['natural_question_count']
        chosen = [qs[key] for key in sorted(qs)[:8]]
        assert [q['question_sha256'] for q in chosen[:4]] == [q['question_sha256'] for q in original['questions']]
        for q in chosen:
            primary = min(q['source_occurrences'], key=lambda o: (['train', 'validation', 'test'].index(o['split']), o['row']))
            q.update(answer=primary['answer'], answer_source=dict(split=primary['split'], row=primary['row']),
                source_answer_variants=sorted({o['answer'] for o in q['source_occurrences']}))
            q['conflicting_source_answers'] = len(q['source_answer_variants']) > 1
        assert sha((ROOT / original['document_path']).read_bytes()) == original['document_sha256']
        groups.append(dict(original, questions=chosen, B=8, original_training_group_index=index))
    output = ROOT / 'data/manifests/docfinqa_training_large_group_pilot_v1.json'
    report = dict(schema=1, scope='Exploratory natural B8 capture only; no formal quality or speedup claim',
        source_manifest=str(path.relative_to(ROOT)), source_manifest_sha256=sha(path.read_bytes()),
        source_parquet_sha256=source_hashes,
        selection_rule='First two groups in frozen training167 order with natural_question_count>=8; first eight exact-context questions sorted by question SHA. No length, answer, mask or performance selection.',
        original_B_cap_changed=True, held_out_manifests_unchanged=True,
        all_training_natural_question_count_histogram=dict(collections.Counter(g['natural_question_count'] for g in source['groups'])),
        groups=groups)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    jobs = []
    for slot, group in enumerate(groups):
        name = f'main30b_tp2_train{group["original_training_group_index"]}_b8_256'
        jobs.append(dict(name='capture_' + name, category='P1', gpus=[2 + slot*2, 3 + slot*2], timeout_seconds=600,
            argv=['.venv-engine/bin/python', 'scripts/probe_engine_docfinqa.py', '--model',
                'data/models/Qwen3-30B-A3B-Thinking-2507-FP8', '--manifest', str(output.relative_to(ROOT)),
                '--document-id', group['document_id'], '--output', f'results/platform/capture_{name}.json',
                '--max-model-len', '262144', '--gpu-memory-utilization', '.93', '--max-new-tokens', '256',
                '--max-num-seqs', '8', '--capture-dir', f'results/captures/{name}'],
            env=dict(VLLM_WORKER_MULTIPROC_METHOD='spawn', NCCL_DEBUG='INFO', NCCL_P2P_DISABLE='1',
                HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(ROOT))))
    (ROOT / 'configs/capture_main30b_training_b8.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), selections=[dict(index=g['original_training_group_index'], document=g['document_id'], natural_questions=g['natural_question_count']) for g in groups]), indent=2))


if __name__ == '__main__':
    main()
