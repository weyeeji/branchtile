"""Preserve the failed v1 pilot; recover SC1 only as a posthoc diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_application import extract_final
from src.financial_answer_metric import score_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = ROOT / args.output
    assert not out.exists()
    path = ROOT / 'results/native_self_consistency_train26_v1/summary.json'
    summary = json.loads(path.read_text())
    assert summary['status'] == 'failed' and "KeyError: '2'" in summary['error']
    assert len(summary['runs']) == 1 and summary['runs'][0]['samples_per_question'] == 1
    run = summary['runs'][0]
    assert 'quality' not in run
    recovery = []
    for worker in run['workers']:
        queue = json.loads(Path(worker['queue_metadata']).read_text())
        admissions = queue['sample_admissions']
        assert len(admissions) == len(run['answers']) == 21
        matched = []
        for answer in run['answers']:
            rows = [r for r in admissions if re.fullmatch(re.escape(answer['request_id']) + r'-[0-9a-f]{8}', r['request_id'])]
            assert len(rows) == 1
            row = rows[0]
            assert row['seed'] == answer['sample']['seed'] and row['question_index'] == answer['sample']['question_index']
            matched.append({'external_id': answer['request_id'], 'internal_id': row['request_id'],
                'question_index': row['question_index'], 'seed': row['seed']})
        recovery.append({'rank': worker['rank'], 'queue_sha256': sha(Path(worker['queue_metadata'])), 'matched': matched})
    assert recovery[0]['matched'] == recovery[1]['matched']
    base = json.loads((ROOT / summary['protocol']['base_protocol']).read_text())
    group = next(g for g in json.loads((ROOT / base['manifest']).read_text())['groups'] if g['document_id'] == summary['protocol']['document_id'])
    predictions = []
    for answer in run['answers']:
        final, _, state = extract_final(answer['text'], thinking=True)
        predictions.append(final if answer['finish_reason'] == 'stop' and state == 'closed_thinking' else '')
    scores = score_pairs(predictions, [q['answer'] for q in group['questions']])
    input_source = next((ROOT / '.venv-engine/lib').glob('python*/site-packages/vllm/v1/engine/input_processor.py'))
    lines = input_source.read_text()
    assert 'request.external_req_id = request.request_id' in lines
    assert 'request.request_id = f"{request.external_req_id}-{random_uuid():.8}"' in lines
    report = {'status': 'original_failure_retained_posthoc_SC1_diagnostic_only',
        'source_sha256': sha(Path(__file__)), 'original_summary_sha256': sha(path),
        'original_error': summary['error'], 'completed_utility_phases': 0, 'generated_unscored_phases': 1,
        'SC4_was_run': False, 'input_processor_source_sha256': sha(input_source),
        'posthoc_unique_internal_suffix_recovery': recovery,
        'SC1_posthoc_score': scores, 'original_cost_fields': {k: run[k] for k in
            ['complete_generate_wall_seconds', 'cold_anchor_wall_seconds', 'cold_document_task_wall_seconds', 'total_output_tokens', 'cap_hits', 'preemptions']},
        'limits': ['Runtime validation failed before scoring; original failed status is not reclassified.',
            'Unique suffix reconstruction is posthoc diagnostic supported by observed IDs and local source, not a replacement for recorded runtime transport mapping.',
            'No SC4 samples or self-consistency utility improvement were measured in this process.',
            'New v2 records actual external/internal mapping while scheduler paused, rather than relying on string reconstruction.',
            'No candidate ran and no candidate speed claim is supported.']}
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'SC1_exact': scores['exact_accuracy'], 'SC1_rounded': scores['rounded_accuracy']}))


if __name__ == '__main__':
    main()
