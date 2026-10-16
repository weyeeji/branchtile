"""Frozen FINAL scoring and complete task-cost audit for dense cascade natural runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_application import extract_final
from src.financial_answer_metric import METRIC_ID, score_pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return {'count': len(values), 'sum': sum(values), 'median': statistics.median(values), 'max': max(values)} if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    report = {'status': 'completed', 'metric_id': METRIC_ID,
        'metric_source_sha256': sha(ROOT / 'src/financial_answer_metric.py'),
        'thinking_extraction_source_sha256': sha(ROOT / 'scripts/screen_application.py'),
        'documents': [], 'limitations': [
            'Two training documents, one seed and one natural run per arm do not establish quality noninferiority or general speedup.',
            'All current KV support retained; floating-point differences can change sampling trajectories and task completion length.',
            'Natural wall ratios describe actual observed task outcomes, not isolated operator acceleration. Output token/s is descriptive only.',
            'Strict FINAL exact/rounded match is the frozen metric, not official program execution accuracy; unknown gold units remain unverified.',
            'The existing prompt requests a numerical answer even for boolean questions; answer-type failures are preserved, not rescored.',
            'First-use candidate setup is added once. Amortized scenarios are arithmetic projections, not independently measured sessions.',
            'Rank prepare times and CPU step intervals overlap other work; do not sum ranks or equate these with GPU kernel time.']}
    for source in sorted(Path(args.directory).glob('*/summary.json')):
        d = json.loads(source.read_text())
        assert d['status'] == 'completed'
        manifest = Path(d['protocol']['manifest'])
        group = next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id'] == d['args']['document_id'])
        assert group['document_sha256'] == d['document_sha256']
        questions = group['questions']
        item = {'document_id': d['args']['document_id'], 'source': str(source), 'source_sha256': sha(source),
            'manifest': str(manifest), 'manifest_sha256': sha(manifest), 'B': d['B'],
            'common_prefix_tokens': d['common_prefix_tokens'], 'rows': [],
            'one_time_costs': {k: d[k] for k in ['input_preparation_wall_seconds', 'initialization_wall_seconds', 'candidate_plan_and_capture_wall_seconds']}}
        for run in d['runs']:
            if not run['natural_eos_enabled']:
                continue
            assert not run['validation'] and len(run['answers']) == len(questions)
            predictions, answers = [], []
            for index, (q, a) in enumerate(zip(questions, run['answers'])):
                assert a['question_index'] == index
                final, reasoning, parse = extract_final(a['text'], thinking=True)
                eligible = a['finish_reason'] == 'stop' and parse == 'closed_thinking'
                predictions.append(final if eligible else '')
                answers.append({'question_index': index, 'question_id': q['question_sha256'], 'question': q['question'],
                    'gold': q['answer'], 'final_text': final, 'final_last_line': final.splitlines()[-1] if final else '',
                    'finish_reason': a['finish_reason'], 'stop_reason': a['stop_reason'],
                    'output_tokens': len(a['tokens']), 'completion_eligible': eligible, 'thinking_parse': parse,
                    'text_sha256': hashlib.sha256(a['text'].encode()).hexdigest(), 'num_cached_tokens': a['num_cached_tokens']})
            scores = score_pairs(predictions, [q['answer'] for q in questions])
            for a, score in zip(answers, scores.pop('per_example')):
                a['score'] = score
            assert sum(a['output_tokens'] for a in answers) == run['total_output_tokens']
            workers = []
            for worker in run['workers']:
                graph_path = Path(worker['graph_metadata'])
                graph = json.loads(graph_path.read_text())
                replays = graph['replays']
                transitions = []
                for replay in replays:
                    state = (replay['batch'], replay['questions'])
                    if not transitions or state != (transitions[-1]['batch'], transitions[-1]['questions']):
                        transitions.append({'batch': state[0], 'questions': state[1]})
                by_batch = {}
                for batch in range(1, 9):
                    entries = [r for r in replays if r['batch'] == batch]
                    intervals = [(b['step_prepare_started_monotonic_seconds'] - a['step_prepare_started_monotonic_seconds']) * 1000 for a, b in zip(replays, replays[1:]) if a['batch'] == b['batch'] == batch]
                    by_batch[str(batch)] = {'replays': len(entries), 'step_interval_ms': stats(intervals),
                        'additional_cascade_prepare_ms': stats([r['additional_cascade_prepare_host_seconds'] * 1000 for r in entries if 'additional_cascade_prepare_host_seconds' in r])}
                workers.append({'rank': worker['rank'], 'graph': str(graph_path), 'sha256': sha(graph_path),
                    'batch_replay_counts': dict(Counter(r['batch'] for r in replays)), 'transitions': transitions,
                    'candidate_graph_replays': sum(r.get('candidate_graph_replayed', False) for r in replays), 'by_batch': by_batch})
            setup = d['candidate_plan_and_capture_wall_seconds'] if run['method'] == 'cascade' else 0
            wall, anchor = run['complete_generate_wall_seconds'], run['anchor_wall_seconds']
            assert abs(wall + anchor - run['cold_shared_document_total_wall_seconds']) < 1e-9
            item['rows'].append({'phase': run['phase'], 'method': run['method'], 'sampling': run['sampling'],
                'JCT_seconds': wall, 'anchor_seconds': anchor, 'anchor_plus_JCT_seconds': wall + anchor,
                'first_use_incremental_candidate_setup_seconds': setup,
                'first_use_anchor_setup_JCT_seconds': wall + anchor + setup,
                'total_output_tokens': run['total_output_tokens'], 'per_question_output_tokens': [a['output_tokens'] for a in answers],
                'EOS_count': sum(a['finish_reason'] == 'stop' for a in answers), 'cap_hit_count': sum(a['finish_reason'] == 'length' for a in answers),
                'eligible_count': sum(a['completion_eligible'] for a in answers), 'score': scores, 'answers': answers, 'workers': workers})
        native = next(r for r in item['rows'] if r['method'] == 'native')
        candidate = next(r for r in item['rows'] if r['method'] == 'cascade')
        assert native['sampling'] == candidate['sampling']
        item['observed_native_over_cascade'] = {k: native[k] / candidate[k] for k in ['JCT_seconds', 'anchor_plus_JCT_seconds', 'first_use_anchor_setup_JCT_seconds']}
        item['amortized_setup_scenarios'] = [{'groups': n, 'native_seconds_per_group': native['anchor_plus_JCT_seconds'],
            'cascade_seconds_per_group': candidate['anchor_plus_JCT_seconds'] + d['candidate_plan_and_capture_wall_seconds'] / n} for n in [1, 2, 10, 100]]
        item['paired_question_changes'] = [{'question_index': i, 'native_final': a['final_last_line'], 'cascade_final': b['final_last_line'],
            'native_score_status': a['score']['status'], 'cascade_score_status': b['score']['status'],
            'exact_delta': int(b['score']['exact_match']) - int(a['score']['exact_match']),
            'rounded_delta': int(b['score']['rounded_match']) - int(a['score']['rounded_match']),
            'native_tokens': a['output_tokens'], 'cascade_tokens': b['output_tokens']} for i, (a, b) in enumerate(zip(native['answers'], candidate['answers']))]
        report['documents'].append(item)
    assert report['documents']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{'document': d['document_id'], 'ratios': d['observed_native_over_cascade'],
        'rows': [{k: r[k] for k in ['method', 'JCT_seconds', 'anchor_plus_JCT_seconds', 'first_use_anchor_setup_JCT_seconds', 'EOS_count', 'cap_hit_count', 'total_output_tokens', 'score']} for r in d['rows']]} for d in report['documents']], indent=2))


if __name__ == '__main__':
    main()
