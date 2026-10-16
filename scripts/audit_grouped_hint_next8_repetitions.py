"""Frozen whole-document repetition audit; phases/layers/tokens never count as independent n."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_grouped_hint_next8_production import summarize


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(allow_incomplete):
    path = ROOT / 'results/grouped_hint_next8_repetition_launch_v1.json'
    launch = json.loads(path.read_text())
    for name, digest in launch['frozen_analysis_sources'].items():
        assert sha(ROOT / name) == digest
    plans = [(0, ROOT / launch['primary_first_run_plan'], launch['primary_first_run_plan_sha256'])]
    plans += [(r['repeat_index'], ROOT / r['plan'], r['plan_sha256']) for r in launch['repeats']]
    rounds = []
    pending = []
    ledger = [json.loads(line) for line in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines()]
    for index, plan_path, digest in plans:
        assert sha(plan_path) == digest
        plan = json.loads(plan_path.read_text())
        report = summarize(plan_path, allow_incomplete=True)
        report['evaluation_role'] = 'original_primary_first_run' if index == 0 else 'same_inputs_repeat_stability_not_new_unseen_data'
        report['repeat_index'] = index
        for document, row in zip(plan['documents'], report['documents'], strict=True):
            if row['status'] == 'pending':
                ends = [r for r in ledger if r.get('event') == 'end' and r.get('name') == document['job_name']]
                if ends:
                    row.update(status='runtime_failure_retained', reason='Process terminated without completed paired result', terminal_receipts=ends)
                else:
                    pending.append([index, document['evaluation_index']])
            if row['status'] == 'paired_completed':
                summary_path = ROOT / document['summary_path']
                summary = json.loads(summary_path.read_text())
                first = summary['runs'][0]
                stop_equal = all([a['stop_reason'] for a in r['answers']] == [a['stop_reason'] for a in first['answers']] for r in summary['runs'])
                row['all_stop_reasons_equal'] = stop_equal
                if not stop_equal:
                    row.update(status='audit_failure_retained', reason='Stop reasons differ within natural ABBA')
                row['raw_phase_seconds'] = [{key:r[key] for key in ['phase', 'complete_generate_wall_seconds', 'anchor_wall_seconds', 'cold_shared_document_total_wall_seconds', 'full_phase_control_wall_seconds']} for r in summary['runs']]
        rounds.append({'repeat_index': index, 'plan': plan, 'report': report})
    if pending and not allow_incomplete:
        raise RuntimeError(f'Pending prescribed repetitions: {pending}')
    documents = []
    complete = []
    for position, original in enumerate(rounds[0]['plan']['documents']):
        number = original['evaluation_index']
        row = {'evaluation_index': number, 'document_id': original['document_id'], 'B': original['B'],
            'round_statuses': [r['report']['documents'][position]['status'] for r in rounds]}
        if not original['context_feasible']:
            row.update(status='configuration_failure_retained_all_rounds', reason=original['reason'])
            documents.append(row)
            continue
        results = [r['report']['documents'][position] for r in rounds]
        if any(r['status'] != 'paired_completed' for r in results):
            row['status'] = 'pending' if any(r['status'] == 'pending' for r in results) else 'repetition_failure_retained'
            documents.append(row)
            continue
        summaries = [json.loads((ROOT / r['plan']['documents'][position]['summary_path']).read_text()) for r in rounds]
        first = summaries[0]
        equality = {}
        for index, summary in enumerate(summaries[1:], 1):
            fields = ['document_sha256', 'prompt_token_hashes', 'prompt_lengths', 'common_prefix_tokens', 'B']
            checks = {key: summary[key] == first[key] for key in fields}
            phase_checks = {}
            for original_run, repeated_run in zip(first['runs'], summary['runs'], strict=True):
                assert original_run['phase'] == repeated_run['phase']
                entry = {key:[a[key] for a in original_run['answers']] == [a[key] for a in repeated_run['answers']] for key in ['tokens', 'num_cached_tokens', 'finish_reason', 'stop_reason']}
                for key in ['quality', 'sampling', 'graph_trajectory_sha256_by_rank']:
                    entry[key] = original_run[key] == repeated_run[key]
                phase_checks[original_run['phase']] = entry
            equality[str(index)] = {'inputs': checks, 'phases': phase_checks,
                'all_equal': all(checks.values()) and all(all(c.values()) for c in phase_checks.values())}
        row['cross_round_equality_to_original'] = equality
        if not all(r['all_equal'] for r in equality.values()):
            row['status'] = 'cross_round_reproduction_failure_retained'
            documents.append(row)
            continue
        metrics = {}
        for metric in results[0]['timings']:
            quantities = [r['timings'][metric] for r in results]
            ratios = [r['native_over_candidate'] for r in quantities]
            native = statistics.mean(r['native_seconds'] for r in quantities)
            candidate = statistics.mean(r['candidate_seconds'] for r in quantities)
            metrics[metric] = {'whole_process_runs': 3, 'original_primary_ratio_unchanged': ratios[0],
                'run_ratios': ratios, 'min_ratio': min(ratios), 'max_ratio': max(ratios),
                'arithmetic_mean_ratio': statistics.mean(ratios),
                'geometric_mean_ratio': math.exp(statistics.mean(math.log(v) for v in ratios)),
                'mean_native_seconds': native, 'mean_candidate_seconds': candidate,
                'ratio_of_three_run_mean_times': native / candidate}
        row.update(status='all_three_runs_reproduced', metrics=metrics,
            AA_max_over_min_by_run=[r['native_AA_generation_max_over_min'] for r in results],
            raw_phase_seconds_by_run=[r['raw_phase_seconds'] for r in results],
            setup_seconds_by_run=[r['unamortized_candidate_setup_seconds'] for r in results],
            quality_by_run=[r['quality'] for r in results],
            caps_by_run=[r['caps_per_arm'] for r in results],
            preemptions_by_run=[r['preemptions_per_arm_repeats'] for r in results])
        documents.append(row)
        complete.append(row)
    aggregate = {}
    if complete:
        for metric in complete[0]['metrics']:
            values = [r['metrics'][metric] for r in complete]
            aggregate[metric] = {'documents': len(values), 'whole_process_runs_per_document': 3,
                'geometric_mean_of_document_three_run_mean_time_ratios': math.exp(statistics.mean(math.log(v['ratio_of_three_run_mean_times']) for v in values)),
                'ratio_of_summed_document_mean_times': sum(v['mean_native_seconds'] for v in values) / sum(v['mean_candidate_seconds'] for v in values)}
    return {'status': 'incomplete' if pending else 'complete_repetition_audit', 'launch_sha256': sha(path),
        'analysis_sha256': sha(Path(__file__)), 'selected_document_denominator': 8, 'selected_question_denominator': 98,
        'requested_process_runs': 18, 'additional_process_runs': 12, 'pending': pending,
        'all_three_runs_reproduced_documents': len(complete), 'documents': documents,
        'round_reports': [r['report'] for r in rounds], 'conditional_all_three_complete_aggregates': aggregate,
        'limits': ['Original first run remains the predeclared primary result; supplemental repeats do not replace it.',
            'All six feasible inputs receive both extra runs. Two context failures remain in every eight-document denominator.',
            'Whole documents measure cross-document generalization. Repeated processes on identical inputs describe timing variability, not new unseen examples.',
            'No layer/token/rank/phase confidence interval or independent n is inferred. Report all ratios and ranges.',
            'Conditional complete-case aggregates do not assign invented speedups to context, runtime, numerical, audit or reproduction failures.',
            'Recorded allocation/visibility overlap and host contention require separate audits; this report does not establish resource isolation.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT / args.output
    assert not output.exists()
    result = audit(args.allow_incomplete)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'complete_documents': result['all_three_runs_reproduced_documents'], 'pending': result['pending']}))


if __name__ == '__main__':
    main()
