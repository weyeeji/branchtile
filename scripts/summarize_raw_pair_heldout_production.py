"""Frozen heldout summary: eight selected documents, all failures retained."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_native_raw_pair_production_setup import audit as audit_production


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(plan_path,allow_incomplete=False):
    plan=json.loads(plan_path.read_text())
    for name,expected in plan['frozen_source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    manifest=json.loads((ROOT/plan['selection_manifest']).read_text())
    assert sha(ROOT/plan['selection_manifest'])==plan['selection_manifest_sha256']
    assert [g['document_id'] for g in manifest['groups']]==[d['document_id'] for d in plan['documents']]
    assert len(plan['documents'])==8
    rows=[];paired=[];pending=[]
    for document,group in zip(plan['documents'],manifest['groups'],strict=True):
        row={'document_id':document['document_id'],'evaluation_index':document['evaluation_index'],'B':document['B']}
        assert row['B']==group['natural_question_count']==len(group['questions'])
        if not document['context_feasible']:
            row.update(status='configuration_failure_retained',reason=document['reason'])
            rows.append(row);continue
        summary_path=ROOT/document['summary_path']
        if not summary_path.exists():
            row.update(status='pending',reason='No output summary yet');pending.append(row['document_id']);rows.append(row);continue
        summary=json.loads(summary_path.read_text())
        if summary['status'] not in ('completed','failed'):
            terminal=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()
                if json.loads(line).get('event')=='end' and json.loads(line).get('name')==document['job_name']]
            if not terminal:
                row.update(status='pending',reason=summary['status']);pending.append(row['document_id']);rows.append(row);continue
            row.update(status='runtime_failure_retained',reason='Terminal process without completed result',terminal_receipts=terminal)
            rows.append(row);continue
        row['summary_sha256']=sha(summary_path)
        if summary['status']=='failed':
            row.update(status='runtime_failure_retained',reason=summary.get('error','failed'));rows.append(row);continue
        try:
            assert summary['args']['document_id']==document['document_id']
            assert summary['args']['protocol']==plan['protocol'] and summary['protocol_sha256']==plan['protocol_sha256']
            assert summary['args']['phase_kind']=='natural' and not summary['args']['setup_only']
            assert summary['source_sha256']['scripts/benchmark_native_raw_pair_production_setup.py']==plan['frozen_source_sha256']['scripts/benchmark_native_raw_pair_production_setup.py']
            assert summary['offline_inputs']==plan['offline_inputs']
            audited=audit_production(summary_path)
            assert audited['status']=='qualified'
        except (AssertionError,KeyError,ValueError,OSError) as exc:
            row.update(status='audit_failure_retained',reason=f'{type(exc).__name__}: {exc}')
            rows.append(row);continue
        native=[r for r in summary['runs'] if r['method']=='native']
        candidate=[r for r in summary['runs'] if r['method']=='cascade']
        assert len(native)==len(candidate)==2
        means={arm:{name:statistics.mean(r[name] for r in runs) for name in
            ('complete_generate_wall_seconds','cold_shared_document_total_wall_seconds','full_phase_control_wall_seconds')}
            for arm,runs in (('native',native),('candidate',candidate))}
        setup_components={name:summary[name] for name in ('offline_preflight_wall_seconds',
            'candidate_configuration_wall_seconds','candidate_graph_capture_wall_seconds','setup_receipt_export_wall_seconds')}
        setup=sum(setup_components.values())
        quantities={
            'ready_generation':(means['native']['complete_generate_wall_seconds'],means['candidate']['complete_generate_wall_seconds']),
            'cold_anchor_plus_generation':(means['native']['cold_shared_document_total_wall_seconds'],means['candidate']['cold_shared_document_total_wall_seconds']),
            'net_cold_with_unamortized_per_document_setup':(means['native']['cold_shared_document_total_wall_seconds'],means['candidate']['cold_shared_document_total_wall_seconds']+setup),
            'full_phase_control':(means['native']['full_phase_control_wall_seconds'],means['candidate']['full_phase_control_wall_seconds']),
            'full_phase_control_with_unamortized_per_document_setup':(means['native']['full_phase_control_wall_seconds'],means['candidate']['full_phase_control_wall_seconds']+setup)}
        row.update(status='paired_completed',setup_components_seconds=setup_components,unamortized_candidate_setup_seconds=setup,
            timings={name:{'native_seconds':a,'candidate_seconds':b,'native_over_candidate':a/b} for name,(a,b) in quantities.items()},
            all_tokens_APC_finishes_graph_trajectories_equal=True,
            candidate_executed_in_both_arms=summary['comparisons']['natural']['candidate_speed_claim_eligible'],
            native_AA_generation_max_over_min=summary['comparisons']['natural']['native_AA_wall_max_over_min'],
            output_lengths=[len(a['tokens']) for a in native[0]['answers']],
            quality={arm:{k:runs[0]['quality']['score'][k] for k in ('denominator','exact_accuracy','rounded_accuracy','prediction_parse_failures','gold_parse_failures')}
                     for arm,runs in (('native',native),('candidate',candidate))},
            caps_per_arm={arm:len(runs[0]['cap_hits']) for arm,runs in (('native',native),('candidate',candidate))},
            preemptions_per_arm_repeats={arm:[r['total_request_preemptions'] for r in runs] for arm,runs in (('native',native),('candidate',candidate))},
            candidate_replay_coverage=[r['ranks'] for r in audited['phases'] if r['phase'] in ('natural_cascade_1','natural_cascade_2')],
            fresh_process_component_estimate=summary['first_use_component_estimates']['natural'])
        rows.append(row);paired.append(row)
    if pending and not allow_incomplete:
        raise RuntimeError(f'Pending selected documents; refuse final aggregate: {pending}')
    aggregate={}
    if paired:
        for metric in paired[0]['timings']:
            ratios=[r['timings'][metric]['native_over_candidate'] for r in paired]
            n=sum(r['timings'][metric]['native_seconds'] for r in paired)
            c=sum(r['timings'][metric]['candidate_seconds'] for r in paired)
            aggregate[metric]={'paired_documents':len(paired),'geometric_mean_native_over_candidate':math.exp(statistics.mean(math.log(v) for v in ratios)),
                'total_task_time_native_over_candidate':n/c,'sum_native_seconds':n,'sum_candidate_seconds':c,
                'min_document_ratio':min(ratios),'max_document_ratio':max(ratios)}
    questions=sum(d['B'] for d in plan['documents'])
    verified_questions=sum(d['B'] for d in paired)
    correct={arm:{kind:sum(round(r['quality'][arm][kind]*r['B']) for r in paired) for kind in ('exact_accuracy','rounded_accuracy')}
        for arm in ('native','candidate')}
    return {'status':'incomplete' if pending else 'complete_selected_evaluation','analysis_source_sha256':sha(Path(__file__)),
        'launch_plan_sha256':sha(plan_path),'primary_metric':'net_cold_with_unamortized_per_document_setup',
        'selected_document_denominator':8,'selected_question_denominator':questions,
        'context_feasible_documents':sum(d['context_feasible'] for d in plan['documents']),
        'paired_completed_documents':len(paired),'paired_completed_document_fraction':len(paired)/8,
        'verified_question_coverage':verified_questions/questions,
        'verified_correct_answer_counts':correct,
        'verified_correct_answer_fraction_of_all_selected_questions':{arm:{kind:n/questions for kind,n in by_kind.items()} for arm,by_kind in correct.items()},
        'documents':rows,'conditional_paired_timing_aggregates':aggregate,'offline_qualification_cost':plan['offline_qualification_cost'],
        'limits':['Timing aggregates are conditional on paired completed documents; all eight selected documents and 101 questions remain in coverage/failure denominators.',
            'Configuration/runtime/audit failures have no fabricated speed ratio. Native-only policy fallback cases remain included among paired completed documents.',
            'No confidence interval is inferred from tokens or two same-session ABBA repeats. Geometric mean and total-time ratio were both frozen before heldout inference.',
            'Primary net cold charges every document full measured preflight/configuration/production-graph/setup-receipt costs once; no amortization across documents.',
            'Common model initialization and tokenization are excluded from warm-service document latency; separate first-use component estimates are not independently timed cold services.',
            'Offline training states establish per-layer bitwise evidence. Heldout verifies complete output tokens and graph input trajectories, not every heldout hidden value.',
            'Correct-answer fractions across all selected questions are verified lower bounds if any runtime/audit result is missing or rejected; configuration failures receive no completed answers.',
            'Any external GPU/compilation overlap must be reported alongside timing; clean process exit does not prove an interference-free timing environment.']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='results/native_raw_pair_heldout_production_launch_v1.json')
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/args.output;assert not output.exists()
    report=summarize(ROOT/args.plan,args.allow_incomplete)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'paired_completed_documents':report['paired_completed_documents'],'conditional_paired_timing_aggregates':report['conditional_paired_timing_aggregates']},indent=2))


if __name__=='__main__':
    main()
