"""Secondary sensitivity only: faster observed native cold task, unchanged candidate/setup."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_grouped_hint_next8_production import summarize


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def aggregate(rows):
    if not rows:
        return {'completed_process_runs': 0}
    return {'completed_process_runs': len(rows),
        'geometric_mean_primary': math.exp(statistics.mean(math.log(r['primary_net']) for r in rows)),
        'geometric_mean_faster_native_sensitivity': math.exp(statistics.mean(math.log(r['faster_native_net']) for r in rows)),
        'primary_ratio_of_summed_times': sum(r['primary_native_cold_mean_seconds'] for r in rows)/sum(r['candidate_mean_plus_setup_seconds'] for r in rows),
        'faster_native_ratio_of_summed_times': sum(r['faster_native_cold_seconds'] for r in rows)/sum(r['candidate_mean_plus_setup_seconds'] for r in rows),
        'generation_A0_slower_than_A3_count': sum(r['generation_A0_over_A3'] > 1 for r in rows),
        'generation_A0_over_A3_values': [r['generation_A0_over_A3'] for r in rows],
        'cold_A0_slower_than_A3_count': sum(r['cold_A0_over_A3'] > 1 for r in rows)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/args.output
    assert not output.exists()
    paths=[ROOT/'results/grouped_hint_next8_production_launch_v1.json']+[ROOT/f'results/grouped_hint_next8_repeat{i}_plan_v1.json' for i in (1,2)]
    all_rows=[];rounds=[];pending=[]
    for repeat,plan_path in enumerate(paths):
        plan=json.loads(plan_path.read_text())
        report=summarize(plan_path,allow_incomplete=True)
        rows=[]
        for document,review in zip(plan['documents'],report['documents'],strict=True):
            row={'repeat_index':repeat,'evaluation_index':document['evaluation_index'],
                'document_id':document['document_id'],'B':document['B'],'status':review['status']}
            if row['status']=='pending':pending.append([repeat,row['evaluation_index']])
            if row['status']!='paired_completed':
                row['reason']=review.get('reason');rows.append(row);continue
            summary_path=ROOT/document['summary_path'];summary=json.loads(summary_path.read_text())
            native=[r for r in summary['runs'] if r['method']=='native']
            candidate=[r for r in summary['runs'] if r['method']=='cascade']
            assert [r['phase'] for r in native]==['natural_native_0','natural_native_3']
            assert len(candidate)==2
            setup=sum(summary[k] for k in ('offline_preflight_wall_seconds','candidate_configuration_wall_seconds',
                'candidate_graph_capture_wall_seconds','setup_receipt_export_wall_seconds'))
            candidate_mean=statistics.mean(r['cold_shared_document_total_wall_seconds'] for r in candidate)
            native_mean=statistics.mean(r['cold_shared_document_total_wall_seconds'] for r in native)
            faster=min(native,key=lambda r:r['cold_shared_document_total_wall_seconds'])
            denominator=candidate_mean+setup
            primary=native_mean/denominator;secondary=faster['cold_shared_document_total_wall_seconds']/denominator
            assert primary==review['timings']['net_cold_with_unamortized_per_document_setup']['native_over_candidate']
            assert secondary<=primary
            row.update(summary_sha256=sha(summary_path),primary_net=primary,faster_native_net=secondary,
                primary_native_cold_mean_seconds=native_mean,faster_native_cold_seconds=faster['cold_shared_document_total_wall_seconds'],
                faster_native_phase=faster['phase'],candidate_cold_mean_seconds=candidate_mean,
                full_unamortized_setup_seconds=setup,candidate_mean_plus_setup_seconds=denominator,
                generation_A0_over_A3=native[0]['complete_generate_wall_seconds']/native[1]['complete_generate_wall_seconds'],
                cold_A0_over_A3=native[0]['cold_shared_document_total_wall_seconds']/native[1]['cold_shared_document_total_wall_seconds'],
                all_phase_seconds=[{k:r[k] for k in ('phase','complete_generate_wall_seconds','anchor_wall_seconds','cold_shared_document_total_wall_seconds')} for r in summary['runs']])
            rows.append(row);all_rows.append(row)
        rounds.append({'repeat_index':repeat,'plan_sha256':sha(plan_path),'selected_document_denominator':8,
            'selected_question_denominator':98,'rows':rows,'conditional_completed_aggregate':aggregate([r for r in rows if r['status']=='paired_completed'])})
    if pending and not args.allow_incomplete:raise RuntimeError(f'Pending planned runs: {pending}')
    by_document=[]
    for document in json.loads(paths[0].read_text())['documents']:
        rows=[r for r in all_rows if r['document_id']==document['document_id']]
        row={'document_id':document['document_id'],'evaluation_index':document['evaluation_index'],'completed_runs':len(rows)}
        if len(rows)==3:
            denominator=statistics.mean(r['candidate_mean_plus_setup_seconds'] for r in rows)
            row.update(status='all_three_timing_runs_available',primary_doc_mean_ratio=statistics.mean(r['primary_native_cold_mean_seconds'] for r in rows)/denominator,
                faster_native_doc_mean_ratio=statistics.mean(r['faster_native_cold_seconds'] for r in rows)/denominator,
                primary_run_ratios=[r['primary_net'] for r in rows],secondary_run_ratios=[r['faster_native_net'] for r in rows])
        else:row['status']='context_failure_retained' if not document['context_feasible'] else 'incomplete_or_failed_repeat_set'
        by_document.append(row)
    result={'status':'incomplete' if pending else 'complete_secondary_sensitivity','analysis_source_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(ROOT/'docs/grouped_hint_faster_native_sensitivity_20261017.md'),
        'definition':'min(A0 cold anchor+generate,A3 cold anchor+generate) / (mean(B1 cold,B2 cold)+same complete setup cost)',
        'rounds':rounds,'documents':by_document,'pending':pending,
        'all_completed_process_runs_descriptive_only':aggregate(all_rows),
        'limits':['Secondary sensitivity added after some timings were observed; never replaces the frozen ABBA mean primary.',
            'Uses the faster observed native whole cold task, not a fixed baseline or a confidence lower bound. Candidate denominator and full setup remain identical.',
            'All six feasible documents and all three process runs are included uniformly. Two context failures remain in each eight-document denominator.',
            'First-native slowness is an observation, not causal identification of JIT, warming, thermal drift or contention.',
            'Pooled process-run description is not independent sample count: repeated runs share documents. Cross-run equality is separately required by the frozen repetition audit.']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'completed_process_runs':len(all_rows),'round_sensitivity':[r['conditional_completed_aggregate'] for r in rounds]},indent=2))


if __name__=='__main__':main()
