"""All-seed device control audit with frozen scoring and historical trajectories."""
import argparse
import json
from pathlib import Path
import statistics

from audit_dense_cascade_device_model import audit as audit_device
from audit_dense_cascade_four_seeds import ROOT, METRIC_ID, score_run, sha
from summarize_dense_cascade_device_natural import terminal


def audit(initial_root, additional_root):
    ledger = [json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    result = {'status': 'completed', 'metric_id': METRIC_ID, 'conditions': [],
              'analysis_source_sha256': sha(Path(__file__)),
              'metric_source_sha256': sha(ROOT/'src/financial_answer_metric.py'),
              'extraction_source_sha256': sha(ROOT/'scripts/screen_application.py'),
              'limits': [
                  'Four seeds on two documents, not eight independent documents. No population confidence interval.',
                  'Raw generation excludes phase-end device validation and record export; both raw and inclusive wall ratios are retained.',
                  'All candidate setup across the two actual sessions is separately charged in a conservative ratio; this includes checked graphs not needed for production.',
                  'Natural outputs may differ between native and candidate, changing output length and MoE work. Task ratios are not isolated algorithm speedups.',
                  'Old/new implementation output trajectories are compared exactly; old/new wall times are historical and not a contemporaneously paired performance test.',
                  'Strict FINAL scoring is an internal financial-answer metric, not official program-execution accuracy.',
                  'Existing cascade and GPU metadata are engineering controls, not claimed algorithmic novelty.']}
    for doc in ('train1','train7'):
        for dtype in ('bf16','fp8'):
            condition = f'{doc}_{dtype}'
            paths = [root/condition/'summary.json' for root in (initial_root,additional_root)]
            reports = [json.loads(p.read_text()) for p in paths]
            audits = [audit_device(p) for p in paths]
            first, extra = reports
            for key in ('protocol','prompt_token_hashes','document_sha256','engine','common_prefix_tokens'):
                assert first[key] == extra[key], (condition,key)
            for source in ('scripts/vllm_dense_cascade_device_worker.py','src/vllm_dense_cascade_device_plan.py'):
                assert first['source_sha256'][source] == extra['source_sha256'][source]
            protocol = extra['seed_protocol']
            seeds = [protocol['already_measured_sampling_seed'],*protocol['additional_sampling_seeds']]
            assert len(seeds) == len(set(seeds)) == 4
            expected_order = [(seed,method) for seed,order in zip(
                protocol['additional_sampling_seeds'],protocol['method_orders']) for method in order]
            assert [(r['sampling']['seed'],r['method']) for r in extra['runs'] if r['natural_eos_enabled']] == expected_order
            assert [(r['sampling']['seed'],r['method']) for r in first['runs'] if r['natural_eos_enabled']] == [
                (seeds[0],'native'),(seeds[0],'cascade')]
            group = next(g for g in json.loads((ROOT/first['protocol']['manifest']).read_text())['groups']
                         if g['document_id'] == first['args']['document_id'])
            assert group['document_sha256'] == first['document_sha256']
            old_initial = ROOT/'results'/('dense_cascade_reference_natural_v1' if dtype=='bf16' else 'dense_cascade_natural_v1')/condition/'summary.json'
            old_extra = ROOT/'results/dense_cascade_multiseed_v1'/condition/'summary.json'
            old_paths = [old_initial,old_extra]
            old_reports = [json.loads(p.read_text()) for p in old_paths]
            old_runs = {}
            for old in old_reports:
                assert old['status'] == 'completed'
                assert old['protocol'] == first['protocol'] and old['prompt_token_hashes'] == first['prompt_token_hashes']
                assert {k:v for k,v in old['engine'].items() if k!='worker_extension_cls'} == {
                    k:v for k,v in first['engine'].items() if k!='worker_extension_cls'}
                for run in old['runs']:
                    if run['natural_eos_enabled']:
                        key=(run['sampling']['seed'],run['method'])
                        assert key not in old_runs
                        old_runs[key]=run
            buckets = {}
            for source,report,checked in zip(paths,reports,audits):
                for run in report['runs']:
                    if not run['natural_eos_enabled']:
                        continue
                    key=(run['sampling']['seed'],run['method'])
                    assert key not in buckets
                    assert {k:v for k,v in run['sampling'].items() if k!='seed'} == {
                        k:v for k,v in first['protocol']['sampling'].items() if k!='seed'}
                    assert len(run['answers']) == 8
                    previous=old_runs[key]
                    assert previous['sampling'] == run['sampling']
                    assert [a['num_cached_tokens'] for a in previous['answers']] == [a['num_cached_tokens'] for a in run['answers']]
                    score,details=score_run(run,group['questions'])
                    phase=next(p for p in checked['phases'] if p['phase']==run['phase'])
                    device=next(p for p in checked['device_metadata_phases'] if p['phase']==run['phase'])
                    buckets[key]={'source':str(source),'phase':run['phase'],'sampling':run['sampling'],
                        'ready_seconds':run['complete_generate_wall_seconds'],
                        'cold_seconds':run['cold_shared_document_total_wall_seconds'],
                        'ready_inclusive_seconds':run['generate_plus_phase_validation_wall_seconds'],
                        'cold_inclusive_seconds':device['anchor_plus_generate_plus_phase_validation_wall_seconds'],
                        'validation_export_seconds':run['phase_validation_and_record_export_wall_seconds'],
                        'output_tokens':run['total_output_tokens'],'EOS':sum(a['finish_reason']=='stop' for a in run['answers']),
                        'cap_hits':len(phase['cap_hits']),'APC_hits':phase['num_cached_tokens'],
                        'score':score,'answers':details,'batch_steps':phase['batch_steps'],
                        'candidate_batch_steps':phase['candidate_batch_steps'],'device_records':device['ranks'],
                        'old_control_tokens_equal_per_question':[a['tokens']==b['tokens'] for a,b in zip(run['answers'],previous['answers'])],
                        'old_control_ready_seconds':previous['complete_generate_wall_seconds'],
                        'old_control_cold_seconds':previous['cold_shared_document_total_wall_seconds']}
            assert set(buckets)=={(seed,method) for seed in seeds for method in ('native','cascade')}
            pairs=[]
            timing_keys=('ready_seconds','cold_seconds','ready_inclusive_seconds','cold_inclusive_seconds')
            for seed in seeds:
                arms={m:buckets[seed,m] for m in ('native','cascade')}
                n,c=arms['native'],arms['cascade']
                assert n['sampling']==c['sampling'] and n['APC_hits']==c['APC_hits']
                pairs.append({'seed':seed,'arms':arms,'native_over_candidate':{key:n[key]/c[key] for key in timing_keys}})
            totals={m:{key:sum(p['arms'][m][key] for p in pairs) for key in
                (*timing_keys,'output_tokens','EOS','cap_hits','validation_export_seconds')} for m in ('native','cascade')}
            setup=sum(r['candidate_plan_and_capture_wall_seconds'] for r in reports)
            result['conditions'].append({'condition':condition,'document_id':first['args']['document_id'],
                'source_sha256':{str(p):sha(p) for p in paths+old_paths},
                'clean_exit_receipts':[terminal(p,ledger) for p in paths+old_paths],
                'qualified_layer_checks_per_session':[a['phases'][0]['same_Q_layer_checks_two_ranks'] for a in audits],
                'pairs':pairs,'four_seed_totals':totals,
                'four_seed_ratios':{key:totals['native'][key]/totals['cascade'][key] for key in timing_keys},
                'four_seed_quality':{m:{metric:statistics.mean(p['arms'][m]['score'][metric] for p in pairs)
                    for metric in ('exact_accuracy','rounded_accuracy')} for m in ('native','cascade')},
                'candidate_setup_both_sessions_seconds':setup,
                'conservative_cold_inclusive_with_all_candidate_setup_ratio':totals['native']['cold_inclusive_seconds']/(
                    totals['cascade']['cold_inclusive_seconds']+setup),
                'old_control_all_tokens_equal':all(all(b['old_control_tokens_equal_per_question']) for b in buckets.values())})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial-root',default='results/dense_cascade_device_natural_v1')
    p.add_argument('--additional-root',default='results/dense_cascade_device_multiseed_v1')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    path=Path(args.output)
    assert not path.exists(),path
    report=audit(Path(args.initial_root),Path(args.additional_root))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:c[k] for k in ('condition','four_seed_ratios','four_seed_quality','old_control_all_tokens_equal')}
                      for c in report['conditions']],indent=2))


if __name__=='__main__':
    main()
