"""Independent exact-trajectory, costs, and execution-coverage ABBA audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_native_raw_pair_model_qualification import audit as qualification_audit
from scripts.screen_application import extract_final
from src.financial_answer_metric import score_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    qualified=qualification_audit(path)
    assert qualified['status']=='qualified'
    summary=json.loads(path.read_text())
    group=next(g for g in json.loads((ROOT/summary['manifest']).read_text())['groups']
               if g['document_id']==summary['args']['document_id'])
    expected=['qualification','fixed_native_0','fixed_cascade_1','fixed_cascade_2','fixed_native_3',
              'natural_native_0','natural_cascade_1','natural_cascade_2','natural_native_3']
    assert [r['phase'] for r in summary['runs']]==expected
    result={'status':'qualified','summary_sha256':sha(path),'audit_source_sha256':sha(Path(__file__)),
        'model_qualification_audit':qualified,'phases':[],'comparisons':{},
        'candidate_configuration_and_graph_setup_seconds':summary['candidate_total_setup_wall_seconds'],
        'qualification_full_phase_control_seconds':summary['runs'][0]['full_phase_control_wall_seconds'],
        'limits':['Two ABBA repeats per arm in one session/seed are descriptive, not a population confidence interval.',
            'Candidate executes only actual B8. Native fallback at smaller batches is part of the measured candidate policy.',
            'Graph counts measure execution coverage, not GPU-time attribution, and exclude non-full-graph prefill.',
            'Host prepare includes synchronized physical checks and may wait for previous GPU work; its sum is not an independent additive CPU tax.',
            'Raw generation, phase validation/export, anchor, and complete phase control scopes are reported separately.',
            'Current prototype requires native-state qualification before candidate activation; conservative first-activation cost includes it and all three captured graph modes.']}
    trajectories={}
    for run in summary['runs'][1:]:
        assert run['method'] in ('native','cascade') and not run['validation']
        assert len(run['workers'])==2 and len(run['answers'])==8
        ranks=[]
        for worker in run['workers']:
            record_path=Path(worker['raw_pair_record'])
            graph_path=Path(worker['graph_metadata'])
            record=json.loads(record_path.read_text())
            rows=json.loads(graph_path.read_text())['replays']
            assert record['phase']==run['phase'] and record['method']==run['method'] and not record['validation']
            assert worker['phase_valid'] and record['phase_valid']
            assert not record['checks'] and record['qualification_steps']==0
            candidates=[r for r in rows if r['candidate_graph_replayed']]
            native=[r for r in rows if r['native_graph_replayed']]
            assert len(candidates)==worker['candidate_replays']==record['candidate_replays']
            assert len(native)==worker['native_replays']==record['native_replays']
            assert len(native)+len(candidates)==len(rows)
            assert all(r['batch']==8 and not r['native_graph_replayed'] for r in candidates)
            assert all(not r['candidate_graph_replayed'] for r in native)
            if run['method']=='native':
                assert len(candidates)==0 and not record['prepare_records']
            else:
                assert len(candidates)>0 and len(record['prepare_records'])==len(candidates)
                for graph,prep in zip(candidates,record['prepare_records'],strict=True):
                    assert graph['positions']==prep['positions']
                    assert prep['lengths']==[p+1 for p in prep['positions']]
                    assert prep['actual_requests']==prep['padded_tokens']==prep['batch']==8
                    assert prep['host_physical_prefix_check'] and prep['metadata_copy_cost_included']
                    assert prep['layers_sharing_state_and_scratch']==48
            trajectory=[(r['batch'],r['questions'],r['positions']) for r in rows]
            trajectories[run['phase'],worker['rank']]=trajectory
            assert hashlib.sha256(json.dumps(trajectory).encode()).hexdigest()==run['graph_trajectory_sha256_by_rank'][worker['rank']]
            times=[p['complete_prepare_host_seconds'] for p in record['prepare_records']]
            ranks.append({'rank':worker['rank'],'record_sha256':sha(record_path),'graph_sha256':sha(graph_path),
                'candidate_graph_replays':len(candidates),'native_graph_replays':len(native),
                'candidate_fraction_of_full_graphs':len(candidates)/len(rows),
                'all_graph_batch_steps':dict(sorted(Counter(r['batch'] for r in rows).items())),
                'candidate_graph_batch_steps':dict(sorted(Counter(r['batch'] for r in candidates).items())),
                'prepare_host_seconds_sum':sum(times),'prepare_host_median_ms':statistics.median(times)*1000 if times else None,
                'fallback_counts':dict(Counter(f['reason'] for f in record['fallbacks'])),
                'fallback_batch_counts':dict(sorted(Counter(f['batch'] for f in record['fallbacks']).items()))})
        assert trajectories[run['phase'],0]==trajectories[run['phase'],1]
        assert sum(len(a['tokens']) for a in run['answers'])==run['total_output_tokens']
        assert sum(a['request_metrics']['num_preemptions'] for a in run['answers'])==run['total_request_preemptions']
        assert abs(run['anchor_wall_seconds']+run['complete_generate_wall_seconds']-run['cold_shared_document_total_wall_seconds'])<1e-8
        assert abs(run['complete_generate_wall_seconds']+run['phase_validation_and_record_export_wall_seconds']-run['generate_plus_phase_validation_wall_seconds'])<1e-8
        if run['natural_eos_enabled']:
            assert run['sampling']==summary['protocol']['sampling']
            predictions=[]
            for answer in run['answers']:
                final,_,parse=extract_final(answer['text'],thinking=True)
                predictions.append(final if answer['finish_reason']=='stop' and parse=='closed_thinking' else '')
            assert json.loads(json.dumps(score_pairs(predictions,[q['answer'] for q in group['questions']])))==run['quality']['score']
        else:
            assert run['sampling']['ignore_eos'] and run['sampling']['min_tokens']==run['sampling']['max_tokens']==128
            assert all(len(a['tokens'])==128 for a in run['answers'])
        result['phases'].append({'phase':run['phase'],'ranks':ranks,'natural_eos_enabled':run['natural_eos_enabled'],
            'output_lengths':[len(a['tokens']) for a in run['answers']],
            'cap_hits':run['cap_hits'],'preemptions':run['total_request_preemptions'],
            'quality':run.get('quality'),'complete_generate_seconds':run['complete_generate_wall_seconds'],
            'cold_anchor_plus_generate_seconds':run['cold_shared_document_total_wall_seconds'],
            'phase_validation_export_seconds':run['phase_validation_and_record_export_wall_seconds'],
            'full_phase_control_seconds':run['full_phase_control_wall_seconds']})
    for kind in ('fixed','natural'):
        runs=[r for r in summary['runs'] if r['phase'].startswith(kind+'_')]
        assert [r['method'] for r in runs]==['native','cascade','cascade','native']
        for run in runs[1:]:
            assert [a['tokens'] for a in run['answers']]==[a['tokens'] for a in runs[0]['answers']]
            assert [a['num_cached_tokens'] for a in run['answers']]==[a['num_cached_tokens'] for a in runs[0]['answers']]
            assert [a['finish_reason'] for a in run['answers']]==[a['finish_reason'] for a in runs[0]['answers']]
            assert trajectories[run['phase'],0]==trajectories[runs[0]['phase'],0]
        ratios={key:statistics.mean(runs[i][key] for i in (0,3))/statistics.mean(runs[i][key] for i in (1,2))
            for key in ('complete_generate_wall_seconds','generate_plus_phase_validation_wall_seconds',
                        'cold_shared_document_total_wall_seconds','full_phase_control_wall_seconds')}
        assert ratios==summary['comparisons'][kind]['native_over_candidate']
        native=statistics.mean(runs[i]['cold_shared_document_total_wall_seconds'] for i in (0,3))
        candidate=statistics.mean(runs[i]['cold_shared_document_total_wall_seconds'] for i in (1,2))
        setup=summary['candidate_total_setup_wall_seconds']
        qualification=summary['runs'][0]['full_phase_control_wall_seconds']
        result['comparisons'][kind]={'all_output_tokens_APC_finish_and_graph_trajectories_equal':True,
            'native_over_candidate':ratios,
            'native_AA_wall_max_over_min':max(runs[i]['complete_generate_wall_seconds'] for i in (0,3))/min(runs[i]['complete_generate_wall_seconds'] for i in (0,3)),
            'conservative_first_activation_cold_including_candidate_setup_and_required_qualification':native/(candidate+setup+qualification),
            'counterfactual_offline_qualified_cold_including_all_graph_setup_only':native/(candidate+setup)}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    assert not output.exists(),output
    report=audit(Path(args.input))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['comparisons'],indent=2))


if __name__=='__main__':
    main()
