"""Independent CPU audit: frozen-policy general-batch strict natural qualification and E2E."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vllm_native_raw_pair_general_setup import REQUIRED_QUALIFICATION_FIELDS
from scripts.screen_application import extract_final
from src.financial_answer_metric import score_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    summary=json.loads(path.read_text())
    assert summary['mechanism']=='raw_pair_native_state_continuation_grouped_hint_v2'
    assert 'scripts/vllm_native_raw_pair_grouped_hint_worker.py' in summary['source_sha256']
    assert 'src/xqa_raw_grouped_hint_build.py' in summary['source_sha256']
    assert summary['status']=='completed',summary.get('error',summary['status'])
    for name,expected in summary['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    assert sha(ROOT/summary['args']['protocol'])==summary['protocol_sha256']
    assert sha(ROOT/summary['manifest'])==summary['manifest_sha256']
    group=next(g for g in json.loads((ROOT/summary['manifest']).read_text())['groups'] if g['document_id']==summary['args']['document_id'])
    batch=summary['B']
    assert batch==group['B']==group['natural_question_count']==len(group['questions'])
    enabled=list(range(6,batch+1))
    assert summary['enabled_batches']==enabled
    assert summary['engine']['max_num_seqs']==batch
    assert summary['engine']['compilation_config']['cudagraph_capture_sizes']==list(range(1,batch+1))
    quota=summary['args']['qualification_states_per_batch']
    assert quota==8
    expected=['qualification']
    if not summary['args']['qualify_only']:
        expected += [f'{kind}_{method}_{i}' for kind in ('fixed','natural') for i,method in enumerate(('native','cascade','cascade','native'))]
    assert [r['phase'] for r in summary['runs']]==expected
    ends=[r for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()
        if (r:=json.loads(line)).get('event')=='end' and '--output' in r.get('argv',[]) and
        (ROOT/r['argv'][r['argv'].index('--output')+1]).resolve()==path.parent.resolve()]
    assert len(ends)==1 and ends[0]['exit_code']==0 and not ends[0]['timed_out']
    result={'status':'qualified','summary_sha256':sha(path),'audit_source_sha256':sha(Path(__file__)),
        'all_runtime_source_hashes_verified':True,'ledger_terminal_receipts':ends,'B':batch,
        'enabled_batches':enabled,'qualification':[],'phases':[],'comparisons':{},
        'limits':['One session/seed and ABBA means are descriptive, not a population confidence interval.',
            'Only naturally observed batch states are model-qualified; unobserved or ineligible states fall back to native.',
            'Candidate graph fractions count replay steps, not GPU-time fractions; prefill is excluded from graph counts.',
            'Strict current-session qualification and all three graph-mode setup costs are mandatory first-activation work.',
            'Full phase control excludes subsequent local scoring/hash/report analysis. No hidden inference warmup is subtracted.']}
    trajectories={}
    qualified_by_rank={}
    for run in summary['runs']:
        validation=run['validation']
        assert validation==(run['phase']=='qualification')
        assert len(run['answers'])==batch and len(run['workers'])==2
        ranks=[]
        for worker in run['workers']:
            source=Path(worker['raw_pair_record'])
            record=json.loads(source.read_text())
            graph=json.loads(Path(worker['graph_metadata']).read_text())
            rank=worker['rank']
            assert rank==record['rank']==graph['rank']
            assert record['phase']==run['phase'] and record['method']==run['method'] and record['validation']==validation
            assert record['phase_valid'] and worker['phase_valid']
            assert record['enabled_batches']==enabled and record['setup_mode']=='strict'
            policy=record['dispatch_policy']
            assert policy['parameters']=={'min_batch':6,'min_prefix':65536,'max_private_prefix_ratio':.125}
            assert sha(ROOT/policy['path'])==policy['sha256']
            result['dispatch_policy']=policy
            assert record['captured_layers_by_batch']=={mode:{str(b):list(range(48)) for b in enabled} for mode in ('witness','checked','production')}
            for name,expected_hash in record['source_sha256'].items():
                assert sha(ROOT/name)==expected_hash,name
            build=record['build_receipt']
            assert build['native_group']==8 and build['fold']==2
            assert build['patch_sha256']==sha(ROOT/'src/xqa_raw_grouped_hint_build.py')
            assert 'scripts/vllm_native_raw_pair_grouped_hint_worker.py' in record['source_sha256']
            for name,expected_hash in build['generated_sha256'].items():
                assert sha(ROOT/'build/xqa_raw_grouped_hint'/build['identity']/name)==expected_hash
            plan=record['plan']
            assert plan['layers']==48 and plan['workspace_allocation_sets']==plan['raw_state_allocation_sets']==1
            assert plan['enabled_batches']==enabled
            for b in enabled:
                p=plan['plans'][str(b)]
                assert p['native_batch']==b and p['native_table_shape'][0]==b
                assert p['native_launch_extent']==p['native_table_shape'][1]*16
                assert p['native_table_stride']==[p['native_table_shape'][1],1]
                splits=min(max(1,p['native_SM']//(b*2)),(p['native_launch_extent']+255)//256)
                assert p['native_splits']==splits and p['prefix_SM']==splits*((b+1)//2)*2
                assert p['dummy_prefix_rows']==b%2 and p['prefix_tiles']==summary['common_prefix_tokens']//256
            rows=graph['replays']
            candidate=[r for r in rows if r['candidate_graph_replayed']]
            native=[r for r in rows if r['native_graph_replayed']]
            assert len(candidate)+len(native)==len(rows)
            assert all(not r['native_graph_replayed'] for r in candidate) and all(not r['candidate_graph_replayed'] for r in native)
            assert worker['native_replays']==record['native_replays']==len(native)
            prepared=record['prepare_records']
            bindings=[p for p in prepared if p['host_physical_prefix_check']]
            assert len(bindings)==(1 if prepared else 0)
            if prepared:
                assert bindings[0] is prepared[0] and bindings[0]['prefix_binding']['physical_prefix_checked']
            for prep in prepared:
                b=prep['batch']
                assert prep['actual_requests']==prep['padded_tokens']==b and b in enabled
                assert prep['lengths']==[p+1 for p in prep['positions']]
                assert prep['cohort_request_slot_epoch_subset_guard'] and prep['native_GPU_lengths_aliased']
                assert prep['host_length_copy'] is False and prep['metadata_copy_cost_included']
                assert prep['layers_sharing_state_and_scratch']==48
                prefix=summary['common_prefix_tokens']
                private=max(prep['lengths'])-prefix
                expected_eligible=b>=6 and prefix>=65536 and 0<private*8<=prefix
                assert prep['policy_eligible']==expected_eligible
                assert prep['policy_prefix_tokens']==prefix and prep['policy_max_private_tokens']==private
                if not validation:
                    assert expected_eligible
                assert prep['native_splits']==plan['plans'][str(b)]['native_splits']
                assert prep['native_table_columns']==plan['plans'][str(b)]['native_table_shape'][1]
            if validation:
                assert run['method']=='native' and run['natural_eos_enabled'] and not candidate
                checks=record['checks']
                assert record['qualification_passed']==worker['qualification_passed']==bool(checks)
                assert len(checks)==record['qualification_steps']==worker['qualification_steps']
                assert len(prepared)==len(checks)
                assert record['qualification_requested_steps']==quota*len(enabled)
                assert worker['candidate_replays']==record['candidate_replays']==2*len(checks)
                actual=Counter(r['batch'] for r in rows if r['batch'] in enabled)
                tested=Counter(c['batch'] for c in checks)
                assert tested==Counter({b:min(quota,n) for b,n in actual.items()})
                assert record['qualification_states_by_batch']=={str(b):tested[b] for b in enabled}
                assert record['qualified_batches']==worker['qualified_batches']==sorted(tested)
                qualified_by_rank[rank]=sorted(tested)
                fields={field:0 for field in REQUIRED_QUALIFICATION_FIELDS}
                for index,(check,prep) in enumerate(zip(checks,prepared,strict=True)):
                    assert check['step']==index and check['batch']==prep['batch'] and check['positions']==prep['positions']
                    assert check['qualification_passed'] and all(check.get(field) is True for field in REQUIRED_QUALIFICATION_FIELDS)
                    for field in fields: fields[field]+=1
                    assert [l['layer'] for l in check['layer_checks']]==list(range(48))
                    assert all(l['Q_bits_equal'] and l['O_bits_equal'] for l in check['layer_checks'])
                    assert [m['mode'] for m in check['modes']]==['checked','production']
                    assert all(v is True for m in check['modes'] for k,v in m.items() if k!='mode')
                for b in enabled:
                    assert [c['positions'] for c in checks if c['batch']==b]==[r['positions'] for r in rows if r['batch']==b][:quota]
                result['qualification'].append({'rank':rank,'tested_states_by_batch':dict(tested),
                    'actual_native_graph_states_by_enabled_batch':dict(actual),'verified_layer_QO_pairs':len(checks)*48,
                    'required_field_true_counts':fields,'unobserved_batches_disabled':sorted(set(enabled)-set(tested))})
            else:
                assert not record['checks'] and record['qualification_steps']==0
                assert record['qualified_batches']==worker['qualified_batches']==qualified_by_rank[rank]
                assert worker['candidate_replays']==record['candidate_replays']==len(candidate)
                if run['method']=='native':
                    assert not candidate and not prepared
                else:
                    assert len(prepared)==len(candidate)
                    assert all(r['batch'] in qualified_by_rank[rank] for r in candidate)
                    assert [(r['batch'],r['positions']) for r in candidate]==[(p['batch'],p['positions']) for p in prepared]
            for fallback in record['fallbacks']:
                if fallback['reason'] in ('development_policy_short_prefix','development_policy_private_prefix_ratio'):
                    assert not validation and run['method']=='cascade'
                    prefix=summary['common_prefix_tokens']
                    private=max(fallback['positions'])+1-prefix
                    assert fallback['policy_eligible'] is False
                    assert fallback['policy_prefix_tokens']==prefix and fallback['max_private_tokens']==private
                    expected_reason='development_policy_short_prefix' if prefix<65536 else 'development_policy_private_prefix_ratio'
                    assert fallback['reason']==expected_reason
                    assert prefix<65536 or private*8>prefix
            trajectory=[(r['batch'],r['questions'],r['positions']) for r in rows]
            trajectories[run['phase'],rank]=trajectory
            if not validation:
                assert hashlib.sha256(json.dumps(trajectory).encode()).hexdigest()==run['graph_trajectory_sha256_by_rank'][rank]
            times=[p['complete_prepare_host_seconds'] for p in prepared]
            ranks.append({'rank':rank,'record_sha256':sha(source),'graph_sha256':sha(Path(worker['graph_metadata'])),
                'candidate_replays':len(candidate),'native_replays':len(native),
                'candidate_fraction_of_full_graphs':len(candidate)/len(rows),
                'all_graph_batch_steps':dict(Counter(r['batch'] for r in rows)),
                'candidate_graph_batch_steps':dict(Counter(r['batch'] for r in candidate)),
                'fallback_counts':dict(Counter(r['reason'] for r in record['fallbacks'])),
                'prepare_host_seconds_sum':sum(times),'prepare_host_median_us':statistics.median(times)*1e6 if times else None,
                'cohort_prefix_bind_count':len(bindings),'build_identity':build['identity']})
        assert trajectories[run['phase'],0]==trajectories[run['phase'],1]
        assert qualified_by_rank[0]==qualified_by_rank[1]
        assert sum(len(a['tokens']) for a in run['answers'])==run['total_output_tokens']
        assert sum(a['request_metrics']['num_preemptions'] for a in run['answers'])==run['total_request_preemptions']
        assert run['cap_hits']==[a['question_index'] for a in run['answers'] if a['finish_reason']=='length']
        assert abs(run['anchor_wall_seconds']+run['complete_generate_wall_seconds']-run['cold_shared_document_total_wall_seconds'])<1e-8
        assert abs(run['complete_generate_wall_seconds']+run['phase_validation_and_record_export_wall_seconds']-run['generate_plus_phase_validation_wall_seconds'])<1e-8
        if run['natural_eos_enabled']:
            assert run['sampling']==summary['protocol']['sampling']
            assert [a['tokens'] for a in run['answers']]==[a['tokens'] for a in summary['runs'][0]['answers']]
            assert trajectories[run['phase'],0]==trajectories['qualification',0]
            predictions=[]
            for a in run['answers']:
                final,_,parse=extract_final(a['text'],thinking=True)
                predictions.append(final if a['finish_reason']=='stop' and parse=='closed_thinking' else '')
            assert json.loads(json.dumps(score_pairs(predictions,[q['answer'] for q in group['questions']])))==run['quality']['score']
        else:
            assert run['sampling']['min_tokens']==run['sampling']['max_tokens']==128 and run['sampling']['ignore_eos']
            assert all(len(a['tokens'])==128 for a in run['answers'])
        result['phases'].append({'phase':run['phase'],'ranks':ranks,'output_lengths':[len(a['tokens']) for a in run['answers']],
            'cap_hits':run['cap_hits'],'preemptions':run['total_request_preemptions'],'quality':run.get('quality'),
            'complete_generate_seconds':run['complete_generate_wall_seconds'],
            'cold_anchor_plus_generate_seconds':run['cold_shared_document_total_wall_seconds'],
            'full_phase_control_seconds':run['full_phase_control_wall_seconds']})
    for kind in summary['comparisons']:
        runs=[r for r in summary['runs'] if r['phase'].startswith(kind+'_')]
        assert [r['method'] for r in runs]==['native','cascade','cascade','native']
        for r in runs[1:]:
            for key in ('tokens','num_cached_tokens','finish_reason'):
                assert [a[key] for a in r['answers']]==[a[key] for a in runs[0]['answers']]
            assert trajectories[r['phase'],0]==trajectories[runs[0]['phase'],0]
        ratios={key:statistics.mean(runs[i][key] for i in (0,3))/statistics.mean(runs[i][key] for i in (1,2))
            for key in ('complete_generate_wall_seconds','generate_plus_phase_validation_wall_seconds',
                'cold_shared_document_total_wall_seconds','full_phase_control_wall_seconds')}
        assert ratios==summary['comparisons'][kind]['native_over_candidate']
        native=statistics.mean(runs[i]['cold_shared_document_total_wall_seconds'] for i in (0,3))
        candidate=statistics.mean(runs[i]['cold_shared_document_total_wall_seconds'] for i in (1,2))
        setup=summary['candidate_total_setup_wall_seconds']
        qualification=summary['runs'][0]['full_phase_control_wall_seconds']
        has_candidate=all(runs[i]['candidate_graph_executed'] for i in (1,2))
        assert summary['comparisons'][kind]['candidate_speed_claim_eligible']==has_candidate
        result['comparisons'][kind]={'all_tokens_APC_finishes_graph_trajectories_equal':True,
            'candidate_speed_claim_eligible':has_candidate,'native_over_candidate':ratios,
            'conservative_first_activation_including_required_setup_and_qualification':native/(candidate+setup+qualification),
            'counterfactual_offline_qualified_including_all_graph_setup':native/(candidate+setup)}
    result['candidate_states_qualified']=bool(qualified_by_rank[0])
    if not result['candidate_states_qualified']:
        result['status']='valid_native_only_no_candidate_qualification'
    result['costs']={key:summary[key] for key in ('native_initialization_and_graph_setup_wall_seconds',
        'candidate_configuration_wall_seconds','candidate_graph_capture_wall_seconds','candidate_total_setup_wall_seconds')}
    result['costs']['qualification_full_phase_control_wall_seconds']=summary['runs'][0]['full_phase_control_wall_seconds']
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    assert not output.exists(),output
    result=audit(Path(args.input))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'qualification':result['qualification'],'comparisons':result['comparisons']},indent=2))


if __name__=='__main__':
    main()
