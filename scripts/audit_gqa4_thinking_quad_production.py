"""Independent CPU audit: production-only setup, offline qualification, and actual ABBA execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vllm_gqa4_raw_pair_setup import REQUIRED_QUALIFICATION_FIELDS
from scripts.screen_application import extract_final
from src.financial_answer_metric import score_pairs
from src.gqa4_quad_scope import in_scope, receipt as scope_receipt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    summary=json.loads(path.read_text())
    assert summary['mechanism']=='raw_quad_native_state_continuation_grouped_hint_v2'
    assert 'scripts/vllm_gqa4_quad_worker.py' in summary['source_sha256']
    assert 'src/xqa_raw_grouped_hint_build.py' in summary['source_sha256']
    assert summary['status'] in ('completed','completed_setup_only'),summary.get('error',summary['status'])
    for name,expected in summary['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    assert sha(ROOT/summary['args']['protocol'])==summary['protocol_sha256']
    assert sha(ROOT/summary['manifest'])==summary['manifest_sha256']
    group=next(g for g in json.loads((ROOT/summary['manifest']).read_text())['groups'] if g['document_id']==summary['args']['document_id'])
    batch=summary['B']
    assert batch==group['B']==group['natural_question_count']==len(group['questions'])
    enabled=summary['enabled_batches']
    assert batch==8 and enabled==[4,5,6,7,8]
    assert summary['common_prefix_tokens']==120528
    assert summary['engine']['max_num_seqs']==batch
    assert summary['engine']['compilation_config']['cudagraph_capture_sizes']==list(range(1,batch+1))
    expected=[]
    if not summary['args']['setup_only']:
        kinds=('fixed','natural') if summary['args']['phase_kind']=='both' else (summary['args']['phase_kind'],)
        expected=[f'{kind}_{method}_{i}' for kind in kinds for i,method in enumerate(('native','cascade','cascade','native'))]
    assert [r['phase'] for r in summary['runs']]==expected
    ends=[r for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()
        if (r:=json.loads(line)).get('event')=='end' and '--output' in r.get('argv',[]) and
        (ROOT/r['argv'][r['argv'].index('--output')+1]).resolve()==path.parent.resolve()]
    assert len(ends)==1 and ends[0]['exit_code']==0 and not ends[0]['timed_out']
    result={'status':'qualified','summary_sha256':sha(path),'audit_source_sha256':sha(Path(__file__)),
        'all_runtime_source_hashes_verified':True,'ledger_terminal_receipts':ends,'B':batch,
        'enabled_batches':enabled,'offline_qualification':[],'phases':[],'comparisons':{},
        'limits':['One session/seed and ABBA means are descriptive, not a population confidence interval.',
            'Only the frozen development prefix120528 and B4..8/private<=8192 scope is covered; not a general policy or heldout result.',
            'Offline native-state numerical evidence is specific to the recorded source/model/device; it is not an exhaustive proof.',
            'Candidate graph fractions count replay steps, not GPU-time fractions; prefill is excluded from graph counts.',
            'This process captures only production graphs; qualification happened offline and is explicitly linked, not hidden or rerun.',
            'Full phase control excludes subsequent local scoring/hash/report analysis. No hidden inference warmup is subtracted.']}
    trajectories={}
    qualified_by_rank={0:list(enabled),1:list(enabled)}
    offline=[]
    covered={0:set(),1:set()}
    for item in summary['offline_inputs']:
        source=Path(item['path'])
        assert sha(source)==item['sha256']
        record=json.loads(source.read_text())
        assert record['rank'] in covered and record['setup_mode']=='strict' and record['qualification_passed']
        assert record['qualification_steps']==len(record['checks'])>0
        for name,expected_hash in record['source_sha256'].items():
            assert sha(ROOT/name)==expected_hash
        for check in record['checks']:
            assert check['qualification_passed'] and all(check.get(k) is True for k in REQUIRED_QUALIFICATION_FIELDS)
            assert [l['layer'] for l in check['layer_checks']]==list(range(36))
            assert all(l['Q_bits_equal'] and l['O_bits_equal'] for l in check['layer_checks'])
            assert [m['mode'] for m in check['modes']]==['checked','production']
            assert all(v is True for m in check['modes'] for k,v in m.items() if k!='mode')
            for mode in ('witness','checked','production'):
                assert record['captured_layers_by_batch'][mode][str(check['batch'])]==list(range(36))
            covered[record['rank']].add(check['batch'])
        offline.append(record)
        result['offline_qualification'].append({'path':str(source),'sha256':sha(source),'rank':record['rank'],'verified_states':len(record['checks']),'verified_batches':sorted({c['batch'] for c in record['checks']})})
    assert summary['offline_jointly_qualified_batches']==sorted(covered[0]&covered[1])
    assert set(enabled)<=covered[0]&covered[1]
    assert summary['unqualified_batches_native_fallback']==sorted(set(range(4,batch+1))-set(enabled))
    assert len(summary['setup_workers'])==2 and {w['rank'] for w in summary['setup_workers']}=={0,1}
    for worker in summary['setup_workers']:
        raw=json.loads(Path(worker['raw_pair_record']).read_text())
        assert raw['setup_mode']=='production_only' and raw['qualified_batches']==enabled
        assert raw['rank']==worker['rank'] and raw['phase_valid']
        assert raw['development_scope']==scope_receipt()
        result['development_scope']=raw['development_scope']
        build=raw['build_receipt']
        assert build['native_group']==4 and build['fold']==4
        assert build['patch_sha256']==sha(ROOT/'src/xqa_raw_grouped_hint_build.py')
        assert 'scripts/vllm_gqa4_quad_worker.py' in raw['source_sha256']
        for name,expected_hash in build['generated_sha256'].items():
            assert sha(ROOT/'build/xqa_raw_grouped_hint'/build['identity']/name)==expected_hash
        assert raw['phase']=='setup_receipt' and raw['method']=='native'
        assert not raw['checks'] and not raw['prepare_records'] and raw['qualification_steps']==0
        assert raw['candidate_replays']==raw['native_replays']==0
        assert raw['captured_layers_by_batch']=={'production':{str(b):list(range(36)) for b in enabled}}
        assert raw['offline_qualification']['source_model_device_match']
        for rec in offline:
            assert rec['execution_fingerprint']==raw['execution_fingerprint']
            assert rec['source_sha256']==raw['source_sha256']
    assert all(c['captured_modes']==['production'] and c['setup_mode']=='production_only' for c in summary['capture'])
    for run in summary['runs']:
        validation=run['validation']
        assert validation is False
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
            assert record['enabled_batches']==enabled and record['setup_mode']=='production_only'
            assert record['development_scope']==scope_receipt()
            assert record['captured_layers_by_batch']=={'production':{str(b):list(range(36)) for b in enabled}}
            for name,expected_hash in record['source_sha256'].items():
                assert sha(ROOT/name)==expected_hash,name
            build=record['build_receipt']
            assert build['native_group']==4 and build['fold']==4
            assert build['patch_sha256']==sha(ROOT/'src/xqa_raw_grouped_hint_build.py')
            assert 'scripts/vllm_gqa4_quad_worker.py' in record['source_sha256']
            for name,expected_hash in build['generated_sha256'].items():
                assert sha(ROOT/'build/xqa_raw_grouped_hint'/build['identity']/name)==expected_hash
            plan=record['plan']
            assert plan['layers']==36 and plan['workspace_allocation_sets']==plan['raw_state_allocation_sets']==1
            assert plan['enabled_batches']==enabled
            for b in enabled:
                p=plan['plans'][str(b)]
                assert p['native_batch']==b and p['native_table_shape'][0]==b
                assert p['native_launch_extent']==p['native_table_shape'][1]*16
                assert p['native_table_stride']==[p['native_table_shape'][1],1]
                splits=min(max(1,p['native_SM']//(b*4)),(p['native_launch_extent']+255)//256)
                assert p['native_splits']==splits and p['prefix_SM']==splits*((b+3)//4)*4
                assert p['dummy_prefix_rows']==(4-b%4)%4 and p['prefix_tiles']==summary['common_prefix_tokens']//256
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
                assert prep['layers_sharing_state_and_scratch']==36
                prefix=summary['common_prefix_tokens']
                private=max(prep['lengths'])-prefix
                expected_eligible=in_scope(b,prefix,prep['lengths'])
                assert prep['development_scope_eligible']==expected_eligible
                assert prep['scope_nominal_prefix']==prefix and prep['scope_max_private_tokens']==private
                if not validation:
                    assert expected_eligible
                assert prep['native_splits']==plan['plans'][str(b)]['native_splits']
                assert prep['native_table_columns']==plan['plans'][str(b)]['native_table_shape'][1]
            assert not record['checks'] and record['qualification_steps']==0
            assert record['qualified_batches']==worker['qualified_batches']==qualified_by_rank[rank]
            assert worker['candidate_replays']==record['candidate_replays']==len(candidate)
            assert record['offline_qualification']['source_model_device_match']
            assert record['execution_fingerprint']==offline[0]['execution_fingerprint']
            assert record['execution_fingerprint']['attention_shape']==[36,16,4,128,16]
            assert record['execution_fingerprint']['model_path']=='data/models/Qwen3-4B-Thinking-2507'
            if run['method']=='native':
                assert not candidate and not prepared
            else:
                assert len(prepared)==len(candidate)
                assert all(r['batch'] in qualified_by_rank[rank] for r in candidate)
                assert [(r['batch'],r['positions']) for r in candidate]==[(p['batch'],p['positions']) for p in prepared]
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
            original=next(r for r in summary['runs'] if r['phase']=='natural_native_0')
            assert [a['tokens'] for a in run['answers']]==[a['tokens'] for a in original['answers']]
            assert trajectories[run['phase'],0]==trajectories['natural_native_0',0]
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
            for key in ('tokens','num_cached_tokens','finish_reason','stop_reason'):
                assert [a[key] for a in r['answers']]==[a[key] for a in runs[0]['answers']]
            assert trajectories[r['phase'],0]==trajectories[runs[0]['phase'],0]
        ratios={key:statistics.mean(runs[i][key] for i in (0,3))/statistics.mean(runs[i][key] for i in (1,2))
            for key in ('complete_generate_wall_seconds','generate_plus_phase_validation_wall_seconds',
                'cold_shared_document_total_wall_seconds','full_phase_control_wall_seconds')}
        assert ratios==summary['comparisons'][kind]['native_over_candidate']
        native=statistics.mean(runs[i]['cold_shared_document_total_wall_seconds'] for i in (0,3))
        candidate=statistics.mean(runs[i]['cold_shared_document_total_wall_seconds'] for i in (1,2))
        setup=summary['candidate_total_setup_wall_seconds']
        preflight=summary['offline_preflight_wall_seconds']
        common=summary['input_preparation_wall_seconds']+summary['native_initialization_and_graph_setup_wall_seconds']
        components=summary['first_use_component_estimates'][kind]
        assert abs(components['native_first_use_component_sum_seconds']-(common+native))<1e-8
        assert abs(components['candidate_first_use_component_sum_seconds']-(common+preflight+setup+candidate))<1e-8
        assert abs(components['native_over_candidate_first_use_component_sum']-(common+native)/(common+preflight+setup+candidate))<1e-10
        has_candidate=all(runs[i]['candidate_graph_executed'] for i in (1,2))
        assert summary['comparisons'][kind]['candidate_speed_claim_eligible']==has_candidate
        result['comparisons'][kind]={'all_tokens_APC_finishes_graph_trajectories_equal':True,
            'candidate_speed_claim_eligible':has_candidate,'native_over_candidate':ratios,
            'cold_document_plus_actual_candidate_setup_ratio':native/(candidate+setup+preflight),
            'primary_net_cold_including_setup_and_export':native/(candidate+setup+preflight+summary['setup_receipt_export_wall_seconds']),
            'native_AA_generation_max_over_min':max(runs[i]['complete_generate_wall_seconds'] for i in (0,3))/min(runs[i]['complete_generate_wall_seconds'] for i in (0,3)),
            'first_use_component_estimate':components}
    result['all_enabled_batches_offline_qualified']=True
    result['production_modes_only_captured']=True
    if summary['args']['setup_only']:
        assert not summary['runs'] and not summary['comparisons']
        result['status']='qualified_setup_only_no_performance_measurement'
    result['costs']={key:summary[key] for key in ('native_initialization_and_graph_setup_wall_seconds',
        'candidate_configuration_wall_seconds','candidate_graph_capture_wall_seconds','candidate_total_setup_wall_seconds')}
    result['costs']['offline_preflight_wall_seconds']=summary['offline_preflight_wall_seconds']
    result['costs']['setup_receipt_export_wall_seconds']=summary['setup_receipt_export_wall_seconds']
    result['first_use_scope']='Component-sum estimate only, using actual new-process single-production-graph setup and same-session task means. No independently timed cold service or repeated-seed confidence claim.'
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
    print(json.dumps({'status':result['status'],'offline_qualification':result['offline_qualification'],'comparisons':result['comparisons']},indent=2))


if __name__=='__main__':
    main()
