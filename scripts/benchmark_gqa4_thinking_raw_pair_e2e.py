"""Strict bitwise-qualified Qwen3-4B-Thinking GQA4 B8 ABBA, native survivor fallback."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer(index,output):
    token=output.outputs[0]
    return {'question_index':index,'tokens':list(token.token_ids),'text':token.text,
        'finish_reason':token.finish_reason,'stop_reason':token.stop_reason,
        'num_cached_tokens':output.num_cached_tokens,'request_id':output.request_id,
        'request_metrics':dataclasses.asdict(output.metrics) if dataclasses.is_dataclass(output.metrics) else None}


def compare_abba(runs):
    assert [r['method'] for r in runs]==['native','cascade','cascade','native']
    token_rows=[[a['tokens'] for a in r['answers']] for r in runs]
    cached=[[a['num_cached_tokens'] for a in r['answers']] for r in runs]
    finishes=[[a['finish_reason'] for a in r['answers']] for r in runs]
    trajectories=[r['graph_trajectory_sha256_by_rank'] for r in runs]
    result={'native_AA_tokens_equal':token_rows[0]==token_rows[3],
        'candidate_BB_tokens_equal':token_rows[1]==token_rows[2],
        'native_candidate_all_tokens_equal':all(t==token_rows[0] for t in token_rows[1:]),
        'all_APC_hits_equal':all(c==cached[0] for c in cached[1:]),
        'all_finish_reasons_equal':all(f==finishes[0] for f in finishes[1:]),
        'all_graph_step_positions_and_questions_equal':all(t==trajectories[0] for t in trajectories[1:]),
        'native_AA_wall_max_over_min':max(runs[i]['complete_generate_wall_seconds'] for i in (0,3))/min(runs[i]['complete_generate_wall_seconds'] for i in (0,3))}
    result['identical_output_comparison_valid']=all(result[k] for k in
        ('native_AA_tokens_equal','candidate_BB_tokens_equal','native_candidate_all_tokens_equal','all_APC_hits_equal','all_finish_reasons_equal','all_graph_step_positions_and_questions_equal'))
    # Preserve descriptive timings even when equality fails, but gate their interpretation.
    result['native_over_candidate']={key:statistics.mean(runs[i][key] for i in (0,3))/statistics.mean(runs[i][key] for i in (1,2))
        for key in ('complete_generate_wall_seconds','generate_plus_phase_validation_wall_seconds',
                    'cold_shared_document_total_wall_seconds','full_phase_control_wall_seconds')}
    result['scope']='Same-session ABBA means, not a population confidence interval; speed interpretation requires bitwise qualification and output equality.'
    return result


def audit_qualification(workers,enabled_batches,per_batch_steps):
    from collections import Counter
    from src.vllm_gqa4_raw_pair_setup import REQUIRED_QUALIFICATION_FIELDS
    receipts=[]
    for worker in workers:
        path=Path(worker['raw_pair_record'])
        record=json.loads(path.read_text())
        rows=json.loads(Path(worker['graph_metadata']).read_text())['replays']
        assert record['rank']==worker['rank'] and record['method']=='native' and record['validation']
        assert record['qualification_steps']==len(record['checks'])>0
        assert record['qualification_passed'] and worker['qualification_passed']
        assert record['enabled_batches']==enabled_batches and record['setup_mode']=='strict'
        assert record['captured_layers_by_batch']=={mode:{str(b):list(range(36)) for b in enabled_batches} for mode in ('witness','checked','production')}
        observed=Counter(row['batch'] for row in rows if row['batch'] in enabled_batches)
        tested=Counter(step['batch'] for step in record['checks'])
        assert tested==Counter({batch:min(per_batch_steps,count) for batch,count in observed.items()}),'Every naturally observed eligible batch must receive its predeclared checks'
        assert record['qualified_batches']==worker['qualified_batches']==sorted(tested)
        assert record['qualification_states_by_batch']=={str(b):tested[b] for b in enabled_batches}
        assert record['candidate_replays']==2*len(record['checks'])
        assert len(record['prepare_records'])==len(record['checks'])
        for index,step in enumerate(record['checks']):
            assert step['step']==index and step['batch'] in enabled_batches
            assert all(step.get(field) is True for field in REQUIRED_QUALIFICATION_FIELDS)
            assert step['qualification_passed']
            assert [row['layer'] for row in step['layer_checks']]==list(range(36))
            assert all(row['Q_bits_equal'] and row['O_bits_equal'] for row in step['layer_checks'])
            assert [m['mode'] for m in step['modes']]==['checked','production']
            assert all(all(value is True for key,value in m.items() if key!='mode') for m in step['modes'])
        receipts.append({'rank':worker['rank'],'path':str(path),'sha256':digest(path),
            'verified_states':len(record['checks']),'verified_layer_QO_pairs':len(record['checks'])*36,
            'native_graph_states_by_enabled_batch':dict(observed),'verified_states_by_batch':dict(tested),
            'qualified_batches':sorted(tested),'unobserved_batches_fallback_native':sorted(set(enabled_batches)-set(tested)),
            'all_required_bitwise_fields_passed':True})
    assert receipts[0]['verified_states_by_batch']==receipts[1]['verified_states_by_batch']
    return receipts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',default='configs/gqa4_thinking_raw_pair_b8_training_v1.json')
    p.add_argument('--document-id',required=True)
    p.add_argument('--qualification-states-per-batch',type=int,default=8)
    p.add_argument('--fixed-tokens',type=int,default=128)
    p.add_argument('--qualify-only',action='store_true')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
    assert args.qualification_states_per_batch==8 and args.fixed_tokens==128
    protocol_path=ROOT/args.protocol
    protocol=json.loads(protocol_path.read_text())
    out=ROOT/args.output
    assert not (out/'summary.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    sources=['scripts/benchmark_gqa4_thinking_raw_pair_e2e.py','scripts/vllm_gqa4_raw_pair_worker.py',
        'src/vllm_gqa4_raw_pair_plan.py','src/vllm_gqa4_raw_pair_setup.py',
        'src/xqa_raw_gqa4_pair_runtime.py','src/xqa_raw_pair_gqa4_build.py',
        'src/vllm_gqa4_raw_pair_contract.py',
        'src/xqa_raw_pair_build.py','src/xqa_raw_continuation_build.py',
        'scripts/vllm_gqa4_native_precision_worker.py','scripts/vllm_gqa4_native_backend_worker.py',
        'scripts/probe_engine_docfinqa.py','scripts/vllm_batch_admission.py','scripts/capture_model.py',
        'scripts/screen_application.py','src/financial_answer_metric.py']
    report={'status':'initializing','uid':os.geteuid(),'python':sys.executable,'args':vars(args),
        'protocol':protocol,'protocol_sha256':digest(protocol_path),'runs':[],'comparisons':{},
        'source_sha256':{name:digest(ROOT/name) for name in sources},
        'mechanism':'raw_pair_native_state_continuation','candidate_dispatch_scope':'Qwen3-4B-Thinking actual B8/padded8 only, qualified in this session; all survivors use native. No GQA8 dispatch reused.',
        'limits':['Single session and one sampling seed; ABBA repeats are not independent documents or seeds.',
            'Qualification replays the same actual native states and returns native sampler results; it is not included in performance timings.',
            'Raw generation includes host preparation; phase validation/export and setup/control RPC costs are separately measured.',
            'Graph execution coverage excludes non-full-graph prefill/mixed work and is not a whole-task GPU-time fraction.',
            'Full phase control wall ends after worker receipt export; subsequent local scoring/hash auditing/report writes are experimental analysis, outside that wall.',
            'Host preparation timestamps may include waiting for earlier GPU work and are not a causal decomposition of added CPU overhead.',
            'No candidate speed conclusion is valid unless both ranks pass bitwise state qualification and every compared output token matches.']}
    def save():
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        from scripts.screen_application import extract_final
        from src.financial_answer_metric import METRIC_ID,score_pairs
        start=time.perf_counter()
        manifest_path=ROOT/protocol['manifest']
        group=next(g for g in json.loads(manifest_path.read_text())['groups'] if g['document_id']==args.document_id)
        item=next(d for d in protocol['documents'] if d['document_id']==args.document_id)
        prompts,begin,end=make_inputs(AutoTokenizer.from_pretrained(ROOT/protocol['model_path'],local_files_only=True),group)
        hashes=[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        batch=len(prompts)
        assert batch==len(set(hashes))==8, 'This independent second-model pilot is B8 only'
        enabled_batches=[8]
        qualification_steps=args.qualification_states_per_batch*len(enabled_batches)
        report['enabled_batches']=enabled_batches
        report['qualification_max_states']=qualification_steps
        assert hashes==item['prompt_token_hashes'] and list(map(len,prompts))==item['prompt_lengths']
        assert [begin,end]==item['document_token_range']
        common=0
        for values in zip(*prompts):
            if len(set(values))!=1:
                break
            common+=1
        common=common//16*16
        assert common>=256 and max(map(len,prompts))+protocol['sampling']['max_tokens']<=protocol['engine']['max_model_len']
        report.update(B=batch,document_sha256=group['document_sha256'],manifest=protocol['manifest'],manifest_sha256=digest(manifest_path),
            prompt_token_hashes=hashes,prompt_lengths=list(map(len,prompts)),common_prefix_tokens=common,
            input_preparation_wall_seconds=time.perf_counter()-start)
        engine=copy.deepcopy(protocol['engine'])
        engine.update(model=protocol['model_path'],kv_cache_dtype='auto',max_num_seqs=batch,disable_log_stats=False,
            worker_extension_cls='scripts.vllm_gqa4_raw_pair_worker.GQA4RawPairWorker')
        engine['compilation_config']['cudagraph_capture_sizes']=list(range(1,batch+1))
        report['engine']=engine
        save()
        start=time.perf_counter()
        llm=LLM(**engine)
        report['native_initialization_and_graph_setup_wall_seconds']=time.perf_counter()-start
        config={'output':str(out.resolve()),'backend':'native_XQA_or_RawPair','prompt_token_hashes':hashes,
                'batch':batch,'enabled_batches':enabled_batches,'setup_mode':'strict','common_prefix_tokens':common,
                'max_tail_tokens':max(map(len,prompts))-common+protocol['sampling']['max_tokens']+64}
        start=time.perf_counter()
        report['configuration']=llm.collective_rpc('configure_dense_cascade',args=(config,))
        report['candidate_configuration_wall_seconds']=time.perf_counter()-start
        start=time.perf_counter()
        report['capture']=llm.collective_rpc('capture_dense_cascade_graphs')
        report['candidate_graph_capture_wall_seconds']=time.perf_counter()-start
        report['candidate_total_setup_wall_seconds']=report['candidate_configuration_wall_seconds']+report['candidate_graph_capture_wall_seconds']
        report['native_before']=llm.collective_rpc('native_precision_receipt',args=('auto',False))
        save()
        phases=[('qualification','native',True,True,None)]
        if not args.qualify_only:
            phases += [(f'fixed_{method}_{i}',method,False,False,128) for i,method in enumerate(('native','cascade','cascade','native'))]
            phases += [(f'natural_{method}_{i}',method,True,False,None) for i,method in enumerate(('native','cascade','cascade','native'))]
        for phase,method,natural,validation,tokens in phases:
            phase_started=time.perf_counter()
            start=time.perf_counter()
            llm.collective_rpc('set_dense_cascade_phase',args=(phase+'_anchor','native',False))
            assert llm.reset_prefix_cache()
            anchor_control=time.perf_counter()-start
            start=time.perf_counter()
            anchors=llm.generate([{'prompt_token_ids':prompts[0]}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=False,seed=protocol['seed']),use_tqdm=False)
            anchor_wall=time.perf_counter()-start
            assert anchors[0].num_cached_tokens==0
            start=time.perf_counter()
            llm.collective_rpc('set_dense_cascade_phase',args=(phase,method,validation))
            if validation:
                report['qualification_arm']=llm.collective_rpc('arm_raw_pair_qualification',args=(qualification_steps,args.qualification_states_per_batch))
            group_control=time.perf_counter()-start
            sampling=dict(protocol['sampling'])
            if not natural:
                sampling.update(max_tokens=tokens,min_tokens=tokens,ignore_eos=True)
            start=time.perf_counter()
            outputs=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in prompts],SamplingParams(**sampling))
            wall=time.perf_counter()-start
            answers=[answer(i,o) for i,o in enumerate(outputs)]
            start=time.perf_counter()
            workers=llm.collective_rpc('finish_dense_cascade_phase')
            finalize=time.perf_counter()-start
            control_wall=time.perf_counter()-phase_started
            run={'phase':phase,'method':method,'validation':validation,'natural_eos_enabled':natural,
                'sampling':sampling,'anchor_wall_seconds':anchor_wall,'anchor_control_wall_seconds':anchor_control,
                'group_control_wall_seconds':group_control,'complete_generate_wall_seconds':wall,
                'cold_shared_document_total_wall_seconds':anchor_wall+wall,
                'phase_validation_and_record_export_wall_seconds':finalize,
                'generate_plus_phase_validation_wall_seconds':wall+finalize,'full_phase_control_wall_seconds':control_wall,
                'answers':answers,'total_output_tokens':sum(len(a['tokens']) for a in answers),'workers':workers}
            report['runs'].append(run)
            save()
            assert len(answers)==batch and all(a['request_metrics'] is not None for a in answers)
            if not natural:
                assert all(len(a['tokens'])==tokens for a in answers)
            assert len(workers)==2 and all(w['phase_valid'] for w in workers),'Both rank receipts preserved; invalid RawPair phase'
            run['total_request_preemptions']=sum(a['request_metrics']['num_preemptions'] for a in answers)
            run['cap_hits']=[a['question_index'] for a in answers if a['finish_reason']=='length']
            run['rank_graph_execution_counts']=[{'rank':w['rank'],'candidate_replays':w['candidate_replays'],
                'native_replays':w['native_replays'],'candidate_fraction_of_counted_graphs':
                w['candidate_replays']/(w['candidate_replays']+w['native_replays']) if w['candidate_replays']+w['native_replays'] else 0} for w in workers]
            if natural:
                if not validation:
                    assert [a['tokens'] for a in answers]==[a['tokens'] for a in report['runs'][0]['answers']],'Performance trajectory differs from native qualification trajectory'
                predictions=[]
                parses=[]
                for a in answers:
                    final,_,parse=extract_final(a['text'],thinking=True)
                    predictions.append(final if a['finish_reason']=='stop' and parse=='closed_thinking' else '')
                    parses.append({'question_index':a['question_index'],'final':final,'parse':parse})
                run['quality']={'metric_id':METRIC_ID,'score':score_pairs(predictions,[q['answer'] for q in group['questions']]),'parses':parses}
            if validation:
                run['qualification_max_states_requested']=qualification_steps
                run['qualification_states_per_batch']=args.qualification_states_per_batch
                run['qualification_passed']=all(w['qualification_passed'] and w['qualification_steps']>0 for w in workers)
                save()
                assert run['qualification_passed'],'Required actual native-state bitwise qualification did not pass in both ranks'
                run['qualification_joint_audit']=audit_qualification(workers,enabled_batches,args.qualification_states_per_batch)
                save()
                continue
            if method=='cascade':
                assert all(w['candidate_replays']>0 for w in workers),'No candidate execution; no speed comparison is valid'
            signatures=[]
            prepare_totals=[]
            for worker in workers:
                rows=json.loads(Path(worker['graph_metadata']).read_text())['replays']
                raw=json.loads(Path(worker['raw_pair_record']).read_text())
                assert len(rows)==worker['candidate_replays']+worker['native_replays']
                assert sum(r['candidate_graph_replayed'] for r in rows)==worker['candidate_replays']
                assert all(r['batch'] in worker['qualified_batches'] for r in rows if r['candidate_graph_replayed'])
                if method=='native':
                    assert worker['candidate_replays']==0
                signatures.append(hashlib.sha256(json.dumps([(r['batch'],r['questions'],r['positions']) for r in rows]).encode()).hexdigest())
                prepare_totals.append({'rank':worker['rank'],'candidate_prepare_records':len(raw['prepare_records']),
                    'total_recorded_prepare_host_seconds':sum(r['complete_prepare_host_seconds'] for r in raw['prepare_records']),
                    'raw_pair_record_sha256':digest(Path(worker['raw_pair_record'])),
                    'fallbacks':raw['fallbacks']})
            assert signatures[0]==signatures[1],'TP rank native trajectories differ'
            run['graph_trajectory_sha256_by_rank']=signatures
            run['candidate_prepare_receipts']=prepare_totals
            if phase in ('fixed_native_3','natural_native_3'):
                kind='natural' if natural else 'fixed'
                selected=[r for r in report['runs'] if r['phase'].startswith(kind+'_')]
                report['comparisons'][kind]=compare_abba(selected)
                save()
                assert report['comparisons'][kind]['identical_output_comparison_valid'],'Outputs/APC differ; keep results, reject speed interpretation'
            save()
        start=time.perf_counter()
        report['native_after']=llm.collective_rpc('native_precision_receipt',args=('auto',True))
        report['final_precision_receipt_wall_seconds']=time.perf_counter()-start
        report['status']='completed'
        save()
    except BaseException:
        report['status']='failed'
        report['error']=traceback.format_exc()
        save()
        raise


if __name__=='__main__':
    main()
