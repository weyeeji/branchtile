"""Production-only multi-document graph reuse under explicit training-shape transfer."""
import argparse,copy,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase_energy_counter import PhaseEnergyCounter,interval
from scripts.benchmark_native_raw_pair_general_policy_e2e import answer,compare_abba,policy_eligible


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_task_components(report):
    """Account actual controls once; this is an ABBA component estimate, not a fresh service run."""
    runs=report['runs']
    def mean_pair(method,read):
        sums=[]
        for i in ((0,3) if method=='native' else (1,2)):
            selected=[r for r in runs if r['phase']==f'natural_{method}_{i}']
            assert len(selected)==2
            values=[read(r) for r in selected]
            if any(v is None for v in values):return None
            sums.append(sum(values))
        return statistics.mean(sums)
    full={m:mean_pair(m,lambda r:r['full_phase_control_wall_seconds']) for m in ('native','cascade')}
    extra=report['offline_preflight_wall_seconds']+report['candidate_setup_instrumented_wall_seconds']
    common=report['common_energy_initialization_and_identity_RPC_wall_seconds']+report['common_precision_export_instrumented_wall_seconds']
    native=full['native']+common;candidate=full['cascade']+common+extra
    cold=report['input_preparation_seconds']+report['native_initialization_and_graph_setup_seconds']
    phase_j={m:mean_pair(m,lambda r:r['energy_intervals']['complete_document_through_export']['total_GPU_joules']) for m in ('native','cascade')}
    setup_j=report['setup_energy_intervals']['candidate_configuration_and_capture']['total_GPU_joules']
    common_j=report['setup_energy_intervals']['common_precision_receipt_export']['total_GPU_joules']
    measured=all(v is not None for v in [*phase_j.values(),setup_j,common_j])
    n_j=phase_j['native']+common_j if measured else None
    c_j=phase_j['cascade']+common_j+setup_j if measured else None
    return {'full_phase_pair_means_seconds':full,'common_control_seconds':common,'candidate_extra_setup_seconds':extra,
            'native_seconds':native,'candidate_seconds':candidate,'native_over_candidate':native/candidate,
            'common_input_native_setup_seconds':cold,'cold_component_native_seconds':native+cold,
            'cold_component_candidate_seconds':candidate+cold,'cold_component_native_over_candidate':(native+cold)/(candidate+cold),
            'phase_pair_means_GPU_joules':phase_j,'candidate_setup_GPU_joules':setup_j,'common_precision_export_GPU_joules':common_j,
            'native_GPU_joules':n_j,'candidate_GPU_joules':c_j,'GPU_joule_native_over_candidate':n_j/c_j if measured and c_j>0 else None,
            'scope':'Component sum of measured one-time controls/setup plus same-session two-document ABBA means. Full phases include activation/reset/anchor/group control/generation/export and six counter samples. Not a matched independent fresh-service experiment.',
            'energy_scope':'Verified board counters begin after common engine and energy initialization. Candidate configuration/capture and common precision export are included once. Offline CPU preflight and common engine/init energy are not measured; no full-process Joule claim.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',default='configs/raw_pair_direct_multidoc_next8_shape_transfer_energy_v2.json')
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--setup-only',action='store_true');ap.add_argument('--training-qualification-summary',required=True);ap.add_argument('--task-index',type=int,choices=(0,1,2),required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert args.prepare_only or len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
    out=Path(args.output);assert not (out/'summary.json').exists();out.mkdir(parents=True,exist_ok=True)
    proto=json.loads(Path(args.protocol).read_text());document_map={d['document_id']:d for d in proto['documents']}
    docs=[document_map[k] for k in proto['fixed_two_document_tasks'][args.task_index]];assert len(docs)==2
    from scripts.vllm_raw_pair_direct_transfer_worker import SOURCE_FILES
    sources=['scripts/benchmark_raw_pair_direct_next8_transfer.py','src/phase_energy_counter.py',*SOURCE_FILES,
             'scripts/benchmark_native_raw_pair_general_policy_e2e.py','scripts/capture_model.py','scripts/vllm_batch_admission.py','src/financial_answer_metric.py','scripts/screen_application.py']
    r={'status':'initializing','args':vars(args),'protocol':proto,'protocol_sha256':digest(args.protocol),
       'source_sha256':{p:digest(p) for p in sources},'runs':[],'comparisons':{},
       'limits':proto['limitations']+['Explicit training-shape transfer: evaluation documents do not have independent per-layer strict qualification.','The exact eight-document list includes two preserved context failures; this selected fixed pair is a conditional feasible task.']+['This session captures production graphs only, once; prior strict qualification on the two frozen training documents is required; new evaluation documents are not locally qualified.','Energy counters include the full device; external activity invalidates task-only energy attribution. Unsupported counters remain null.']}
    def save():(out/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        from scripts.vllm_raw_pair_direct_multidoc_worker import source_hashes as strict_source_hashes
        from src.raw_pair_direct_training_shape_transfer import load_training_shapes
        tick=time.perf_counter();transfer=load_training_shapes(args.training_qualification_summary,strict_source_hashes(),proto['candidate_enabled_batches'])
        offline=transfer['loaded']
        r['offline_preflight_wall_seconds']=time.perf_counter()-tick
        r['offline_qualification']={key:value for key,value in offline.items() if key not in ('records','native_tokens_by_document')}
        r['training_shape_transfer']=transfer['contract']
        r['selected_document_ids']=[d['document_id'] for d in docs];r['preserved_full8_context_failures']=proto['predeclared_input_failures']
        from transformers import AutoTokenizer
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        from scripts.screen_application import extract_final
        from src.financial_answer_metric import METRIC_ID,score_pairs
        tick=time.perf_counter();tok=AutoTokenizer.from_pretrained(proto['model_path'],local_files_only=True)
        groups={g['document_id']:g for g in json.loads(Path(proto['manifest']).read_text())['groups']};inputs={}
        for d in docs:
            prompts,begin,end=make_inputs(tok,groups[d['document_id']]);hashes=[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
            assert hashes==d['prompt_token_hashes'] and list(map(len,prompts))==d['prompt_lengths'] and [begin,end]==d['document_token_range']
            common=0
            for values in zip(*prompts):
                if len(set(values))!=1:break
                common+=1
            assert common//16*16==d['common_complete_page_tokens'] and len(prompts)==d['B']
            assert max(map(len,prompts))+proto['sampling']['max_tokens']<=proto['engine']['max_model_len']
            inputs[d['document_id']]=prompts
        r['input_preparation_seconds']=time.perf_counter()-tick;r['manifest_sha256']=digest(proto['manifest'])
        if args.prepare_only:
            import torch
            assert not torch.cuda.is_initialized();r['status']='CPU_prepared';save();return
        from vllm import LLM,SamplingParams
        engine=copy.deepcopy(proto['engine']);engine.update(model=proto['model_path'],kv_cache_dtype='auto',disable_log_stats=False,
            worker_extension_cls='scripts.vllm_raw_pair_direct_transfer_worker.RawPairDirectTransferWorker')
        r['engine']=engine;tick=time.perf_counter();llm=LLM(**engine);r['native_initialization_and_graph_setup_seconds']=time.perf_counter()-tick
        enabled=proto['candidate_enabled_batches'];first=docs[0]
        cfg={'output':str(out.resolve()),'backend':'native_multidoc_dynamic_cut','batch':proto['engine']['max_num_seqs'],
             'enabled_batches':enabled,'setup_mode':'production_only','offline_training_records':offline['records'],'training_protocol':transfer['training_protocol'],'common_prefix_tokens':first['common_complete_page_tokens'],
             'prompt_token_hashes':first['prompt_token_hashes'],'documents':docs}
        tick=time.perf_counter();identities=llm.collective_rpc('raw_pair_device_identity')
        r['energy_worker_devices']=identities
        energy=PhaseEnergyCounter([x['device_UUID'] for x in sorted(identities,key=lambda x:x['rank'])])
        r['common_energy_initialization_and_identity_RPC_wall_seconds']=time.perf_counter()-tick
        setup_tick=time.perf_counter();setup_samples=[energy.sample('before_candidate_configuration')]
        tick=time.perf_counter();r['configuration']=llm.collective_rpc('configure_dense_cascade',args=(cfg,));r['candidate_configuration_seconds']=time.perf_counter()-tick
        setup_samples.append(energy.sample('after_candidate_configuration'))
        tick=time.perf_counter();r['capture']=llm.collective_rpc('capture_dense_cascade_graphs');r['candidate_capture_seconds']=time.perf_counter()-tick
        setup_samples.append(energy.sample('after_candidate_capture'))
        r['candidate_setup_instrumented_wall_seconds']=time.perf_counter()-setup_tick
        export_tick=time.perf_counter();r['native_before']=llm.collective_rpc('native_precision_receipt',args=('auto',False))
        r['common_precision_receipt_export_RPC_wall_seconds']=time.perf_counter()-export_tick
        setup_samples.append(energy.sample('after_common_precision_receipt_export'))
        r['common_precision_export_instrumented_wall_seconds']=time.perf_counter()-export_tick
        r['setup_energy_samples']=setup_samples
        r['setup_energy_intervals']={'candidate_configuration':interval(setup_samples[0],setup_samples[1]),
            'candidate_capture':interval(setup_samples[1],setup_samples[2]),'candidate_configuration_and_capture':interval(setup_samples[0],setup_samples[2]),
            'common_precision_receipt_export':interval(setup_samples[2],setup_samples[3])}
        r['setup_NVML_sampling_wall_seconds']=sum(x['NVML_sampling_wall_seconds'] for x in setup_samples)
        r['energy_setup_accounting']='Instrumentation identity/init and precision evidence export are common controls charged equally. Candidate config/capture including their three samples is candidate setup. GPU energy begins after common engine/instrument initialization; CPU offline preflight has no measured GPU counter boundary.'
        save()
        assert all(c['captured_modes']==['production'] for c in r['capture'])
        qualified_answers={}
        phases=[] if args.setup_only else [(f'natural_{m}_{i}',m,False) for i,m in enumerate(('native','cascade','cascade','native'))]
        for phase,method,validation in phases:
            for doc_index,d in enumerate(docs):
                docid=d['document_id'];prefix=d['common_complete_page_tokens'];prompts=inputs[docid];tag=f'{phase}_doc{doc_index}'
                assert not llm.llm_engine.has_unfinished_requests(),'Document activation requires an empty completed scheduler group'
                start=time.perf_counter();samples=[energy.sample('document_start')]
                tick=time.perf_counter();llm.collective_rpc('set_dense_cascade_phase',args=(tag+'_anchor','native',False))
                activations=llm.collective_rpc('activate_raw_pair_document',args=(docid,method=='cascade' or validation))
                activation_wall=time.perf_counter()-tick;samples.append(energy.sample('after_document_activation'))
                tick=time.perf_counter();assert llm.reset_prefix_cache();reset_wall=time.perf_counter()-tick
                tick=time.perf_counter();anchor=llm.generate([{'prompt_token_ids':prompts[0]}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=False,seed=proto['seed']),use_tqdm=False)
                anchor_wall=time.perf_counter()-tick;assert anchor[0].num_cached_tokens==0;samples.append(energy.sample('after_anchor'))
                tick=time.perf_counter();llm.collective_rpc('set_dense_cascade_phase',args=(tag,method,validation))
                assert validation is False
                group_control=time.perf_counter()-tick;samples.append(energy.sample('before_group_generate'))
                tick=time.perf_counter();outputs=generate_ready_group(llm,[{'prompt_token_ids':x} for x in prompts],SamplingParams(**proto['sampling']));wall=time.perf_counter()-tick
                assert not llm.llm_engine.has_unfinished_requests();samples.append(energy.sample('after_group_generate'))
                answers=[answer(i,o) for i,o in enumerate(outputs)];tick=time.perf_counter();workers=llm.collective_rpc('finish_dense_cascade_phase');finish_wall=time.perf_counter()-tick
                samples.append(energy.sample('after_phase_export'));total=time.perf_counter()-start
                run={'phase':phase,'document_id':docid,'document_index':doc_index,'method':method,'validation':validation,'natural_eos_enabled':True,
                     'sampling':proto['sampling'],'B':d['B'],'P':prefix,'activations':activations,'activation_control_wall_seconds':activation_wall,
                     'prefix_reset_wall_seconds':reset_wall,'anchor_wall_seconds':anchor_wall,'group_control_wall_seconds':group_control,
                     'complete_generate_wall_seconds':wall,'generate_plus_phase_validation_wall_seconds':wall+finish_wall,
                     'cold_shared_document_total_wall_seconds':activation_wall+reset_wall+anchor_wall+group_control+wall,
                     'full_phase_control_wall_seconds':total,'answers':answers,'total_output_tokens':sum(len(a['tokens']) for a in answers),'workers':workers,
                     'energy_samples':samples,'energy_intervals':{'activation':interval(samples[0],samples[1]),'anchor_and_reset':interval(samples[1],samples[2]),
                       'group_generate':interval(samples[3],samples[4]),'complete_document_through_export':interval(samples[0],samples[5])},
                     'NVML_sampling_wall_seconds':sum(s['NVML_sampling_wall_seconds'] for s in samples)}
                r['runs'].append(run);save()
                assert len(answers)==d['B'] and all(w['phase_valid'] for w in workers)
                assert all(a['request_metrics'] is not None for a in answers)
                run['total_request_preemptions']=sum(a['request_metrics']['num_preemptions'] for a in answers)
                run['cap_hits']=[a['question_index'] for a in answers if a['finish_reason']=='length']
                pred=[];parses=[]
                for a in answers:
                    final,_,parse=extract_final(a['text'],thinking=True);pred.append(final if a['finish_reason']=='stop' and parse=='closed_thinking' else '')
                    parses.append({'question_index':a['question_index'],'final':final,'parse':parse})
                run['quality']={'metric_id':METRIC_ID,'score':score_pairs(pred,[q['answer'] for q in groups[docid]['questions']]),'parses':parses}
                if phase=='natural_native_0':qualified_answers[docid]=[a['tokens'] for a in answers]
                run['all_tokens_equal_to_first_native']=[a['tokens'] for a in answers]==qualified_answers[docid]
                signatures=[];run['candidate_graph_executed']=all(w['candidate_replays']>0 for w in workers)
                for w in workers:
                    raw=json.loads(Path(w['raw_pair_record']).read_text());rows=json.loads(Path(w['graph_metadata']).read_text())['replays']
                    assert raw['document_id']==docid and raw['capture_invocations']==1
                    assert raw['setup_mode']=='production_only' and raw['checks']==[] and raw['qualification_steps']==0
                    assert raw['qualification_kind']=='explicit_training_shape_transfer' and raw['document_locally_qualified'] is False
                    assert raw['qualified_batches']==transfer['contract']['eligible_training_qualified_shapes']
                    assert set(raw['captured_layers_by_batch'])=={'production'}
                    assert raw['captured_layers_by_batch']['production']=={str(b):list(range(48)) for b in enabled}
                    assert raw['plan']['shared_cut_ptr']==r['capture'][w['rank']]['plan']['shared_cut_ptr']
                    assert sum(x['candidate_graph_replayed'] for x in rows)==(0 if validation else w['candidate_replays'])
                    if not validation:assert len(rows)==w['candidate_replays']+w['native_replays']
                    assert all(x['batch'] in w['qualified_batches'] and policy_eligible(x,prefix,enabled) for x in rows if x['candidate_graph_replayed'])
                    signatures.append(hashlib.sha256(json.dumps([(x['batch'],x['questions'],x['positions']) for x in rows]).encode()).hexdigest())
                run['graph_trajectory_sha256_by_rank']=signatures
                assert len(signatures)==2 and len(set(signatures))==1, 'TP ranks must replay the same request/position trajectory'
                save()
        if not args.setup_only:
            for d in docs:
                rows=[x for x in r['runs'] if x['document_id']==d['document_id'] and not x['validation']]
                comparison=compare_abba(rows);comparison['candidate_executed_in_both_candidate_arms']=all(x['candidate_graph_executed'] for x in rows if x['method']=='cascade')
                comparison['candidate_speed_claim_eligible']=comparison['identical_output_comparison_valid'] and comparison['candidate_executed_in_both_candidate_arms']
                r['comparisons'][d['document_id']]=comparison;save()
            totals={m:statistics.mean(sum(x['cold_shared_document_total_wall_seconds'] for x in r['runs'] if x['phase']==f'natural_{m}_{i}') for i in ((0,3) if m=='native' else (1,2))) for m in ('native','cascade')}
            r['two_document_task_means_seconds']=totals;r['two_document_native_over_candidate']=totals['native']/totals['cascade']
            r['two_document_speed_claim_eligible']=all(c['candidate_speed_claim_eligible'] for c in r['comparisons'].values())
        r['setup_scope']='One production graph bank captured once. Explicit training-shape transfer; no per-evaluation-document strict qualification. Missing training-qualified shapes fall back to native. New-doc output identity is independently tested under ABBA.'
        if not args.setup_only:
            r['first_two_document_task_component_sums']=first_task_components(r)
            save()
            assert all(c['identical_output_comparison_valid'] for c in r['comparisons'].values()), 'Preserved all ABBA runs; output/APC/trajectory identity failed'
        r['status']='completed';save();print(json.dumps({'status':r['status'],'output':str(out)}))
    except Exception:r['status']='failed';r['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
