"""Full-document TP1 capacity and natural EOS pilot, followed by separate native profiling."""
import argparse,copy,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_native_raw_pair_general_policy_e2e import answer

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',default='configs/thinking_4b_tp1_native_probe_v1.json')
    ap.add_argument('--output',required=True);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--capacity-only',action='store_true')
    args=ap.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert (args.prepare_only and os.environ.get('CUDA_VISIBLE_DEVICES')=='') or (not args.prepare_only and len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==1 and os.environ.get('CUDA_VISIBLE_DEVICES'))
    out=Path(args.output);assert not (out/'summary.json').exists();out.mkdir(parents=True,exist_ok=True)
    proto=json.loads(Path(args.protocol).read_text());assert len(proto['documents'])==1
    sources=['scripts/probe_thinking_4b_tp1_native.py','scripts/vllm_thinking_tp1_native_worker.py','scripts/vllm_gqa4_native_backend_worker.py','scripts/vllm_native_opportunity_worker.py','scripts/vllm_native_precision_worker.py','scripts/vllm_native_backend_worker.py','scripts/probe_engine_docfinqa.py','scripts/capture_model.py','scripts/vllm_batch_admission.py','scripts/benchmark_native_raw_pair_general_policy_e2e.py','scripts/screen_application.py','src/financial_answer_metric.py']
    r={'status':'initializing','args':vars(args),'protocol':proto,'protocol_sha256':digest(args.protocol),'source_sha256':{p:digest(p) for p in sources},'runs':[],'limits':proto['limitations']}
    def save():(out/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        import torch
        from transformers import AutoTokenizer
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        from scripts.screen_application import extract_final
        from src.financial_answer_metric import METRIC_ID,score_pairs
        config=json.loads((Path(proto['model_path'])/'config.json').read_text())
        assert (config['num_hidden_layers'],config['num_attention_heads'],config['num_key_value_heads'],config['head_dim'])==(36,32,8,128)
        for name,expected in proto['model_artifact_sha256'].items():assert digest(Path(proto['model_path'])/name)==expected
        tick=time.perf_counter();doc=proto['documents'][0]
        group=next(g for g in json.loads(Path(proto['manifest']).read_text())['groups'] if g['document_id']==doc['document_id'])
        prompts,start,end=make_inputs(AutoTokenizer.from_pretrained(proto['model_path'],local_files_only=True),group)
        hashes=[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert len(prompts)==len(group['questions'])==8 and hashes==doc['prompt_token_hashes']
        assert list(map(len,prompts))==doc['prompt_lengths'] and [start,end]==doc['document_token_range']
        assert max(map(len,prompts))+proto['sampling']['max_tokens']<=proto['engine']['max_model_len']
        common=0
        for values in zip(*prompts):
            if len(set(values))!=1:break
            common+=1
        prefix=common//16*16;assert prefix==120528
        bytes_per_token=36*8*128*2*2
        initial_token_slots=prefix+sum(((len(p)-prefix+15)//16)*16 for p in prompts)
        all_cap_token_slots=prefix+sum(((len(p)+proto['sampling']['max_tokens']-prefix+15)//16)*16 for p in prompts)
        r.update(input_preparation_seconds=time.perf_counter()-tick,document_id=doc['document_id'],B=8,P=prefix,
                 manifest_sha256=digest(proto['manifest']),model_attention_shape=[36,32,8,128],KV_payload_bytes_per_token=bytes_per_token,
                 initial_shared_group_KV_token_slots=initial_token_slots,initial_shared_group_KV_payload_bytes=initial_token_slots*bytes_per_token,
                 all_eight_cap_KV_token_slots=all_cap_token_slots,all_eight_cap_KV_payload_bytes=all_cap_token_slots*bytes_per_token,
                 capacity_estimate_scope='Logical page-rounded KV payload only, before allocator/graph/workspace overhead. All-cap upper bound is not a prediction of natural output length.')
        if args.prepare_only:
            assert not torch.cuda.is_initialized();r['status']='CPU_prepared';r['cuda_initialized']=False;save();return
        from vllm import LLM,SamplingParams
        trace=out/'torch_profile';assert not trace.exists()
        engine=copy.deepcopy(proto['engine']);engine.update(model=proto['model_path'],kv_cache_dtype='auto',worker_extension_cls='scripts.vllm_thinking_tp1_native_worker.ThinkingTP1NativeWorker',
            profiler_config={'profiler':'torch','torch_profiler_dir':str(trace.resolve()),'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False,'torch_profiler_activities':['CPU','CUDA'],'delay_iterations':64,'max_iterations':32,'ignore_frontend':True})
        assert engine['tensor_parallel_size']==1 and engine['max_model_len']==131072 and engine['max_num_seqs']==8 and engine['gpu_memory_utilization']==.9
        r['engine']=engine;save();tick=time.perf_counter();llm=LLM(**engine);r['initialization_and_native_graph_seconds']=time.perf_counter()-tick
        r['probe_configuration']=llm.collective_rpc('configure_opportunity_probe',args=({'output':str(out.resolve()),'backend':'native_TP1_BF16','prompt_token_hashes':hashes},))
        r['capacity_receipt']=llm.collective_rpc('native_precision_receipt',args=('auto',False))
        assert len(r['capacity_receipt'])==1
        capacity=r['capacity_receipt'][0]['allocated_cache_token_capacity']
        r['capacity_initial_token_slot_margin']=capacity-initial_token_slots;r['capacity_all_cap_token_slot_margin']=capacity-all_cap_token_slots
        r['capacity_initial_group_payload_fits']=capacity>=initial_token_slots
        r['capacity_all_cap_payload_fits']=capacity>=all_cap_token_slots
        save()
        if args.capacity_only:r['status']='capacity_measured';save();return
        warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=False,seed=proto['seed'])
        for phase,profile in [('natural_0',False),('profile_fixed128',True)]:
            assert not llm.llm_engine.has_unfinished_requests();assert llm.reset_prefix_cache()
            llm.collective_rpc('set_native_phase',args=(phase+'_anchor',));tick=time.perf_counter()
            anchor=llm.generate([{'prompt_token_ids':prompts[0]}],warm,use_tqdm=False);anchor_wall=time.perf_counter()-tick
            assert anchor[0].num_cached_tokens==0
            llm.collective_rpc('set_native_phase',args=(phase,));sampling=copy.deepcopy(proto['sampling'])
            if profile:
                sampling.update(max_tokens=128,min_tokens=128,ignore_eos=True)
                llm.collective_rpc('set_opportunity_instrumentation',args=(True,));llm.start_profile(profile_prefix='thinking_TP1_native_decode')
            tick=time.perf_counter();outputs=generate_ready_group(llm,[{'prompt_token_ids':p} for p in prompts],SamplingParams(**sampling));wall=time.perf_counter()-tick
            if profile:
                llm.stop_profile();r['profile_graph_events']=llm.collective_rpc('finish_opportunity_instrumentation')
            answers=[answer(i,o) for i,o in enumerate(outputs)];assert len(answers)==8
            workers=llm.collective_rpc('finish_native_phase')
            run={'phase':phase,'profiled':profile,'natural_eos_enabled':not profile,'sampling':sampling,'anchor_wall_seconds':anchor_wall,
                 'complete_generate_wall_seconds':wall,'cold_shared_document_total_wall_seconds':anchor_wall+wall,
                 'answers':answers,'workers':workers,'lengths':[len(a['tokens']) for a in answers],
                 'total_output_tokens':sum(len(a['tokens']) for a in answers),'cap_hits':[a['question_index'] for a in answers if a['finish_reason']=='length'],
                 'preemptions':sum(a['request_metrics']['num_preemptions'] for a in answers) if all(a['request_metrics'] is not None for a in answers) else None}
            r['runs'].append(run);save()
            if profile:assert all(len(a['tokens'])==128 for a in answers)
            else:
                predictions=[];parses=[]
                for a in answers:
                    final,_,parse=extract_final(a['text'],thinking=True)
                    predictions.append(final if a['finish_reason']=='stop' and parse=='closed_thinking' else '')
                    parses.append({'question_index':a['question_index'],'final':final,'parse':parse})
                run['quality']={'metric_id':METRIC_ID,'score':score_pairs(predictions,[q['answer'] for q in group['questions']]),'parses':parses}
                run['all_eight_finished_naturally']=not run['cap_hits'] and all(a['finish_reason']=='stop' for a in answers)
            run['native_receipt']=llm.collective_rpc('native_precision_receipt',args=('auto',True));save()
        r['trace_files']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(trace.rglob('*')) if p.is_file()]
        natural=r['runs'][0]
        r['natural_feasible_without_preemption_or_cap']=natural['all_eight_finished_naturally'] and natural['preemptions']==0
        r['status']='completed';save()
    except BaseException:r['status']='failed';r['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
