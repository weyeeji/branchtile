"""One actual native natural cohort, full text/history, with preemption metrics."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer_record(index,output):
    item=output.outputs[0]
    metrics=dataclasses.asdict(output.metrics) if dataclasses.is_dataclass(output.metrics) else None
    return {'question_index':index,'tokens':list(item.token_ids),'text':item.text,
            'finish_reason':item.finish_reason,'stop_reason':item.stop_reason,
            'num_cached_tokens':output.num_cached_tokens,'request_metrics':metrics,
            'request_id':output.request_id,'prompt_tokens':len(output.prompt_token_ids),
            'scheduled_to_first_token_seconds':metrics['first_token_ts']-metrics['scheduled_ts'] if metrics else None,
            'first_to_last_token_seconds':metrics['last_token_ts']-metrics['first_token_ts'] if metrics else None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='configs/native_natural_cohort16plus_training_v1.json')
    parser.add_argument('--document-id',required=True)
    parser.add_argument('--kv-dtype',choices=['auto','fp8_e4m3'],default='auto')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
    protocol_path=ROOT/args.protocol
    protocol=json.loads(protocol_path.read_text())
    out=ROOT/args.output
    assert not (out/'summary.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    sources=['scripts/benchmark_native_natural_cohort.py','scripts/vllm_native_natural_cohort_worker.py',
        'scripts/vllm_native_precision_worker.py','scripts/vllm_native_backend_worker.py',
        'scripts/probe_engine_docfinqa.py','scripts/vllm_batch_admission.py','scripts/capture_model.py',
        'scripts/screen_application.py','src/financial_answer_metric.py']
    report={'status':'initializing','uid':os.geteuid(),'python':sys.executable,'args':vars(args),
        'protocol':protocol,'protocol_sha256':sha(protocol_path),'runs':[],
        'source_sha256':{p:sha(ROOT/p) for p in sources},
        'scope':'Unchanged native FlashInfer/XQA natural ready-group workload. No replacement attention, profile, or cached-output intervention.',
        'limitations':['Question counts do not establish real-world simultaneous arrival. All selected natural questions are admitted together explicitly.',
            'CPU identity and step instrumentation plus native logging statistics remain charged in generation wall.',
            'Request scheduled-to-first-token timing includes queue/engine effects and is not isolated GPU prefill kernel time.',
            'Native initialization includes graph setup; no additional candidate graphs exist.',
            'B8 measured pool was only a planning reference. Actual pool, preemptions, recomputation, cap hits, and failures must be retained.']}
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
        assert hashes==item['prompt_token_hashes'] and len(set(hashes))==len(hashes)
        assert list(map(len,prompts))==item['prompt_lengths'] and [begin,end]==item['document_token_range']
        assert len(prompts)==len(group['questions'])==group['natural_question_count']>=16
        assert max(map(len,prompts))+protocol['sampling']['max_tokens']<=protocol['engine']['max_model_len'], 'Preselected context failure; no replacement/truncation allowed'
        report.update(B=len(prompts),manifest=protocol['manifest'],manifest_sha256=sha(manifest_path),
            document_sha256=group['document_sha256'],prompt_token_hashes=hashes,prompt_lengths=list(map(len,prompts)),
            input_preparation_wall_seconds=time.perf_counter()-start)
        engine=copy.deepcopy(protocol['engine'])
        engine.update(model=protocol['model_path'],kv_cache_dtype=args.kv_dtype,max_num_seqs=len(prompts),
            disable_log_stats=False,
            worker_extension_cls='scripts.vllm_native_natural_cohort_worker.NativeNaturalCohortWorker')
        engine['compilation_config']['cudagraph_capture_sizes']=list(range(1,len(prompts)+1))
        report['engine']=engine
        save()
        start=time.perf_counter()
        llm=LLM(**engine)
        report['native_engine_initialization_and_graph_setup_wall_seconds']=time.perf_counter()-start
        start=time.perf_counter()
        config={'output':str(out.resolve()),'backend':'native_XQA_full_history','prompt_token_hashes':hashes}
        report['configuration']=llm.collective_rpc('configure_native_cohort_probe',args=(config,))
        report['native_before']=llm.collective_rpc('native_precision_receipt',args=(args.kv_dtype,False))
        report['configuration_and_initial_receipt_wall_seconds']=time.perf_counter()-start
        save()
        assert llm.reset_prefix_cache()
        llm.collective_rpc('set_native_cohort_phase',args=('cold_anchor',))
        sampling=SamplingParams(temperature=0,max_tokens=1,ignore_eos=False,seed=protocol['seed'])
        start=time.perf_counter()
        anchors=llm.generate([{'prompt_token_ids':prompts[0]}],sampling,use_tqdm=False)
        anchor_wall=time.perf_counter()-start
        anchor=answer_record(0,anchors[0])
        assert anchor['num_cached_tokens']==0
        start=time.perf_counter()
        anchor_workers=llm.collective_rpc('finish_native_cohort_phase')
        anchor_export=time.perf_counter()-start
        report['anchor']={'wall_seconds':anchor_wall,'request':anchor,'workers':anchor_workers,
                          'record_export_wall_seconds':anchor_export}
        save()
        llm.collective_rpc('set_native_cohort_phase',args=('native_natural',))
        start=time.perf_counter()
        outputs=generate_ready_group(llm,[{'prompt_token_ids':p} for p in prompts],SamplingParams(**protocol['sampling']))
        wall=time.perf_counter()-start
        answers=[answer_record(i,o) for i,o in enumerate(outputs)]
        assert len(answers)==len(prompts)
        start=time.perf_counter()
        workers=llm.collective_rpc('finish_native_cohort_phase')
        export=time.perf_counter()-start
        run={'phase':'native_natural','method':'native','natural_eos_enabled':True,'sampling':protocol['sampling'],
            'anchor_wall_seconds':anchor_wall,'complete_generate_wall_seconds':wall,
            'cold_shared_document_total_wall_seconds':anchor_wall+wall,
            'phase_record_export_wall_seconds':export,'generate_plus_record_export_wall_seconds':wall+export,
            'answers':answers,'total_output_tokens':sum(len(a['tokens']) for a in answers),'workers':workers,
            'request_metrics_all_present':all(a['request_metrics'] is not None for a in answers)}
        report['runs'].append(run)
        save()
        assert run['request_metrics_all_present'],'Preemption metrics required; preserve failure rather than assume zero'
        run['total_request_preemptions']=sum(a['request_metrics']['num_preemptions'] for a in answers)
        run['cap_hits']=[a['question_index'] for a in answers if a['finish_reason']=='length']
        predictions=[]
        parses=[]
        for answer in answers:
            final,_,parse=extract_final(answer['text'],thinking=True)
            predictions.append(final if answer['finish_reason']=='stop' and parse=='closed_thinking' else '')
            parses.append({'question_index':answer['question_index'],'final':final,'parse':parse})
        run['quality']={'metric_id':METRIC_ID,'score':score_pairs(predictions,[q['answer'] for q in group['questions']]),'parses':parses}
        report['native_after']=llm.collective_rpc('native_precision_receipt',args=(args.kv_dtype,True))
        report['status']='completed'
        save()
    except BaseException:
        report['status']='failed'
        report['error']=traceback.format_exc()
        save()
        raise


if __name__=='__main__':
    main()
