"""Official native backend natural generation control with matched resources/APC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',default='FLASHINFER')
    p.add_argument('--steps',type=int,default=8192);p.add_argument('--repeats',type=int,default=2);args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from capture_model import make_inputs
    from scripts.vllm_batch_admission import generate_ready_group
    model='data/models/Qwen3-30B-A3B-Thinking-2507-FP8';manifest=ROOT/'data/manifests/docfinqa_exploratory_dev8.json'
    group=next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id']=='docfinqa-c0f11a67aae850fc')
    prep=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    inputs,start,end=make_inputs(tokenizer,group);input_seconds=time.perf_counter()-prep
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','uid':os.geteuid(),'python':sys.executable,'document_id':group['document_id'],
        'document_sha256':group['document_sha256'],'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'B':len(inputs),'prompt_lengths':list(map(len,inputs)),'budget':end-start,'budget_request':'full','steps_cap':args.steps,
        'sampling':{'temperature':.6,'top_p':.95,'seed':20261016,'max_tokens':args.steps,'ignore_eos':False},
        'methods':[args.backend],'repeats':args.repeats,'runs':[],'native_official_backend_control':True,
        'generator_input_preparation_wall_seconds':input_seconds,
        'limitations':['One exploratory document is not a held-out task-quality experiment.','Natural trajectories differ; token/s does not establish task speedup.',
            'Native backend uses its own officially selected cache layout/workspace; same GPU memory fraction and model/batch/context limits as sparse runs.',
            'No candidate graphs retained, no selector, no logit copies or shadows. Minimal CPU step identity/timestamps remain charged.']}
    report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/benchmark_vllm_native_backend.py','scripts/vllm_native_backend_worker.py']}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    engine=dict(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',max_model_len=98304,
        max_num_seqs=3,max_num_batched_tokens=4096,enable_prefix_caching=True,enforce_eager=False,gpu_memory_utilization=.80,
        disable_custom_all_reduce=True,enable_batch_sharded_sampling=False,
        compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,3]},
        attention_config={'backend':args.backend},worker_extension_cls='scripts.vllm_native_backend_worker.NativeBackendWorkerExtension',seed=20261016)
    report['engine_configuration']=engine;save();started=time.perf_counter();llm=LLM(**engine)
    report['initialization_wall_seconds']=time.perf_counter()-started;save()
    config={'output':str(out.resolve()),'backend':args.backend,
        'prompt_token_hashes':[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs]}
    report['configured_workers']=llm.collective_rpc('configure_native_probe',args=(config,));save()
    warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=False)
    started=time.perf_counter();llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(temperature=.6,top_p=.95,seed=20261016,max_tokens=1))
    report['stochastic_sampler_warmup_wall_seconds']=time.perf_counter()-started;save()
    for repeat in range(args.repeats):
        phase=f'{args.backend}_natural_{repeat}';assert llm.reset_prefix_cache()
        llm.collective_rpc('set_native_phase',args=(phase+'_anchor',))
        started=time.perf_counter();llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False);anchor=time.perf_counter()-started
        llm.collective_rpc('set_native_phase',args=(phase,))
        started=time.perf_counter()
        generated=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(**report['sampling']))
        wall=time.perf_counter()-started
        run={'phase':phase,'method':args.backend,'repeat':repeat,'anchor_wall_seconds':anchor,'complete_generate_wall_seconds':wall,
            'teacher_forced':False,'answers':[{'tokens':list(x.outputs[0].token_ids),'text':x.outputs[0].text,
            'finish_reason':x.outputs[0].finish_reason,'stop_reason':x.outputs[0].stop_reason,'num_cached_tokens':x.num_cached_tokens,'question_index':i} for i,x in enumerate(generated)]}
        run['total_output_tokens']=sum(len(a['tokens']) for a in run['answers']);run['tokens_per_generate_wall_second']=run['total_output_tokens']/wall
        run['workers']=llm.collective_rpc('finish_native_phase');report['runs'].append(run);save()
        print(json.dumps({'phase':phase,'tokens':run['total_output_tokens'],'wall_seconds':wall,'finish_reasons':[a['finish_reason'] for a in run['answers']]}),flush=True)
    report['status']='completed';save()


if __name__=='__main__':main()
