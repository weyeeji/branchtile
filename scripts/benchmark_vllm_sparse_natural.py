"""Actual native/E0/union generation with shadows/forcing/copies disabled.

Single exploratory document, fixed shared history masks, independent natural
trajectories. Full JCT includes per-request prefill and every metadata update.
One-time model load, dense selector preparation and graph construction are
reported separately and must be charged in any service break-even argument.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=8192);p.add_argument('--compile-mode',type=int,default=3)
    p.add_argument('--methods',default='dense,e0,union');p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--budget',default='4096',help='Frozen document token budget, or full')
    args=p.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from capture_model import make_inputs
    from scripts.vllm_batch_admission import generate_ready_group
    model='data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    manifest=ROOT/'data/manifests/docfinqa_exploratory_dev8.json'
    group=next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id']=='docfinqa-c0f11a67aae850fc')
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True);inputs,start,end=make_inputs(tokenizer,group)
    budget=end-start if args.budget=='full' else int(args.budget)
    assert 36<=budget<=end-start and 1<=args.steps<=8192
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','uid':os.geteuid(),'python':sys.executable,'document_id':group['document_id'],
        'document_sha256':group['document_sha256'],'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'B':len(inputs),'prompt_lengths':list(map(len,inputs)),'budget':budget,'budget_request':args.budget,'steps_cap':args.steps,
        'sampling':{'temperature':.6,'top_p':.95,'seed':20261016,'max_tokens':args.steps,'ignore_eos':False},
        'methods':args.methods.split(','),'repeats':args.repeats,'runs':[],
        'limitations':['One exploratory document is not a held-out task-quality experiment.','No forced tokens: method trajectories and total work may differ; compare complete JCT and actual token counts together.','Minimal CPU step records remain enabled and charged. No logit copies, shadows, layer snapshots or teacher forcing.',
            'Both candidate graph sets remain resident for every method; native standalone deployment can reclaim their memory.',
            'Fixed histories are prepared once from actual dense suffix Q and held constant across all methods; one-time costs are separately reported, not free.']}
    report['source_sha256']={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'scripts/benchmark_vllm_sparse_natural.py','scripts/vllm_sparse_production_worker.py',
        'scripts/vllm_shared_plan_graph_worker.py','scripts/vllm_sparse_graph_worker.py',
        'src/vllm_shared_decode_plan.py','src/vllm_shared_union_plan.py')}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    engine=dict(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',max_model_len=98304,
        max_num_seqs=3,max_num_batched_tokens=4096,enable_prefix_caching=True,enforce_eager=False,gpu_memory_utilization=.80,
        disable_custom_all_reduce=True,enable_batch_sharded_sampling=False,
        compilation_config={'mode':args.compile_mode,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,3]},
        worker_extension_cls='scripts.vllm_sparse_production_worker.ProductionSparseWorkerExtension',seed=20261016)
    report['engine_configuration']=engine;save();started=time.perf_counter();llm=LLM(**engine)
    report['initialization_wall_seconds']=time.perf_counter()-started;save()
    config={'output':str(out.resolve()),'document_token_range':[start,end],'prompt_tokens':inputs,
        'prompt_token_hashes':[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
        'budget':budget,'check_dense':False,'check_sparse_reference':False,'capture_logits':False,
        'capture_layer_outputs':False,'graph_shadow_check':False}
    report['configured_workers']=llm.collective_rpc('configure_sparse_attention',args=(config,));save()
    warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=False)
    started=time.perf_counter()
    llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],warm)
    report['selector_dense_warmup_wall_seconds']=time.perf_counter()-started
    report['graph_capture']=llm.collective_rpc('capture_sparse_graphs',args=(('e0','union'),));save()
    report['production_configuration']=llm.collective_rpc('enable_production_sparse');save()
    # Warm the same stochastic sampler before any measured arm. The historical
    # 4K run predates this step and keeps its first-dense JIT limitation.
    llm.collective_rpc('set_sparse_phase',args=('stochastic_sampler_warmup','dense'))
    started=time.perf_counter()
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(temperature=.6,top_p=.95,seed=20261016,max_tokens=1))
    report['stochastic_sampler_warmup_wall_seconds']=time.perf_counter()-started;save()
    for repeat in range(args.repeats):
        methods=report['methods'] if repeat%2==0 else list(reversed(report['methods']))
        for method in methods:
            phase=f'{method}_natural_{repeat}'
            assert llm.reset_prefix_cache()
            llm.collective_rpc('set_sparse_phase',args=(phase+'_anchor','dense'))
            started=time.perf_counter();anchor=llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
            anchor_wall=time.perf_counter()-started
            llm.collective_rpc('set_sparse_phase',args=(phase,method,None))
            started=time.perf_counter()
            generated=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(**report['sampling']))
            wall=time.perf_counter()-started
            run={'phase':phase,'method':method,'repeat':repeat,'anchor_wall_seconds':anchor_wall,'complete_generate_wall_seconds':wall,
                'teacher_forced':False,'answers':[{'tokens':list(x.outputs[0].token_ids),'text':x.outputs[0].text,
                'finish_reason':x.outputs[0].finish_reason,'stop_reason':x.outputs[0].stop_reason,
                'num_cached_tokens':x.num_cached_tokens,'question_index':i} for i,x in enumerate(generated)]}
            run['total_output_tokens']=sum(len(a['tokens']) for a in run['answers'])
            run['tokens_per_generate_wall_second']=run['total_output_tokens']/wall
            run['workers']=llm.collective_rpc('finish_sparse_phase');report['runs'].append(run);save()
            print(json.dumps({'phase':phase,'wall_seconds':wall,'output_tokens':run['total_output_tokens'],
                'finish_reasons':[a['finish_reason'] for a in run['answers']]}),flush=True)
    report['status']='completed';save();print(json.dumps({'status':'completed','output':str(out)}),flush=True)


if __name__=='__main__':main()
