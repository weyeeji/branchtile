"""Bounded frozen-mask variants on one official-FI native E0 graph/buffer set.

Qualification shadows use the currently installed *same new mask*. Natural
mode is an exploratory complete-output diagnostic, not a novelty/quality gate.
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
    p.add_argument('--variants',required=True);p.add_argument('--qualify',action='store_true')
    p.add_argument('--document-id',default='docfinqa-c0f11a67aae850fc')
    p.add_argument('--manifest',default='data/manifests/docfinqa_exploratory_dev8.json')
    p.add_argument('--budget',type=int,default=32768);p.add_argument('--steps',type=int,default=8192)
    p.add_argument('--seed',type=int,default=20261016);args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    specification_path=ROOT/args.variants;protocol=json.loads(specification_path.read_text())
    specifications=protocol['variants']
    assert specifications[0]=={'name':'independent','configuration':None}
    names=[s['name'] for s in specifications];assert len(names)==len(set(names))
    assert 1<=args.steps<=8192 and args.budget==protocol['budget']
    assert args.qualify == (protocol['mode']=='qualification')
    if args.qualify:
        assert names==['independent','epsilon0','epsilon001','epsilon01']
        phases=[('independent_A','independent'),('epsilon0_A','epsilon0'),
                ('epsilon001','epsilon001'),('epsilon01','epsilon01'),('epsilon0_B','epsilon0')]
    else:
        assert names==['independent','epsilon001','epsilon01']
        phases=[('dense','dense')]+[(s['name'],s['name']) for s in specifications]
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from scripts.capture_model import make_inputs
    from scripts.vllm_batch_admission import generate_ready_group
    model='data/models/Qwen3-30B-A3B-Thinking-2507-FP8';manifest=ROOT/args.manifest
    group=next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id']==args.document_id)
    inputs,start,end=make_inputs(AutoTokenizer.from_pretrained(model,local_files_only=True),group)
    assert 1<=len(inputs)<=3 and max(map(len,inputs))+args.steps<=98304 and 36<=args.budget<=end-start
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','qualification':args.qualify,'uid':os.geteuid(),'python':sys.executable,
        'document_id':group['document_id'],'document_sha256':group['document_sha256'],
        'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'B':len(inputs),'prompt_lengths':list(map(len,inputs)),'budget':args.budget,'budget_request':str(args.budget),
        'steps_cap':args.steps,'variant_protocol':str(specification_path),
        'variant_protocol_sha256':hashlib.sha256(specification_path.read_bytes()).hexdigest(),
        'variants':specifications,'fixed_phase_order':phases,'methods':[v for _,v in phases], 'repeats':1,'runs':[],
        'sampling':{'temperature':0 if args.qualify else .6,'top_p':1 if args.qualify else .95,
                    'seed':args.seed,'max_tokens':args.steps,'ignore_eos':False},
        'scope':'Bounded engineering/quality development diagnostic. No novel algorithm claim; classical coverage control still required before comparative advantage claims.',
        'limitations':['Pooled-score epsilon does not bound decode mass/O/LSE/task error.',
            'All variants freeze before any candidate output; one E0 graph/buffer set shared, all installations charged.',
            'Native FI prefill/XQA dense remain original. No physical full-cache capacity release.',
            'This run captures E0 only, unlike historical E0+union resident grids; do not attribute resource/capture savings to exchange.',
            'Classical coverage currently improves union geometry more; it must enter future strong system controls.',
            'Natural trajectories differ: report complete FINAL/EOS/caps and costs; no token/s quality or speed proof.']}
    sources=['scripts/benchmark_vllm_joint_variants.py','scripts/vllm_joint_support_worker.py',
        'scripts/vllm_sparse_production_worker.py','scripts/vllm_shared_plan_graph_worker.py',
        'scripts/vllm_sparse_graph_worker.py','scripts/vllm_sparse_worker.py','src/vllm_sparse_selector.py',
        'src/joint_support_exchange.py','src/selector_baseline.py','src/vllm_sparse_executor.py','src/vllm_shared_decode_plan.py']
    report['source_sha256']={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sources}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    engine=dict(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',max_model_len=98304,
        max_num_seqs=len(inputs),max_num_batched_tokens=4096,enable_prefix_caching=True,enforce_eager=False,gpu_memory_utilization=.80,
        disable_custom_all_reduce=True,enable_batch_sharded_sampling=False,
        compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':list(range(1,len(inputs)+1))},
        attention_config={'backend':'FLASHINFER'},worker_extension_cls='scripts.vllm_joint_support_worker.JointSupportWorkerExtension',seed=args.seed)
    report['engine_configuration']=engine;save();started=time.perf_counter();llm=LLM(**engine)
    report['initialization_wall_seconds']=time.perf_counter()-started
    config={'output':str(out.resolve()),'document_token_range':[start,end],'prompt_tokens':inputs,
        'prompt_token_hashes':[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
        'budget':args.budget,'check_dense':False,'check_sparse_reference':args.qualify,'capture_logits':args.qualify,
        'capture_layer_outputs':args.qualify,'graph_shadow_check':args.qualify,
        'joint_support_exchange':next(s['configuration'] for s in specifications if s['configuration'] is not None)}
    report['configured_workers']=llm.collective_rpc('configure_sparse_attention',args=(config,));save()
    warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=False);started=time.perf_counter()
    llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],warm)
    report['selector_dense_warmup_wall_seconds']=time.perf_counter()-started
    started=time.perf_counter();report['frozen_variants']=llm.collective_rpc('freeze_joint_variants',args=(specifications,))
    report['joint_finalize_RPC_wall_seconds']=time.perf_counter()-started;save()
    report['graph_capture']=llm.collective_rpc('capture_sparse_graphs',args=(('e0',),));save()
    if not args.qualify:
        report['production_configuration']=llm.collective_rpc('enable_production_sparse')
        llm.collective_rpc('set_sparse_phase',args=('stochastic_warmup','dense'))
        started=time.perf_counter()
        generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(temperature=.6,top_p=.95,seed=args.seed,max_tokens=1))
        report['stochastic_sampler_warmup_wall_seconds']=time.perf_counter()-started
    for phase,variant in phases:
        assert llm.reset_prefix_cache()
        llm.collective_rpc('set_sparse_phase',args=(phase+'_anchor','dense'))
        started=time.perf_counter();llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
        anchor=time.perf_counter()-started
        teacher=[a['tokens'] for a in report['runs'][0]['answers']] if args.qualify and report['runs'] else None
        started=time.perf_counter()
        installation=llm.collective_rpc('set_joint_phase',args=(phase,None if variant=='dense' else variant,teacher))
        install_wall=time.perf_counter()-started
        started=time.perf_counter()
        generated=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(**report['sampling']))
        wall=time.perf_counter()-started
        run={'phase':phase,'variant':variant,'method':variant,'repeat':0,'teacher_forced':teacher is not None,
            'anchor_wall_seconds':anchor,'variant_install_RPC_wall_seconds':install_wall,'variant_installation':installation,
            'complete_generate_wall_seconds':wall,'anchor_install_generate_wall_seconds':anchor+install_wall+wall,
            'answers':[{'question_index':i,'tokens':list(x.outputs[0].token_ids),'text':x.outputs[0].text,
                'finish_reason':x.outputs[0].finish_reason,'stop_reason':x.outputs[0].stop_reason,'num_cached_tokens':x.num_cached_tokens} for i,x in enumerate(generated)]}
        run['total_output_tokens']=sum(len(a['tokens']) for a in run['answers'])
        run['tokens_per_generate_wall_second']=run['total_output_tokens']/wall
        started=time.perf_counter();run['workers']=llm.collective_rpc('finish_sparse_phase')
        run['audit_export_RPC_wall_seconds']=time.perf_counter()-started
        report['runs'].append(run);save();print(json.dumps({'phase':phase,'wall':wall,'tokens':run['total_output_tokens']}),flush=True)
    report['status']='completed';save()


if __name__=='__main__':main()
