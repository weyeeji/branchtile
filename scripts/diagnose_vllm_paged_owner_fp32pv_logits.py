"""Independent model diagnostic: E0 natural, E0 forced AA, owner forced.

One selected exploratory document, fixed masks and a common forced trajectory.
All shadow, snapshot and logit-copy costs are enabled. Timings are diagnostics only.
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
    p.add_argument('--steps',type=int,default=16);p.add_argument('--compile-mode',type=int,default=3)
    p.add_argument('--methods',default='e0,owner');p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--budget',default='4096',help='Frozen document token budget, or full')
    p.add_argument('--manifest',default='data/manifests/docfinqa_exploratory_dev8.json')
    p.add_argument('--document-id',default='docfinqa-c0f11a67aae850fc')
    p.add_argument('--max-model-len',type=int,default=98304)
    p.add_argument('--attention-backend',choices=['FLASH_ATTN','FLASHINFER'],default=None)
    p.add_argument('--qualification',action='store_true',help='Full layer shadow and logits; performance invalid')
    p.add_argument('--owner-chunk-tokens',type=int,default=128)
    args=p.parse_args();args.qualification=True;assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    requested_methods=args.methods.split(',')
    assert requested_methods==['e0','owner'] and 2<=args.steps<=64, 'Diagnostic has fixed three arms and a short qualification cap'
    assert set(requested_methods)<=set(('dense','e0','union','owner')) and len(set(requested_methods))==len(requested_methods)
    capture_methods=tuple(m for m in ('e0','union','owner') if m in requested_methods)
    if args.qualification:
        assert requested_methods[0]=='e0' and 'owner' in requested_methods and args.repeats==1, 'Qualification requires e0 first, owner and one repeat'
    assert capture_methods, 'Use benchmark_vllm_native_backend.py for native-only execution'
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from capture_model import make_inputs
    from scripts.vllm_batch_admission import generate_ready_group
    model='data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    manifest=ROOT/args.manifest
    group=next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id']==args.document_id)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True);inputs,start,end=make_inputs(tokenizer,group)
    assert 1<=len(inputs)<=3, 'This bridge only supports natural groups B1/B2/B3'
    assert max(map(len,inputs))+args.steps<=args.max_model_len, 'Context capacity must cover complete prompt and full generation cap'
    budget=end-start if args.budget=='full' else int(args.budget)
    assert 36<=budget<=end-start and 1<=args.steps<=8192
    out=Path(args.output);assert not (out/'summary.json').exists(), 'Refuse to overwrite existing run';out.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','uid':os.geteuid(),'python':sys.executable,'document_id':group['document_id'],
        'document_sha256':group['document_sha256'],'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'B':len(inputs),'prompt_lengths':list(map(len,inputs)),'budget':budget,'budget_request':args.budget,'steps_cap':args.steps,
        'sampling':{'temperature':.6,'top_p':.95,'seed':20261016,'max_tokens':args.steps,'ignore_eos':False},
        'methods':args.methods.split(','),'repeats':args.repeats,'runs':[],
        'diagnostic_arms':['e0_natural','e0_forced_AA','owner_forced'],'performance_evidence':False,'pv_arithmetic':'FP32 probabilities and V, IEEE dot; original BF16 QK; diagnostic only',
        'owner_integration_scope':'New paged ABI, frozen history owner groups, dynamic exact private suffix, cached subset metadata. Mixed batches use charged exact E0 fallback. Not the old contiguous operator ABI.',
        'limitations':['One exploratory document is not a held-out task-quality experiment.','First E0 arm samples naturally; E0 AA and owner arms follow exactly those forced tokens. No performance or natural task-quality claim.','Minimal CPU step records remain enabled and charged. No logit copies, shadows, layer snapshots or teacher forcing.',
            'All requested candidate graph sets remain resident for every method; native standalone deployment can reclaim their memory.',
            'Fixed histories are prepared once from actual dense suffix Q and held constant across all methods; one-time costs are separately reported, not free.']}
    report['source_sha256']={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'scripts/diagnose_vllm_paged_owner_fp32pv_logits.py','scripts/vllm_paged_owner_fp32pv_worker.py',
        'src/vllm_paged_owner_fp32pv_plan.py','src/branchtile/triton_kernel_paged_owner_fp32pv.py','scripts/vllm_sparse_production_worker.py',
        'scripts/vllm_shared_plan_graph_worker.py','scripts/vllm_sparse_graph_worker.py',
        'scripts/vllm_sparse_worker.py','src/vllm_sparse_selector.py','src/selector_baseline.py',
        'src/vllm_sparse_executor.py','src/vllm_shared_decode_plan.py','src/vllm_shared_union_plan.py')}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    engine=dict(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',max_model_len=args.max_model_len,
        max_num_seqs=len(inputs),max_num_batched_tokens=4096,enable_prefix_caching=True,enforce_eager=False,gpu_memory_utilization=.80,
        disable_custom_all_reduce=True,enable_batch_sharded_sampling=False,
        compilation_config={'mode':args.compile_mode,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':list(range(1,len(inputs)+1))},
        worker_extension_cls='scripts.vllm_paged_owner_fp32pv_worker.PagedOwnerFP32PVWorker',seed=20261016)
    if args.attention_backend is not None:engine['attention_config']={'backend':args.attention_backend}
    report['engine_configuration']=engine;save();started=time.perf_counter();llm=LLM(**engine)
    report['initialization_wall_seconds']=time.perf_counter()-started;save()
    config={'output':str(out.resolve()),'document_token_range':[start,end],'prompt_tokens':inputs,
        'prompt_token_hashes':[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
        'budget':budget,'check_dense':False,'check_sparse_reference':False,'capture_logits':False,
        'capture_layer_outputs':False,'graph_shadow_check':False,'max_generated':args.steps,'owner_chunk_tokens':args.owner_chunk_tokens}
    report['configured_workers']=llm.collective_rpc('configure_sparse_attention',args=(config,));save()
    warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=False)
    started=time.perf_counter()
    llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],warm)
    report['selector_dense_warmup_wall_seconds']=time.perf_counter()-started
    report['requested_sparse_graph_methods']=list(capture_methods)
    report['graph_capture']=llm.collective_rpc('capture_sparse_graphs',args=(capture_methods,));save()
    report['production_configuration']=llm.collective_rpc('enable_production_sparse');save()
    report['qualification_mode']=args.qualification
    if args.qualification:
        report['limitations']=[x for x in report['limitations'] if 'No logit copies' not in x]
        report['limitations'].append('Qualification enables full attention-output shadows, CPU logits copies and E0 teacher-forced candidate tokens. These runs are not performance or natural-EOS evidence.')
    if args.qualification:
        report['qualification_configuration']=llm.collective_rpc('enable_owner_qualification');save()
    # Warm the same stochastic sampler before any measured arm. The historical
    # 4K run predates this step and keeps its first-dense JIT limitation.
    llm.collective_rpc('set_sparse_phase',args=('stochastic_sampler_warmup','dense'))
    started=time.perf_counter()
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(temperature=.6,top_p=.95,seed=20261016,max_tokens=1))
    report['stochastic_sampler_warmup_wall_seconds']=time.perf_counter()-started;save()
    qualification_teacher=None
    for repeat in range(args.repeats):
        arms=[('e0_natural_0','e0',False),('e0_aa_0','e0',True),('owner_forced_0','owner',True)]
        for phase,method,use_teacher in arms:
            assert llm.reset_prefix_cache()
            llm.collective_rpc('set_sparse_phase',args=(phase+'_anchor','dense'))
            started=time.perf_counter();anchor=llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
            anchor_wall=time.perf_counter()-started
            teacher=qualification_teacher if use_teacher else None
            llm.collective_rpc('set_sparse_phase',args=(phase,method,teacher))
            started=time.perf_counter()
            generated=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(**report['sampling']))
            wall=time.perf_counter()-started
            run={'phase':phase,'method':method,'repeat':repeat,'anchor_wall_seconds':anchor_wall,'complete_generate_wall_seconds':wall,'anchor_plus_generate_wall_seconds':anchor_wall+wall,
                'teacher_forced':teacher is not None,'answers':[{'tokens':list(x.outputs[0].token_ids),'text':x.outputs[0].text,
                'finish_reason':x.outputs[0].finish_reason,'stop_reason':x.outputs[0].stop_reason,
                'num_cached_tokens':x.num_cached_tokens,'question_index':i} for i,x in enumerate(generated)]}
            if phase=='e0_natural_0':qualification_teacher=[a['tokens'] for a in run['answers']]
            run['total_output_tokens']=sum(len(a['tokens']) for a in run['answers'])
            run['tokens_per_generate_wall_second']=run['total_output_tokens']/wall
            run['workers']=llm.collective_rpc('finish_sparse_phase');report['runs'].append(run);save()
            print(json.dumps({'phase':phase,'wall_seconds':wall,'output_tokens':run['total_output_tokens'],
                'finish_reasons':[a['finish_reason'] for a in run['answers']]}),flush=True)
    import torch
    graphs={};all_shadow=[];coverage={};layer_trends=[]
    for path in sorted(out.glob('rank*/*_graph.json')):
        record=json.loads(path.read_text());phase=record['phase'];rank=record['rank']
        graphs[phase,rank]=record
        for step,row in enumerate(record.get('replays',[])):
            shadow=row.get('shadow_checks',[]);all_shadow.extend(shadow)
            coverage.setdefault((phase,rank),set()).update(x['layer'] for x in shadow)
            for x in shadow:
                layer_trends.append({'phase':phase,'rank':rank,'replay_index':step,'layer':x['layer'],
                    'max_abs':x['max_abs'],'output_pass':x['frozen_output_tolerance_passed'],
                    'query_l2':x['query_l2'],'output_l2':x['output_l2'],'positions':row['positions']})
    phases=['e0_natural_0','e0_aa_0','owner_forced_0']
    shadow_pass=bool(all_shadow) and all(x['frozen_output_tolerance_passed'] for x in all_shadow)
    coverage_pass=all(coverage.get((phase,rank),set())==set(range(48)) for phase in phases for rank in (0,1))
    report['qualification_layer_output_gate']={'checks':len(all_shadow),'all_passed':shadow_pass,'full_coverage':coverage_pass}
    report['layer_shadow_trends']=layer_trends
    baseline=torch.load(out/'rank0/e0_natural_0.pt',map_location='cpu',weights_only=True)
    comparisons={}
    for phase in ('e0_aa_0','owner_forced_0'):
        candidate=torch.load(out/f'rank0/{phase}.pt',map_location='cpu',weights_only=True)
        rows=[]
        for key in sorted(set(baseline)&set(candidate)):
            a,b=baseline[key],candidate[key]
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            cosine=float(torch.nn.functional.cosine_similarity(a.double(),b.double(),dim=0))
            rows.append({'key':key,'finite':finite,'cosine':cosine,'max_abs':float((a-b).abs().max()),
                'top1_equal':int(a.argmax())==int(b.argmax()),'baseline_top1':int(a.argmax()),'candidate_top1':int(b.argmax())})
        passed=set(baseline)==set(candidate) and bool(rows) and any(not x['key'].endswith('_s1') for x in rows) and all(x['finite'] and x['cosine']>=.999 for x in rows)
        comparisons[phase]={'checks':rows,'cosine_threshold':.999,'passed':passed,'key_sets_equal':set(baseline)==set(candidate)}
    report['qualification_same_teacher_path_logits']=comparisons
    # Q/output norms across layers provide localization evidence; they do not
    # identify router decisions or establish a floating-point error cause.
    differences=[]
    for phase in ('e0_aa_0','owner_forced_0'):
        for rank in (0,1):
            base=graphs['e0_natural_0',rank]['replays'];other=graphs[phase,rank]['replays']
            for step,(a,b) in enumerate(zip(base,other)):
                for x,y in zip(a.get('shadow_checks',[]),b.get('shadow_checks',[])):
                    assert x['layer']==y['layer']
                    differences.append({'phase':phase,'rank':rank,'replay_index':step,'layer':x['layer'],
                        'query_norm_abs_difference':abs(x['query_l2']-y['query_l2']),
                        'output_norm_abs_difference':abs(x['output_l2']-y['output_l2'])})
    report['layer_norm_differences_from_e0']=differences
    passed=shadow_pass and coverage_pass and all(x['passed'] for x in comparisons.values())
    report['qualification_joint_gate_passed']=passed
    if not passed:
        report['status']='qualification_failed';save()
        raise AssertionError('Model diagnostic gate failed; inspect separate layer-output, E0-AA and owner-logit gates')
    report['status']='completed';save();print(json.dumps({'status':'completed','output':str(out)}),flush=True)


if __name__=='__main__':main()
