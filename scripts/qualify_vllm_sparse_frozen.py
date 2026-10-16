"""Real engine frozen QC sparse trajectories and same-mask native comparisons.

Teacher-forced short numerical qualification, not natural answer quality or
performance. Pure decode bypasses dense attention; extra sparse references and
FP64 probes are explicitly instrumented.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--budget', type=int, default=4096)
    p.add_argument('--atomic-admission', action='store_true')
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from capture_model import make_inputs
    from scripts.vllm_batch_admission import generate_ready_group
    model = 'data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    manifest = ROOT/'data/manifests/docfinqa_exploratory_dev8.json'
    group = next(x for x in json.loads(manifest.read_text())['groups'] if x['document_id']=='docfinqa-c0f11a67aae850fc')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    inputs, start, end = make_inputs(tokenizer, group)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'status':'initializing','uid':os.geteuid(),'python':sys.executable,'runs':[],
        'scope':'Short teacher-forced same-mask numerical qualification only; natural task quality and graph application performance remain pending.',
        'document_id':group['document_id'],'document_sha256':group['document_sha256'],'B':len(inputs),
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'steps_cap':args.steps,'budget':args.budget,'reference_phase':'e0_simt_A',
        'prefix_policy':'Freeze QC masks once; before each comparison reset APC and perform identical dense q0 singleton anchor warmup. Logical masks are unchanged.',
        'request_admission':'pause scheduler, enqueue entire ready group, resume' if args.atomic_admission else 'ordinary generate, asynchronous arrivals may differ',
        'prompt_lengths':[len(ids) for ids in inputs],'document_token_range':[start,end]}
    def save():
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    llm = LLM(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',
        max_model_len=98304,max_num_seqs=4,max_num_batched_tokens=4096,
        enable_prefix_caching=True,enforce_eager=True,gpu_memory_utilization=.85,
        disable_custom_all_reduce=True,enable_batch_sharded_sampling=False,
        worker_extension_cls='scripts.vllm_sparse_worker.SparseAttentionWorkerExtension',seed=20261016)
    config = {'output':str(output.resolve()),'document_token_range':[start,end],
        'prompt_tokens':inputs,'prompt_token_hashes':[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
        'budget':args.budget,'check_dense':False,'check_sparse_reference':True,
        'capture_logits':True,'capture_layer_outputs':True,'fp64_probe_layers':[0,15,31,47]}
    report['configured_workers'] = llm.collective_rpc('configure_sparse_attention',args=(config,))
    warm = SamplingParams(temperature=0,max_tokens=1,ignore_eos=False)
    llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    llm.generate([{'prompt_token_ids':ids} for ids in inputs],warm,use_tqdm=False)
    for phase,method in [('e0_simt_A','e0_simt'),('e0_simt_B','e0_simt'),('e0_TC','e0'),('union','union')]:
        assert llm.reset_prefix_cache()
        llm.collective_rpc('set_sparse_phase',args=(phase+'_anchor_prepare','dense'))
        llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
        teacher = [a['tokens'] for a in report['runs'][0]['answers']] if report['runs'] else None
        llm.collective_rpc('set_sparse_phase',args=(phase,method,teacher))
        started = time.perf_counter()
        prompts = [{'prompt_token_ids':ids} for ids in inputs]
        params = SamplingParams(temperature=0,max_tokens=args.steps,ignore_eos=False)
        generated = generate_ready_group(llm,prompts,params) if args.atomic_admission else llm.generate(prompts,params,use_tqdm=False)
        record = {'phase':phase,'method':method,'instrumented_wall_seconds':time.perf_counter()-started,
            'teacher_forced_to_dense_A':False,'teacher_forced_to_reference':teacher is not None,
            'answers':[{'tokens':list(x.outputs[0].token_ids),'text':x.outputs[0].text,'finish_reason':x.outputs[0].finish_reason,'num_cached_tokens':x.num_cached_tokens} for x in generated]}
        record['workers'] = llm.collective_rpc('finish_sparse_phase')
        report['runs'].append(record)
        save()
    base = report['runs'][0]
    report['tokens_equal_reference'] = {r['phase']:[a['tokens']==b['tokens'] for a,b in zip(base['answers'],r['answers'])] for r in report['runs'][1:]}
    report['status']='completed'
    save()
    print(json.dumps({'output':str(output),'status':report['status']}),flush=True)


if __name__ == '__main__':
    main()
