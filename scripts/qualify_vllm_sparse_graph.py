"""Real TP2 FULL graph capture/replay of frozen sparse E0 and native union.

All graph invocations are shadow-checked using their actual Q and cache. This
instrumented run establishes execution correctness, not serving throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=8)
    args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from capture_model import make_inputs
    from scripts.vllm_batch_admission import generate_ready_group
    model='data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    group=next(x for x in json.loads((ROOT/'data/manifests/docfinqa_exploratory_dev8.json').read_text())['groups'] if x['document_id']=='docfinqa-c0f11a67aae850fc')
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    inputs,start,end=make_inputs(tokenizer,group)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','uid':os.geteuid(),'python':sys.executable,'B':len(inputs),'budget':4096,
        'document_id':group['document_id'],'document_sha256':group['document_sha256'],'steps_cap':args.steps,
        'reference_phase':'e0_graph_A','runs':[],
        'scope':'Instrumented true FULL graph sparse attention qualification; graph capture and replay shadow checks, not application throughput.',
        'admission':'Before each phase reset APC, identical q0 anchor; pause scheduler while enqueuing full ready B3 group.'}
    def save():
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    engine=dict(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',
        max_model_len=98304,max_num_seqs=3,max_num_batched_tokens=4096,enable_prefix_caching=True,
        enforce_eager=False,gpu_memory_utilization=.80,disable_custom_all_reduce=True,enable_batch_sharded_sampling=False,
        compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,3]},
        worker_extension_cls='scripts.vllm_sparse_graph_worker.SparseGraphWorkerExtension',seed=20261016)
    report['engine_configuration']=engine;save()
    llm=LLM(**engine)
    config={'output':str(out.resolve()),'document_token_range':[start,end],
        'prompt_tokens':inputs,'prompt_token_hashes':[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
        'budget':4096,'check_dense':False,'check_sparse_reference':True,'capture_logits':True,
        'capture_layer_outputs':True,'graph_shadow_check':True}
    report['configured_workers']=llm.collective_rpc('configure_sparse_attention',args=(config,));save()
    warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=False)
    llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],warm)
    report['graph_capture']=llm.collective_rpc('capture_sparse_graphs',args=(('e0','union'),));save()
    for phase,method in [('e0_graph_A','e0'),('e0_graph_B','e0'),('union_graph','union')]:
        assert llm.reset_prefix_cache()
        llm.collective_rpc('set_sparse_phase',args=(phase+'_anchor_prepare','dense'))
        llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
        teacher=[a['tokens'] for a in report['runs'][0]['answers']] if report['runs'] else None
        llm.collective_rpc('set_sparse_phase',args=(phase,method,teacher))
        started=time.perf_counter()
        generated=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(temperature=0,max_tokens=args.steps,ignore_eos=False))
        run={'phase':phase,'method':method,'instrumented_wall_seconds':time.perf_counter()-started,
            'teacher_forced_to_reference':teacher is not None,
            'answers':[{'tokens':list(x.outputs[0].token_ids),'text':x.outputs[0].text,
                        'finish_reason':x.outputs[0].finish_reason,'num_cached_tokens':x.num_cached_tokens} for x in generated]}
        run['workers']=llm.collective_rpc('finish_sparse_phase')
        report['runs'].append(run);save()
    base=report['runs'][0]
    report['tokens_equal_reference']={r['phase']:[a['tokens']==b['tokens'] for a,b in zip(base['answers'],r['answers'])] for r in report['runs'][1:]}
    report['status']='completed';save()
    print(json.dumps({'status':report['status'],'output':str(out)}),flush=True)


if __name__=='__main__':
    main()
