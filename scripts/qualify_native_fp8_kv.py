"""One bounded official FP8-KV unit-scale qualification, not task quality."""
import argparse
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
    protocol_path=ROOT/args.protocol;protocol=json.loads(protocol_path.read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in protocol['artifact_sha256'].items():assert sha(ROOT/name)==digest,name
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','uid':os.geteuid(),'pid':os.getpid(),'python':sys.executable,
        'protocol':args.protocol,'protocol_sha256':sha(protocol_path),'protocol_contents':protocol,
        'runs':[],'scope':'Instrumented unit-scale native FP8 KV execution qualification only; not natural quality, application speedup or a joint-selector result.'}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        model=protocol['model_path']
        manifest=ROOT/protocol['manifest']
        group=next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id']==protocol['document_id'])
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        prompts,start,end=make_inputs(tokenizer,group)
        hashes=[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in prompts]
        assert hashes==protocol['prompt_token_hashes'] and list(map(len,prompts))==protocol['prompt_lengths']
        assert [start,end]==protocol['document_token_range'] and end%16!=0
        engine=dict(protocol['engine'],model=model,
            worker_extension_cls='scripts.vllm_native_fp8_kv_worker.NativeFP8KVWorkerExtension')
        report['engine_configuration']=engine;save();started=time.perf_counter()
        llm=LLM(**engine);report['initialization_wall_seconds']=time.perf_counter()-started;save()
        config={'output':str(out.resolve()),'prompt_token_hashes':hashes,
            'common_prefix_tokens':protocol['common_prefix_tokens'],'layers':protocol['reference_layers'],
            'reference_chunk':protocol['reference_chunk']}
        report['configured_workers']=llm.collective_rpc('configure_fp8_probe',args=(config,));save()
        warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,seed=protocol['seed'])
        llm.generate([{'prompt_token_ids':prompts[0]}],warm,use_tqdm=False)
        generate_ready_group(llm,[{'prompt_token_ids':x} for x in prompts],warm)
        report['instrumented_native_graph_capture']=llm.collective_rpc('capture_fp8_native_graphs');save()
        for phase,mode in [('full_A','native'),('full_B','native'),('eager_teacher','eager')]:
            assert llm.reset_prefix_cache()
            llm.collective_rpc('set_fp8_phase',args=(phase+'_anchor',mode))
            started=time.perf_counter()
            anchor=llm.generate([{'prompt_token_ids':prompts[0]}],warm,use_tqdm=False)
            anchor_seconds=time.perf_counter()-started
            assert anchor[0].num_cached_tokens==0,'Each A/A sequence must start with cold APC'
            teacher=[a['tokens'] for a in report['runs'][0]['answers']] if mode=='eager' else None
            llm.collective_rpc('set_fp8_phase',args=(phase,mode,teacher))
            params=[SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,seed=protocol['seed']) for n in protocol['tokens_by_question']]
            started=time.perf_counter()
            outputs=generate_ready_group(llm,[{'prompt_token_ids':x} for x in prompts],params)
            elapsed=time.perf_counter()-started
            run={'phase':phase,'mode':mode,'teacher_forced':teacher is not None,
                'cold_anchor_wall_seconds':anchor_seconds,'instrumented_generate_wall_seconds':elapsed,
                'anchor_cached_tokens':anchor[0].num_cached_tokens,
                'answers':[{'question':i,'tokens':list(o.outputs[0].token_ids),
                    'finish_reason':o.outputs[0].finish_reason,'num_cached_tokens':o.num_cached_tokens} for i,o in enumerate(outputs)]}
            run['workers']=llm.collective_rpc('finish_fp8_phase')
            report['runs'].append(run);save()
            assert [len(a['tokens']) for a in run['answers']]==protocol['tokens_by_question']
            assert all(a['num_cached_tokens']>0 for a in run['answers'])
            assert all(w['pure_batches']==[1,2,3] for w in run['workers'])
        a,b=report['runs'][:2]
        report['cold_AA']={'same_tokens':[x['tokens']==y['tokens'] for x,y in zip(a['answers'],b['answers'])],
            'same_APC_hits':[x['num_cached_tokens']==y['num_cached_tokens'] for x,y in zip(a['answers'],b['answers'])],
            'same_steps':[]}
        expected_keys={f'q{q}_s{s}' for q,n in enumerate(protocol['tokens_by_question']) for s in range(1,n+1)}
        report['logit_diagnostics']={}
        for phase in ('full_A','full_B','eager_teacher'):
            logits=json.loads((out/f'rank0/{phase}.json').read_text())['logits']
            assert set(logits)==expected_keys,(phase,'missing or extra logit records',set(logits)^expected_keys)
            assert all(r['all_finite'] for r in logits.values())
            item={'expected_keys':len(expected_keys),'observed_keys':len(logits),'all_finite':True}
            if phase!='full_A':
                item.update(max_abs_vs_full_A=max(r['max_abs_vs_full_A'] for r in logits.values()),
                    max_relative_l2_vs_full_A=max(r['relative_l2_vs_full_A'] for r in logits.values()),
                    all_top1_equal_full_A=all(r['top1_equal_full_A'] for r in logits.values()),
                    bitwise_equal_full_A=all(r['max_abs_vs_full_A']==0 for r in logits.values()),
                    scope='Descriptive FP8 same-stack logit drift; no BF16 quality inference or invented noninferiority threshold')
            report['logit_diagnostics'][phase]=item
        report['eager_teacher_same_tokens_A']=[x['tokens']==y['tokens'] for x,y in zip(a['answers'],report['runs'][2]['answers'])]
        assert all(report['eager_teacher_same_tokens_A'])
        report['eager_teacher_same_APC_A']=[x['num_cached_tokens']==y['num_cached_tokens'] for x,y in zip(a['answers'],report['runs'][2]['answers'])]
        assert all(report['eager_teacher_same_APC_A'])
        report['eager_teacher_same_steps_A']=[]
        for rank in (0,1):
            def steps(phase):
                data=json.loads((out/f'rank{rank}/{phase}.json').read_text())
                return [(r['batch'],r['questions'],r['positions']) for r in data['steps']]
            report['cold_AA']['same_steps'].append(steps('full_A')==steps('full_B'))
            report['eager_teacher_same_steps_A'].append(steps('full_A')==steps('eager_teacher'))
            checks=json.loads((out/f'rank{rank}/checks.json').read_text())['checks']
            assert len(checks)==2*3*len(protocol['reference_layers'])
            assert all(r['passed'] for r in checks)
        assert all(all(v) for v in report['cold_AA'].values()),report['cold_AA']
        assert all(report['eager_teacher_same_steps_A'])
        report['status']='qualified_instrumented_native_FP8KV_unit';save()
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()
