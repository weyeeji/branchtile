"""Bounded native-trajectory late-state localization; never a performance run."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default='results/native_raw_pair_heldout_production_v1/eval2_bf16/summary.json')
    parser.add_argument('--start-graph',type=int,default=1760);parser.add_argument('--stop-graph',type=int,default=1960)
    parser.add_argument('--max-tokens',type=int,default=1970);parser.add_argument('--batches',default='9')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
    assert 0<=args.start_graph<=args.stop_graph<args.max_tokens-1
    out=Path(args.output);assert not (out/'summary.json').exists();out.mkdir(parents=True,exist_ok=True)
    reference=json.loads(Path(args.reference).read_text());protocol=reference['protocol']
    ref_run=next(r for r in reference['runs'] if r['method']=='native' and r['natural_eos_enabled'])
    r={'status':'initializing','args':vars(args),'reference_sha256':hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
       'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['scripts/diagnose_raw_pair_late_same_state_grouped_hint.py','scripts/vllm_raw_pair_same_state_grouped_hint_worker.py','scripts/vllm_native_raw_pair_grouped_hint_worker.py','src/xqa_raw_grouped_hint_build.py']}}
    def save():(out/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        group=next(g for g in json.loads(Path(protocol['manifest']).read_text())['groups'] if g['document_id']==reference['args']['document_id'])
        prompts,_,_=make_inputs(AutoTokenizer.from_pretrained(protocol['model_path'],local_files_only=True),group)
        hashes=[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert hashes==reference['prompt_token_hashes']
        engine=copy.deepcopy(reference['engine']);engine['worker_extension_cls']='scripts.vllm_raw_pair_same_state_grouped_hint_worker.RawPairSameStateGroupedHintWorker'
        llm=LLM(**engine)
        cfg={'output':str(out.resolve()),'backend':'native_late_same_state_diagnosis','prompt_token_hashes':hashes,
             'batch':len(prompts),'enabled_batches':[int(x) for x in args.batches.split(',')],'setup_mode':'strict',
             'common_prefix_tokens':reference['common_prefix_tokens'],'diagnostic_start_graph':args.start_graph,
             'diagnostic_stop_graph':args.stop_graph}
        r['configuration']=llm.collective_rpc('configure_dense_cascade',args=(cfg,))
        r['capture']=llm.collective_rpc('capture_dense_cascade_graphs');save()
        llm.collective_rpc('set_dense_cascade_phase',args=('native_anchor','native',False));assert llm.reset_prefix_cache()
        llm.generate([{'prompt_token_ids':prompts[0]}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=False,seed=protocol['seed']),use_tqdm=False)
        llm.collective_rpc('set_dense_cascade_phase',args=('native_diagnostic','native',True))
        llm.collective_rpc('arm_raw_pair_qualification',args=(100000,100000))
        sampling=dict(ref_run['sampling']);sampling.update(max_tokens=args.max_tokens)
        assert not sampling.get('ignore_eos',False) and sampling.get('min_tokens',0)==0
        outputs=generate_ready_group(llm,[{'prompt_token_ids':p} for p in prompts],SamplingParams(**sampling))
        r['answers']=[{'question':i,'tokens':list(o.outputs[0].token_ids),'finish_reason':o.outputs[0].finish_reason} for i,o in enumerate(outputs)]
        r['native_reference_prefix_checks']=[]
        for answer,prior in zip(r['answers'],ref_run['answers']):
            a,b=answer['tokens'],prior['tokens'];first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
            r['native_reference_prefix_checks'].append({'question':answer['question'],'equal_observed_prefix':a==b[:len(a)],'first_difference':first,'tokens':len(a)})
        r['phase_workers']=llm.collective_rpc('finish_dense_cascade_phase')
        r['diagnostics']=llm.collective_rpc('finish_same_state_diagnostic')
        assert all(x['diagnosed_states']>0 for x in r['diagnostics']),'No actual window state diagnosed'
        r['status']='diagnosis_complete'
        r['regression_scope']='Eval2 known arithmetic regression, not unseen heldout validation.'
        r['native_reference_prefix_all_equal']=all(x['equal_observed_prefix'] for x in r['native_reference_prefix_checks'])
        r['grouped_hint_regression_passed']=r['native_reference_prefix_all_equal'] and all(x['full_model_failures']==0 and x['same_Q_candidate_failures']==0 and x['native_AA_failures']==0 and not x['snapshots'] for x in r['diagnostics'])
        assert r['grouped_hint_regression_passed'], 'Preserved late-state regression failed'
        r['status']='regression_qualified'
    except Exception:r['status']='failed';r['error']=traceback.format_exc();save();raise
    save();print(json.dumps({'status':r['status'],'output':str(out)}))
if __name__=='__main__':main()
