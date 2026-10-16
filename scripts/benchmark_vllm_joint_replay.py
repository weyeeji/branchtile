"""Frozen 32K real-cache replay, bounded independently of natural quality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--document-id',required=True);parser.add_argument('--replica',type=int,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    protocol_path=ROOT/args.protocol;protocol=json.loads(protocol_path.read_text())
    assert protocol['budget']==32768 and protocol['blocks']==6 and args.replica in (0,1)
    for name,expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from scripts.capture_model import make_inputs
    from scripts.vllm_batch_admission import generate_ready_group
    model='data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    manifest=ROOT/protocol['manifest'];assert hashlib.sha256(manifest.read_bytes()).hexdigest()==protocol['manifest_sha256']
    group=next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id']==args.document_id)
    assert group['document_sha256']==next(d['document_sha256'] for d in protocol['documents'] if d['document_id']==args.document_id)
    inputs,start,end=make_inputs(AutoTokenizer.from_pretrained(model,local_files_only=True),group)
    assert len(inputs)==3
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','uid':os.geteuid(),'python':sys.executable,'document_id':args.document_id,
        'document_sha256':group['document_sha256'],'device_replica':args.replica,'protocol':args.protocol,
        'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'source_sha256':protocol['source_sha256'],
        'budget':32768,'B':3,'prompt_lengths':list(map(len,inputs)),'runs':[]}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    engine=dict(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',max_model_len=98304,
        max_num_seqs=3,max_num_batched_tokens=4096,enable_prefix_caching=True,enforce_eager=False,gpu_memory_utilization=.80,
        disable_custom_all_reduce=True,enable_batch_sharded_sampling=False,
        compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,3]},
        attention_config={'backend':'FLASHINFER'},worker_extension_cls='scripts.vllm_joint_replay_worker.JointReplayWorkerExtension',seed=20261016)
    report['engine_configuration']=engine;save();t0=time.perf_counter();llm=LLM(**engine)
    report['initialization_wall_seconds']=time.perf_counter()-t0
    config={'output':str(out.resolve()),'document_token_range':[start,end],'prompt_tokens':inputs,
        'prompt_token_hashes':[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
        'budget':32768,'check_dense':False,'check_sparse_reference':False,'capture_logits':False,
        'capture_layer_outputs':False,'graph_shadow_check':False,
        'joint_support_exchange':protocol['variants'][1]['configuration']}
    report['configured_workers']=llm.collective_rpc('configure_sparse_attention',args=(config,));save()
    warm=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True);t0=time.perf_counter()
    llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],warm)
    report['selector_dense_warmup_wall_seconds']=time.perf_counter()-t0
    t0=time.perf_counter();report['frozen_variants']=llm.collective_rpc('freeze_joint_variants',args=(protocol['variants'],))
    report['joint_finalize_RPC_wall_seconds']=time.perf_counter()-t0;save()
    report['graph_capture']=llm.collective_rpc('capture_sparse_graphs',args=(('e0',),));save()
    report['production_configuration']=llm.collective_rpc('enable_production_sparse')
    assert llm.reset_prefix_cache()
    llm.collective_rpc('set_sparse_phase',args=('actual_snapshot_anchor','dense'))
    t0=time.perf_counter();llm.generate([{'prompt_token_ids':inputs[0]}],warm,use_tqdm=False)
    report['cold_anchor_wall_seconds']=time.perf_counter()-t0
    t0=time.perf_counter();report['initial_installation']=llm.collective_rpc('set_joint_phase',args=('actual_snapshot','independent'))
    report['initial_install_RPC_wall_seconds']=time.perf_counter()-t0
    report['profile_armed']=llm.collective_rpc('arm_replay_profile');save()
    t0=time.perf_counter()
    generated=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in inputs],SamplingParams(temperature=0,max_tokens=8,ignore_eos=True))
    report['instrumented_generate_wall_seconds']=time.perf_counter()-t0
    report['bounded_outputs']=[{'question_index':i,'token_ids':list(x.outputs[0].token_ids),
        'finish_reason':x.outputs[0].finish_reason,'num_cached_tokens':x.num_cached_tokens} for i,x in enumerate(generated)]
    report['snapshots']=llm.collective_rpc('freeze_replay_snapshot');save()
    # From this point no engine request/forward is allowed: physical private KV pages remain untouched.
    report['replay_workers']=llm.collective_rpc('run_joint_replay',args=(protocol,));report['status']='completed';save()


if __name__=='__main__':main()
