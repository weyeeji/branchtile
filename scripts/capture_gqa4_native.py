"""Capture actual native Qwen3-4B-Instruct-2507 TP2 BF16 state for GQA4 qualification."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--document-id',required=True);p.add_argument('--output',required=True)
    p.add_argument('--manifest',default='data/manifests/docfinqa_training_large_group_pilot_v1.json')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    out=Path(a.output);assert not (out/'summary.json').exists();out.mkdir(parents=True,exist_ok=True)
    model='data/models/Qwen3-4B-Instruct-2507'
    report={'status':'initializing','args':vars(a),'model':model,'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
        ['scripts/capture_gqa4_native.py','scripts/vllm_gqa4_native_capture_worker.py','scripts/capture_model.py','scripts/vllm_batch_admission.py']}}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        from transformers import AutoTokenizer
        from scripts.capture_model import make_inputs
        manifest=Path(a.manifest);group=next(x for x in json.loads(manifest.read_text())['groups'] if x['document_id']==a.document_id)
        prompts,start,end=make_inputs(AutoTokenizer.from_pretrained(model,local_files_only=True),group)
        assert len(prompts)==8
        hashes=[hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        report.update(prompt_lengths=list(map(len,prompts)),prompt_token_hashes=hashes,document_token_range=[start,end],
                      manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),document_sha256=group['document_sha256'],
                      model_config_sha256=hashlib.sha256((Path(model)/'config.json').read_bytes()).hexdigest())
        engine=dict(model=model,tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',block_size=16,
                    max_model_len=163840,max_num_seqs=8,max_num_batched_tokens=4096,enable_prefix_caching=True,
                    enforce_eager=True,gpu_memory_utilization=.9,disable_custom_all_reduce=True,
                    enable_batch_sharded_sampling=False,attention_config={'backend':'FLASHINFER'},seed=20261017,
                    worker_extension_cls='scripts.vllm_gqa4_native_capture_worker.GQA4NativeCaptureWorker')
        assert max(map(len,prompts))+8<engine['max_model_len']
        report['engine']=engine;save()
        if a.prepare_only:
            report['status']='prepared_cpu_only';save();return
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
        from vllm import LLM,SamplingParams
        from scripts.vllm_batch_admission import generate_ready_group
        started=time.perf_counter();llm=LLM(**engine);report['initialization_seconds']=time.perf_counter()-started
        report['configured']=llm.collective_rpc('configure_gqa4_capture',args=({'output':str(out.resolve()),'prompt_token_hashes':hashes},))
        assert llm.reset_prefix_cache()
        llm.generate([{'prompt_token_ids':prompts[0]}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
        report['arm']=llm.collective_rpc('arm_gqa4_capture')
        outputs=generate_ready_group(llm,[{'prompt_token_ids':ids} for ids in prompts],SamplingParams(temperature=0,max_tokens=8,min_tokens=8,ignore_eos=True))
        report['outputs']=[{'tokens':list(x.outputs[0].token_ids),'finish_reason':x.outputs[0].finish_reason} for x in outputs]
        report['workers']=llm.collective_rpc('gqa4_capture_result');report['status']='completed'
    except Exception:report['status']='failed';report['error']=traceback.format_exc();save();raise
    save();print(json.dumps({'status':report['status'],'output':str(out)}))
if __name__=='__main__':main()
