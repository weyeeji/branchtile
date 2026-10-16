"""Capture real consecutive AR states for causal multi-query attention replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default='configs/native_opportunity_b8_training_v1.json')
    p.add_argument('--document-id', required=True)
    p.add_argument('--kv-dtype', choices=['auto', 'fp8_e4m3'], default='fp8_e4m3')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    assert 2 <= args.steps <= 8
    out = ROOT/args.output
    assert not out.exists(), 'Preserve previous capture and failures'
    out.mkdir(parents=True)
    protocol = json.loads((ROOT/args.protocol).read_text())
    report = dict(status='initializing', args=vars(args), protocol=protocol, uid=os.geteuid(),
                  python=sys.executable, scope=__doc__, source_sha256={})
    for f in ('scripts/capture_ragged_verify_state.py', 'scripts/vllm_ragged_state_capture_worker.py',
              'scripts/vllm_native_precision_worker.py', 'scripts/vllm_native_backend_worker.py'):
        report['source_sha256'][f] = hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    def save():
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        group = next(g for g in json.loads((ROOT/protocol['manifest']).read_text())['groups'] if g['document_id']==args.document_id)
        item = next(d for d in protocol['documents'] if d['document_id']==args.document_id)
        prompts, start, end = make_inputs(AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True), group)
        hashes = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert hashes == item['prompt_token_hashes'] and len(prompts) == 8
        engine = dict(protocol['engine'], model=protocol['model_path'], kv_cache_dtype=args.kv_dtype,
                      enforce_eager=True, worker_extension_cls='scripts.vllm_ragged_state_capture_worker.RaggedStateCaptureWorker')
        report['engine'] = engine
        started = time.perf_counter()
        llm = LLM(**engine)
        report['initialization_seconds'] = time.perf_counter()-started
        config = dict(output=str(out.resolve()), backend='native_'+args.kv_dtype,
                      prompt_token_hashes=hashes, batch=8, steps=args.steps, layers=[0,15,31,47])
        report['configured'] = llm.collective_rpc('configure_ragged_state_capture', args=(config,))
        save()
        assert llm.reset_prefix_cache()
        llm.generate([{'prompt_token_ids':prompts[0]}], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        llm.collective_rpc('set_native_phase', args=('actual_consecutive_ar_capture',))
        report['armed'] = llm.collective_rpc('arm_ragged_state_capture')
        outputs = generate_ready_group(llm, [{'prompt_token_ids':ids} for ids in prompts],
            SamplingParams(temperature=0, max_tokens=args.steps+3, ignore_eos=False))
        report['outputs'] = [dict(tokens=list(o.outputs[0].token_ids), finish_reason=o.outputs[0].finish_reason) for o in outputs]
        report['workers'] = llm.collective_rpc('ragged_state_capture_result')
        report['native_receipts'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype, True))
        report['status'] = 'completed'
        save()
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
