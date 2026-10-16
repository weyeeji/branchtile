"""Bounded real-state model cached-O opportunity diagnostic, not natural quality."""
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
    p.add_argument('--kv-dtype', choices=['auto'], default='auto')
    p.add_argument('--blocks', type=int, default=12)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    protocol_path = ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text())
    out = ROOT / args.output
    assert not (out / 'summary.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'uid': os.geteuid(), 'python': sys.executable,
              'args': vars(args), 'protocol': protocol,
              'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in (
                  'scripts/capture_dense_shared_b8_bf16.py', 'scripts/vllm_cached_o_b8_diagnostic_worker.py', 'scripts/vllm_dense_shared_bf16_capture_worker.py', 'scripts/capture_model.py', 'scripts/vllm_batch_admission.py',
                  'scripts/vllm_native_precision_worker.py', 'scripts/vllm_native_backend_worker.py')},
              'scope': 'Actual native BF16 KV/Q/O capture after bitwise-qualified fixed-state diagnostic. No FP8 artifact cast; capture I/O excluded from diagnostic timing, not online copy performance.'}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        group = next(g for g in json.loads((ROOT / protocol['manifest']).read_text())['groups'] if g['document_id'] == args.document_id)
        item = next(d for d in protocol['documents'] if d['document_id'] == args.document_id)
        prompts, start, end = make_inputs(AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True), group)
        hashes = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert hashes == item['prompt_token_hashes'] and len(prompts) == 8
        report['prompt_token_hashes'] = hashes
        report['document_sha256'] = group['document_sha256']
        report['source_manifest_sha256'] = hashlib.sha256((ROOT/protocol['manifest']).read_bytes()).hexdigest()
        engine = dict(protocol['engine'], model=protocol['model_path'], kv_cache_dtype=args.kv_dtype,
                      worker_extension_cls='scripts.vllm_dense_shared_bf16_capture_worker.DenseSharedBF16CaptureWorker')
        report['engine'] = engine
        started = time.perf_counter()
        llm = LLM(**engine)
        report['initialization_wall_seconds'] = time.perf_counter() - started
        config = {'output': str(out.resolve()), 'backend': 'native_' + args.kv_dtype,
                  'prompt_token_hashes': hashes, 'blocks': args.blocks}
        report['configured'] = llm.collective_rpc('configure_cached_o_diagnostic', args=(config,))
        report['capture'] = llm.collective_rpc('capture_cached_o_graphs')
        save()
        warm = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        assert llm.reset_prefix_cache()
        llm.collective_rpc('set_native_phase', args=('fresh_real_anchor',))
        llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
        llm.collective_rpc('set_native_phase', args=('real_fixed_diagnostic',))
        report['arm'] = llm.collective_rpc('arm_cached_o_diagnostic')
        outputs = generate_ready_group(llm, [{'prompt_token_ids': ids} for ids in prompts],
                                       SamplingParams(temperature=0, max_tokens=8, min_tokens=8, ignore_eos=True))
        report['bounded_outputs'] = [{'tokens': list(o.outputs[0].token_ids), 'finish_reason': o.outputs[0].finish_reason} for o in outputs]
        report['workers'] = llm.collective_rpc('cached_o_diagnostic_result')
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
