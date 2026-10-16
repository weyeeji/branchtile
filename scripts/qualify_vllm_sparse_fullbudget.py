"""Official FP8 TP2 full-budget E0/union bridge qualification, no speed claims."""
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
    p.add_argument('--methods', default='e0,union')
    p.add_argument('--manifest', default='data/manifests/docfinqa_exploratory_dev8.json')
    p.add_argument('--document-id', default='docfinqa-c0f11a67aae850fc')
    p.add_argument('--max-model-len', type=int, default=98304)
    p.add_argument('--teacher-force-native-calibration', action='store_true')
    p.add_argument('--fp64-probes', action='store_true')
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from capture_model import make_inputs
    model = 'data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    manifest_path = ROOT/args.manifest
    manifest = json.loads(manifest_path.read_text())
    group = next(x for x in manifest['groups'] if x['document_id'] == args.document_id)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    inputs, start, end = make_inputs(tokenizer, group)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'uid': os.geteuid(), 'python': sys.executable, 'runs': [],
              'scope': 'Full-budget numerical qualification; instrumented and greedy capped to a short test, not quality or speedup.',
              'document_id': group['document_id'], 'document_sha256': group['document_sha256'], 'B': len(inputs),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_manifest': str(manifest_path), 'input_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'teacher_forced_native_calibration': args.teacher_force_native_calibration,
              'prompt_lengths': [len(ids) for ids in inputs], 'document_token_range': [start,end],
              'steps_cap': args.steps, 'budget': 'full', 'sampling': {'temperature': 0, 'ignore_eos': False}}
    def save():
        (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    llm = LLM(model=model, tensor_parallel_size=2, dtype='bfloat16', kv_cache_dtype='auto',
              max_model_len=args.max_model_len, max_num_seqs=4, max_num_batched_tokens=4096,
              enable_prefix_caching=True, enforce_eager=True, gpu_memory_utilization=.85,
              disable_custom_all_reduce=True, enable_batch_sharded_sampling=False,
              worker_extension_cls='scripts.vllm_sparse_worker.SparseAttentionWorkerExtension', seed=20261016)
    config = {'output': str(output.resolve()), 'document_token_range': [start, end],
              'prompt_token_hashes': [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
              'budget': 'full', 'check_dense': True, 'capture_logits': True, 'capture_layer_outputs': True,
              'fp64_probe_layers': [0,15,31,47] if args.fp64_probes else []}
    report['configured_workers'] = llm.collective_rpc('configure_sparse_attention', args=(config,))
    warm = SamplingParams(temperature=0, max_tokens=1, ignore_eos=False)
    llm.generate([{'prompt_token_ids': inputs[0]}], warm, use_tqdm=False)
    # Prime every request's complete prompt before matched APC-warm A/A.
    llm.generate([{'prompt_token_ids': ids} for ids in inputs], warm, use_tqdm=False)
    methods = args.methods.split(',')
    if args.teacher_force_native_calibration:
        assert all(m in ('e0_simt','e0') for m in methods), 'Independent calibration excludes union candidates'
    for phase, method in [('dense_A','dense'), ('dense_B','dense')] + [(m+'_full', m) for m in methods]:
        teacher = [answer['tokens'] for answer in report['runs'][0]['answers']] if args.teacher_force_native_calibration and report['runs'] else None
        llm.collective_rpc('set_sparse_phase', args=(phase, method, teacher))
        started = time.perf_counter()
        generated = llm.generate([{'prompt_token_ids': ids} for ids in inputs], SamplingParams(temperature=0, max_tokens=args.steps, ignore_eos=False), use_tqdm=False)
        record = {'phase': phase, 'method': method, 'instrumented_wall_seconds': time.perf_counter()-started,
                  'teacher_forced_to_dense_A': teacher is not None,
                  'answers': [{'tokens': list(x.outputs[0].token_ids), 'text': x.outputs[0].text, 'finish_reason': x.outputs[0].finish_reason} for x in generated]}
        record['workers'] = llm.collective_rpc('finish_sparse_phase')
        report['runs'].append(record)
        save()
    base = report['runs'][0]
    report['greedy_tokens_equal_dense_A'] = {r['phase']: [a['tokens'] == b['tokens'] for a,b in zip(base['answers'], r['answers'])] for r in report['runs'][1:]}
    reference = torch.load(output/'rank0/dense_A.pt', map_location='cpu', weights_only=True)
    report['logit_comparisons'] = {}
    for phase in ['dense_B'] + [m+'_full' for m in methods]:
        actual = torch.load(output/f'rank0/{phase}.pt', map_location='cpu', weights_only=True)
        common = sorted(set(reference) & set(actual))
        report['logit_comparisons'][phase] = {'common_keys': common, 'key_sets_equal': set(reference) == set(actual),
            'missing': sorted(set(reference)-set(actual)), 'extra': sorted(set(actual)-set(reference)),
            'rows': [{'key': key, 'max_abs': float((actual[key]-reference[key]).abs().max()),
                      'rmse': float((actual[key]-reference[key]).square().mean().sqrt()),
                      'top1_equal': int(actual[key].argmax()) == int(reference[key].argmax())} for key in common]}
    report['status'] = 'completed'
    save()
    print(json.dumps({'output': str(output), 'greedy': report['greedy_tokens_equal_dense_A']}), flush=True)


if __name__ == '__main__':
    main()
