"""Full native model wall timing and separately instrumented decode attribution.

No attention substitution, sparse support, or invalid-output bypass is used.
Fixed-length generation is bounded valid-model execution, not natural quality.
"""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='configs/native_precision_natural_development_v1.json')
    parser.add_argument('--document-id', required=True)
    parser.add_argument('--kv-dtype', choices=['auto', 'fp8_e4m3'], required=True)
    parser.add_argument('--fixed-tokens', type=int, default=128)
    parser.add_argument('--natural-repeats', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    assert args.fixed_tokens >= 128 and args.natural_repeats >= 0
    protocol_path = ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text())
    out = ROOT / args.output
    assert not (out / 'summary.json').exists(), 'Refusing to overwrite an existing run'
    out.mkdir(parents=True, exist_ok=True)
    method = 'native_BF16_KV' if args.kv_dtype == 'auto' else 'native_FP8_KV_unit'
    report = {'schema': 1, 'status': 'initializing', 'uid': os.geteuid(),
              'python': sys.executable, 'argv': vars(args), 'protocol_sha256': sha(protocol_path),
              'protocol_contents': protocol,
              'document_id': args.document_id, 'method': method, 'runs': [],
              'source_sha256': {p: sha(ROOT / p) for p in (
                  'scripts/benchmark_native_opportunity.py', 'scripts/vllm_native_opportunity_worker.py',
                  'scripts/vllm_native_precision_worker.py', 'scripts/vllm_native_backend_worker.py',
                  'scripts/vllm_batch_admission.py', 'scripts/capture_model.py')},
              'scope': 'Baseline opportunity measurement, not candidate speedup or publication success.',
              'limitations': ['Fixed token runs ignore EOS and do not measure natural task quality.',
                             'Profile and graph event timings are separate from unprofiled wall time.',
                             'Native CPU metadata/identity hooks remain charged in wall time.',
                             'BF16/FP8 can follow different natural trajectories; cross-precision time is not same-work speedup.']}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        item = next(d for d in protocol['documents'] if d['document_id'] == args.document_id)
        manifest = ROOT / protocol['manifest']
        group = next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id'] == args.document_id)
        prompts, start, end = make_inputs(AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True), group)
        hashes = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert hashes == item['prompt_token_hashes']
        assert list(map(len, prompts)) == item['prompt_lengths']
        assert [start, end] == item['document_token_range']
        report.update(manifest=protocol['manifest'], manifest_sha256=sha(manifest),
                      document_sha256=group['document_sha256'], B=len(prompts),
                      prompt_lengths=list(map(len, prompts)), prompt_token_hashes=hashes)
        trace_dir = ROOT / 'results/raw/native_opportunity' / out.name
        assert not trace_dir.exists(), 'Refusing to reuse an existing trace directory'
        engine = dict(protocol['engine'], model=protocol['model_path'], kv_cache_dtype=args.kv_dtype,
                      worker_extension_cls='scripts.vllm_native_opportunity_worker.NativeOpportunityWorkerExtension',
                      profiler_config={'profiler': 'torch', 'torch_profiler_dir': str(trace_dir.resolve()),
                                       'torch_profiler_with_stack': False, 'torch_profiler_record_shapes': False,
                                       'torch_profiler_activities': ['CPU', 'CUDA'], 'delay_iterations': 64,
                                       'max_iterations': 32, 'ignore_frontend': True})
        report['engine_configuration'] = engine
        save()
        started = time.perf_counter()
        llm = LLM(**engine)
        report['initialization_wall_seconds'] = time.perf_counter() - started
        config = {'output': str(out.resolve()), 'backend': method, 'prompt_token_hashes': hashes}
        report['configuration'] = llm.collective_rpc('configure_opportunity_probe', args=(config,))
        warm = SamplingParams(temperature=0, max_tokens=1, ignore_eos=False, seed=protocol['seed'])
        llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
        generate_ready_group(llm, [{'prompt_token_ids': ids} for ids in prompts],
                             SamplingParams(**dict(protocol['sampling'], max_tokens=2)))
        report['native_receipt_before'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype, True))
        save()
        phases = [('fixed_0', False, False), ('fixed_1', False, False)]
        phases += [(f'natural_{i}', True, False) for i in range(args.natural_repeats)]
        phases += [('profile_fixed', False, True)]
        for phase, natural, profiled in phases:
            assert llm.reset_prefix_cache()
            llm.collective_rpc('set_native_phase', args=(phase + '_anchor',))
            started = time.perf_counter()
            anchor = llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
            anchor_wall = time.perf_counter() - started
            assert anchor[0].num_cached_tokens == 0
            llm.collective_rpc('set_native_phase', args=(phase,))
            sampling = dict(protocol['sampling'])
            if not natural:
                sampling.update(max_tokens=args.fixed_tokens, min_tokens=args.fixed_tokens, ignore_eos=True)
            if profiled:
                llm.collective_rpc('set_opportunity_instrumentation', args=(True,))
                llm.start_profile(profile_prefix='native_opportunity_decode')
            started = time.perf_counter()
            outputs = generate_ready_group(llm, [{'prompt_token_ids': ids} for ids in prompts], SamplingParams(**sampling))
            wall = time.perf_counter() - started
            if profiled:
                llm.stop_profile()
                report['profile_graph_events'] = llm.collective_rpc('finish_opportunity_instrumentation')
            answers = [{'question_index': i, 'tokens': list(o.outputs[0].token_ids),
                        'text': o.outputs[0].text, 'finish_reason': o.outputs[0].finish_reason,
                        'stop_reason': o.outputs[0].stop_reason, 'num_cached_tokens': o.num_cached_tokens}
                       for i, o in enumerate(outputs)]
            assert len(answers) == len(prompts)
            if not natural:
                assert all(len(a['tokens']) == args.fixed_tokens for a in answers)
            run = {'phase': phase, 'natural_eos_enabled': natural, 'profiled': profiled,
                   'sampling': sampling, 'anchor_wall_seconds': anchor_wall,
                   'complete_generate_wall_seconds': wall,
                   'cold_shared_document_total_wall_seconds': anchor_wall + wall,
                   'total_output_tokens': sum(len(a['tokens']) for a in answers), 'answers': answers,
                   'workers': llm.collective_rpc('finish_native_phase')}
            run['output_tokens_per_generate_wall_second'] = run['total_output_tokens'] / wall
            report['runs'].append(run)
            save()
        report['fixed_AA_tokens_equal'] = [a['tokens'] == b['tokens'] for a, b in zip(report['runs'][0]['answers'], report['runs'][1]['answers'])]
        report['fixed_AA_APC_hits_equal'] = [a['num_cached_tokens'] == b['num_cached_tokens'] for a, b in zip(report['runs'][0]['answers'], report['runs'][1]['answers'])]
        report['fixed_AA_scheduled_steps_equal'] = []
        report['fixed_AA_trajectory_files'] = []
        for rank in (0, 1):
            sequences = []
            for phase in ('fixed_0', 'fixed_1'):
                path = out / f'rank{rank}' / f'{phase}_graph.json'
                graph = json.loads(path.read_text())
                sequences.append([(r['batch'], r['questions'], r['positions']) for r in graph['replays']])
                report['fixed_AA_trajectory_files'].append({'rank': rank, 'phase': phase,
                    'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'steps': len(sequences[-1])})
            report['fixed_AA_scheduled_steps_equal'].append(sequences[0] == sequences[1])
        report['fixed_AA_all_equal'] = all(report['fixed_AA_tokens_equal'] + report['fixed_AA_APC_hits_equal'] + report['fixed_AA_scheduled_steps_equal'])
        report['native_receipt_after'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype, True))
        report['trace_files'] = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                                 for p in sorted(trace_dir.rglob('*')) if p.is_file()]
        report['status'] = 'completed'
        save()
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
