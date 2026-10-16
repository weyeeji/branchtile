"""Frozen same-stack natural development comparison of BF16 and FP8 KV."""
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
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--document-id', required=True)
    parser.add_argument('--kv-dtype', choices=['auto', 'fp8_e4m3'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    protocol_path = ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in protocol['artifact_sha256'].items():
        assert sha(ROOT / path) == digest, path
    qualification = json.loads((ROOT / protocol['qualification_audit']).read_text())
    assert qualification['numerical_and_trajectory_contract_complete']
    assert qualification['process_clean_exit']
    assert qualification['frozen_protocol_overall_status'] == 'passed_clean_exit'
    item = next(row for row in protocol['documents'] if row['document_id'] == args.document_id)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    method = 'native_BF16_KV' if args.kv_dtype == 'auto' else 'native_FP8_KV_unit'
    report = {'status': 'initializing', 'uid': os.geteuid(), 'python': sys.executable,
              'protocol': args.protocol, 'protocol_sha256': sha(protocol_path),
              'protocol_contents': protocol, 'document_id': args.document_id,
              'manifest': str(ROOT / protocol['manifest']),
              'manifest_sha256': sha(ROOT / protocol['manifest']),
              'methods': [method], 'repeats': protocol['repeats'], 'runs': [],
              'sampling': protocol['sampling'], 'steps_cap': protocol['sampling']['max_tokens'],
              'budget_request': 'full', 'native_official_backend_control': True,
              'limitations': protocol['limitations']}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        started = time.perf_counter()
        group = next(group for group in json.loads((ROOT / protocol['manifest']).read_text())['groups']
                     if group['document_id'] == args.document_id)
        tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
        prompts, start, end = make_inputs(tokenizer, group)
        hashes = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert hashes == item['prompt_token_hashes']
        assert list(map(len, prompts)) == item['prompt_lengths']
        assert [start, end] == item['document_token_range']
        report.update(generator_input_preparation_wall_seconds=time.perf_counter()-started,
                      document_sha256=group['document_sha256'], B=len(prompts), budget=end-start,
                      prompt_lengths=list(map(len, prompts)))
        engine = dict(protocol['engine'], model=protocol['model_path'], kv_cache_dtype=args.kv_dtype,
                      worker_extension_cls='scripts.vllm_native_precision_worker.NativePrecisionWorkerExtension')
        report['engine_configuration'] = engine
        save()
        started = time.perf_counter()
        llm = LLM(**engine)
        report['initialization_wall_seconds'] = time.perf_counter()-started
        report['sampling_params_repr'] = repr(SamplingParams(**protocol['sampling']))
        save()
        config = {'output': str(out.resolve()), 'backend': method, 'prompt_token_hashes': hashes}
        report['configured_workers'] = llm.collective_rpc('configure_native_precision_probe', args=(config,))
        report['precision_before_warmup'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype,))
        save()
        warm = SamplingParams(temperature=0, max_tokens=1, ignore_eos=False, seed=protocol['seed'])
        started = time.perf_counter()
        llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
        warm_stochastic = dict(protocol['sampling'], max_tokens=2)
        generate_ready_group(llm, [{'prompt_token_ids': ids} for ids in prompts], SamplingParams(**warm_stochastic))
        report['stochastic_sampler_warmup_wall_seconds'] = time.perf_counter()-started
        report['precision_after_warmup'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype, True))
        save()
        for repeat in range(protocol['repeats']):
            phase = f'{method}_cold_{repeat}'
            started = time.perf_counter()
            assert llm.reset_prefix_cache()
            reset_seconds = time.perf_counter()-started
            llm.collective_rpc('set_native_phase', args=(phase+'_anchor',))
            started = time.perf_counter()
            anchor = llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
            anchor_seconds = time.perf_counter()-started
            assert anchor[0].num_cached_tokens == 0
            llm.collective_rpc('set_native_phase', args=(phase,))
            started = time.perf_counter()
            outputs = generate_ready_group(llm, [{'prompt_token_ids': ids} for ids in prompts],
                                           SamplingParams(**protocol['sampling']))
            wall = time.perf_counter()-started
            run = {'phase': phase, 'method': method, 'repeat': repeat, 'teacher_forced': False,
                   'reset_APC_wall_seconds': reset_seconds, 'anchor_wall_seconds': anchor_seconds,
                   'anchor_cached_tokens': anchor[0].num_cached_tokens,
                   'complete_generate_wall_seconds': wall,
                   'answers': [{'question_index': i, 'tokens': list(o.outputs[0].token_ids),
                                'text': o.outputs[0].text, 'finish_reason': o.outputs[0].finish_reason,
                                'stop_reason': o.outputs[0].stop_reason, 'num_cached_tokens': o.num_cached_tokens}
                               for i, o in enumerate(outputs)]}
            assert len(outputs) == len(prompts)
            assert all(answer['num_cached_tokens'] > 0 for answer in run['answers'])
            run['total_output_tokens'] = sum(len(answer['tokens']) for answer in run['answers'])
            run['tokens_per_generate_wall_second'] = run['total_output_tokens'] / wall
            started = time.perf_counter()
            run['workers'] = llm.collective_rpc('finish_native_phase')
            run['audit_export_RPC_wall_seconds'] = time.perf_counter()-started
            report['runs'].append(run)
            save()
        a, b = report['runs']
        report['cold_AA'] = {'tokens_equal': [x['tokens'] == y['tokens'] for x, y in zip(a['answers'], b['answers'])],
                             'APC_hits_equal': [x['num_cached_tokens'] == y['num_cached_tokens'] for x, y in zip(a['answers'], b['answers'])],
                             'scheduled_steps_equal': []}
        for rank in (0, 1):
            def steps(run):
                graph = json.loads((out / f'rank{rank}' / f'{run["phase"]}_graph.json').read_text())
                return [(r['batch'], r['questions'], r['positions']) for r in graph['replays']]
            report['cold_AA']['scheduled_steps_equal'].append(steps(a) == steps(b))
        report['cold_AA']['all_equal'] = all(all(value) for value in report['cold_AA'].values())
        report['precision_after_generation'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype, True))
        report['status'] = 'completed'
        save()
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
