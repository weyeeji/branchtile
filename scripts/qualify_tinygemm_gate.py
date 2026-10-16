"""Prepare or run a bounded existing TinyGEMM direct-replacement qualification."""
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


def validate_protocol(protocol):
    for group in ('source_sha256', 'artifact_sha256'):
        for name, digest in protocol[group].items():
            assert sha(ROOT / name) == digest, (group, name)
    assert protocol['phases'] == ['full_A', 'full_B']
    assert protocol['tokens_by_question'] == [8, 12, 16]
    assert protocol['timing']['blocks'] == 20
    assert protocol['engine']['tensor_parallel_size'] == 2
    assert protocol['budgets']['allocated_GPU_hours_upper_bound'] >= 2 * (240 + 15) / 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    path = ROOT / args.protocol; protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    if args.validate_only:
        print(json.dumps({'status': 'CPU_source_artifact_hashes_validated',
                          'protocol_sha256': sha(path), 'GPU_run': False}))
        return
    assert args.output and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'summary.json').exists(), 'Refuse to overwrite an earlier attempt; freeze a new output path first'
    report = {'status': 'initializing', 'uid': os.geteuid(), 'pid': os.getpid(),
        'python': sys.executable, 'protocol': args.protocol, 'protocol_sha256': sha(path),
        'protocol_contents': protocol, 'runs': [],
        'scope': 'Existing TinyGEMM baseline conservative bitwise drop-in gate, not a new algorithm or task-quality/application-speed experiment.'}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        manifest = json.loads((ROOT / protocol['manifest']).read_text())
        group = next(g for g in manifest['groups'] if g['document_id'] == protocol['document_id'])
        tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
        prompts, start, end = make_inputs(tokenizer, group)
        hashes = [hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in prompts]
        assert hashes == protocol['prompt_token_hashes']
        assert list(map(len, prompts)) == protocol['prompt_lengths']
        assert [start, end] == protocol['document_token_range']
        engine = dict(protocol['engine'], model=protocol['model_path'],
            worker_extension_cls='scripts.vllm_tinygemm_gate_worker.TinyGemmGateWorkerExtension')
        report['engine_configuration'] = engine; save(); started = time.perf_counter()
        llm = LLM(**engine)
        report['initialization_wall_seconds'] = time.perf_counter() - started; save()
        config = {'output': str(out.resolve()), 'prompt_token_hashes': hashes, 'timing': protocol['timing']}
        report['configured_workers'] = llm.collective_rpc('configure_tiny_gate_probe', args=(config,)); save()
        warm = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, seed=protocol['seed'])
        started = time.perf_counter()
        llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
        generate_ready_group(llm, [{'prompt_token_ids': x} for x in prompts], warm)
        report['warmup_wall_seconds'] = time.perf_counter() - started
        report['instrumented_native_graph_capture'] = llm.collective_rpc('capture_tiny_gate_native_graphs'); save()
        for phase in protocol['phases']:
            assert llm.reset_prefix_cache()
            llm.collective_rpc('set_tiny_gate_phase', args=(phase + '_anchor',))
            started = time.perf_counter()
            anchor = llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
            anchor_seconds = time.perf_counter() - started
            assert anchor[0].num_cached_tokens == 0
            llm.collective_rpc('set_tiny_gate_phase', args=(phase,))
            params = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True, seed=protocol['seed'])
                      for n in protocol['tokens_by_question']]
            started = time.perf_counter()
            outputs = generate_ready_group(llm, [{'prompt_token_ids': x} for x in prompts], params)
            run = {'phase': phase, 'cold_anchor_wall_seconds': anchor_seconds,
                'instrumented_generate_wall_seconds': time.perf_counter() - started,
                'anchor_cached_tokens': anchor[0].num_cached_tokens,
                'answers': [{'question': i, 'tokens': list(o.outputs[0].token_ids),
                    'finish_reason': o.outputs[0].finish_reason, 'num_cached_tokens': o.num_cached_tokens}
                    for i, o in enumerate(outputs)],
                'workers': llm.collective_rpc('finish_tiny_gate_phase')}
            report['runs'].append(run); save()
            assert [len(a['tokens']) for a in run['answers']] == protocol['tokens_by_question']
            assert all(a['num_cached_tokens'] > 0 for a in run['answers'])
            assert all(w['batches'] == [1, 2, 3] for w in run['workers'])
        a, b = report['runs']
        report['cold_AA'] = {'same_tokens': [x['tokens'] == y['tokens'] for x, y in zip(a['answers'], b['answers'])],
            'same_APC_hits': [x['num_cached_tokens'] == y['num_cached_tokens'] for x, y in zip(a['answers'], b['answers'])],
            'same_steps': []}
        for rank in (0, 1):
            def steps(phase):
                data = json.loads((out / f'rank{rank}/{phase}.json').read_text())
                return [(r['batch'], r['questions'], r['positions']) for r in data['steps']]
            report['cold_AA']['same_steps'].append(steps('full_A') == steps('full_B'))
        save()
        assert all(all(values) for values in report['cold_AA'].values()), report['cold_AA']
        report['qualification_workers'] = llm.collective_rpc('qualify_and_time_tiny_gate'); save()
        for rank in (0, 1):
            result = json.loads((out / f'rank{rank}/qualification.json').read_text())
            keys = {(r['phase'], r['batch'], r['layer']) for r in result['checks']}
            assert len(result['checks']) == 288 and keys == {
                (phase, batch, layer) for phase in protocol['phases'] for batch in (1, 2, 3) for layer in range(48)}
            assert len(result['cold_AA_points']) == 144
        qualified = all(w['qualified'] for w in report['qualification_workers'])
        report['timing_workers'] = (llm.collective_rpc('time_qualified_tiny_gate') if qualified else
            {'status': 'skipped_on_both_ranks_because_at_least_one_rank_not_bitwise_qualified'})
        report['qualified_bitwise_drop_in'] = qualified
        report['status'] = 'completed_qualified_bitwise_drop_in' if qualified else 'completed_not_qualified_bitwise_drop_in'
        report['operationally_completed'] = True
        report['interpretation'] = ('Bitwise qualification is conservative; a difference does not imply mathematically incorrect TinyGEMM. '
            'No candidate was installed in the live model. A non-bitwise implementation needs separately frozen model-level validation.')
        save()
    except BaseException:
        report['status'] = 'operational_failure'; report['error'] = traceback.format_exc(); save(); raise


if __name__ == '__main__':
    main()
