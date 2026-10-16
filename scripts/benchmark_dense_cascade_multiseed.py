"""Native XQA versus existing full-history shared-prefix cascade in real vLLM."""
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
    p.add_argument('--kv-dtype', choices=['auto', 'fp8_e4m3'], default='auto')
    p.add_argument('--qualification-tokens', type=int, default=16)
    p.add_argument('--seed-protocol', default='configs/dense_cascade_multiseed_natural_v1.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    assert args.qualification_tokens >= 2
    seed_protocol = json.loads((ROOT / args.seed_protocol).read_text())
    assert args.protocol == seed_protocol['base_protocol']
    assert args.qualification_tokens == seed_protocol['qualification_tokens']
    assert len(seed_protocol['additional_sampling_seeds']) == len(seed_protocol['method_orders'])
    protocol = json.loads((ROOT / args.protocol).read_text())
    out = ROOT / args.output
    assert not (out / 'summary.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/benchmark_dense_cascade_multiseed.py', args.seed_protocol,
               'scripts/benchmark_dense_cascade_reference.py', 'scripts/vllm_dense_cascade_reference_worker.py',
               'src/dense_cascade_fp64_reference.py', 'scripts/benchmark_dense_cascade_native.py', 'scripts/vllm_dense_cascade_worker.py',
               'src/vllm_dense_cascade_plan.py', 'scripts/vllm_native_precision_worker.py',
               'scripts/vllm_native_backend_worker.py', 'scripts/vllm_batch_admission.py']
    report = {'status': 'initializing', 'uid': os.geteuid(), 'python': sys.executable, 'args': vars(args),
              'protocol': protocol, 'seed_protocol':seed_protocol, 'runs': [],
              'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in sources},
              'scope': 'Existing dense shared-prefix control. Original pairwise screen retained; failed screens adjudicated for both arms against FP64 with unchanged .006/.015 tolerances.',
              'limitations': ['Candidate floating-point reduction order may change tokens; fixed caps and natural task outcomes are distinct.',
                             'Cold shared-document wall includes one anchor and full group generation; graph/plan setup is separate and must be charged for first use.',
                             'B1 and mixed/nonpure decode use the original XQA/prefill path; actual candidate/fallback records retained.',
                             'Candidate host synchronization and native plan preparation remain included in real generation wall time.']}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from scripts.capture_model import make_inputs
        from scripts.vllm_batch_admission import generate_ready_group
        started = time.perf_counter()
        group = next(g for g in json.loads((ROOT / protocol['manifest']).read_text())['groups'] if g['document_id'] == args.document_id)
        item = next(d for d in protocol['documents'] if d['document_id'] == args.document_id)
        prompts, start, end = make_inputs(AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True), group)
        hashes = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert hashes == item['prompt_token_hashes']
        common = 0
        for tokens in zip(*prompts):
            if len(set(tokens)) != 1:
                break
            common += 1
        common = (common // 16) * 16
        assert common > 0 and common < min(map(len, prompts))
        report.update(B=len(prompts), prompt_token_hashes=hashes, prompt_lengths=list(map(len, prompts)),
                      document_sha256=group['document_sha256'], common_prefix_tokens=common,
                      input_preparation_wall_seconds=time.perf_counter()-started)
        engine = dict(protocol['engine'], model=protocol['model_path'], kv_cache_dtype=args.kv_dtype,
                      worker_extension_cls='scripts.vllm_dense_cascade_reference_worker.DenseCascadeReferenceWorker')
        report['engine'] = engine
        started = time.perf_counter()
        llm = LLM(**engine)
        report['initialization_wall_seconds'] = time.perf_counter()-started
        config = {'output': str(out.resolve()), 'backend': 'native_XQA_or_dense_cascade',
                  'prompt_token_hashes': hashes, 'batch': len(prompts), 'common_prefix_tokens': common,
                  'max_tail_tokens': max(map(len, prompts))-common+protocol['sampling']['max_tokens']+64}
        started = time.perf_counter()
        report['configuration'] = llm.collective_rpc('configure_dense_cascade', args=(config,))
        report['capture'] = llm.collective_rpc('capture_dense_cascade_graphs')
        report['candidate_plan_and_capture_wall_seconds'] = time.perf_counter()-started
        report['native_before'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype,))
        save()
        warm = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        phases = [('qualification', 'cascade', False, True, args.qualification_tokens, seed_protocol['qualification_seed'])]
        for seed,order in zip(seed_protocol['additional_sampling_seeds'],seed_protocol['method_orders']):
            assert sorted(order) == ['cascade','native']
            for method in order:
                phases.append((f'{method}_natural_seed{seed}',method,True,False,None,seed))
        for phase, method, natural, validation, tokens, seed in phases:
            llm.collective_rpc('set_dense_cascade_phase', args=(phase+'_anchor', 'native', False))
            assert llm.reset_prefix_cache()
            started = time.perf_counter()
            anchor = llm.generate([{'prompt_token_ids': prompts[0]}], warm, use_tqdm=False)
            anchor_wall = time.perf_counter()-started
            assert anchor[0].num_cached_tokens == 0
            llm.collective_rpc('set_dense_cascade_phase', args=(phase, method, validation))
            sampling = dict(protocol['sampling'],seed=seed)
            if not natural:
                sampling.update(max_tokens=tokens, min_tokens=tokens, ignore_eos=True)
            started = time.perf_counter()
            outputs = generate_ready_group(llm, [{'prompt_token_ids': ids} for ids in prompts], SamplingParams(**sampling))
            wall = time.perf_counter()-started
            answers = [{'question_index': i, 'tokens': list(o.outputs[0].token_ids), 'text': o.outputs[0].text,
                        'finish_reason': o.outputs[0].finish_reason, 'stop_reason': o.outputs[0].stop_reason,
                        'num_cached_tokens': o.num_cached_tokens} for i, o in enumerate(outputs)]
            run = {'phase': phase, 'method': method, 'validation': validation, 'natural_eos_enabled': natural,
                   'sampling': sampling, 'sampling_seed':seed, 'anchor_wall_seconds': anchor_wall, 'complete_generate_wall_seconds': wall,
                   'cold_shared_document_total_wall_seconds': anchor_wall+wall, 'answers': answers,
                   'total_output_tokens': sum(len(a['tokens']) for a in answers),
                   'workers': llm.collective_rpc('finish_dense_cascade_phase')}
            if phase == 'qualification':
                checks = [json.loads(Path(w['cascade_record']).read_text()) for w in run['workers']]
                assert all(c['checks'] for c in checks)
                run['same_Q_native_all_layers_passed'] = all(all(x['same_Q_native_O_passed'] for x in step['layers']) for c in checks for step in c['checks'])
                run['all_layers_qualification_passed'] = all(all(x['qualification_passed'] for x in step['layers']) for c in checks for step in c['checks'])
                run['pair_failed_but_reference_passed_count'] = sum(x.get('pair_failed_but_reference_passed',False) for c in checks for step in c['checks'] for x in step['layers'])
                report['runs'].append(run)
                save()
                assert run['all_layers_qualification_passed'], 'Reference qualification failed; both rank records are saved.'
                continue
            report['runs'].append(run)
            save()
        report['native_after'] = llm.collective_rpc('native_precision_receipt', args=(args.kv_dtype, True))
        report['status'] = 'completed'
        save()
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
