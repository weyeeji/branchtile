"""Finite native self-consistency utility/cost pilot; no candidate performance claim."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.financial_self_consistency import sample_seed, vote
from scripts.benchmark_native_raw_pair_direct_production import answer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/native_self_consistency_train26_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.prepare_only else len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
    out = Path(args.output)
    assert not (out / 'summary.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(args.protocol).read_text())
    sources = ['scripts/probe_native_self_consistency.py', 'scripts/vllm_native_self_consistency_worker.py',
               'scripts/vllm_native_precision_worker.py', 'scripts/vllm_native_backend_worker.py',
               'scripts/capture_model.py', 'scripts/vllm_batch_admission.py', 'scripts/screen_application.py',
               'src/financial_answer_metric.py', 'src/financial_self_consistency.py']
    report = {'status': 'initializing', 'protocol': protocol, 'protocol_sha256': digest(args.protocol),
              'sources': {p: digest(p) for p in sources}, 'runs': [], 'args': vars(args)}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        import torch
        from transformers import AutoTokenizer
        from scripts.capture_model import make_inputs
        from scripts.screen_application import extract_final
        from src.financial_answer_metric import score_pairs
        start = time.perf_counter()
        base = json.loads(Path(protocol['base_protocol']).read_text())
        assert digest(protocol['base_protocol']) == protocol['base_protocol_sha256']
        document = next(d for d in base['documents'] if d['document_id'] == protocol['document_id'])
        group = next(g for g in json.loads(Path(base['manifest']).read_text())['groups'] if g['document_id'] == protocol['document_id'])
        prompts, begin, end = make_inputs(AutoTokenizer.from_pretrained(base['model_path'], local_files_only=True), group)
        hashes = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
        assert hashes == document['prompt_token_hashes']
        assert list(map(len, prompts)) == document['prompt_lengths']
        assert [begin, end] == document['document_token_range']
        assert len(prompts) == 21 and max(map(len, prompts)) + 8192 <= base['engine']['max_model_len']
        requests = [{'question_index': q, 'sample_index': s, 'prompt_sha256': hashes[q],
                     'seed': sample_seed(protocol['document_id'], hashes[q], s, protocol['study_seed'])}
                    for s in range(4) for q in range(len(prompts))]
        assert len({r['seed'] for r in requests}) == len(requests)
        report.update(request_manifest=requests, B=21, P=document['common_complete_page_tokens'],
                      input_preparation_wall_seconds=time.perf_counter()-start,
                      manifest_sha256=digest(base['manifest']))
        if args.prepare_only:
            assert not torch.cuda.is_initialized()
            report.update(status='CPU_prepared', cuda_initialized=False); save(); return
        from vllm import LLM, SamplingParams
        from scripts.vllm_batch_admission import generate_ready_group
        engine = copy.deepcopy(base['engine'])
        engine.update(model=base['model_path'], kv_cache_dtype='auto', max_num_seqs=21,
                      worker_extension_cls='scripts.vllm_native_self_consistency_worker.NativeSelfConsistencyWorker')
        engine['compilation_config']['cudagraph_capture_sizes'] = list(range(1, 22))
        start = time.perf_counter(); llm = LLM(**engine)
        report['engine_initialization_wall_seconds'] = time.perf_counter()-start
        report['engine'] = engine
        report['configuration'] = llm.collective_rpc('configure_self_consistency_probe', args=({'output': str(out.resolve()), 'backend': 'native_XQA_SelfConsistency', 'prompt_token_hashes': hashes},))
        report['capacity'] = llm.collective_rpc('native_precision_receipt', args=('auto', False))
        # Compile the exact stochastic sampler before either utility phase.
        warm = copy.deepcopy(base['sampling']); warm.update(max_tokens=1, seed=requests[0]['seed'])
        start = time.perf_counter()
        llm.generate([{'prompt_token_ids': prompts[0]}], SamplingParams(**warm), use_tqdm=False)
        report['shared_sampling_warmup_wall_seconds'] = time.perf_counter()-start; save()
        for n in (1, 4):
            phase = f'native_SC{n}'
            manifest = [r for r in requests if r['sample_index'] < n]
            assert llm.reset_prefix_cache()
            llm.collective_rpc('set_native_phase', args=(phase+'_anchor',))
            start = time.perf_counter()
            anchors = llm.generate([{'prompt_token_ids': prompts[0]}], SamplingParams(temperature=0, max_tokens=1, seed=protocol['study_seed']), use_tqdm=False)
            anchor_wall = time.perf_counter()-start; assert anchors[0].num_cached_tokens == 0
            llm.collective_rpc('set_native_phase', args=(phase,))
            parameters = [SamplingParams(**dict(base['sampling'], seed=r['seed'])) for r in manifest]
            started = time.perf_counter()
            outputs = generate_ready_group(llm, [{'prompt_token_ids': prompts[r['question_index']]} for r in manifest], parameters)
            ended = time.perf_counter()
            assert len(outputs) == len(manifest)
            answers = [dict(answer(i, output), sample=manifest[i]) for i, output in enumerate(outputs)]
            workers = llm.collective_rpc('finish_self_consistency_phase')
            run = {'phase': phase, 'samples_per_question': n, 'answers': answers, 'workers': workers,
                   'generation_started_monotonic_seconds': started, 'generation_ended_monotonic_seconds': ended,
                   'complete_generate_wall_seconds': ended-start, 'cold_anchor_wall_seconds': anchor_wall,
                   'cold_document_task_wall_seconds': anchor_wall+ended-start,
                   'total_output_tokens': sum(len(a['tokens']) for a in answers),
                   'cap_hits': [a['sample'] for a in answers if a['finish_reason']=='length'],
                   'preemptions': sum(a['request_metrics']['num_preemptions'] for a in answers) if all(a['request_metrics'] is not None for a in answers) else None}
            report['runs'].append(run); save()
            # Resolve outputs against actual CPU admission seeds; never rely only on positional IDs.
            for worker in workers:
                data = json.loads(Path(worker['queue_metadata']).read_text())
                admissions = {a['request_id']: a for a in data['sample_admissions']}
                assert len(admissions) == len(answers)
                for a in answers:
                    receipt = admissions[a['request_id']]
                    assert receipt['seed'] == a['sample']['seed']
                    assert receipt['question_index'] == a['sample']['question_index']
            predictions = [[] for _ in prompts]
            for a in answers:
                final, _, parsed = extract_final(a['text'], thinking=True)
                predictions[a['sample']['question_index']].append(final if a['finish_reason']=='stop' and parsed=='closed_thinking' else '')
            voting_started = time.perf_counter()
            votes = [vote(values) for values in predictions]
            run['vote_wall_seconds'] = time.perf_counter()-voting_started
            golds = [q['answer'] for q in group['questions']]
            run['quality'] = {'votes': votes, 'vote_score': score_pairs([v['prediction'] for v in votes], golds),
                              'sample0_within_this_queue': score_pairs([p[0] for p in predictions], golds),
                              'individual_sample_scores': [score_pairs([p[s] for p in predictions], golds) for s in range(n)]}
            run['receipt'] = llm.collective_rpc('native_precision_receipt', args=('auto', True)); save()
        report['status'] = 'completed'; save()
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    main()
