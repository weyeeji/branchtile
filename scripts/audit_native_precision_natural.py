"""Independent CPU receipt/denominator audit of the frozen precision pilot."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_application import extract_final
from src.financial_answer_metric import METRIC_ID, score_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='configs/native_precision_natural_development_v1.json')
    parser.add_argument('--jobs', default='configs/native_precision_natural_jobs_v1.json')
    parser.add_argument('--output', default='results/native_precision_independent_audit_v1')
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    protocol = json.loads((ROOT / args.protocol).read_text())
    jobs = json.loads((ROOT / args.jobs).read_text())
    ledger = [json.loads(line) for line in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines()]
    groups = {g['document_id']: g for g in json.loads((ROOT / protocol['manifest']).read_text())['groups']}
    findings, checks, raw_hashes, rows, job_rows = [], {}, {}, [], []
    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            findings.append(name)
    for category in ('source_sha256', 'artifact_sha256'):
        for name, digest in protocol[category].items():
            check(f'{category}:{name}', sha(ROOT / name) == digest)
    def load(path):
        raw_hashes[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())
    check('four_jobs_cover_eight_distinct_GPUs', len(jobs) == 4 and sorted(g for job in jobs for g in job['gpus']) == list(range(8)))
    for job in jobs:
        name = job['name']; argv = job['argv']
        directory = ROOT / argv[argv.index('--output') + 1]
        summary = load(directory / 'summary.json')
        document = summary['document_id']; dtype = summary['engine_configuration']['kv_cache_dtype']
        question_group = groups[document]['questions']; n = len(question_group)
        events = [r for r in ledger if r['name'] == name and r['argv'] == argv]
        check(f'{name}:unique_start_end', len(events) == 2 and [r['event'] for r in events] == ['start', 'end']
              and events[0]['pid'] == events[1]['pid'])
        terminal = next((r for r in reversed(events) if r['event'] == 'end'), None)
        if terminal is None:
            raise RuntimeError(f'{name} is still active; defer final audit until ledger end')
        check(f'{name}:clean_terminal', terminal['exit_code'] == 0 and not terminal['timed_out'])
        check(f'{name}:complete_two_repeats', summary['status'] == 'completed' and len(summary['runs']) == 2
              and {r['repeat'] for r in summary['runs']} == {0, 1})
        check(f'{name}:source_protocol', summary['protocol_sha256'] == sha(ROOT / args.protocol)
              and summary['protocol_contents'] == protocol)
        check(f'{name}:natural_sampling', summary['sampling'] == protocol['sampling']
              and not summary['sampling']['ignore_eos'] and summary['sampling']['max_tokens'] == 8192)
        engine = dict(summary['engine_configuration'])
        for extra in ('model', 'kv_cache_dtype', 'worker_extension_cls'):
            engine.pop(extra)
        check(f'{name}:same_engine_parameters', engine == protocol['engine'])
        precision = []
        for phase in ('precision_before_warmup', 'precision_after_warmup', 'precision_after_generation'):
            receipts = summary.get(phase, [])
            check(f'{name}:{phase}:two_ranks', sorted(r['rank'] for r in receipts) == [0, 1])
            for receipt in receipts:
                tag = f'{name}:{phase}:rank{receipt["rank"]}'
                layers = receipt['layers']; metadata = receipt['actual_metadata']
                check(tag + ':48_layers', sorted(r['layer'] for r in layers) == list(range(48)))
                check(tag + ':dtype_layout', all(r['backend'] == 'FLASHINFER'
                    and r['logical_cache_dtype'] == dtype and r['shape'][1:] == [2, 16, 256]
                    and r['storage_dtype'] in (['torch.bfloat16'] if dtype == 'auto' else ['torch.uint8', 'torch.float8_e4m3fn']) for r in layers))
                check(tag + ':unit_scales', all(set(r['scales']) == {'_q_scale', '_k_scale', '_v_scale'}
                    and all(v == 1 for v in r['scales'].values())
                    and all(r['host_scales'][k + '_float'] == v for k, v in r['scales'].items()) for r in layers))
                check(tag + ':BF16_query', all(m['q_data_type_prefill'] == m['q_data_type_decode'] == 'torch.bfloat16' for m in metadata))
                if phase != 'precision_before_warmup':
                    check(tag + ':XQA_decode', any('decode' in m for m in metadata)
                          and all('XQA' in m['decode']['kernel'].upper() for m in metadata if 'decode' in m))
                check(tag + ':FULL_graphs_B123', all(any(f'num_tokens={b},' in g for g in receipt['graphs']) for b in (1, 2, 3)))
                check(tag + ':pool_byte_identity', sum(r['payload_bytes'] for r in layers) == receipt['cache_unique_storage_bytes'])
                precision.append({'phase': phase, 'rank': receipt['rank'],
                    **{k: receipt[k] for k in ('cache_unique_storage_bytes', 'allocated_cache_token_capacity',
                        'allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes')},
                    'graph_descriptors': receipt['graphs'],
                    'admission_identity_CPU_seconds': sum(r['CPU_identity_wall_seconds'] for r in receipt['admission_identity_records']),
                    'admission_record_count': len(receipt['admission_identity_records'])})
        run_signatures = []
        for run in summary['runs']:
            tag = f'{name}:repeat{run["repeat"]}'; answers = run['answers']
            check(tag + ':all_questions', len(answers) == n and [a['question_index'] for a in answers] == list(range(n)))
            check(tag + ':not_forced', run['teacher_forced'] is False)
            check(tag + ':cold_anchor', run['anchor_cached_tokens'] == 0)
            check(tag + ':positive_APC_hits', all(a['num_cached_tokens'] > 0 for a in answers))
            predictions, answer_rows = [], []
            for question, answer in zip(question_group, answers):
                final, reasoning, parse = extract_final(answer['text'], thinking=True)
                eligible = answer['finish_reason'] == 'stop' and parse == 'closed_thinking'
                predictions.append(final if eligible else '')
                answer_rows.append({'question_id': question['question_sha256'], 'question_index': answer['question_index'],
                    'gold': question['answer'], 'finish_reason': answer['finish_reason'], 'stop_reason': answer['stop_reason'],
                    'output_tokens': len(answer['tokens']), 'APC_hits': answer['num_cached_tokens'],
                    'thinking_parse': parse, 'completion_eligible': eligible,
                    'raw_text_sha256': hashlib.sha256(answer['text'].encode()).hexdigest(), 'final_text': final})
            scores = score_pairs(predictions, [q['answer'] for q in question_group])
            check(tag + ':fixed_denominator', scores['denominator'] == n)
            for answer, score in zip(answer_rows, scores.pop('per_example')):
                answer['score'] = score
            graph_signatures, workers = [], []
            for rank in (0, 1):
                graph = load(directory / f'rank{rank}/{run["phase"]}_graph.json')
                meta = load(directory / f'rank{rank}/{run["phase"]}.json')
                replays = graph['replays']
                signature = [(r['batch'], r['questions'], r['positions']) for r in replays]
                graph_signatures.append(signature)
                check(tag + f':rank{rank}:unmodified_native', meta['native_full_history'] and not meta['records']
                    and meta['logits_count'] == 0 and all(r['native_graph_replayed'] and not r['candidate_graph_replayed']
                    and r['dense_attention_executed_for_pure_decode'] for r in replays))
                check(tag + f':rank{rank}:no_shadows', all(not r.get('shadow_checks') for r in replays))
                workers.append({'rank': rank, 'FULL_replays': len(replays),
                    'batch_counts': dict(Counter(r['batch'] for r in replays)),
                    'graph_replay_questions': dict(Counter(q for r in replays for q in r['questions'])),
                    'host_attention_prepare_seconds': sum(r['complete_attention_prepare_host_seconds'] for r in replays),
                    'native_full_history': meta['native_full_history'], 'logit_copy_count': meta['logits_count'],
                    'metadata': meta['metadata_types']})
            check(tag + ':rank_step_agreement', graph_signatures[0] == graph_signatures[1])
            run_signatures.append(graph_signatures)
            rows.append({'job': name, 'document': document, 'dtype': dtype, 'repeat': run['repeat'],
                'score': scores, 'answers': answer_rows, 'workers': workers,
                'total_output_tokens': sum(len(a['tokens']) for a in answers),
                'natural_stop_count': sum(a['finish_reason'] == 'stop' for a in answers),
                'cap_hit_count': sum(a['finish_reason'] == 'length' for a in answers),
                **{k: run[k] for k in ('anchor_wall_seconds', 'complete_generate_wall_seconds',
                    'reset_APC_wall_seconds', 'audit_export_RPC_wall_seconds')},
                'anchor_plus_generate_sum_seconds': run['anchor_wall_seconds'] + run['complete_generate_wall_seconds']})
        a, b = summary['runs']
        aa = {'tokens': [x['tokens'] == y['tokens'] for x, y in zip(a['answers'], b['answers'])],
              'APC': [x['num_cached_tokens'] == y['num_cached_tokens'] for x, y in zip(a['answers'], b['answers'])],
              'steps_by_rank': [run_signatures[0][r] == run_signatures[1][r] for r in (0, 1)]}
        aa['all_equal'] = all(all(v) for v in aa.values())
        check(f'{name}:AA_summary_faithful', aa['all_equal'] == summary['cold_AA']['all_equal'])
        log = ROOT / 'logs' / f'{name}.log'; raw_hashes[str(log.relative_to(ROOT))] = sha(log)
        graph_pool = re.findall(r'CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB \(estimated\)', log.read_text())
        job_rows.append({'name': name, 'document': document, 'dtype': dtype, 'gpus': job['gpus'],
            'ledger_events': events, 'cold_AA_independently_recomputed': aa,
            'precision_receipts': precision, 'graph_pool_GiB_log_rounded_actual_estimated': graph_pool,
            **{k: summary.get(k) for k in ('generator_input_preparation_wall_seconds',
                'initialization_wall_seconds', 'stochastic_sampler_warmup_wall_seconds')},
            'allocation_GPU_hours': terminal['gpu_hours']})
    cross_precision = []
    for document in protocol['documents']:
        docid = document['document_id']
        for repeat in range(2):
            pair = {r['dtype']: r for r in rows if r['document'] == docid and r['repeat'] == repeat}
            check(f'{docid}:repeat{repeat}:two_precisions', set(pair) == {'auto', 'fp8_e4m3'})
            aa, bb = pair['auto'], pair['fp8_e4m3']
            hits = [x['APC_hits'] == y['APC_hits'] for x, y in zip(aa['answers'], bb['answers'])]
            check(f'{docid}:repeat{repeat}:same_APC_work', all(hits))
            cross_precision.append({'document': docid, 'repeat': repeat, 'same_APC_hits_by_question': hits,
                'BF16_APC_hits': [a['APC_hits'] for a in aa['answers']],
                'FP8_APC_hits': [a['APC_hits'] for a in bb['answers']],
                'BF16_score': aa['score'], 'FP8_score': bb['score'],
                'output_tokens_BF16_FP8': [aa['total_output_tokens'], bb['total_output_tokens']],
                'anchor_plus_generate_sum_BF16_FP8': [aa['anchor_plus_generate_sum_seconds'], bb['anchor_plus_generate_sum_seconds']]})
    check('all_24_answer_records_retained', sum(len(r['answers']) for r in rows) == 24)
    result = {'schema': 1, 'status': 'audit_passed' if not findings else 'audit_has_findings',
        'independent_auditor': 'Second agent, same model family; CPU source-and-receipt audit, not human review.',
        'protocol': args.protocol, 'protocol_sha256': sha(ROOT / args.protocol), 'metric_id': METRIC_ID,
        'checks': checks, 'findings': findings, 'jobs': job_rows, 'runs': rows,
        'cross_precision': cross_precision, 'P1_allocated_GPU_hours': sum(j['allocation_GPU_hours'] for j in job_rows),
        'unique_documents': 2, 'unique_questions': 6, 'answer_records_including_repeats_and_precisions': 24,
        'static_source_review': {
            'no_forcing_or_Q_O_logit_shadow_copies': True,
            'CPU_prompt_hash_before_original_add_requests': True,
            'graph_wrapper_asserts_preexisting_request_map': True,
            'inherited_GPU_prompt_readback_branch_unreachable_given_runtime_assertions': True,
            'attention_forward_and_graphs_unmodified': True,
            'CPU_identity_and_step_metadata_overhead_charged': True},
        'limitations': protocol['limitations'] + [
            'Two cold repeats are implementation repeatability checks, not new questions or independent quality samples.',
            'Anchor+generation is a sum of separately timed phases, not a measured end-to-end client interval.',
            'Different token trajectories and device pairs confound any executor-speed attribution.',
            'Same pool fraction means larger FP8 token capacity, not half total allocated process VRAM.',
            'Graph pool values parsed from logs are rounded to 0.01 GiB; receipt bytes are exact allocation counters.',
            'Strict/rounded FINAL matches are frozen proxy metrics, not official execution accuracy or source factual verification.'],
        'raw_sha256': raw_hashes, 'audit_source_sha256': sha(Path(__file__))}
    out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': result['status'], 'findings': findings,
                      'checks': len(checks), 'P1_GPU_hours': result['P1_allocated_GPU_hours'],
                      'answer_records': result['answer_records_including_repeats_and_precisions']}, indent=2))


if __name__ == '__main__':
    main()
