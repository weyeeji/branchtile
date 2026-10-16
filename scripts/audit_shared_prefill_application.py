#!/usr/bin/env python3
"""CPU-only accounting of already measured native cold-group preparation.

Reads execution metadata, token counts and timings; reproduces tokenization
on CPU. Does not score answers, change an experiment or launch inference.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/shared_prefill_application_audit_v1/accounting.json'


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    from transformers import AutoTokenizer
    from capture_model import make_inputs
    import torch
    assert not torch.cuda.is_initialized()
    inputs = {}
    reproduced = {}

    def read(relative):
        path = ROOT / relative
        data = path.read_bytes()
        inputs[str(relative)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    rows = []
    for arm in ('original_bf16', 'original_fp8', 'new_bf16', 'new_fp8'):
        base = Path('results/native_precision_natural_development_v1') / arm
        source = read(base / 'summary.json')
        assert source['status'] == 'completed' and source['cold_AA']['all_equal']
        document = next(x for x in source['protocol_contents']['documents']
                        if x['document_id'] == source['document_id'])
        if source['document_id'] not in reproduced:
            protocol = source['protocol_contents']
            manifest = read(protocol['manifest'])
            group = next(x for x in manifest['groups'] if x['document_id'] == source['document_id'])
            tokenizer = AutoTokenizer.from_pretrained(ROOT / protocol['model_path'], local_files_only=True)
            prompts, start, end = make_inputs(tokenizer, group)
            assert [start, end] == document['document_token_range']
            assert [len(p) for p in prompts] == document['prompt_lengths']
            assert [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts] == document['prompt_token_hashes']
            common = 0
            for token_tuple in zip(*prompts):
                if len(set(token_tuple)) != 1:
                    break
                common += 1
            reproduced[source['document_id']] = common
            path = ROOT / group['document_path']
            inputs[group['document_path']] = hashlib.sha256(path.read_bytes()).hexdigest()
        common = reproduced[source['document_id']]
        lengths = source['prompt_lengths']
        block_size = source['engine_configuration']['block_size']
        doc_end = document['document_token_range'][1]
        runs = []
        for run in source['runs']:
            assert run['anchor_cached_tokens'] == 0 and not run['teacher_forced']
            hits = [a['num_cached_tokens'] for a in run['answers']]
            assert all(h > 0 and h % block_size == 0 for h in hits)
            assert all(h <= n - 1 for h, n in zip(hits, lengths))
            common_full_pages = min(hits)
            # This is a token-work accounting bound for document replay, not
            # a GPU latency bound. Anchor computed the document once already.
            replayed_document_tokens = sum(max(0, doc_end - h) for h in hits)
            q0_replayed_prompt_tokens = lengths[0] - hits[0]
            assert 0 < q0_replayed_prompt_tokens <= block_size
            assert hits[0] >= common
            repeated_common_tail = sum(max(0, common-h) for h in hits[1:])
            per_rank = []
            for rank in (0, 1):
                graph = read(base / f'rank{rank}' / f'{run["phase"]}_graph.json')
                replay = graph['replays']
                assert replay and all(x['native_graph_replayed'] for x in replay)
                admission = [x for x in source['precision_after_generation'][rank]
                             ['admission_identity_records'] if x['phase'] == run['phase']]
                assert sorted(x['question'] for x in admission) == [0, 1, 2]
                per_rank.append({
                    'rank': rank,
                    'full_decode_replay_count': len(replay),
                    'batch_size_counts': dict(sorted(collections.Counter(
                        str(x['batch']) for x in replay).items())),
                    'prepare_attn_host_seconds_sum': sum(
                        x['complete_attention_prepare_host_seconds'] for x in replay),
                    'prepare_attn_host_seconds_median': statistics.median(
                        x['complete_attention_prepare_host_seconds'] for x in replay),
                    'admission_identity_host_seconds_sum': sum(
                        x['CPU_identity_wall_seconds'] for x in admission),
                })
            runs.append({
                'phase': run['phase'],
                'APC_hit_tokens': hits,
                'uncached_prompt_tokens_after_anchor': [n-h for n, h in zip(lengths, hits)],
                'common_full_page_prefix_tokens': common_full_pages,
                'q0_prompt_tokens_recomputed_after_anchor': q0_replayed_prompt_tokens,
                'document_token_replays_after_anchor_across_all_requests': replayed_document_tokens,
                'common_prompt_token_replays_in_followers': repeated_common_tail,
                'removable_duplicate_prompt_token_work_upper_bound':
                    q0_replayed_prompt_tokens + repeated_common_tail,
                'anchor_wall_seconds': run['anchor_wall_seconds'],
                'generate_wall_seconds': run['complete_generate_wall_seconds'],
                'anchor_plus_generate_seconds': run['anchor_wall_seconds'] + run['complete_generate_wall_seconds'],
                'output_lengths': [len(a['tokens']) for a in run['answers']],
                'per_rank_host_scopes_do_not_sum_across_ranks': per_rank,
            })
        cold = statistics.median(x['anchor_plus_generate_seconds'] for x in runs)
        prep = source['generator_input_preparation_wall_seconds']
        rows.append({
            'arm': arm, 'document_id': source['document_id'],
            'document_sha256': source['document_sha256'],
            'document_token_range': document['document_token_range'],
            'exact_common_prompt_prefix_tokens_reproduced_from_input_hashes': common,
            'prompt_lengths': lengths, 'block_size': block_size,
            'input_preparation_seconds_including_tokenizer_load': prep,
            'median_anchor_plus_generate_seconds': cold,
            'optimistic_fraction_if_entire_input_preparation_disappeared': prep / (prep + cold),
            'runs': runs,
        })
    for rel in (
        'scripts/benchmark_native_precision.py', 'scripts/vllm_native_precision_worker.py',
        'scripts/vllm_native_backend_worker.py', 'scripts/vllm_batch_admission.py',
        'scripts/capture_model.py', 'docs/pivot_candidates_20261016.md',
        'docs/post_stop_research_decision_20261016.md',
        '.venv-engine/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py',
        '.venv-engine/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py',
        'data/models/Qwen3-30B-A3B-Thinking-2507-FP8/tokenizer.json',
        'data/models/Qwen3-30B-A3B-Thinking-2507-FP8/tokenizer_config.json',
    ):
        inputs[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    report = {
        'schema': 1, 'scope': 'Existing development metadata only; CPU-only; no new experiments.',
        'uid': os.getuid(), 'python': sys.executable, 'GPU_hours': 0,
        'cuda_initialized': torch.cuda.is_initialized(),
        'novel_candidate_count': 0, 'input_sha256': inputs, 'arms': rows,
        'limitations': [
            'The anchor computes the long prefix once; it is not removable duplicated full-prefill work.',
            'The removable-token bound counts q0 replay and all exact-common-prefix replays in followers; it is not a time bound or proof those rows are avoidable by an available API.',
            'The discarded anchor output is sampled from prefill logits; max_tokens=1 does not imply an extra autoregressive decode forward.',
            'Input preparation includes tokenizer loading, rendering, offsets, all prompt tokenization and identity checks; it is not all per-request repeated work.',
            'The all-input-preparation fraction is a deliberately optimistic conditional accounting sensitivity, not measured client JCT or a general lower bound.',
            'prepare_attn CPU scopes can overlap GPU work; rank sums must not be added or treated as recoverable critical-path latency.',
            'The record contains no multi-document arrivals, queueing delay, eviction pressure, or cross-replica placement experiment.',
            'Natural output lengths and device assignment differ across precisions; no cross-precision executor-speed causal claim.',
        ],
    }
    assert not torch.cuda.is_initialized()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'arms': len(rows), 'inputs': len(inputs),
                      'novel_candidates': 0, 'GPU_hours': 0}))


if __name__ == '__main__':
    main()
