"""CPU-only native model opportunity audit; no inferred candidate speedup."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_duration(events):
    total = 0.0
    left = right = None
    for start, stop in sorted((float(e['ts']), float(e['ts']) + float(e['dur'])) for e in events):
        if left is None:
            left, right = start, stop
        elif start <= right:
            right = max(right, stop)
        else:
            total += right - left
            left, right = start, stop
    return total + (right - left if left is not None else 0.0)


def classify(name):
    lower = name.lower()
    if 'nccl' in lower or 'allreduce' in lower or 'all_reduce' in lower:
        return 'tensor_parallel_communication'
    if 'xqa' in lower:
        return 'attention_xqa'
    if 'flashinfer::' in name and any(x in name for x in ('BatchDecodeWithPagedKVCache', 'BatchPrefillWithPagedKVCache', 'SingleDecodeWithKVCache', 'SinglePrefillWithKVCache', 'MergeStates', 'MergeState')):
        return 'attention_flashinfer_non_xqa_or_merge'
    if 'flash_fwd' in lower:
        return 'attention_flashattention'
    if 'reshape_and_cache' in lower:
        return 'kv_cache_write'
    if 'gemm' in lower or 'cublas' in lower:
        return 'matrix_multiply'
    return 'other'


def summarize_trace(path):
    raw = path.read_bytes()
    obj = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
    kernels = [e for e in obj['traceEvents'] if e.get('cat') == 'kernel' and e.get('dur', 0) > 0]
    assert kernels, path
    groups = collections.defaultdict(list)
    names = collections.Counter()
    for event in kernels:
        groups[classify(event['name'])].append(event)
        names[event['name']] += event['dur']
    total = sum(e['dur'] for e in kernels)
    attention = [e for e in kernels if classify(e['name']).startswith('attention_')]
    span = max(e['ts'] + e['dur'] for e in kernels) - min(e['ts'] for e in kernels)
    attention_us = sum(e['dur'] for e in attention)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'kernel_count': len(kernels), 'sum_kernel_us': total,
            'union_kernel_busy_us': union_duration(kernels), 'first_to_last_kernel_span_us': span,
            'recognized_attention_sum_us': attention_us,
            'recognized_attention_fraction_of_kernel_sum': attention_us / total,
            'recognized_attention_sum_over_timeline_span': attention_us / span,
            'groups': {k: {'count': len(v), 'sum_us': sum(e['dur'] for e in v),
                           'fraction_of_kernel_sum': sum(e['dur'] for e in v) / total}
                       for k, v in groups.items()},
            'all_kernel_names': [{'name': k, 'category': classify(k), 'sum_us': v} for k, v in names.most_common()],
            'classification_status': 'Name-based recognizable attention only; inspect all kernel names before declaring complete attribution.',
            'scope': 'Instrumented trace, not unprofiled wall time or causal speedup. No graph-event denominator from a different window is mixed in.'}


def summarize_run(path):
    data = json.loads(path.read_text())
    assert data['status'] == 'completed', (path, data['status'])
    rows = []
    for run in data['runs']:
        tokens = sum(len(a['tokens']) for a in run['answers'])
        assert tokens == run['total_output_tokens']
        wall = run['complete_generate_wall_seconds']
        assert wall > 0 and math.isclose(tokens / wall, run['output_tokens_per_generate_wall_second'], rel_tol=1e-12)
        assert math.isclose(run['anchor_wall_seconds'] + wall, run['cold_shared_document_total_wall_seconds'], rel_tol=1e-12)
        rows.append({'phase': run['phase'], 'profiled': run['profiled'],
                     'natural_eos_enabled': run['natural_eos_enabled'],
                     'generation_wall_seconds': wall, 'anchor_wall_seconds': run['anchor_wall_seconds'],
                     'cold_shared_document_total_wall_seconds': run['cold_shared_document_total_wall_seconds'],
                     'tokens_by_request': [len(a['tokens']) for a in run['answers']],
                     'finish_reason_by_request': [a['finish_reason'] for a in run['answers']],
                     'total_output_tokens': tokens, 'output_tokens_per_generate_wall_second': tokens / wall})
    assert [r['phase'] for r in rows[:2]] == ['fixed_0', 'fixed_1']
    equal = [a['tokens'] == b['tokens'] for a, b in zip(data['runs'][0]['answers'], data['runs'][1]['answers'])]
    assert equal == data['fixed_AA_tokens_equal']
    apc_equal = [a['num_cached_tokens'] == b['num_cached_tokens'] for a, b in zip(data['runs'][0]['answers'], data['runs'][1]['answers'])]
    assert apc_equal == data['fixed_AA_APC_hits_equal']
    step_rows = {}
    for meta in data['fixed_AA_trajectory_files']:
        graph_path = ROOT / meta['path']
        assert sha(graph_path) == meta['sha256']
        graph = json.loads(graph_path.read_text())
        steps = [(r['batch'], r['questions'], r['positions']) for r in graph['replays']]
        assert len(steps) == meta['steps']
        step_rows[meta['rank'], meta['phase']] = steps
    steps_equal = [step_rows[r, 'fixed_0'] == step_rows[r, 'fixed_1'] for r in (0, 1)]
    assert steps_equal == data['fixed_AA_scheduled_steps_equal']
    assert all(equal + apc_equal + steps_equal) == data['fixed_AA_all_equal']
    fixed = [r['generation_wall_seconds'] for r in rows[:2]]
    trace_rows = []
    for meta in data['trace_files']:
        trace = ROOT / meta['path']
        assert sha(trace) == meta['sha256'], trace
        if trace.name.endswith('.json.gz') or trace.suffix == '.json':
            trace_rows.append(summarize_trace(trace))
    event_rows = []
    for rank in data['profile_graph_events']:
        times = [e['model_graph_cuda_us'] for e in rank['events']]
        assert times and all(t > 0 for t in times)
        event_rows.append({'rank': rank['rank'], 'count': len(times),
                           'median_instrumented_model_graph_us': statistics.median(times),
                           'sum_instrumented_model_graph_us': sum(times),
                           'batch_histogram': dict(collections.Counter(e['batch'] for e in rank['events'])),
                           'scope': rank['scope']})
    return {'summary': str(path.relative_to(ROOT)), 'sha256': sha(path), 'document_id': data['document_id'],
            'method': data['method'], 'B': data['B'], 'runs': rows, 'fixed_AA_tokens_equal': equal,
            'fixed_AA_APC_hits_equal': apc_equal, 'fixed_AA_scheduled_steps_equal': steps_equal,
            'fixed_AA_max_over_min_wall_ratio': max(fixed) / min(fixed),
            'trace_rows': trace_rows, 'instrumented_graph_events': event_rows,
            'native_metadata_before': [r['actual_metadata'] for r in data['native_receipt_before']],
            'limitations': data['limitations']}


def historical_evidence():
    path = ROOT / 'results/joint_real48_replay_v1/trace_attribution.json'
    data = json.loads(path.read_text())
    rows = []
    for row in data['ranks']:
        trace = ROOT / row['trace']
        assert sha(trace) == row['trace_file_sha256']
        attn = sum(row['duration_us'].values())
        assert math.isclose(attn, row['attention_with_required_copy_us'], rel_tol=1e-12)
        denominators = {'matched_model_kernel_sum': row['matched_four_model_graph_kernel_sum_us'],
                        'all_trace_kernel_sum': row['kernel_sum_us'],
                        'instrumented_model_graph_events': row['profiled_model_graph_cuda_events_us']}
        rows.append({'rank': row['rank'], 'snapshot': row['snapshot'],
                     'fractions': {key: attn / value for key, value in denominators.items()}})
    return {'source': str(path.relative_to(ROOT)), 'sha256': sha(path), 'rows': rows,
            'scope': 'Historical independent sparse32K B3 only; not dense native XQA and not whole engine/request wall time.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Directory containing completed new opportunity summary files')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = sorted((ROOT / args.root).rglob('summary.json')) if args.root else []
    rows = [summarize_run(path) for path in paths]
    report = {'schema': 1, 'script_sha256': sha(Path(__file__)), 'historical_sparse_evidence': historical_evidence(),
              'new_native_runs': rows,
              'interpretation': ['No candidate is executed by this experiment; it establishes a real baseline and opportunity attribution.',
                                 'Attention kernel fraction is not an observed end-to-end speedup or rigorous removable critical-path bound.',
                                 'Cold total includes the one shared-document anchor and group completion; startup is separately recorded.',
                                 'Do not use cross-precision natural wall ratios as same-work acceleration.',
                                 'Two AA repeats quantify observed drift only; not sufficient for confidence intervals or publication claims.']}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'completed_native_runs': len(rows), 'historical_rows': len(report['historical_sparse_evidence']['rows'])}))


if __name__ == '__main__':
    main()
