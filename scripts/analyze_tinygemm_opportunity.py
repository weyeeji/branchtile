"""Recompute router/MoE cost evidence from existing traces; never touches CUDA."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt_path = ROOT / 'results/joint_real48_replay_v1/trace_attribution.json'
    receipt = json.loads(receipt_path.read_text())
    sources = {str(receipt_path.relative_to(ROOT)): sha(receipt_path)}
    rows = []
    for rank in receipt['ranks']:
        path = ROOT / rank['trace']; sources[rank['trace']] = sha(path)
        assert sha(path) == rank['trace_file_sha256']
        events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
        graph_ids = {(g['correlation'], g['graph_id']) for g in rank['graph_launches']}
        kernels = [e for e in events if e.get('cat') == 'kernel' and
                   (e['args'].get('correlation'), e['args'].get('graph id')) in graph_ids]
        assert len(kernels) == 4 * 1348
        costs = {}
        tests = {
            'inferred_gate_BF16_GEMM': lambda n: 'cutlass_80_wmma_tensorop_bf16' in n,
            'router_topk': lambda n: 'topkGating<' in n,
            'MoE_grouped_GEMM': lambda n: 'deep_gemm::sm120_fp8_fp4_gemm' in n and '(deep_gemm::GemmType)1' in n,
            'MoE_scatter_gather': lambda n: 'scatter' in n.lower() or 'gather' in n.lower(),
            'NCCL_allreduce': lambda n: 'ncclDevKernel_AllReduce' in n,
        }
        for label, pred in tests.items():
            selected = [e for e in kernels if pred(e['name'])]
            costs[label] = {'count_per_step': len(selected) / 4,
                           'kernel_sum_us_per_step': sum(e['dur'] for e in selected) / 4}
        assert costs['inferred_gate_BF16_GEMM']['count_per_step'] == 48
        assert costs['router_topk']['count_per_step'] == 48
        assert costs['MoE_grouped_GEMM']['count_per_step'] == 96
        adjacency = []; first_collective = []; other_collectives = []
        for key in sorted(graph_ids):
            group = sorted([e for e in kernels if (e['args']['correlation'], e['args']['graph id']) == key], key=lambda e: e['ts'])
            for i, e in enumerate(group):
                if tests['inferred_gate_BF16_GEMM'](e['name']):
                    adjacency.append(tests['router_topk'](group[i + 1]['name']))
            collective = [e for e in group if tests['NCCL_allreduce'](e['name'])]
            assert len(collective) == 97
            first_collective.append(collective[0]['dur'])
            other_collectives.append(sum(e['dur'] for e in collective[1:]))
        assert len(adjacency) == 192 and all(adjacency)
        rows.append({'trace': rank['trace'], 'rank': rank['rank'], 'profiled_steps': 4,
            'profiled_model_event_us_per_step': rank['profiled_model_graph_cuda_events_us'] / 4,
            'attention_including_matched_copy_us_per_step': rank['attention_with_required_copy_per_model_step_us'],
            'costs': costs, 'all_192_BF16_GEMMs_immediately_followed_by_actual_topk_kernel': all(adjacency),
            'mean_first_collective_us': sum(first_collective) / 4,
            'mean_remaining_96_collectives_sum_us': sum(other_collectives) / 4})
    natural_path = ROOT / 'results/native_precision_natural_development_v1/analysis.json'
    sources[str(natural_path.relative_to(ROOT))] = sha(natural_path)
    natural = json.loads(natural_path.read_text()); scenarios = []
    # Deliberately conditional: a B3 sparse trace cannot establish B1/B2 FP8 cost.
    max_gate_us = max(r['costs']['inferred_gate_BF16_GEMM']['kernel_sum_us_per_step'] for r in rows)
    for row in natural['rows']:
        if row['arm'] not in ('original_fp8', 'new_fp8'):
            continue
        run = row['runs'][0]
        graph_path = ROOT / Path(row['source']).parent / 'rank0' / (run['phase'] + '_graph.json')
        sources[str(graph_path.relative_to(ROOT))] = sha(graph_path)
        graph = json.loads(graph_path.read_text())
        steps = graph['replays']
        count = len(steps); cold = run['cold_anchor_plus_generate_seconds']
        scenarios.append({'arm': row['arm'], 'natural_source': row['source'], 'repeat': 0,
            'graph_receipt': str(graph_path.relative_to(ROOT)), 'actual_graph_steps': count,
            'B_counts': dict(collections.Counter(s['batch'] for s in steps)),
            'cold_anchor_plus_generate_seconds': cold, 'generate_seconds': run['generate_seconds'],
            'required_us_per_step_for_1_05x_cold_speedup': cold * (1 - 1 / 1.05) * 1e6 / count,
            'required_us_per_step_for_5pct_cold_latency_reduction': cold * .05 * 1e6 / count,
            'conditional_eliminate_whole_B3_gate_every_step_cold_latency_fraction': max_gate_us * count / 1e6 / cold,
            'conditional_assumptions': 'All B1/B2/B3 gate costs equal largest observed B3 sparse trace and disappear entirely, with no extra cost or trajectory change. Neither assumption has been verified on full native FP8.'})
    result = {'schema': 1, 'status': 'CPU_existing_trace_review',
        'script_sha256': sha(Path(__file__)), 'source_sha256': sources, 'trace_rows': rows,
        'natural_conditional_scenarios': scenarios,
        'interpretation': ['Gate attribution is supported by 48 BF16 GEMMs each immediately followed by topk; actual X/W and output receipts remain a pending qualification.',
            'Kernel duration sums are not critical-path latency. First all-reduce includes rank-arrival wait; it is not a removable fabric-cost estimate.',
            'These are 32K sparse-E0 BF16-KV instrumented graph traces, not a full native FP8 profile.',
            '1348 graph GPU kernels do not imply 1348 Python launches. CUPTI graph-launch runtime durations are not production host-overhead measurements.',
            'No new research mechanism is identified. Existing FlashInfer TinyGEMM integration is a mature baseline coverage check.',
            'Cold natural denominator includes anchor and generate, excludes initialization/warmup; no complete application speedup follows from this gate-only scenario.']}
    target = ROOT / 'results/platform/tinygemm_opportunity_20261016.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(target), 'ranks': len(rows), 'GPU_run': False}))


if __name__ == '__main__':
    main()
