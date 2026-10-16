"""CPU-only audit of actual native B3/B8 MoE trace intervals, not speedup."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

from summarize_native_opportunity import union_duration

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(meta, run):
    path = ROOT / meta['path']
    assert sha(path) == meta['sha256']
    obj = json.loads(gzip.decompress(path.read_bytes()))
    kernels = sorted((e for e in obj['traceEvents'] if e.get('cat') == 'kernel'), key=lambda e: e['ts'])
    topks = [i for i, e in enumerate(kernels) if 'topkGating<' in e['name']]
    assert len(topks) == 32 * 48
    groups = collections.defaultdict(list)
    chains = []
    for i in topks:
        start = i - 1
        if 'cublasLt::splitKreduce_kernel' in kernels[start]['name']:
            groups['gate_splitk_reduce'].append(kernels[start])
            start -= 1
        assert 'cutlass_80_wmma_tensorop_bf16' in kernels[start]['name']
        groups['gate_GEMM'].append(kernels[start])
        end = i + 1
        while '_fwd_kernel_ep_gather' not in kernels[end]['name']:
            assert 'topkGating<' not in kernels[end]['name']
            end += 1
        chain = kernels[start:end + 1]
        key = lambda e: (e['args']['correlation'], e['args']['graph id'], e['args']['stream'])
        assert len({key(e) for e in chain}) == 1
        chains.append({'ts': chain[0]['ts'], 'dur': max(e['ts'] + e['dur'] for e in chain) - chain[0]['ts']})
        groups['whole_MoE_chain_kernels'].extend(chain)
    tests = {
        'router_topk': lambda n: 'topkGating<' in n,
        'MoE_grouped_GEMM': lambda n: 'deep_gemm::sm120_fp8_fp4_gemm' in n and '(deep_gemm::GemmType)1' in n,
        'MoE_scatter_gather_count': lambda n: n.startswith(('_fwd_kernel_ep_scatter_', '_fwd_kernel_ep_gather', '_count_expert_num_tokens')),
        'MoE_silu_quant': lambda n: n.startswith('_silu_mul_quant_fp8_packed_kernel'),
        'attention_XQA': lambda n: n.startswith('kernel_mha('),
        'PDL_packed_quant': lambda n: 'per_token_group_quant_8bit_packed_register_kernel' in n,
        'NCCL_allreduce': lambda n: 'ncclDevKernel_AllReduce' in n,
    }
    for label, pred in tests.items():
        groups[label] = [e for e in kernels if pred(e['name'])]
    assert len(groups['MoE_grouped_GEMM']) == 32 * 96
    assert len(groups['attention_XQA']) == 32 * 48
    span = max(e['ts'] + e['dur'] for e in kernels) - kernels[0]['ts']
    all_busy = union_duration(kernels)
    costs = {}
    for label, events in groups.items():
        selected_ids = {id(e) for e in events}
        other = [e for e in kernels if id(e) not in selected_ids]
        selected_union = union_duration(events)
        # Timeline occupied only by this category, not a removable causal bound.
        exclusive = all_busy - union_duration(other)
        costs[label] = {
            'count_per_step': len(events) / 32,
            'kernel_sum_us_per_step': sum(e['dur'] for e in events) / 32,
            'interval_union_us_per_step': selected_union / 32,
            'exclusive_interval_us_per_step': exclusive / 32,
            'union_fraction_of_trace_span': selected_union / span,
            'names': sorted({e['name'] for e in events}),
        }
    return {'trace': meta['path'], 'sha256': meta['sha256'], 'document_id': run['document_id'],
            'method': run['method'], 'B': run['B'], 'steps': 32,
            'gate_attribution': 'Each actual topk predecessor is BF16 GEMM, optionally followed by cuBLAS split-K reduce; same stream/graph/correlation through expert gather.',
            'mean_trace_span_us_per_step': span / 32,
            'MoE_gate_through_gather_span_us_per_step': sum(e['dur'] for e in chains) / 32,
            'MoE_chain_interkernel_gap_us_per_step': (sum(e['dur'] for e in chains) - union_duration(groups['whole_MoE_chain_kernels'])) / 32,
            'costs': costs}


def main():
    sources = {}; rows = []
    for relative in ('results/native_opportunity_wave26/analysis.json', 'results/native_opportunity_b8_v1/analysis.json'):
        path = ROOT / relative
        sources[relative] = sha(path)
        data = json.loads(path.read_text())
        for run in data['new_native_runs']:
            for meta in run['trace_rows']:
                rows.append(audit(meta, run))
    result = {'schema': 1, 'status': 'CPU_only_existing_trace_audit', 'script_sha256': sha(Path(__file__)),
              'source_sha256': sources, 'trace_rows': rows,
              'limitations': ['Profiled fixed B3/B8 graph window only, not natural B1/B2 tail or production wall latency.',
                              'Interval union removes double counting but neither union, exclusive occupancy, nor kernel sum is a causal removable latency bound.',
                              'PDL may charge producer waiting to a consumer kernel; no independent quantization savings inferred.',
                              'MoE whole-chain includes router, quantization, counts, scatter, packing, grouped GEMMs, activation and gather, excludes preceding RMSNorm and following NCCL.',
                              'No claim of a new mechanism, candidate qualification, GPU experiment, or measured application acceleration.']}
    target = ROOT / 'results/platform/native_moe_opportunity_20261017.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(target.relative_to(ROOT)), 'traces': len(rows), 'GPU_run': False}))
    for row in rows:
        c = row['costs']
        print(row['B'], row['method'], row['document_id'],
              {k: round(c[k]['kernel_sum_us_per_step'], 2) for k in ('gate_GEMM', 'gate_splitk_reduce', 'MoE_grouped_GEMM', 'MoE_scatter_gather_count', 'whole_MoE_chain_kernels') if k in c},
              'chain_span', round(row['MoE_gate_through_gather_span_us_per_step'], 2),
              'trace_step_span', round(row['mean_trace_span_us_per_step'], 2))


if __name__ == '__main__':
    main()
