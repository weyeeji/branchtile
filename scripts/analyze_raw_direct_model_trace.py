"""CPU interval accounting for profiled model graphs; no E2E/causal attribution."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics


def merged(events):
    spans = sorted((e['ts'], e['ts'] + e['dur']) for e in events)
    result = []
    for start, end in spans:
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def union_us(events):
    return sum(end-start for start, end in merged(events))


def category(name):
    name = name.lower()
    if 'kernel_mha(' in name:
        return 'attention_XQA'
    if 'nccl' in name:
        return 'NCCL'
    if 'per_token_group_quant_8bit_packed_register_kernel' in name:
        return 'standalone_token_quant_PDL_sensitive'
    if 'deep_gemm' in name or 'cutlass' in name or 'gemm' in name:
        return 'GEMM_and_reduce_named'
    if 'copy' in name or 'memcpy' in name or 'cache' in name:
        return 'copy_or_cache_named'
    if 'quant' in name:
        return 'other_quant_or_fused_quant_named'
    return 'other'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    r = json.loads(Path(a.summary).read_text())
    assert r['status'] == 'completed_trace_collection_pending_CUPTI_audit'
    output = Path(a.output)
    assert not output.exists()
    report = {'status': 'profile_interval_accounting_completed', 'runs': [], 'limitations': [
        'All durations are instrumented trace observations, not E2E latency or speed claims.',
        'Kernel sum includes overlapped execution and dependency waiting, particularly native PDL quant kernels.',
        'Interval union is device timeline coverage, not exclusive compute time or a causal removable bound.',
        'First-to-last graph kernel span is not host generation walltime and does not reconstruct the dependency critical path.',
        'Category names group symbols, not semantic operation ownership; normal framework KV/cache copies remain.',
        'Only graphs containing decode XQA are included; prefill, sampling outside the graph, host scheduling and setup are excluded.']}
    for run in r['runs']:
        for trace in run['traces']:
            path = Path(trace['trace'])
            events = json.loads(path.read_text())['traceEvents']
            launches = {e['args']['correlation']: e for e in events
                        if e.get('cat') == 'cuda_runtime' and e.get('name') == 'cudaGraphLaunch'}
            groups = collections.defaultdict(list)
            for e in events:
                if e.get('cat') == 'kernel' and e['args'].get('graph id', 0):
                    groups[e['args']['correlation']].append(e)
            graphs, aggregate = [], collections.defaultdict(lambda: {'count': 0, 'sum_us': 0.})
            for corr, ks in groups.items():
                if not any('kernel_mha(' in e['name'] for e in ks):
                    continue
                cats = collections.defaultdict(list)
                for e in ks:
                    cats[category(e['name'])].append(e)
                    aggregate[e['name']]['count'] += 1
                    aggregate[e['name']]['sum_us'] += e['dur']
                attn = cats['attention_XQA']
                other = [e for e in ks if category(e['name']) != 'attention_XQA']
                quant = cats['standalone_token_quant_PDL_sensitive']
                span = max(e['ts'] + e['dur'] for e in ks) - min(e['ts'] for e in ks)
                row = {'correlation': corr, 'kernel_count': len(ks), 'XQA_count': len(attn),
                       'graph_kernel_span_us': span, 'all_kernel_sum_us': sum(e['dur'] for e in ks),
                       'all_kernel_interval_union_us': union_us(ks),
                       'attention_interval_union_us': union_us(attn),
                       'other_interval_union_us': union_us(other),
                       'attention_other_overlap_us': union_us(attn) + union_us(other) - union_us(ks),
                       'attention_quant_overlap_us': union_us(attn) + union_us(quant) - union_us(attn + quant),
                       'uncovered_inside_graph_kernel_span_us': span - union_us(ks),
                       'cudaGraphLaunch_CPU_duration_us': launches[corr]['dur'],
                       'categories': {k: {'count': len(v), 'sum_us': sum(e['dur'] for e in v), 'union_us': union_us(v)} for k, v in cats.items()}}
                graphs.append(row)
            medians = {k: statistics.median(g[k] for g in graphs) for k in graphs[0] if k not in ('correlation', 'categories')}
            catmedians = {k: {metric: statistics.median(g['categories'].get(k, {}).get(metric, 0) for g in graphs)
                              for metric in ('count', 'sum_us', 'union_us')}
                          for k in set().union(*(g['categories'] for g in graphs))}
            report['runs'].append({'method': run['method'], 'rank': trace['rank'], 'trace': str(path),
                'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'graph_count': len(graphs),
                'medians_per_graph': medians, 'category_medians_per_graph': catmedians, 'graphs': graphs,
                'top_kernel_names_by_sum': [{'name': k, **v} for k, v in sorted(aggregate.items(), key=lambda x: -x[1]['sum_us'])[:20]]})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'traces': len(report['runs'])}))


if __name__ == '__main__':
    main()
