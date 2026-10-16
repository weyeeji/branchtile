"""O-only deployment contract versus diagnostic LSE contracts on real captures.

Same source, native TC=True fixed in advance, fixed owner C128/N64; no oracle.
This remains an operator diagnostic, not an engine or request speedup.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--budget', type=int, default=4096)
    p.add_argument('--blocks', type=int, default=30)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = Path(args.output)
    assert not out.exists()
    import numpy as np
    import torch
    from compare_attention import load_capture, graph_of, elapsed
    from branchtile.attention import make_plan, reference
    from branchtile.attention_owner_tasks import Attention
    from flashinfer_baseline import NativeSparseDecode
    torch.set_num_threads(4)
    q, k, v, owners, b, g, provenance = load_capture(args.capture, args.layer, args.budget, args.step, 2)
    ref, ref_lse = reference(q, k, v, owners, b, g)
    supports = [[torch.where(owners[h] & (1 << r))[0] for r in range(b)] for h in range(k.shape[0])]
    native = NativeSparseDecode(q, k, v, supports, use_tensor_cores=True, normalize_lse=True)
    owner = Attention(make_plan(owners, b, g, q.device), chunk_tokens=128, block_n=64)

    def native_o():
        native.wrapper.run(native.q, native.kv, out=native.out, return_lse=False)
        return native.out.view(q.shape)

    def native_raw():
        native.wrapper.run(native.q, native.kv, out=native.out, lse=native.lse, return_lse=True)
        return native.out.view(q.shape)

    methods = {'native_O_only': native_o, 'native_O_log2LSE': native_raw,
               'native_O_lnLSE': lambda: native.run()[0],
               'owner_O_with_internal_lnLSE': lambda: owner(q, k, v)[0]}
    checks = {}
    for name, fn in methods.items():
        value = fn()
        torch.testing.assert_close(value.float(), ref.float(), atol=.006, rtol=.015)
        checks[name] = {'max_O_abs_error': float((value.double()-ref).abs().max()), 'passed': True}
    native_raw()
    torch.testing.assert_close(native.lse.view(ref_lse.shape)*math.log(2), ref_lse.float(), atol=.0003, rtol=.0001)
    _, native_ln = native.run()
    torch.testing.assert_close(native_ln, ref_lse.float(), atol=.0003, rtol=.0001)
    actual, lse = owner(q, k, v)
    torch.testing.assert_close(lse, ref_lse.float(), atol=.0003, rtol=.0001)
    flush = torch.zeros(max(256*1024*1024, 2*torch.cuda.get_device_properties(0).L2_cache_size), device='cuda', dtype=torch.uint8)
    rows, summaries = [], {}
    rng = random.Random(20261016)
    for cache, reps in [('hot', 20), ('cold', 1)]:
        graphs = {name: graph_of(fn, reps) for name, fn in methods.items()}
        graphs['native_O_only_AA'] = graph_of(native_o, reps)
        for name, graph in graphs.items():
            if cache == 'cold':
                flush.add_(1)
            graph.replay()
            actual = owner.output if name.startswith('owner_') else native.out.view(q.shape)
            torch.testing.assert_close(actual.float(), ref.float(), atol=.006, rtol=.015)
            checks[cache+'_'+name] = {'graph_O_passed': True}
        for block in range(args.blocks):
            order = list(graphs)
            rng.shuffle(order)
            for direction, ordered in [('random', order), ('reverse', list(reversed(order)))]:
                rows.append({'cache': cache, 'block': block, 'direction': direction,
                             'us': {name: elapsed(graphs[name], reps, flush if cache == 'cold' else None) for name in ordered}})
        selected = [r for r in rows if r['cache'] == cache]
        stats = {'median_us': {name: statistics.median(r['us'][name] for r in selected) for name in graphs}, 'paired_ratios': {}}
        draws = np.random.default_rng(71).integers(0, args.blocks, (10000, args.blocks))
        for numerator, denominator in [('native_O_only', 'owner_O_with_internal_lnLSE'),
                                       ('native_O_lnLSE', 'owner_O_with_internal_lnLSE'),
                                       ('native_O_lnLSE', 'native_O_only'),
                                       ('native_O_only', 'native_O_only_AA')]:
            ratios = np.array([[r['us'][numerator]/r['us'][denominator] for r in selected if r['block'] == block] for block in range(args.blocks)])
            ci = np.quantile(np.median(ratios[draws], axis=(1,2)), [.025,.975])
            stats['paired_ratios'][numerator+'_over_'+denominator] = {'median': float(np.median(ratios)), 'bootstrap95': ci.tolist()}
        summaries[cache] = stats
    sources = ['scripts/bench_owner_output_contract.py', 'src/flashinfer_baseline.py',
               'src/branchtile/attention_owner_tasks.py', 'src/branchtile/triton_kernel_owner_tasks.py']
    report = {'status': 'completed', 'args': vars(args), 'provenance': provenance, 'checks': checks,
              'flush_allocated_bytes': flush.numel()*flush.element_size(),
              'source_sha256': {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
              'summary': summaries, 'measurements': rows, 'uid': os.geteuid(), 'python': sys.executable,
              'scope': 'O-only application output contract diagnostic; native fixed TC=True, owner fixed C128/N64; no preparation in graph times.',
              'limitations': ['Hot fixed snapshot, partial heads, correlated development inputs, not whole-model execution.',
                              'Owner still computes its internal LSE; no claimed optimal O-only owner kernel.',
                              'Same page-one sparse native adapter, not official dense XQA deployment.',
                              'Cold event quantization retained; confidence intervals do not include systematic timing error.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'summary': summaries}), flush=True)


if __name__ == '__main__':
    main()
