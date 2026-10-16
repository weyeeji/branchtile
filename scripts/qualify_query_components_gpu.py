"""Bounded three-arm shared-prefix-only qualification; root assigns GPU.

Native XQA O-only, same-shape BF16-QK control, two-component FP8-QK.
All prefix tokens participate; private tails and engine integration are excluded.
Compact CPU artifacts are not claimed to be free online preparation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--capture', help='One actual layer*.pt compact capture; omitted uses synthetic data')
    p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--block-m', type=int, choices=(32, 64), default=64)
    p.add_argument('--blocks', type=int, default=20)
    p.add_argument('--replays', type=int, default=10)
    p.add_argument('--cold', action='store_true')
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = Path(args.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'args': vars(args), 'scope': __doc__, 'uid': os.geteuid(),
              'python': sys.executable, 'O_tolerance': {'atol': .006, 'rtol': .015},
              'QK_tolerance': {'atol': .0001, 'rtol': .0001},
              'checks': [], 'timing_blocks': []}
    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    started = time.perf_counter()
    try:
        import torch
        import triton
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        from src.branchtile.triton_kernel_query_components import SharedPrefixAttention, qk_probe
        torch.set_num_threads(2)
        torch.manual_seed(20261017)
        assert torch.cuda.get_device_capability() == (12, 0)
        if args.capture:
            path = Path(args.capture)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            report['capture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            q_native = payload['q'].cuda()
            kv = payload['prefix_kv']
            assert kv.shape[0] == 2 and kv.shape[2:] == (2, 128)
            if kv.dtype == torch.uint8:
                kv = kv.view(torch.float8_e4m3fn)
            assert kv.dtype == torch.float8_e4m3fn
            k = kv[0].permute(1, 0, 2).contiguous().cuda()
            v = kv[1].permute(1, 0, 2).contiguous().cuda()
        else:
            assert args.tokens > 0 and args.tokens % 16 == 0
            q_native = torch.randn((args.batch, 16, 128), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((2, args.tokens, 128), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            v = torch.randn_like(k, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        batch, qheads, dim = q_native.shape
        heads, tokens, _ = k.shape
        assert q_native.dtype == torch.bfloat16 and (heads, qheads, dim) == (2, 16, 128) and tokens % 16 == 0
        q = q_native.reshape(batch, heads, 8, dim).permute(1, 0, 2, 3).reshape(heads, batch * 8, dim).contiguous()
        report['shape'] = {'batch': batch, 'heads': heads, 'GQA': 8, 'D': dim, 'prefix_tokens': tokens,
                           'kv_scale': 1., 'prefix_only': True}
        page = 16
        # Each request aliases precisely the same physical prefix pages.
        kp = k.reshape(heads, tokens // page, page, dim).permute(1, 0, 2, 3).contiguous()
        vp = v.reshape(heads, tokens // page, page, dim).permute(1, 0, 2, 3).contiguous()
        table = torch.arange(tokens // page, device='cuda', dtype=torch.int32).repeat(batch, 1)
        lengths = torch.full((batch,), tokens, device='cuda', dtype=torch.uint32)
        workspace = torch.zeros(256 * 1024 * 1024, device='cuda', dtype=torch.uint8)
        native_out = torch.empty_like(q_native)
        def native():
            xqa_batch_decode_with_kv_cache(q_native, (kp, vp), workspace, table, lengths,
                                          max_seq_len=tokens, bmm1_scale=dim ** -.5,
                                          bmm2_scale=1., kv_layout='HND', out=native_out)
        control = SharedPrefixAttention(q, k, v, False, block_m=args.block_m)
        candidate = SharedPrefixAttention(q, k, v, True, block_m=args.block_m)
        methods = {'native_xqa_O_only': native, 'BF16_QK_control': control, 'two_FP8_QK': candidate}
        def err(value, expected, atol, rtol):
            vv = value.double(); diff = vv - expected
            den = expected.norm()
            return {'passed': bool(torch.isclose(vv, expected, atol=atol, rtol=rtol).all()),
                    'finite': bool(torch.isfinite(vv).all()), 'max_abs': float(diff.abs().max()),
                    'relative_l2': float(diff.norm() / den.clamp_min(1e-30)),
                    'max_row_l2': float(diff.norm(dim=-1).max())}
        # Deterministic span-wide sample, all 128 contraction dimensions.
        sample = torch.linspace(0, tokens - 1, min(tokens, 1024), device='cuda').long().unique()
        ks = k.view(torch.uint8)[:, sample].contiguous().view(torch.float8_e4m3fn)
        reference_scores = torch.matmul(q.double(), ks.double().transpose(-1, -2)) * dim ** -.5
        for flag in (False, True):
            scores = torch.empty(reference_scores.shape, device='cuda', dtype=torch.float32)
            qk_probe[(heads, triton.cdiv(batch * 8, args.block_m), triton.cdiv(len(sample), 64))](
                q, ks, scores, ROWS=batch * 8, TOKENS=len(sample), M=args.block_m, N=64, D=dim,
                COMPONENTS=flag, num_warps=4, num_stages=1)
            report['checks'].append({'stage': 'QK_sample', 'components': flag, 'sample_tokens': len(sample),
                                     **err(scores, reference_scores, .0001, .0001)})
        # Complete prefix FP64 reference, not captured native O (which includes tails).
        reference = torch.empty_like(q, dtype=torch.float64)
        for h in range(heads):
            scores = (q[h].double() @ k[h].double().T) * dim ** -.5
            reference[h] = torch.softmax(scores, -1) @ v[h].double()
        for name, fn in methods.items():
            fn()
            value = native_out.reshape(batch, heads, 8, dim).permute(1, 0, 2, 3).reshape_as(q) if name.startswith('native') else fn.output
            report['checks'].append({'stage': 'complete_prefix_O', 'method': name,
                                     **err(value, reference, .006, .015)})
        torch.cuda.synchronize()
        report['setup_and_qualification_seconds'] = time.perf_counter() - started
        report['resources'] = {name: {'workspace_bytes': arm.workspace_bytes,
                                      'producer_n_regs': arm.last_producer.n_regs,
                                      'producer_n_spills': arm.last_producer.n_spills,
                                      'producer_shared': arm.last_producer.metadata.shared}
                               for name, arm in (('control', control), ('candidate', candidate))}
        report['source_sha256'] = {x: hashlib.sha256((ROOT / x).read_bytes()).hexdigest() for x in
                                   ('scripts/qualify_query_components_gpu.py', 'src/branchtile/triton_kernel_query_components.py')}
        report['all_numerical_checks_passed'] = all(x['passed'] for x in report['checks'])
        save()
        if not report['all_numerical_checks_passed']:
            report['status'] = 'numerical_gate_failed_no_timing'; save(); return
        graphs = {}
        for name, fn in methods.items():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
        cold = torch.empty(256 * 1024 * 1024, device='cuda', dtype=torch.uint8) if args.cold else None
        rng = random.Random(20261017)
        for block in range(args.blocks):
            order = list(methods); rng.shuffle(order)
            for name in order:
                if cold is not None:
                    cold.zero_(); torch.cuda.synchronize()
                begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.replays):
                    graphs[name].replay()
                end.record(); end.synchronize()
                report['timing_blocks'].append({'block': block, 'method': name,
                                                'cuda_us_per_call': begin.elapsed_time(end) * 1000 / args.replays,
                                                'replays': args.replays,
                                                'cache': 'flush_before_block_then_hot_replays' if args.cold else 'hot'})
            save()
        report['status'] = 'qualified_and_benchmarked_prefix_only'
        report['total_seconds'] = time.perf_counter() - started
        save()
        print(json.dumps({'status': report['status'], 'checks': len(report['checks']), 'blocks': args.blocks}), flush=True)
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = str(exc); report['traceback'] = traceback.format_exc(); save(); raise


if __name__ == '__main__':
    main()
