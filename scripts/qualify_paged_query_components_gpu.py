"""Bounded original-paged-layout shared-prefix qualification; root assigns GPU.

Native XQA O-only, same-shape BF16-QK control, two-component FP8-QK.
All prefix tokens participate; private tails and engine integration are excluded.
All timed arms use the same fused physical KV storage and Q tensor. Offline
capture reconstruction is initialization, not an online layout transformation.
FA2 shared-prefix is an additional mature same-page baseline. Tails excluded.
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
        import flashinfer
        from src.branchtile.triton_kernel_query_components import qk_probe
        from src.branchtile.triton_kernel_paged_query_components import PagedSharedPrefixAttention
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
        prefix_pages = tokens // page
        if args.capture:
            cache_shape, cache_stride = payload['original_cache_shape'], payload['original_cache_stride']
            table = payload['original_block_table'][:, :prefix_pages].to(device='cuda', dtype=torch.int32).contiguous()
        else:
            cache_shape = (prefix_pages + 19, heads, page, 2 * dim)
            cache_stride = (heads * page * 2 * dim, 2 * dim, heads * 2 * dim, 1)
            table = torch.randperm(cache_shape[0], device='cuda', dtype=torch.int64)[:prefix_pages].int().repeat(batch, 1)
        cache = torch.empty_strided(cache_shape, cache_stride, device='cuda', dtype=torch.uint8)
        assert bool((table == table[:1]).all())
        physical = table[0].long()
        compact_k = k.view(torch.uint8).reshape(heads, prefix_pages, page, dim).permute(1, 0, 2, 3)
        compact_v = v.view(torch.uint8).reshape(heads, prefix_pages, page, dim).permute(1, 0, 2, 3)
        cache[physical] = torch.cat((compact_k, compact_v), dim=-1)
        typed_cache = cache.view(torch.float8_e4m3fn)
        kp, vp = typed_cache[..., :dim], typed_cache[..., dim:]
        assert kp.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        report['paged_alias_contract'] = {'cache_shape': list(cache.shape), 'cache_stride': list(cache.stride()),
            'q_shape': list(q_native.shape), 'q_stride': list(q_native.stride()),
            'original_physical_page_ids_preserved': bool(args.capture),
            'all_timed_arms_same_cache_storage': True, 'candidate_Q_pack_or_KV_copy_in_call': False,
            'common_page_ids_sha256': hashlib.sha256(table[0].cpu().numpy().tobytes()).hexdigest()}
        lengths = torch.full((batch,), tokens, device='cuda', dtype=torch.uint32)
        workspace = torch.zeros(256 * 1024 * 1024, device='cuda', dtype=torch.uint8)
        native_out = torch.empty_like(q_native)
        def native():
            xqa_batch_decode_with_kv_cache(q_native, (kp, vp), workspace, table, lengths,
                                          max_seq_len=tokens, bmm1_scale=dim ** -.5,
                                          bmm2_scale=1., kv_layout='HND', out=native_out)
        control = PagedSharedPrefixAttention(q_native, cache, table[0], tokens, False, block_m=args.block_m)
        candidate = PagedSharedPrefixAttention(q_native, cache, table[0], tokens, True, block_m=args.block_m)
        shared_workspace = torch.empty_like(workspace)
        qptr = torch.tensor([0, batch], device='cuda', dtype=torch.int32)
        kptr = torch.tensor([0, prefix_pages], device='cuda', dtype=torch.int32)
        last = torch.tensor([page], device='cuda', dtype=torch.int32)
        shared_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(shared_workspace, kv_layout='HND', backend='fa2',
            use_cuda_graph=True, qo_indptr_buf=qptr, paged_kv_indptr_buf=kptr,
            paged_kv_indices_buf=table[0], paged_kv_last_page_len_buf=last)
        shared_wrapper.plan(qptr.cpu(), kptr.cpu(), table[0], last.cpu(), qheads, heads, dim, page,
                            causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, sm_scale=dim ** -.5)
        shared_out = torch.empty_like(q_native)
        shared_lse = torch.empty(q_native.shape[:2], device='cuda', dtype=torch.float32)
        def shared():
            shared_wrapper.run(q_native, (kp, vp), out=shared_out, lse=shared_lse, return_lse=True, k_scale=1., v_scale=1.)
        methods = {'native_xqa_O_only': native, 'FA2_shared_prefix': shared,
                   'BF16_QK_control': control, 'two_FP8_QK': candidate}
        outputs = {'native_xqa_O_only': native_out, 'FA2_shared_prefix': shared_out,
                   'BF16_QK_control': control.output, 'two_FP8_QK': candidate.output}
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
        reference_lse = torch.empty(q.shape[:-1], device='cuda', dtype=torch.float64)
        for h in range(heads):
            scores = (q[h].double() @ k[h].double().T) * dim ** -.5
            reference[h] = torch.softmax(scores, -1) @ v[h].double()
            reference_lse[h] = torch.logsumexp(scores, -1) / __import__('math').log(2)
        for name, fn in methods.items():
            fn()
            value = outputs[name].reshape(batch, heads, 8, dim).permute(1, 0, 2, 3).reshape_as(q)
            report['checks'].append({'stage': 'complete_prefix_O', 'method': name,
                                     **err(value, reference, .006, .015)})
            if name != 'native_xqa_O_only':
                lse = shared_lse if name == 'FA2_shared_prefix' else fn.lse
                logical_lse = lse.reshape(batch, heads, 8).permute(1, 0, 2).reshape_as(reference_lse)
                report['checks'].append({'stage': 'prefix_LSE_log2', 'method': name,
                                         **err(logical_lse, reference_lse, .0003, .0001)})
        torch.cuda.synchronize()
        report['setup_and_qualification_seconds'] = time.perf_counter() - started
        report['resources'] = {name: {'workspace_bytes': arm.workspace_bytes,
                                      'producer_n_regs': arm.last_producer.n_regs,
                                      'producer_n_spills': arm.last_producer.n_spills,
                                      'producer_shared': arm.last_producer.metadata.shared}
                               for name, arm in (('control', control), ('candidate', candidate))}
        report['source_sha256'] = {x: hashlib.sha256((ROOT / x).read_bytes()).hexdigest() for x in
                                   ('scripts/qualify_paged_query_components_gpu.py', 'src/branchtile/triton_kernel_paged_query_components.py',
                                    'src/branchtile/triton_kernel_query_components.py')}
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
