"""Native-capture subset v3 XQA pair-padding screen with explicit natural membership.

Existing kernels only: no claim of a new algorithm. Full shared prefix only;
independent private suffixes are deliberately not represented by this adapter.
Keep native per-row split count using public sm_count, test bit equality first,
and charge Q packing + O restoration in complete graph timing. Root owns GPUs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--dtype', choices=['bf16', 'fp8'], required=True)
    p.add_argument('--capture', required=True)
    p.add_argument('--membership-trace', required=True)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--tokens', type=int, default=120528)
    p.add_argument('--blocks', type=int, default=20)
    p.add_argument('--replays', type=int, default=10)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = Path(a.output); assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'scope': __doc__, 'args': vars(a),
              'checks': [], 'timing_blocks': [], 'uid': os.geteuid(), 'python': sys.executable}
    def save(): out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        import flashinfer
        torch.set_num_threads(2); torch.manual_seed(20261017)
        assert torch.cuda.get_device_capability() == (12, 0)
        dtype = torch.bfloat16 if a.dtype == 'bf16' else torch.float8_e4m3fn
        payload = None
        if a.capture:
            path = Path(a.capture)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            report['capture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            trace_path = Path(a.membership_trace)
            trace = json.loads(trace_path.read_text())
            assert trace['replays'][0]['questions'] == list(range(payload['q'].shape[0]))
            matches = [(i, row) for i, row in enumerate(trace['replays']) if row['batch'] == a.batch]
            assert matches, 'Requested batch not observed in natural membership trace'
            replay_index, replay = matches[0]
            subset_rows = replay['questions']
            assert len(subset_rows) == a.batch and len(set(subset_rows)) == a.batch
            assert all(0 <= row < payload['q'].shape[0] for row in subset_rows)
            q = payload['q'][subset_rows].contiguous().cuda()
            report['subset'] = {'source_rows': subset_rows, 'trace': str(trace_path),
                'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                'trace_replay_index': replay_index, 'trace_positions': replay['positions'],
                'scope': 'Captured initial B8 Q subset with naturally observed survivor membership. Q is NOT from the later natural step; prefix-only, no private tails.',
                'native_split_rule': 'Recomputed exactly as native XQA for this subset batch, not held at B8 value.'}
            kv = payload['prefix_kv']
            if kv.dtype == torch.uint8: kv = kv.view(dtype)
            assert kv.dtype == dtype and kv.shape[0] == 2 and kv.shape[2:] == (2, 128)
            tokens = kv.shape[1]; batch = q.shape[0]
            shape, stride = payload['original_cache_shape'], payload['original_cache_stride']
            source_table = payload['original_block_table'][subset_rows, :tokens // 16]
            assert source_table.min() >= 0 and source_table.max() < shape[0]
            table = source_table.to(device='cuda', dtype=torch.int32).contiguous()
            assert float(payload.get('k_scale', 1.)) == 1. and float(payload.get('v_scale', 1.)) == 1., 'This bounded test only supports unit scales'
            if dtype == torch.bfloat16:
                assert payload['original_cache_dtype'] == 'torch.bfloat16'
                assert payload.get('payload_cast_performed', False) is False
                cache = torch.empty_strided(shape, stride, device='cuda', dtype=dtype)
                cache.zero_()
                values = kv.reshape(2, tokens // 16, 16, 2, 128)
                compact = torch.cat((values[0].permute(0, 2, 1, 3),
                                     values[1].permute(0, 2, 1, 3)), dim=-1)
                cache[table[0].long()] = compact.cuda()
                recovered = cache[table[0].long()].cpu()
            else:
                assert payload['original_cache_dtype'] in ('torch.uint8', 'torch.float8_e4m3fn')
                cache = torch.empty_strided(shape, stride, device='cuda', dtype=torch.uint8)
                cache.zero_()
                values = kv.view(torch.uint8).reshape(2, tokens // 16, 16, 2, 128)
                compact = torch.cat((values[0].permute(0, 2, 1, 3),
                                     values[1].permute(0, 2, 1, 3)), dim=-1)
                cache[table[0].long()] = compact.cuda()
                recovered = cache[table[0].long()].cpu()
                cache = cache.view(dtype)
            assert torch.equal(recovered.view(torch.uint8), compact.view(torch.uint8))
            report['capture_ABI'] = {'source_table_dtype': str(source_table.dtype),
                'runtime_table_dtype': str(table.dtype), 'runtime_cache_dtype': str(cache.dtype),
                'source_cache_dtype': payload['original_cache_dtype'],
                'page_roundtrip_bytes_exact': True, 'unreferenced_pages_initialized_zero': True,
                'k_scale': float(payload.get('k_scale', 1.)), 'v_scale': float(payload.get('v_scale', 1.))}

        else:
            tokens, batch = a.tokens, a.batch
            assert tokens > 0 and tokens % 16 == 0
            pages = tokens // 16 + 19
            storage = torch.randn((pages, 16, 2, 256), device='cuda', dtype=torch.bfloat16).to(dtype)
            cache = storage.permute(0, 2, 1, 3)
            table = torch.randperm(pages, device='cuda')[:tokens // 16].int().repeat(batch, 1)
            q = torch.randn((batch, 16, 128), device='cuda', dtype=torch.bfloat16)
        assert batch in (2, 4, 6, 8) and q.shape == (batch, 16, 128)
        assert q.is_contiguous() and table.is_contiguous() and table.dtype == torch.int32
        assert bool((table == table[:1]).all())
        assert cache.stride() == (8192, 256, 512, 1)
        original_q, original_table = q.clone(), table.clone()
        groups = batch // 2
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        native_splits = min(max(1, sm // (batch * 2)), (tokens + 255) // 256)
        paired_sm = native_splits * groups * 2
        assert min(max(1, paired_sm // (groups * 2)), (tokens + 255) // 256) == native_splits
        kp, vp = cache[..., :128], cache[..., 128:]
        pair_table = table[::2].contiguous()
        lengths = torch.full((batch, 1), tokens, device='cuda', dtype=torch.uint32)
        pair_lengths = torch.full((groups, 1), tokens, device='cuda', dtype=torch.uint32)
        packed = torch.empty((groups, 1, 32, 128), device='cuda', dtype=torch.bfloat16)
        packed_outputs = {m: torch.empty_like(packed) for m in ('pair_fixed_split', 'pair_auto_split')}
        outputs = {m: torch.empty_like(q) for m in ('native', 'native_AA', 'pair_fixed_split', 'pair_auto_split')}
        workspaces = {m: torch.zeros(256 * 1024 * 1024, device='cuda', dtype=torch.uint8) for m in outputs}
        def pack():
            packed.view(groups, 2, 2, 8, 128).copy_(q.view(groups, 2, 2, 8, 128).permute(0, 2, 1, 3, 4))
        def restore(name):
            outputs[name].view(groups, 2, 2, 8, 128).copy_(packed_outputs[name].view(groups, 2, 2, 8, 128).permute(0, 2, 1, 3, 4))
        def call(name, paired=False, fixed=False):
            if paired: pack()
            w = workspaces[name]
            xqa(packed if paired else q.unsqueeze(1), kp, vp,
                pair_table if paired else table, pair_lengths if paired else lengths,
                packed_outputs[name] if paired else outputs[name].unsqueeze(1),
                w[8 * 1024 * 1024:], w[:8 * 1024 * 1024], 2, 16,
                q_scale=1., kv_scale=1., kv_layout='HND', sm_count=paired_sm if fixed else sm,
                enable_pdl=True)
            if paired: restore(name)
        methods = {'native': lambda: call('native'), 'native_AA': lambda: call('native_AA'),
                   'pair_fixed_split': lambda: call('pair_fixed_split', True, True),
                   'pair_auto_split': lambda: call('pair_auto_split', True, False)}
        report['shape'] = {'batch': batch, 'pair_groups': groups, 'tokens': tokens, 'kv_heads': 2,
            'original_GQA': 8, 'paired_GQA': 16, 'native_M': 16, 'paired_M': 16,
            'native_SM_parameter': sm, 'paired_fixed_SM_parameter': paired_sm,
            'native_splits': native_splits, 'paired_fixed_splits': native_splits,
            'paired_auto_splits': min(max(1, sm // (groups * 2)), (tokens + 255) // 256),
            'native_CTAs': native_splits * batch * 2, 'paired_fixed_CTAs': native_splits * groups * 2,
            'cache_stride': list(cache.stride()), 'original_capture_Q_stride': list(payload['q'].stride()) if payload else None,
            'Q_used_stride': list(q.stride()), 'full_shared_prefix_only': True,
            'timing_includes_Q_pack_and_O_restore': True}
        save()
        graphs = {}
        for name, fn in methods.items():
            for _ in range(3): fn()
            torch.cuda.synchronize(); graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name]): fn()
        # Confirm the public native adapter matches the production convenience entry.
        bridge_out = torch.empty_like(q); bridge_work = torch.zeros_like(workspaces['native'])
        xqa_batch_decode_with_kv_cache(q, (kp, vp), bridge_work, table, lengths.flatten(),
            max_seq_len=tokens, bmm1_scale=128 ** -.5, bmm2_scale=1., kv_layout='HND', out=bridge_out)
        graphs['native'].replay(); torch.cuda.synchronize()
        assert torch.equal(outputs['native'].view(torch.int16), bridge_out.view(torch.int16))
        report['production_entry_bitwise_match'] = True
        def compare(name):
            reference = outputs['native']; value = outputs[name]
            mismatch = reference.view(torch.int16) != value.view(torch.int16)
            return {'method': name, 'bitwise': not bool(mismatch.any()),
                'different_elements': int(mismatch.sum()), 'max_abs': float((value.float()-reference.float()).abs().max()),
                'lower_rows_different': int(mismatch[::2].sum()), 'upper_rows_different': int(mismatch[1::2].sum()),
                'different_by_row_head': mismatch.sum(-1).cpu().tolist(),
                'finite': bool(torch.isfinite(value).all())}
        # One captured real state plus independent Q states, pair swaps and page rebinding.
        for case in range(6):
            if case == 1: q.copy_(q.view(groups, 2, 16, 128).flip(1).reshape_as(q))
            elif case > 1: q.normal_()
            if case == 4:
                table.copy_(table.flip(1)); pair_table.copy_(table[::2])
            if case == 5:
                # Non-page-aligned shared boundary; no fake private-tail equivalence.
                lengths.fill_(tokens - 3); pair_lengths.fill_(tokens - 3)
            for name in methods: graphs[name].replay()
            torch.cuda.synchronize()
            if case == 0 and payload is not None:
                # Independent mathematical reference, CPU FP64. Never part of timed graphs.
                # Sample both members, both KV heads and boundary GQA heads.
                rows = sorted(set((0, 1, batch - 2, batch - 1)))
                heads = [0, 7, 8, 15]
                q_cpu = q.cpu().double()
                reference = []
                for row in rows:
                    for head in heads:
                        k = kv[0, :, head // 8].double()
                        v = kv[1, :, head // 8].double()
                        logits = torch.mv(k, q_cpu[row, head]) * (128 ** -.5)
                        ref = torch.matmul(torch.softmax(logits, 0), v)
                        errors = {}
                        for method in methods:
                            got = outputs[method][row, head].cpu().double()
                            errors[method] = {'max_abs': float((got-ref).abs().max()),
                                'relative_l2': float((got-ref).norm()/ref.norm().clamp_min(1e-30)),
                                'cosine': float(torch.nn.functional.cosine_similarity(got, ref, dim=0)),
                                'finite': bool(torch.isfinite(got).all())}
                        reference.append({'row': row, 'head': head, 'errors': errors})
                report['FP64_reference'] = {'scope': 'Full actual prefix KV; sampled original Q rows/heads; CPU double softmax and matmul; no private suffix',
                    'samples': reference, 'rounding': 'Mathematical reference does not emulate native BF16 probability or split casts.'}
                save()
            for name in ('native_AA', 'pair_fixed_split', 'pair_auto_split'):
                report['checks'].append({'case': case, **compare(name)})
            save(); assert report['checks'][-3]['bitwise'], 'Native AA is not bitwise stable'
        qualified = all(r['bitwise'] for r in report['checks'] if r['method'] == 'pair_fixed_split')
        report['paired_fixed_split_bitwise_qualified'] = qualified
        # Return to the actual captured Q/page ordering for bounded timing.
        q.copy_(original_q); table.copy_(original_table); pair_table.copy_(table[::2])
        lengths.fill_(tokens); pair_lengths.fill_(tokens)
        for graph in graphs.values(): graph.replay()
        torch.cuda.synchronize()
        rng = random.Random(20261017)
        for block in range(a.blocks):
            order = list(methods); rng.shuffle(order); values = {}
            for name in order:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(a.replays): graphs[name].replay()
                end.record(); end.synchronize()
                values[name] = begin.elapsed_time(end) * 1000 / a.replays
            report['timing_blocks'].append({'block': block, 'order': order, 'us': values}); save()
        report['median_us'] = {name: statistics.median(r['us'][name] for r in report['timing_blocks']) for name in methods}
        # CUPTI activity tracing (not hardware counters) exports the actual grid.
        # Keep this separate from all performance timing above.
        report['launch_profiles'] = {}
        for name in ('native', 'pair_fixed_split', 'pair_auto_split'):
            trace = out.with_name(out.stem + '_' + name + '_launch.json')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                graphs[name].replay(); torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            kernels = [{'name': e['name'], 'args': e.get('args', {})}
                       for e in events if e.get('cat') == 'kernel' and 'kernel_mha' in e.get('name', '')]
            assert kernels, ('Missing actual XQA kernel launch activity', name)
            expected_s = native_splits if name != 'pair_auto_split' else report['shape']['paired_auto_splits']
            expected_grid = [expected_s, 2, batch if name == 'native' else groups]
            for kernel in kernels:
                assert kernel['args'].get('grid') == expected_grid, (name, kernel, expected_grid)
            report['launch_profiles'][name] = {'trace': str(trace), 'kernels': kernels,
                                             'expected_grid_verified': expected_grid}
        report['status'] = 'bitwise_qualified_and_benchmarked_prefix_only' if qualified else 'bitwise_gate_failed_diagnostic_timing_only'
        source_paths = [Path(__file__), Path(flashinfer.__file__).parent / 'xqa.py',
            Path(flashinfer.__file__).parent / 'jit/xqa.py', Path(flashinfer.__file__).parent / 'data/csrc/xqa/mha.cu',
            Path(flashinfer.__file__).parent / 'data/csrc/xqa/defines.h']
        report['sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        report['limitations'] = ['No independent private tails, no model integration, no first-use/encoding cost amortization.',
            'Existing native GQA16 adapter is a diagnostic/control, not a novel kernel.',
            'auto-split arm is a negative control; changed outputs do not establish that either mathematical attention is wrong.',
            'Timing restores original captured Q and page ordering; synthetic jobs remain synthetic.',
            'No end-to-end performance or universal bitwise guarantee from bounded inputs.']
        save(); print(json.dumps({'status': report['status'], 'median_us': report['median_us']}))
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc()); save(); raise


if __name__ == '__main__': main()
