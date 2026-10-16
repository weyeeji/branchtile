"""Fixed S' coverage mode against ORIGINAL native and same-S' counterfactual.

B8 true BF16 captures only. No changed-default-bitwise claim and no shape oracle.
All timings include the complete prefix/continuation/reduction CUDA graph.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_xqa_raw_pair import paired_ratio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', required=True)
    p.add_argument('--blocks', type=int, default=30)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    # Reuse the exact frozen native physical-cache loader, without synthetic arms.
    a.prefix, a.extra_tail = 8192, 0
    batches = [8]
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and a.blocks >= 30
    out = Path(a.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/qualify_benchmark_xqa_raw_retuned_split.py',
               'src/xqa_raw_retuned_split_runtime.py', 'scripts/benchmark_xqa_raw_pair.py',
               'src/xqa_raw_continuation_build.py', 'src/xqa_raw_pair_build.py',
               'src/xqa_raw_pair_general_runtime.py', 'src/xqa_raw_pair_padded_runtime.py',
               'src/xqa_raw_grouped_hint_build.py', 'src/xqa_raw_direct_input_build.py',
               'src/xqa_raw_direct_input_runtime.py']
    r = {'status': 'initializing', 'scope': __doc__, 'args': vars(a), 'cases': [],
         'frozen_rule': "S'=min(max(1,SM//(ceil(B/2)*Hkv)),ceil(table_token_capacity/256)); B8,Hkv2",
         'source_sha256': {x: hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sources}}
    def save():
        out.write_text(json.dumps(r, indent=2)+'\n')
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_direct_input_build import jit_specs
        from src.xqa_raw_direct_input_runtime import RawPairDirectInputPlan
        from src.xqa_raw_retuned_split_runtime import RawPairRetunedSplitPlan, coverage_splits
        from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA
        torch.set_num_threads(2)
        assert torch.cuda.get_device_capability() == (12, 0)
        torch.manual_seed(20261017)
        specs, receipt = jit_specs()
        modules = {g: s.build_and_load() for g, s in specs.items()}
        r['build_identity'] = receipt['identity']
        r['generated_sha256'] = receipt['generated_sha256']
        r['gpu'] = str(torch.cuda.get_device_properties(0))
        payload = None
        if a.capture:
            path = Path(a.capture)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert not payload["payload_cast_performed"]
            assert payload["q"].dtype == payload["prefix_kv"].dtype == torch.bfloat16
            assert payload["k_scale"] == payload["v_scale"] == 1.
            prefix = payload["prefix_kv"].shape[1]
            base_q = payload["q"].contiguous().cuda()
            assert max(batches) <= len(base_q), "Never duplicate captured Q to fake a larger real cohort"
            tails = [t.cuda() for t in payload["tail_kv"]]
            prefix_kv = payload["prefix_kv"].cuda()
            r["capture"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "rank": payload["rank"], "layer": payload["layer"]}
        else:
            prefix = a.prefix
            base_q = torch.randn((max(batches), 16, 128), device="cuda", dtype=torch.bfloat16)
            prefix_kv = torch.randn((2, prefix, 2, 128), device="cuda", dtype=torch.bfloat16)
            tails = [torch.randn((2, 17 + (i * 37) % 257, 2, 128), device="cuda", dtype=torch.bfloat16)
                     for i in range(len(base_q))]
        if a.extra_tail:
            tails = [torch.cat((t, torch.randn((2, a.extra_tail + i % 7, 2, 128), device="cuda", dtype=torch.bfloat16)), dim=1)
                     for i, t in enumerate(tails)]
        lens = [prefix + t.shape[1] for t in tails]
        n = len(base_q)
        if payload is not None and not a.extra_tail:
            # Preserve native physical page identity and stride for unmodified captures.
            shape, stride = payload["original_cache_shape"], payload["original_cache_stride"]
            cache = torch.empty_strided(shape, stride, device="cuda", dtype=torch.bfloat16)
            cache.zero_()
            base_table = payload["original_block_table"].to(device="cuda", dtype=torch.int32).contiguous()
            r["layout"] = "native original physical IDs and stride"
        else:
            # Explicit synthetic/stress layout: random physical page permutation, same fused stride.
            pp = prefix // 16
            tail_pages = max((t.shape[1] + 15) // 16 for t in tails) + 2
            pages = pp + n * tail_pages + 32
            cache = torch.randn((pages, 16, 2, 256), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
            perm = torch.randperm(pages, device="cuda")
            base_table = torch.empty((n, pp + tail_pages), device="cuda", dtype=torch.int32)
            base_table[:, :pp] = perm[:pp].int()
            for row in range(n):
                base_table[row, pp:] = perm[pp + row * tail_pages: pp + (row + 1) * tail_pages].int()
            r["layout"] = "random remapped shared prefix, independent private pages; synthetic extra KV" if payload is not None else "synthetic Q and KV"
        for row in range(n):
            kv = torch.cat((prefix_kv, tails[row]), dim=1)
            t = torch.arange(lens[row], device="cuda")
            cache[base_table[row, t // 16].long(), :, t % 16] = torch.cat((kv[0], kv[1]), dim=-1)
        del kv, t, prefix_kv, tails
        assert n == 8, 'Actual full B8 capture required; no subset or replicated queries'
        q, table, lengths = base_q, base_table, lens
        def fp64_reference():
            refs = []
            actual_lengths = shared_lengths.flatten().to(dtype=torch.int64).cpu().tolist()
            for row, length in enumerate(actual_lengths):
                token = torch.arange(length, device='cuda')
                kv = cache[table[row, token // 16].long(), :, token % 16].double()
                score = torch.einsum('hgd,thd->hgt', q[row].double().reshape(2, 8, 128), kv[..., :128]) / 128**.5
                refs.append(torch.einsum('hgt,thd->hgd', score.softmax(-1), kv[..., 128:]).reshape(16, 128))
            return torch.stack(refs)
        def bits(x, y):
            return torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
        def difference(x, y):
            diff = x.double()-y.double()
            return {'bit_equal': bits(x, y),
                    'mismatched_BF16_elements': int((x.contiguous().view(torch.int16) != y.contiguous().view(torch.int16)).sum()),
                    'max_abs': float(diff.abs().max()), 'rms': float(diff.square().mean().sqrt()),
                    'cosine': float(torch.nn.functional.cosine_similarity(x.double().flatten(), y.double().flatten(), dim=0))}
        workspace = torch.zeros(64*1024*1024, device='cuda', dtype=torch.uint8)
        native_workspace = torch.zeros_like(workspace)
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        sprime = coverage_splits(sm, 8, 2, table.shape[1]*16)
        state = torch.empty(4*2*sprime*STATE_FLOATS_PER_CTA, device='cuda', dtype=torch.float32)
        control_state = torch.empty_like(state)
        native_raw = torch.empty(8*2*sprime*STATE_FLOATS_PER_CTA, device='cuda', dtype=torch.float32)
        start = time.perf_counter()
        plan = RawPairRetunedSplitPlan(modules, q, cache, table, lengths, prefix, workspace=workspace, raw_state=state)
        control = RawPairDirectInputPlan(modules, q, cache, table, lengths, prefix, workspace=workspace, raw_state=control_state)
        control.lengths = plan.lengths
        shared_lengths = plan.lengths
        assert plan.splits == sprime and control.splits == plan.original_splits
        torch.cuda.synchronize()
        case = {'batch': 8, 'prefix': prefix, 'lengths': lengths,
                'table_shape': list(table.shape), 'Q_stride': list(q.stride()), 'cache_stride': list(cache.stride()),
                'native_S': control.splits, 'retuned_S': plan.splits, 'actual_SM': plan.sm,
                'counterfactual_native_SM_parameter': plan.tail_sm,
                'prefix_SM_parameter': plan.prefix_sm, 'tail_SM_parameter': plan.tail_sm,
                'native_active_state_bytes': control.state.numel()*4, 'retuned_active_state_bytes': plan.state.numel()*4,
                'allocated_state_bytes_each': state.numel()*4, 'raw_oracle_state_bytes': native_raw.numel()*4,
                'candidate_workspace_bytes': workspace.numel(),
                'setup_wall_seconds': time.perf_counter()-start, 'checks': {}, 'cut_checks': [],
                'expected_grids': {'native': [control.splits,2,8], 'direct_nativeS_prefix': [control.splits,2,4],
                                   'direct_nativeS_tail': [control.splits,2,8], 'native_retunedS': [plan.splits,2,8],
                                   'direct_retunedS_prefix': [plan.splits,2,4], 'direct_retunedS_tail': [plan.splits,2,8]}}
        r['cases'].append(case)
        native_o, aa_o, retuned_o = (torch.empty_like(q) for _ in range(3))
        def native(output, sm_parameter):
            xqa(q.unsqueeze(1),cache[...,:128],cache[...,128:],table,shared_lengths,output.unsqueeze(1),
                native_workspace[8*1024*1024:],native_workspace[:8*1024*1024],2,16,
                q_scale=1.,kv_scale=1.,kv_layout='HND',sm_count=sm_parameter,enable_pdl=True)
        funcs = {'native': lambda: native(native_o,sm), 'native_AA': lambda: native(aa_o,sm),
                 'direct_nativeS': control.run, 'native_retunedS': lambda: native(retuned_o,plan.tail_sm),
                 'direct_retunedS': plan.run}
        outputs = {'native':native_o,'native_AA':aa_o,'direct_nativeS':control.output,
                   'native_retunedS':retuned_o,'direct_retunedS':plan.output}
        checks = case['checks']
        def qualify(label, functions):
            state.fill_(float('nan'));control_state.fill_(float('nan'))
            for fn in functions.values():fn()
            torch.cuda.synchronize()
            checks[label+'_native_AA_bits'] = bits(native_o,aa_o)
            checks[label+'_default_control_bits'] = bits(native_o,control.output)
            checks[label+'_same_retuned_S_bits'] = bits(retuned_o,plan.output)
            reference = fp64_reference()
            for arm, value in outputs.items():
                checks[label+'_'+arm+'_FP64'] = bool(torch.allclose(value.double(), reference, atol=.006,rtol=.015))
            case.setdefault('default_vs_retuned',{})[label] = difference(native_o,retuned_o)
            case.setdefault('FP64_errors',{})[label] = {name:{'max_abs':float((value.double()-reference).abs().max()),
                       'rms':float((value.double()-reference).square().mean().sqrt())} for name,value in outputs.items()}
            save()
            assert all(checks.values()), case
        qualify('initial',funcs)
        for _ in range(10):
            native(aa_o,sm)
            assert bits(native_o,aa_o), 'Native AA instability'
        checks['AA_10_bits'] = True
        # Audit live native GQA8 raw fragments before normalization against group16.
        # Invalid padded registers/scalar lanes are intentionally not compared.
        native_view = native_raw.view(8,2,sprime,4,32,35)
        valid = torch.zeros_like(native_view,dtype=torch.bool)
        for reg in range(32):
            if reg%4<2:valid[...,reg]=True
        valid[...,:8,32:]=True
        mapped = torch.empty_like(native_view)
        cuts = sorted({x for x in [1,sprime-1,sprime,sprime+1,339,prefix//256] if 1<=x<=prefix//256})
        for cut in cuts:
            plan.prefix_tokens = cut*256
            native_raw.fill_(float('nan'));state.fill_(float('nan'))
            modules[8].xqa_wrapper(False,plan.tail_sm,2,0,1.,None,aa_o.unsqueeze(1),1.,q.unsqueeze(1),
                None,plan.wrapper_k,plan.wrapper_v,None,None,table,plan.max_len,
                shared_lengths,8,1.,None,1,None,None,plan.sem,plan.scratch,False,native_raw,cut,1)
            plan.run()
            src = state.view(4,2,sprime,4,32,35)
            for row in range(8):
                for reg in range(32):
                    if reg%4<2:mapped[row,...,reg]=src[row//2,...,reg+2*(row%2)]
                mapped[row,...,:8,32:]=src[row//2,...,(row%2)*8:(row%2+1)*8,32:]
            mismatch = (mapped.view(torch.int32)!=native_view.view(torch.int32))&valid
            item = {'cut_tiles':cut,'raw_live_bits':not bool(mismatch.any()),
                    'raw_live_finite':bool(torch.isfinite(native_view[valid]).all() and torch.isfinite(mapped[valid]).all()),
                    'candidate_same_S_native_bits':bits(retuned_o,plan.output),
                    'mismatch_fields':{name:int(mismatch[...,lo:hi].sum()) for name,lo,hi in [('U',0,32),('m',32,33),('z',33,34),('hint',34,35)]}}
            case['cut_checks'].append(item);save()
            assert item['raw_live_bits'] and item['raw_live_finite'] and item['candidate_same_S_native_bits'],item
        plan.prefix_tokens = prefix
        graphs = {}
        start = time.perf_counter()
        for name,fn in funcs.items():
            fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            graphs[name]=graph
        torch.cuda.synchronize()
        case['capture_wall_seconds']=time.perf_counter()-start
        replays = {name:graph.replay for name,graph in graphs.items()}
        oldq = q.clone();q.mul_(.875)
        plan.prepare([length-min(3,length-prefix-1) for length in lengths])
        qualify('changed_graph_Q_lengths',replays)
        q.copy_(oldq);plan.prepare(lengths)
        qualify('restored_graph',replays)
        rng = random.Random(20261017)
        ratios = {'native_over_retuned':('native','direct_retunedS'),
                  'native_over_fixed':('native','direct_nativeS'),
                  'fixed_over_retuned':('direct_nativeS','direct_retunedS'),
                  'counterfactual_over_retuned':('native_retunedS','direct_retunedS'),
                  'native_over_counterfactual':('native','native_retunedS'),
                  'AA':('native','native_AA')}
        flush = torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
        for mode in ['hot','cold']:
            blocks,orders=[],[]
            for _ in range(a.blocks):
                order=list(graphs);rng.shuffle(order);orders.append(order);block={}
                for name in order:
                    for _ in range(3):graphs[name].replay()
                    if mode=='cold':flush.zero_()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(10 if mode=='hot' else 1):graphs[name].replay()
                    end.record();end.synchronize()
                    block[name]=begin.elapsed_time(end)/(10 if mode=='hot' else 1)
                blocks.append(block)
            case[mode]={'blocks_ms':blocks,'orders':orders,
                        'median_ms':{name:statistics.median(x[name] for x in blocks) for name in graphs},
                        **{key:paired_ratio(blocks,*arms) for key,arms in ratios.items()}}
            qualify(mode+'_post_measurement',replays)
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for name,graph in graphs.items():
                    with torch.profiler.record_function('retuned_split_arm_'+name):
                        graph.replay();torch.cuda.synchronize()
            trace=out.with_suffix('.trace.json');prof.export_chrome_trace(str(trace));case['trace']=str(trace)
        qualify('final_revisit',replays)
        r['status']='qualified_same_retuned_schedule_and_measured'
        r['limitations']=['Retuned S changes original native tile assignment/reduction and is not default-bitwise mode.',
            'Original PDL native is the primary baseline; native forced to S-prime is only a counterfactual.',
            'Two fixed B8 captures cannot establish model quality, natural output length, or end-to-end benefit.',
            'Host preparation and graph creation are separately reported; timed complete calls have two XQA kernels and no input copies.',
            'Cold timings are single-replay CUDA event intervals; quantization can make bootstrap intervals narrow.',
            'Finite sparse-cut and graph-change qualification is not a proof over all inputs.']
    except Exception:
        r['status']='failed';r['error']=traceback.format_exc();save();raise
    save();print(json.dumps({'status':r['status'],'output':str(out)}))


if __name__=='__main__':main()
