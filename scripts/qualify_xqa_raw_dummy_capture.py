"""Safety-only gate: aliased int32 native lengths, dummy L=1 capture then real replay.

Synthetic independent Q/private KV, physically shared prefix, full capacity table.
No performance claim and no change to frozen runtime/CUDA sources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches', default='6,7,8,19,21')
    p.add_argument('--prefix', type=int, default=120528)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert a.prefix > 0 and a.prefix % 16 == 0
    out = Path(a.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    files = ['scripts/qualify_xqa_raw_dummy_capture.py', 'src/xqa_raw_pair_padded_runtime.py',
             'src/xqa_raw_pair_general_runtime.py', 'src/xqa_raw_pair_build.py']
    report = {'status': 'initializing', 'scope': __doc__, 'args': vars(a), 'cases': [],
              'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}
    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_pair_build import jit_specs
        from src.xqa_raw_pair_padded_runtime import RawPairPaddedPlan
        torch.set_num_threads(2)
        torch.manual_seed(2026101704)
        assert torch.cuda.get_device_capability() == (12, 0)
        specs, receipt = jit_specs()
        modules = {g: s.build_and_load() for g, s in specs.items()}
        report['build_identity'] = receipt['identity']
        workspace = torch.zeros(64 * 1024 * 1024, device='cuda', dtype=torch.uint8)
        native_workspace = torch.zeros_like(workspace)
        state = torch.empty(2 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
        for b in map(int, a.batches.split(',')):
            assert 5 <= b <= 32
            pp, private_pages = a.prefix // 16, 514
            pages = pp + b * private_pages
            cache = torch.randn((pages, 16, 2, 256), device='cuda', dtype=torch.bfloat16).permute(0, 2, 1, 3)
            physical = torch.randperm(pages, device='cuda', dtype=torch.int32)
            table = torch.empty((b, pp + private_pages), device='cuda', dtype=torch.int32)
            table[:, :pp] = physical[:pp]
            for row in range(b):
                table[row, pp:] = physical[pp + row * private_pages:pp + (row + 1) * private_pages]
            q = torch.randn((b, 16, 128), device='cuda', dtype=torch.bfloat16)
            initial = [a.prefix + 17 + row * 31 for row in range(b)]
            plan = RawPairPaddedPlan(modules, q, cache, table, initial, a.prefix, workspace, state)
            # This models the actual runner's int32 device buffer, not a separate copied uint32 vector.
            native_lengths = torch.ones(b, device='cuda', dtype=torch.int32)
            plan.lengths = native_lengths.view(torch.uint32).reshape(b, 1)
            assert plan.lengths.data_ptr() == native_lengths.data_ptr()
            native_o = torch.empty_like(q)
            def native():
                xqa(q.unsqueeze(1), plan.k, plan.v, table, plan.lengths, native_o.unsqueeze(1),
                    native_workspace[8*1024*1024:], native_workspace[:8*1024*1024], 2, 16,
                    q_scale=1., kv_scale=1., kv_layout='HND', sm_count=plan.sm, enable_pdl=True)
            case = {'batch': b, 'S': plan.splits, 'prefix': a.prefix,
                    'cache_shape': list(cache.shape), 'cache_stride': list(cache.stride()),
                    'table_shape': list(table.shape), 'pointer_alias': True, 'checks': {}}
            report['cases'].append(case)
            save()
            # Warm up and capture while every length is 1. Python prepare is deliberately bypassed.
            plan.run()
            native()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                plan.run()
            tests = [('dummy_one', [1] * b), ('real_unequal', initial),
                     ('real_growth', [a.prefix + 8192 - row * 7 for row in range(b)]),
                     ('dummy_again', [1] * b), ('real_again', initial)]
            for name, lengths in tests:
                native_lengths.copy_(torch.tensor(lengths, device='cuda', dtype=torch.int32))
                q.mul_(.875)
                # Same addresses, both new Q and new lengths; checkpoint must self-reset.
                state.fill_(float('nan'))
                graph.replay()
                native()
                torch.cuda.synchronize()
                equal = torch.equal(native_o.view(torch.int16), plan.output.view(torch.int16))
                finite = bool(torch.isfinite(plan.output).all())
                case['checks'][name] = {'bitwise_equal': equal, 'finite': finite,
                                       'max_abs': float((native_o.float()-plan.output.float()).abs().max()),
                                       'min_length': min(lengths), 'max_length': max(lengths)}
                save()
                assert equal and finite, case
            del graph, plan, native_o, q, table, cache, physical, native_lengths
        report['status'] = 'qualified'
        report['limitations'] = ['Synthetic safety gate, not model quality or performance.',
                                 'Only dummy length 1 is qualified; length 0 is outside this gate.']
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({'status': report['status'], 'output': str(out)}))


if __name__ == '__main__':
    main()
