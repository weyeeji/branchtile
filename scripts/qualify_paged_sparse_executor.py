"""Small FP64/physical-page qualification in the exact vLLM engine venv."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from vllm_sparse_executor import PagedSparseExecutor, physical_supports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    torch.manual_seed(1087)
    tables = torch.tensor([[5, 3, 11, 17], [5, 3, 13, 19], [5, 3, 15, 23]])
    records = []
    for layout in ('contiguous', 'layers_interleaved_between_pages'):
        backing = torch.randn(32, 3, 2, 16, 256, device='cuda', dtype=torch.bfloat16)
        cache = backing[:, 1] if layout != 'contiguous' else backing[:, 1].contiguous()
        for batch in (3, 2, 1):
            supports = [[torch.cat((torch.arange(4), torch.arange(6 + r + h, 29, 3), torch.arange(30, 47 + r))) for h in range(2)] for r in range(batch)]
            physical = physical_supports(tables[:batch], supports, 2, 16, cache.stride())
            q = torch.randn(batch, 16, 128, device='cuda', dtype=torch.bfloat16)
            for method in ('e0', 'union'):
                op = PagedSparseExecutor(q.device, method).plan(q, cache, physical)
                for change in (False, True):
                    if change:
                        q.normal_()
                        cache.add_(torch.tensor(.03125, device='cuda', dtype=cache.dtype))
                    out, lse = op.run(q, return_lse=True)
                    expected, expected_lse = torch.empty_like(q, dtype=torch.float64), torch.empty((batch, 16), device='cuda', dtype=torch.float64)
                    for request in range(batch):
                        for head in range(2):
                            logical = supports[request][head]
                            pages = tables[request, logical // 16].to('cuda')
                            offset = (logical % 16).to('cuda')
                            # Independent indexing of original cache, not op's flattened view.
                            kv = cache[pages, head, offset].double()
                            logits = q[request, head*8:(head+1)*8].double() @ kv[:, :128].T / 128**.5
                            expected[request, head*8:(head+1)*8] = logits.softmax(-1) @ kv[:, 128:]
                            expected_lse[request, head*8:(head+1)*8] = logits.logsumexp(-1)
                    checks = {}
                    for name, actual, reference, atol, rtol in [('output', out.float(), expected.float(), .006, .015), ('lse', lse, expected_lse.float(), .0003, .0001)]:
                        try:
                            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
                            checks[name] = {'passed': True}
                        except AssertionError as exc:
                            checks[name] = {'passed': False, 'error': str(exc)}
                    saved_output = out.clone()
                    actual_without_lse = op.run(q)
                    torch.testing.assert_close(actual_without_lse, saved_output, atol=0, rtol=0)
                    records.append({'layout': layout, 'batch': batch, 'method': method, 'changed_q_and_cache': change,
                                    'checks': checks, 'max_abs_error_fp64': float((out.double()-expected).abs().max()),
                                    'max_lse_error_fp64': float((lse.double()-expected_lse).abs().max()),
                                    'cache_stride': list(cache.stride()), 'view_stride': list(op.kv[0].stride()),
                                    'payload_storage_aliases_source': all(x.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr() for x in op.kv)})
                del op
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'status': 'qualified_output' if all(r['checks']['output']['passed'] for r in records) else 'output_failed', 'strict_lse_all_passed': all(r['checks']['lse']['passed'] for r in records), 'uid': os.geteuid(), 'python': sys.executable,
        'packages': {x: importlib.metadata.version(x) for x in ('torch', 'flashinfer-python', 'vllm')},
        'records': records, 'scope': 'Physical-page/strided-payload operator qualification only; no engine generation or full-budget qualification yet.'}, indent=2)+'\n')
    print(json.dumps({'cases': len(records), 'output_passed': sum(r['checks']['output']['passed'] for r in records), 'lse_passed': sum(r['checks']['lse']['passed'] for r in records), 'output': str(path)}))
    assert all(r['checks']['output']['passed'] for r in records), 'Cannot proceed to engine qualification with failed output'


if __name__ == '__main__':
    main()
