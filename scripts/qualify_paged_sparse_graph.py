"""Graph replay after changed Q/cache/support/length, with original-cache FP64."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.vllm_sparse_executor import physical_supports
from src.vllm_sparse_graph_executor import GraphPagedSparseExecutor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.manual_seed(1123)
    torch.set_num_threads(4)
    backing = torch.randn(3000, 3, 2, 16, 256, device='cuda', dtype=torch.bfloat16)
    cache = backing[:, 1]
    tables = torch.stack([torch.cat((torch.arange(64), torch.arange(64+r*600, 64+(r+1)*600))) for r in range(4)])
    records = []
    for batch in (1, 3, 4):
        q = torch.randn(batch, 16, 128, device='cuda', dtype=torch.bfloat16)*.25
        for method in ('e0_simt', 'e0', 'union'):
            op = GraphPagedSparseExecutor(q.device, method, batch, max_tokens=10000)
            graph = None
            for step, length in enumerate((257, 1025, 4097, 129, 8193)):
                q.normal_(std=.25)
                cache.add_(.00390625)
                supports = [[torch.cat((torch.arange(128), torch.arange(min(128+r+h,length), length, 1+((r+h+step)%3)))) for h in range(2)] for r in range(batch)]
                actual_tables = tables[:batch].flip(0) if step % 2 else tables[:batch]
                ids = physical_supports(actual_tables, supports, 2, 16, cache.stride())
                started = time.perf_counter()
                item = {'method':method,'batch':batch,'step':step,'logical_length':length,
                        'support_counts':[[len(s) for s in row] for row in supports], 'changed_query_cache_supports_and_request_mapping':step>0}
                try:
                    op.plan(q, cache, ids)
                    torch.cuda.synchronize()
                    item['plan_complete_seconds'] = time.perf_counter()-started
                    item['plan_info'] = list(op.fixed_plan_info)
                    if graph is None:
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            op.run(q)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            output = op.run(q)
                    graph.replay()
                    actual = output.clone()
                    expected = torch.empty_like(q, dtype=torch.float64)
                    for request in range(batch):
                        for head in range(2):
                            logical = supports[request][head]
                            pages = actual_tables[request, logical//16].to('cuda')
                            offsets = (logical%16).to('cuda')
                            kv = cache[pages,head,offsets].double()
                            logits = q[request,head*8:(head+1)*8].double() @ kv[:,:128].T / 128**.5
                            expected[request,head*8:(head+1)*8] = logits.softmax(-1) @ kv[:,128:]
                    item['max_abs_vs_independent_fp64'] = float((actual.double()-expected).abs().max())
                    item['output_frozen_tolerance_passed'] = bool(torch.isclose(actual.double(),expected,atol=.006,rtol=.015).all())
                    item['status'] = 'qualified' if item['output_frozen_tolerance_passed'] else 'failed'
                except Exception as exc:
                    item.update(status='failed', error=repr(exc))
                records.append(item)
                if item['status'] != 'qualified':
                    break
            del op, graph
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = len(records)==45 and all(r['status']=='qualified' for r in records)
    path.write_text(json.dumps({'status':'qualified' if passed else 'failed','uid':os.geteuid(),'python':sys.executable,
        'packages':{x:importlib.metadata.version(x) for x in ('torch','flashinfer-python','vllm')},
        'source_cache_shape':list(cache.shape),'source_cache_stride':list(cache.stride()),'source_storage_offset':cache.storage_offset(),
        'contract':'Fixed batch/cache/output/workspace bindings. CPU plan occurs outside replay. Native plan_info must remain exactly equal; changed topology explicitly requires recapture.',
        'scope':'Operator graph metadata correctness only, not vLLM graph integration or application speed evidence.', 'records':records},indent=2)+'\n')
    print(json.dumps({'status':'qualified' if passed else 'failed','cases':len(records),'output':str(path)}))
    assert passed


if __name__ == '__main__':
    main()
