#!/usr/bin/env python
"""Reproducible graph microbenchmarks; split job indices across visible GPUs.

The timed region includes producer + stable split merge, and excludes one-time
metadata preparation/JIT compilation. Preparation and workspace are reported
separately. Results are synthetic feasibility evidence, not application speedup.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
import time
import torch
from branchtile.attention import Attention, make_plan, reference, synthetic_owners


def csv_int(value):
    return [int(x) for x in value.split(',')]


def graph_bench(fn, repeats=100, samples=9):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(repeats):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        values.append(a.elapsed_time(b)*1000/repeats)
    return dict(median_us=statistics.median(values), min_us=min(values), max_us=max(values), samples_us=values)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--heads',type=csv_int,default=[2])
    p.add_argument('--tokens',type=csv_int,default=[4096,16384,65536])
    p.add_argument('--patterns',default='cross,full,tree,mixed,disjoint')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--gqa',type=csv_int,default=[8,4])
    p.add_argument('--splits',type=csv_int,default=[1,4,16,64])
    p.add_argument('--block-n',type=csv_int,default=[32,64])
    p.add_argument('--threads',type=csv_int,default=[128])
    p.add_argument('--methods',default='compact,fixed')
    p.add_argument('--job-index',type=int,default=0)
    p.add_argument('--job-count',type=int,default=1)
    p.add_argument('--output',required=True)
    p.add_argument('--source-dir')
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--skip-reference',action='store_true')
    a=p.parse_args()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    device=torch.cuda.get_device_properties(0)
    cases=list(itertools.product(a.heads,a.tokens,a.patterns.split(','),a.gqa,a.splits,a.block_n,a.threads))
    for index,(heads,tokens,pattern,gqa,splits,bn,threads) in enumerate(cases):
        if index%a.job_count!=a.job_index:
            continue
        torch.manual_seed(a.seed)
        owners=synthetic_owners(heads,tokens,pattern,a.batch,a.seed)
        plan=make_plan(owners,a.batch,gqa,'cuda')
        q=torch.randn(heads,a.batch*gqa,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(heads,tokens,128,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        expected,expected_lse=(None,None) if a.skip_reference else reference(q,k,v,owners,a.batch,gqa,compute_dtype=torch.float32)
        for method in a.methods.split(','):
            result=dict(case_index=index,heads=heads,tokens=tokens,pattern=pattern,batch=a.batch,gqa=gqa,
                        splits=splits,block_n=bn,threads=threads,method=method,seed=a.seed,
                        device=device.name,device_uuid=str(device.uuid),pid=os.getpid(),
                        torch_version=torch.__version__,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        **plan.statistics(bn,splits))
            try:
                started=time.perf_counter()
                op=Attention(plan,splits,bn,threads,compact=method.startswith('compact'),region_partials=method.endswith('_partial'))
                output,lse=op(q,k,v)
                torch.cuda.synchronize()
                result['jit_and_init_s']=time.perf_counter()-started
                if expected is not None:
                    result['max_abs_error']=float((output.float()-expected).abs().max())
                    result['max_lse_error']=float((lse-expected_lse).nan_to_num().abs().max())
                    torch.testing.assert_close(output.float(),expected,atol=0.006,rtol=0.02)
                    torch.testing.assert_close(lse,expected_lse,atol=0.0001,rtol=0.0001)
                result['workspace_bytes']=sum(t.numel()*t.element_size() for t in [op.u,op.m,op.l,op.output,op.lse])
                result.update(graph_bench(lambda:op(q,k,v),a.repeats,a.samples))
                if a.source_dir:
                    path=Path(a.source_dir)/f'{index}_{method}.cu';path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(op.kernel.get_kernel_source())
                    result['generated_source']=str(path)
                result['status']='ok'
            except Exception as e:
                result.update(status='error',error=repr(e))
            with open(a.output,'a') as f:
                f.write(json.dumps(result)+'\n')
            print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
