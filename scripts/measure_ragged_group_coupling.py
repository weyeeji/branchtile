"""Interleaved same-K group/depth costs, including independently executed groups.

Consumes a completed real G0 qualification. This is an attention cost-model
experiment, not a speculative decoder or an acceptance/scheduler speedup.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_ratio(blocks,numerator,denominator):
    rng=random.Random(7713);sample=[]
    for _ in range(1000):
        picked=[rng.choice(blocks) for _ in blocks]
        sample.append(statistics.median(x[numerator] for x in picked)/statistics.median(x[denominator] for x in picked))
    sample.sort()
    return dict(ratio=statistics.median(x[numerator] for x in blocks)/statistics.median(x[denominator] for x in blocks),
                paired_bootstrap_95ci=[sample[25],sample[974]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification',required=True)
    p.add_argument('--blocks',type=int,default=30);p.add_argument('--output',required=True)
    args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and args.blocks>=20
    source=json.loads(Path(args.qualification).read_text())
    assert source['status']=='qualified' and source['all_passed'] and source['original_AR_O_fp64_pass']
    assert len(source['args']['capture'])==2 and not source['args'].get('rows')
    for name,digest in source['source_sha256'].items():assert sha(ROOT/name)==digest
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='initializing',args=vars(args),qualification_sha256=sha(args.qualification),
                source_sha256=sha(__file__),cases=[],measurements={},scope=__doc__)
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        from src.ragged_capture_fixture_v2 import load_ragged_fixture
        from src.ragged_shared_verify_plan_v2 import RaggedSharedVerifyPlanV2
        from scripts.qualify_ragged_shared_verify_v2 import reference_all
        torch.set_num_threads(4);assert torch.cuda.get_device_capability()==(12,0)
        fixture=load_ragged_fixture(source['args']['capture'],source['args'].get('capture_recovery',[]))
        report['provenance']=fixture['provenance']
        cache=torch.empty_strided(fixture['cache'].shape,fixture['stride'],device='cuda',dtype=fixture['cache'].dtype)
        cache.copy_(fixture['cache']);queries=fixture['q'].cuda()
        expected,_=reference_all(cache,fixture,queries)
        patterns=[c['query_counts'] for c in source['cases'] if c['total_queries']==32]
        assert len(patterns)==3 and patterns[0]==[4]*8
        retained=[];graphs={}
        for pi,counts in enumerate(patterns):
            for group in (None,0,1):
                rows=list(range(8)) if group is None else [r for r,g in enumerate(fixture['groups']) if g==group]
                local_counts=[counts[r] for r in rows]
                local_groups=fixture['groups'] if group is None else [0]*len(rows)
                local_prefix=fixture['prefixes'] if group is None else [fixture['prefixes'][group]]
                lens=[int(fixture['positions'][r,0])+counts[r] for r in rows]
                q=torch.cat([queries[r,:counts[r]] for r in rows]).contiguous()
                ref=torch.cat([expected[r,:counts[r]] for r in rows])
                name=f'joint_{pi}' if group is None else f'group{group}_{pi}'
                started=time.perf_counter()
                plan=RaggedSharedVerifyPlanV2(cache,local_groups,local_prefix,local_counts,lens,
                    fixture['block_tables'][rows],fixture['k_scale'],fixture['v_scale'])
                actual=plan.shared(q)
                passed=bool(torch.isclose(actual.double(),ref,atol=.006,rtol=.015).all())
                record=dict(name=name,original_rows=rows,query_counts=local_counts,total_queries=sum(local_counts),
                    prefix_tokens=local_prefix,O_fp64_pass=passed,O_fp64_maxabs=float((actual.double()-ref).abs().max()),
                    plan_info={n:list(w._plan_info) for n,w in plan.wrappers.items()},prepare=plan.receipt)
                report['cases'].append(record);save();assert passed,record
                for _ in range(3):plan.shared(q)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):plan.shared(q)
                torch.cuda.synchronize()
                record['prepare_and_capture_wall_seconds']=time.perf_counter()-started
                retained.append((plan,q));graphs[name]=graph
        report['retained_plan_workspace_bytes']=len(retained)*512*1024**2
        report['allocated_gpu_bytes_before_timing']=torch.cuda.memory_allocated()
        flush=torch.empty(256*1024**2,device='cuda',dtype=torch.uint8)
        graphs['joint_0_AA']=graphs['joint_0']
        def timed(name,repeats):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):graphs[name].replay()
            end.record();end.synchronize();return start.elapsed_time(end)/repeats
        for regime in ('hot','cold'):
            blocks=[];rng=random.Random(20261017)
            for _ in range(args.blocks):
                names=list(graphs);rng.shuffle(names);times={}
                for name in names:
                    if regime=='cold':flush.zero_()
                    times[name]=timed(name,20 if regime=='hot' else 1)
                blocks.append(times)
            med={n:statistics.median(x[n] for x in blocks) for n in graphs}
            report['measurements'][regime]=dict(blocks=blocks,median_ms=med,
                AA=paired_ratio(blocks,'joint_0_AA','joint_0'),
                same_K_nonuniform_over_uniform={str(i):paired_ratio(blocks,f'joint_{i}','joint_0') for i in (1,2)},
                sum_separate_group_ms={str(i):med[f'group0_{i}']+med[f'group1_{i}'] for i in range(3)},
                interpretation='Separate-group sums include two calls and are cost-model diagnostics, not fair online scheduler competitors.')
            save()
        report['status']='completed';report['all_FP64_passed']=True
        report['limitations']=['No real draft acceptance trace or whole-model verification here.',
            'Different depth allocations are different attention inputs; equal K is a cost comparison, not same-task speedup.',
            'An ordinary group-aware cost table may fully capture any measured difference.',
            'Nine static plans retained only for controlled interleaving; all 4.5 GiB workspaces and capture time are reported.']
        save()
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()
