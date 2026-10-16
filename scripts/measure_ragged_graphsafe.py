"""Internal graph-safe plans versus three finite, guarded shape graphs.

No scheduler/acceptance speedup: measures graph viability and complete attention
cost, including separately measured prepare+switch wall time.
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification',required=True)
    p.add_argument('--blocks',type=int,default=30);p.add_argument('--profile',action='store_true')
    p.add_argument('--output',required=True);args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and args.blocks>=10
    prior=json.loads(Path(args.qualification).read_text())
    assert prior['status']=='qualified' and prior['all_passed'] and prior['original_AR_O_fp64_pass']
    assert len(prior['args']['capture'])==2 and not prior['args'].get('rows')
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='initializing',args=vars(args),scope=__doc__,qualification_sha256=sha(args.qualification),
        source_sha256={n:sha(ROOT/n) for n in ('src/ragged_verify_graphsafe.py','scripts/measure_ragged_graphsafe.py')},
        checks=[],graph_captures=[],growth_checks=[],timing={})
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        from src.ragged_capture_fixture_v2 import load_ragged_fixture
        from src.ragged_verify_graphsafe import RaggedGraphSafePlan,FiniteShapeVerifyPlan
        from scripts.qualify_ragged_shared_verify_v2 import reference_all
        from scripts.measure_ragged_group_coupling import paired_ratio
        torch.set_num_threads(4);assert torch.cuda.get_device_capability()==(12,0)
        fixture=load_ragged_fixture(prior['args']['capture'],prior['args'].get('capture_recovery',[]))
        report['provenance']=fixture['provenance']
        cache=torch.empty_strided(fixture['cache'].shape,fixture['stride'],device='cuda',dtype=fixture['cache'].dtype)
        cache.copy_(fixture['cache']);all_q=fixture['q'].cuda();expected,_=reference_all(cache,fixture,all_q)
        patterns=[c['query_counts'] for c in prior['cases'] if c['total_queries']==32]
        assert len(patterns)==3 and patterns[0]==[4]*8
        lens=lambda counts,shift=0:[int(fixture['positions'][r,0])+shift+n for r,n in enumerate(counts)]
        queries=lambda counts,shift=0:torch.cat([all_q[r,shift:shift+n] for r,n in enumerate(counts)])
        references=lambda counts,shift=0:torch.cat([expected[r,shift:shift+n] for r,n in enumerate(counts)])
        groups,prefixes,tables=fixture['groups'],fixture['prefixes'],fixture['block_tables']
        qbuffer=torch.empty((32,16,128),device='cuda',dtype=torch.bfloat16)
        dynamic=RaggedGraphSafePlan(cache,groups,prefixes,32,(fixture['positions'][:,-1]+1).tolist(),
            tables,fixture['k_scale'],fixture['v_scale'])
        fixed=[];started=time.perf_counter()
        for counts in patterns:
            fixed.append(FiniteShapeVerifyPlan(cache,groups,prefixes,counts,lens(counts),tables,
                fixture['k_scale'],fixture['v_scale']))
        torch.cuda.synchronize();report['three_finite_plans_setup_seconds']=time.perf_counter()-started
        report['workspace_bytes']=dict(dynamic=512*1024**2,three_finite=3*512*1024**2)
        graphs={}
        def select(arm,pi,shift=0):
            counts=patterns[pi];mode,method=arm.split('_');plan=dynamic if mode=='dynamic' else fixed[pi]
            if mode=='dynamic':
                receipt=plan.prepare(counts,lens(counts,shift),tables)
                signature=tuple(tuple(x) for x in receipt['signature'])
                key=(mode,method,signature)
            else:
                receipt=plan.update(groups,prefixes,counts,lens(counts,shift),tables)
                assert receipt['reusable'],receipt
                key=(mode,method,pi)
            fn=plan.native_full if method=='full' else plan.shared
            qbuffer.copy_(queries(counts,shift))
            if key not in graphs:
                started=time.perf_counter()
                for _ in range(3):fn(qbuffer)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):fn(qbuffer)
                torch.cuda.synchronize();graphs[key]=graph
                report['graph_captures'].append(dict(arm=arm,pattern=pi,shift=shift,
                    seconds=time.perf_counter()-started,plan_info={n:list(w._plan_info) for n,w in plan.wrappers.items()}))
            return plan,graphs[key],receipt
        arms=['dynamic_full','dynamic_shared','fixed_full','fixed_shared']
        for pi,counts in enumerate(patterns):
            for arm in arms:
                plan,graph,receipt=select(arm,pi);graph.replay()
                value=plan.outputs['full' if arm.endswith('full') else 'prefix']
                ref=references(counts)
                passed=bool(torch.isclose(value.double(),ref,atol=.006,rtol=.015).all())
                check=dict(arm=arm,pattern=pi,O_fp64_pass=passed,
                    max_abs=float((value.double()-ref).abs().max()),prepare=receipt)
                report['checks'].append(check);save();assert passed,check
        # Actual continuous suffix growth: one step stays within page counts;
        # two steps cross a row's page boundary in the current frozen inputs.
        for shift in (1,2):
            counts=patterns[0]
            reuse=fixed[0].update(groups,prefixes,counts,lens(counts,shift),tables)
            if not reuse['reusable']:
                start=time.perf_counter()
                fallback=FiniteShapeVerifyPlan(cache,groups,prefixes,counts,lens(counts,shift),tables,
                    fixture['k_scale'],fixture['v_scale'])
                actual=fallback.shared(queries(counts,shift))
                torch.cuda.synchronize();fallback_time=time.perf_counter()-start
                fallback_graph=torch.cuda.CUDAGraph();qbuffer.copy_(queries(counts,shift))
                start=time.perf_counter()
                with torch.cuda.graph(fallback_graph):fallback.shared(qbuffer)
                fallback_graph.replay();torch.cuda.synchronize()
                recapture_time=time.perf_counter()-start
                del fallback_graph
            else:
                fallback_time=recapture_time=0.;actual=fixed[0].shared(queries(counts,shift))
            finite_pass=bool(torch.isclose(actual.double(),references(counts,shift),atol=.006,rtol=.015).all())
            plan,graph,receipt=select('dynamic_shared',0,shift);graph.replay()
            dynamic_pass=bool(torch.isclose(plan.outputs['prefix'].double(),references(counts,shift),atol=.006,rtol=.015).all())
            row=dict(shift=shift,finite_reuse=reuse,finite_FP64_pass=finite_pass,dynamic_FP64_pass=dynamic_pass,
                     finite_fallback_prepare_seconds=fallback_time,finite_fallback_capture_seconds=recapture_time,
                     dynamic_prepare=receipt)
            report['growth_checks'].append(row);save();assert finite_pass and dynamic_pass,row
        # Restore base state before repeated allocation comparisons.
        for pi in range(3):
            for arm in arms:select(arm,pi)
        captures_before=len(graphs)
        # Future-position mutation checks on the actual captured graph paths.
        for arm in ('dynamic_shared','fixed_shared'):
            plan,graph,_=select(arm,0);graph.replay();before=plan.outputs['prefix'].clone()
            pos=lens(patterns[0])[0]-1;physical=int(tables[0,pos//16])
            typed=cache.view(torch.float8_e4m3fn) if cache.dtype==torch.uint8 else cache
            saved=typed[physical,:,pos%16].clone();typed[physical,:,pos%16].fill_(4.)
            graph.replay();mask=torch.ones(32,device='cuda',dtype=torch.bool);mask[3]=False
            unchanged=bool(torch.equal(plan.outputs['prefix'][mask],before[mask]))
            typed[physical,:,pos%16].copy_(saved)
            report['checks'].append(dict(arm=arm,graph_future_KV_no_leak=unchanged));save();assert unchanged
        for regime in ('hot','cold'):
            flush=torch.empty(256*1024**2,device='cuda',dtype=torch.uint8)
            blocks=[];rng=random.Random(20261017)
            for _ in range(args.blocks):
                order=[(arm,pi) for arm in arms for pi in range(3)]+[('dynamic_shared',0)]
                rng.shuffle(order);times={};walls={};seen={}
                for arm,pi in order:
                    base=f'{arm}_{pi}';seen[base]=seen.get(base,0)+1
                    name=base if seen[base]==1 else base+'_AA'
                    torch.cuda.synchronize();started=time.perf_counter()
                    plan,graph,receipt=select(arm,pi);graph.replay();torch.cuda.synchronize()
                    walls[name]=(time.perf_counter()-started)*1000
                    if regime=='cold':flush.zero_()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    repeats=20 if regime=='hot' else 1;begin.record()
                    for _ in range(repeats):graph.replay()
                    end.record();end.synchronize();times[name]=begin.elapsed_time(end)/repeats
                blocks.append(dict(gpu=times,prepare_switch_replay_wall_ms=walls))
            gpu=[x['gpu'] for x in blocks];wall=[x['prepare_switch_replay_wall_ms'] for x in blocks]
            report['timing'][regime]=dict(blocks=blocks,
                median_gpu_ms={n:statistics.median(x[n] for x in gpu) for n in gpu[0]},
                median_prepare_switch_replay_wall_ms={n:statistics.median(x[n] for x in wall) for n in wall[0]},
                dynamic_nonuniform_over_uniform={str(i):paired_ratio(gpu,f'dynamic_shared_{i}','dynamic_shared_0') for i in (1,2)},
                finite_nonuniform_over_uniform={str(i):paired_ratio(gpu,f'fixed_shared_{i}','fixed_shared_0') for i in (1,2)},
                AA=paired_ratio(gpu,'dynamic_shared_0_AA','dynamic_shared_0'))
            save();del flush
        assert len(graphs)==captures_before,'Unexpected graph recapture inside timed base-state loop'
        if args.profile:
            path=out.with_suffix('.trace.json')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for pi in range(3):
                    plan,graph,_=select('dynamic_shared',pi)
                    with torch.profiler.record_function(f'dynamic_shared_pattern_{pi}'):
                        graph.replay();torch.cuda.synchronize()
            prof.export_chrome_trace(str(path));report['profile_trace']=str(path)
        report['status']='completed';report['all_checks_passed']=True
        report['limitations']=['Fixed real AR queries only, no actual verifier whole-model or acceptance trace.',
            'Prepare path conservatively updates/plans all three wrappers in both methods; wall is a diagnostic, not production latency.',
            'Finite pool covers only three predeclared query partitions. Unknown shapes/pages require measured replan or graph-safe fallback.',
            'No superiority to prefix-aware cost-table greedy can be claimed without actual acceptance and held-out policies.']
        save()
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()
