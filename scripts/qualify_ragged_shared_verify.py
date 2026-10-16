"""G0 causal-attention qualification/cost on real AR queries or marked synthetic data."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synthetic_fixture(dtype):
    import torch
    gen = torch.Generator().manual_seed(20261017)
    groups, prefixes = [0]*4+[1]*4, [256,272]
    ids, next_page = [], sum(p//16 for p in prefixes)
    for r,g in enumerate(groups):
        start = 0 if g==0 else prefixes[0]//16
        prefix_pages = list(range(start,start+prefixes[g]//16))
        ids.append(prefix_pages+list(range(next_page,next_page+3)))
        next_page += 3
    table = torch.zeros((8,max(map(len,ids))),dtype=torch.int32)
    for r,row in enumerate(ids):
        table[r,:len(row)] = torch.tensor(row)
    cache = torch.randn((next_page,2,16,256),generator=gen).mul_(.3)
    cache = cache.to(torch.bfloat16 if dtype=='bf16' else torch.float8_e4m3fn)
    if dtype!='bf16':
        cache = cache.view(torch.uint8)
    # Include private-tail page crossings and differing historical tail lengths.
    starts = torch.tensor([prefixes[g]+t for g,t in zip(groups,[1,7,15,16,17,23,31,32])])
    positions = starts[:,None]+torch.arange(8)[None,:]
    return dict(cache=cache,block_tables=table,q=torch.randn((8,8,16,128),generator=gen).bfloat16(),
                native_o=None,positions=positions,groups=groups,prefixes=prefixes,
                k_scale=1.25,v_scale=.75,stride=[8192,256,512,1],
                kv_semantic_dtype='bfloat16' if dtype=='bf16' else 'float8_e4m3fn',
                provenance=[dict(scope='synthetic_only_not_real_model_queries',seed=20261017)])


def reference_all(cache, fixture, query):
    import torch
    typed = cache.view(torch.float8_e4m3fn) if cache.dtype==torch.uint8 else cache
    result = torch.empty_like(query,dtype=torch.float64)
    lse = torch.empty(query.shape[:-1],device=query.device,dtype=torch.float64)
    tables = fixture['block_tables'].to(query.device)
    for r in range(query.shape[0]):
        pos = fixture['positions'][r].to(query.device)
        logical = torch.arange(int(pos[-1])+1,device=query.device)
        payload = typed[tables[r,logical//16].long(),:,logical%16,:].double()
        for h in range(2):
            q = query[r,:,h*8:(h+1)*8].double().reshape(-1,128)
            scores = (q@(payload[:,h,:128]*fixture['k_scale']).T)*(128**-.5)
            scores.masked_fill_(logical[None,:] > pos.repeat_interleave(8)[:,None], float('-inf'))
            result[r,:,h*8:(h+1)*8] = (torch.softmax(scores,dim=-1)@(payload[:,h,128:]*fixture['v_scale'])).reshape(8,8,128)
            lse[r,:,h*8:(h+1)*8] = torch.logsumexp(scores,dim=-1).reshape(8,8)
        del payload
    return result,lse


def summarize(blocks):
    rng = random.Random(7731)
    ratios = []
    for _ in range(1000):
        selected = [rng.choice(blocks) for _ in blocks]
        ratios.append(statistics.median(x['full'] for x in selected)/statistics.median(x['shared'] for x in selected))
    ratios.sort()
    med = {n:statistics.median(x[n] for x in blocks) for n in ('full','full_AA','shared')}
    return dict(median_ms=med,native_over_shared=med['full']/med['shared'],
                paired_bootstrap_95ci=[ratios[25],ratios[974]],AA_ratio=med['full']/med['full_AA'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture',action='append',default=[])
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--dtype',choices=['bf16','fp8'],default='fp8')
    p.add_argument('--depths',help='Optional comma-separated depths for eight captured requests')
    p.add_argument('--blocks',type=int,default=30)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert bool(args.capture) != args.synthetic
    assert args.blocks>=10
    out = Path(args.output)
    assert not out.exists()
    out.parent.mkdir(parents=True,exist_ok=True)
    report = dict(status='initializing',args=vars(args),uid=os.geteuid(),python=sys.executable,
                  scope=__doc__,cases=[],source_sha256={})
    for name in ('scripts/qualify_ragged_shared_verify.py','src/ragged_shared_verify_plan.py','src/ragged_capture_fixture.py'):
        report['source_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    def save():
        out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        from src.ragged_capture_fixture import load_ragged_fixture
        from src.ragged_shared_verify_plan import RaggedSharedVerifyPlan,ragged_metadata
        torch.set_num_threads(4)
        assert torch.cuda.get_device_capability()==(12,0)
        fixture = synthetic_fixture(args.dtype) if args.synthetic else load_ragged_fixture(args.capture)
        report['provenance'] = fixture['provenance']
        report['input_contract'] = 'synthetic qualification' if args.synthetic else 'real consecutive AR Q and final KV, causal multiquery attention replay; not actual whole-model verifier'
        shape = fixture['cache'].shape
        span = (sum((n-1)*s for n,s in zip(shape,fixture['stride']))+1)*fixture['cache'].element_size()
        assert span<22*1024**3
        cache = torch.empty_strided(shape,fixture['stride'],device='cuda',dtype=fixture['cache'].dtype)
        cache.copy_(fixture['cache'])
        queries = fixture['q'].cuda()
        expected,expected_lse = reference_all(cache,fixture,queries)
        if fixture['native_o'] is not None:
            ar = fixture['native_o'].cuda().double()
            report['original_AR_O_fp64_pass'] = bool(torch.isclose(ar,expected,atol=.006,rtol=.015).all())
            report['original_AR_O_fp64_maxabs'] = float((ar-expected).abs().max())
            save()
            assert report['original_AR_O_fp64_pass']
        patterns = [[n]*8 for n in (1,2,4,8)]+[[8,8,8,4,1,1,1,1],[8,1,8,1,8,1,4,1]]
        if args.depths:
            patterns = [list(map(int,args.depths.split(',')))]
        flush = torch.empty(256*1024**2,device='cuda',dtype=torch.uint8)
        for counts in patterns:
            assert len(counts)==8 and all(1<=n<=8 for n in counts)
            lens = [int(fixture['positions'][r,0])+n for r,n in enumerate(counts)]
            q = torch.cat([queries[r,:n] for r,n in enumerate(counts)]).contiguous()
            ref = torch.cat([expected[r,:n] for r,n in enumerate(counts)])
            ref_lse = torch.cat([expected_lse[r,:n] for r,n in enumerate(counts)])
            started = time.perf_counter()
            plan = RaggedSharedVerifyPlan(cache,fixture['groups'],fixture['prefixes'],counts,lens,
                fixture['block_tables'],fixture['k_scale'],fixture['v_scale'])
            torch.cuda.synchronize()
            case = dict(query_counts=counts,total_queries=sum(counts),groups=fixture['groups'],
                        prefix_tokens=fixture['prefixes'],seq_lens=lens,prepare=plan.receipt,
                        first_prepare_synchronized_seconds=time.perf_counter()-started,checks={},timing={})
            report['cases'].append(case)
            functions = {'full':lambda:plan.native_full(q),'shared':lambda:plan.shared(q)}
            for name,fn in functions.items():
                value = fn()
                actual_lse = plan.lse['full' if name=='full' else 'prefix'].double()*math.log(2)
                check = dict(O_pass=bool(torch.isclose(value.double(),ref,atol=.006,rtol=.015).all()),
                    O_max_abs=float((value.double()-ref).abs().max()),
                    LSE_pass=bool(torch.isclose(actual_lse,ref_lse,atol=.0003,rtol=.0001).all()),
                    LSE_max_abs=float((actual_lse-ref_lse).abs().max()))
                case['checks'][name] = check
                save()
                assert check['O_pass'] and check['LSE_pass'],check
            # Corrupt the last real query's K/V in a private page. All earlier
            # queries and all other requests must remain bitwise unchanged.
            if any(n>1 for n in counts):
                row = next(r for r,n in enumerate(counts) if n>1)
                pos = lens[row]-1
                physical = int(fixture['block_tables'][row,pos//16])
                typed = cache.view(torch.float8_e4m3fn) if cache.dtype==torch.uint8 else cache
                original = typed[physical,:,pos%16].clone()
                before = {name:fn().clone() for name,fn in functions.items()}
                typed[physical,:,pos%16].fill_(4.)
                mask = torch.ones(q.shape[0],dtype=torch.bool,device='cuda')
                mask[sum(counts[:row+1])-1] = False
                unchanged = {name:bool(torch.equal(fn()[mask],before[name][mask])) for name,fn in functions.items()}
                typed[physical,:,pos%16].copy_(original)
                case['checks']['future_KV_does_not_leak'] = unchanged
                save()
                assert all(unchanged.values()),unchanged
            # CPU negative contract: a different physical prefix must reject.
            changed = fixture['block_tables'].clone()
            changed[1,0] = changed[1,0]+1
            try:
                ragged_metadata(fixture['groups'],fixture['prefixes'],counts,lens,changed)
            except AssertionError:
                case['checks']['different_prefix_rejected'] = True
            else:
                raise AssertionError('Divergent prefix silently accepted')
            graphs = {}
            started = time.perf_counter()
            for name,fn in functions.items():
                for _ in range(3):fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):fn()
                graphs[name] = graph
            torch.cuda.synchronize()
            case['graph_capture_seconds'] = time.perf_counter()-started
            def timed(name,repeats):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):graphs['full' if name=='full_AA' else name].replay()
                end.record();end.synchronize()
                return start.elapsed_time(end)/repeats
            rng = random.Random(20261017)
            for regime in ('hot','cold'):
                blocks = []
                for _ in range(args.blocks):
                    order = ['full','full_AA','shared'];rng.shuffle(order)
                    times = {}
                    for name in order:
                        if regime=='cold':flush.zero_()
                        times[name] = timed(name,20 if regime=='hot' else 1)
                    blocks.append(times)
                case['timing'][regime] = dict(blocks=blocks,summary=summarize(blocks))
            save()
            del graphs,plan
        report['status'] = 'qualified'
        report['all_passed'] = True
        report['performance_scope'] = 'Complete attention operator only; static offline preparation and graph costs reported separately, not hidden online costs'
        save()
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();save();raise


if __name__=='__main__':
    main()
