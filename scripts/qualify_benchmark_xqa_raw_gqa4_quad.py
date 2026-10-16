"""GQA4 four-request prefix correctness and complete-call benchmark; no E2E or inherited dispatch claim."""
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
from scripts.benchmark_xqa_raw_pair import paired_ratio


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture');p.add_argument('--batch',type=int,default=8)
    p.add_argument('--prefix',type=int,default=120528);p.add_argument('--extra-tail',type=int,default=0)
    p.add_argument('--blocks',type=int,default=30);p.add_argument('--output',required=True)
    a=p.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and a.blocks>=20 and a.batch in (4,5,6,7,8)
    out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    r={'status':'initializing','args':vars(a),'checks':{},'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
       ['scripts/qualify_benchmark_xqa_raw_gqa4_quad.py','src/xqa_raw_quad_gqa4_runtime.py','src/xqa_raw_quad_gqa4_build.py','src/xqa_raw_gqa4_pair_runtime.py','src/xqa_raw_gqa4_padded_runtime.py','src/xqa_raw_pair_gqa4_build.py']}}
    def save():out.write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_quad_gqa4_build import jit_specs
        from src.xqa_raw_quad_gqa4_runtime import RawGQA4QuadPlan
        torch.set_num_threads(2);torch.manual_seed(2026101706)
        assert torch.cuda.get_device_capability()==(12,0)
        specs,receipt=jit_specs();modules={g:s.build_and_load() for g,s in specs.items()};r['build_identity']=receipt['identity']
        payload=None
        if a.capture:
            path=Path(a.capture);payload=torch.load(path,map_location='cpu',weights_only=True)
            assert not payload['payload_cast_performed'] and payload['k_scale']==payload['v_scale']==1.
            assert len(payload['q'])==8, 'Actual captured cohort has eight independent queries'
            selected=[7,0,6,1,5,2,4,3][:a.batch]
            q=payload['q'][selected].cuda().contiguous();pkv=payload['prefix_kv'].cuda();tails=[payload['tail_kv'][i].cuda() for i in selected]
            assert q.dtype==pkv.dtype==torch.bfloat16 and pkv.shape[2:]==(4,128)
            b,prefix=len(q),pkv.shape[1]
            r['selected_original_rows']=selected
            r['captured_native_O_scope']='B8 original S5; subset B6/B7 requires fresh native output, never assumes captured O parity'
            r['capture']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            b,prefix=a.batch,a.prefix
            q=torch.randn((b,16,128),device='cuda',dtype=torch.bfloat16)
            pkv=torch.randn((2,prefix,4,128),device='cuda',dtype=torch.bfloat16)
            tails=[torch.randn((2,17+i*31,4,128),device='cuda',dtype=torch.bfloat16) for i in range(b)]
        assert b in (4,5,6,7,8)
        if a.extra_tail:tails=[torch.cat((t,torch.randn((2,a.extra_tail+i%7,4,128),device='cuda',dtype=torch.bfloat16)),1) for i,t in enumerate(tails)]
        lengths=[prefix+t.shape[1] for t in tails]
        if payload is not None and not a.extra_tail:
            cache=torch.empty_strided(payload['original_cache_shape'],payload['original_cache_stride'],device='cuda',dtype=torch.bfloat16);cache.zero_()
            table=payload['original_block_table'][selected].to(device='cuda',dtype=torch.int32).contiguous()
            r['layout']='preserved original native physical IDs and stride'
        else:
            pp=prefix//16;tp=max((t.shape[1]+15)//16 for t in tails)+2;npages=pp+b*tp
            cache=torch.randn((npages,16,4,256),device='cuda',dtype=torch.bfloat16).permute(0,2,1,3)
            perm=torch.randperm(npages,device='cuda',dtype=torch.int32);table=torch.empty((b,pp+tp),device='cuda',dtype=torch.int32);table[:,:pp]=perm[:pp]
            for i in range(b):table[i,pp:]=perm[pp+i*tp:pp+(i+1)*tp]
            r['layout']='random remapped synthetic layout (and synthetic tail stress if capture supplied)'
        for i in range(b):
            kv=torch.cat((pkv,tails[i]),1);token=torch.arange(lengths[i],device='cuda');cache[table[i,token//16].long(),:,token%16]=torch.cat((kv[0],kv[1]),-1)
        del pkv,tails,kv,token
        started=time.perf_counter();plan=RawGQA4QuadPlan(modules,q,cache,table,lengths,prefix);torch.cuda.synchronize()
        r.update(batch=b,prefix=prefix,lengths=lengths,S=plan.splits,setup_seconds=time.perf_counter()-started)
        nw=torch.zeros_like(plan.workspace);outputs={k:torch.empty_like(q) for k in ['native','AA','native_PDL_off','unshared_checkpoint']}
        raw_unshared=torch.empty(b*4*plan.splits*4*32*35,device='cuda',dtype=torch.float32)
        def native(key,pdl=True):xqa(q.unsqueeze(1),plan.k,plan.v,table,plan.lengths,outputs[key].unsqueeze(1),nw[8*1024*1024:],nw[:8*1024*1024],4,16,q_scale=1.,kv_scale=1.,kv_layout='HND',sm_count=plan.sm,enable_pdl=pdl)
        def unshared():
            for phase in [1,2]:plan.call(4,q.unsqueeze(1),outputs['unshared_checkpoint'].unsqueeze(1),table,plan.lengths,b,phase,raw_unshared)
        funcs={'native':lambda:native('native'),'AA':lambda:native('AA'),'native_PDL_off':lambda:native('native_PDL_off',False),'unshared_checkpoint':unshared,'candidate':plan.run}
        pair=None
        if b in (6,7,8):
            from src.xqa_raw_pair_gqa4_build import jit_specs as pair_specs
            from src.xqa_raw_gqa4_padded_runtime import RawGQA4PaddedPlan
            ps,pr=pair_specs();pm={g:s.build_and_load() for g,s in ps.items()}
            pair=RawGQA4PaddedPlan(pm,q,cache,table,lengths,prefix)
            funcs['pair']=pair.run;outputs['pair']=pair.output;r['pair_build_identity']=pr['identity']
        def bits(x,y):return torch.equal(x.contiguous().view(torch.int16),y.contiguous().view(torch.int16))
        for fn in funcs.values():fn()
        for key,o in {**outputs,'candidate':plan.output}.items():r['checks'][key+'_bits']=bits(outputs['native'],o)
        if payload is not None and not a.extra_tail and b==8:r['checks']['original_native_O_bits']=bits(outputs['native'],payload['o'][selected].cuda())
        for row in [0,b-1]:
            token=torch.arange(lengths[row],device='cuda');kv=cache[table[row,token//16].long(),:,token%16].double()
            score=torch.einsum('hgd,thd->hgt',q[row].double().reshape(4,4,128),kv[...,:128])/128**.5
            ref=torch.einsum('hgt,thd->hgd',score.softmax(-1),kv[...,128:]).reshape(16,128)
            for key,o in [('native',outputs['native']),('candidate',plan.output)]:r['checks'][f'{key}_FP64_row{row}']=torch.allclose(o[row].double(),ref,atol=.006,rtol=.015)
        save();assert all(r['checks'].values()),r['checks']
        qualification_started=time.perf_counter()
        # Native int32 device metadata alias, captured initially at dummy length1.
        # Same pointer then holds real unequal lengths. Do not invoke host prepare here.
        saved_lengths=plan.lengths.clone();q_saved=q.clone()
        native_lengths=torch.ones(b,device='cuda',dtype=torch.int32)
        plan.lengths=native_lengths.view(torch.uint32).reshape(b,1)
        if pair is not None:pair.lengths=plan.lengths
        assert plan.lengths.data_ptr()==native_lengths.data_ptr()
        dummy_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(dummy_graph):plan.run()
        for label,values in [('dummy_one',[1]*b),('changed_real',[x-min(3,x-prefix-1) for x in lengths])]:
            native_lengths.copy_(torch.tensor(values,device='cuda',dtype=torch.int32));q.mul_(.875)
            plan.state.fill_(float('nan'));dummy_graph.replay();native('native');torch.cuda.synchronize()
            r['checks'][label+'_capture_replay_bits']=bits(outputs['native'],plan.output)
        # Same original graph after Q/length tests: swap actual pair membership in-place.
        plan.lengths.copy_(saved_lengths)
        q.copy_(q_saved)
        rows=torch.arange(b,device='cuda').flip(0)
        oldtable=table.clone();oldlens=plan.lengths.clone()
        q.copy_(q.index_select(0,rows));table.copy_(table.index_select(0,rows));native_lengths.copy_(native_lengths.index_select(0,rows))
        plan.state.fill_(float('nan'));dummy_graph.replay();native('native');torch.cuda.synchronize()
        r['checks']['swapped_pair_membership_bits']=bits(outputs['native'],plan.output)
        q.copy_(q_saved);table.copy_(oldtable);plan.lengths.copy_(oldlens)
        r['checks']['all_dummy_Q_zero']=bool((plan.padded_q[b:]==0).all())
        save();assert all(r['checks'].values()),r['checks']
        r['alias_graph_qualification_seconds']=time.perf_counter()-qualification_started
        # Timed graphs all retain the same native-int32 length alias.
        graphs={};started=time.perf_counter()
        for key,fn in funcs.items():
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            graphs[key]=graph
        torch.cuda.synchronize();r['capture_seconds']=time.perf_counter()-started
        eviction=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8);rng=random.Random(2026101706)
        for regime,replays in [('hot',10),('cold',1)]:
            blocks=[];orders=[]
            for _ in range(a.blocks):
                order=list(graphs);rng.shuffle(order);orders.append(order);block={}
                for key in order:
                    if regime=='cold':eviction.fill_(rng.randrange(256))
                    else:graphs[key].replay()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                    for _ in range(replays):graphs[key].replay()
                    end.record();end.synchronize();block[key]=begin.elapsed_time(end)/replays
                blocks.append(block)
            r[regime]={'blocks_ms':blocks,'orders':orders,'median_ms':{k:statistics.median(x[k] for x in blocks) for k in graphs},
                       'native_over_candidate':paired_ratio(blocks,'native','candidate'),'AA':paired_ratio(blocks,'native','AA'),
                       'native_over_unshared':paired_ratio(blocks,'native','unshared_checkpoint')}
            if pair is not None:r[regime]['pair_over_quad']=paired_ratio(blocks,'pair','candidate')
            save()
        for key,o in {**outputs,'candidate':plan.output}.items():r['checks']['final_'+key+'_bits']=bits(outputs['native'],o)
        assert all(r['checks'].values());r['status']='qualified_and_measured'
        r['limitations']=['Complete four-request call includes Q packing/dummy rows/table+length D2D/state export+resume+native merge.',
                          'Setup/capture separate; no whole-model speed claim. New GQA4 geometry does not inherit GQA8 dispatch.',
                          'Original captured O bits checked only for full B8 without synthetic tail. B6/B7 compare fresh same-B native outputs and FP64.',
                          'Membership subset is actual captured Q at one state, not later natural survivor Q; new batch preserves its own native S.',
                          'Batch pads to a multiple of four prefix Q rows; original B/GQA4/S tail has no orphan scan. Pair baseline present only for its already supported B6/7/8.']
    except Exception:r['status']='failed';r['error']=traceback.format_exc();save();raise
    save();print(json.dumps({'status':r['status'],'output':str(out)}))
if __name__=='__main__':main()
