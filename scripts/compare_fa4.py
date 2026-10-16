"""Paired exact-mask FA4/FI/own-union replay, including required copying."""
import argparse
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compare_attention import load_capture,graph_of,elapsed
from branchtile.attention import make_plan,reference
from branchtile.union import FixedUnion
from flashinfer_baseline import NativeSparseDecode
from flashinfer_union import NativeUnionPrefill
from fa4_baseline import FullSourceMasked,GatheredDecode


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--layer',type=int,required=True)
    p.add_argument('--step',type=int,default=256);p.add_argument('--heads',type=int,required=True)
    p.add_argument('--budget',type=int,default=4096);p.add_argument('--blocks',type=int,default=20);p.add_argument('--output',required=True)
    a=p.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    torch.set_num_threads(4);rng=random.Random(20261016)
    q,k,v,owners,b,g,provenance=load_capture(a.capture,a.layer,a.budget,a.step,a.heads)
    ref,ref_lse=reference(q,k,v,owners,b,g)
    plan=make_plan(owners,b,g,'cuda')
    supports=[[torch.where((owners[h]&(1<<i))!=0)[0] for i in range(b)] for h in range(k.shape[0])]
    props=torch.cuda.get_device_properties(0)
    flush=torch.zeros(max(2*props.L2_cache_size,256*1024*1024)//4,dtype=torch.float32,device='cuda')
    factories=[]
    for name,cls in [('fa4_source_mask',FullSourceMasked),('fa4_gather_e0',GatheredDecode)]:
        for pack,tile in itertools.product((True,False),((128,64),(64,64))):
            factories.append((name,{'pack_gqa':pack,'tile_mn':tile},lambda cls=cls,pack=pack,tile=tile:cls(q,k,v,owners,b,pack_gqa=pack,tile_mn=tile)))
    for split in (None,128,256,512,1024):
        factories.append(('fi_decode',{'split':split,'tensor_cores':True},lambda split=split:NativeSparseDecode(q,k,v,supports,fixed_split_size=split,normalize_lse=True)))
        factories.append(('fi_union',{'split':split},lambda split=split:NativeUnionPrefill(q,k,v,owners,b,fixed_split_size=split,normalize_lse=True)))
    factories.append(('fi_decode',{'split':None,'tensor_cores':False},lambda:NativeSparseDecode(q,k,v,supports,use_tensor_cores=False,normalize_lse=True)))
    for bn,split,threads in itertools.product((32,64),(16,32,64,128),(128,256)):
        factories.append(('own_union',{'block_n':bn,'split':split,'threads':threads},lambda bn=bn,split=split,threads=threads:FixedUnion(plan,owners,split,bn,threads)))
    rng.shuffle(factories);best={'hot':{},'cold':{}};tuning=[]
    result={'status':'tuning','uid':os.geteuid(),'python':sys.executable,'gpu_uuid':str(props.uuid),
            'versions':{x:importlib.metadata.version(x) for x in ('torch','flash-attn-4','flashinfer-python','nvidia-cutlass-dsl','apache-tvm-ffi','tilelang')},
            'provenance':provenance,'q_shape':list(q.shape),'k_shape':list(k.shape),'source_bytes':2*k.numel()*k.element_size(),
            'lse_contract':'Natural log for all arms. FA4 produces ln internally, FI log2-to-ln multiplication is timed, and output layout conversions are timed.',
            'fa4_api':'Unmodified official flash_attn.cute.interface._flash_attn_fwd with preallocated output/LSE; SM120 num_splits=0 resolves to 1.',
            'fa4_source_mask_contract':'Shares original physical K/V without payload copy, exact arbitrary token mask, visits whole source; not paged or block-sparse.',
            'fa4_gather_e0_contract':'Independent per-head/request ragged supports; every complete run includes both index_select K/V gathers plus attention, never credited as zero-copy.',
            'scope':'Same-mask operator replay, no application or quality claim; plan and compilation costs separate from repeated complete run.',
            'tuning':tuning}
    outpath=Path(a.output);outpath.parent.mkdir(parents=True,exist_ok=True)
    def save():outpath.write_text(json.dumps(result,indent=2)+'\n')
    for name,config,construct in factories:
        row={'method':name,**config};start=time.monotonic()
        try:
            op=construct();fn=(lambda op=op:op(q,k,v)) if name=='own_union' else op.run
            out,lse=fn();torch.cuda.synchronize()
            row.update(output_max_abs=float((out.double()-ref).abs().max()),lse_max_abs=float((lse.double()-ref_lse).abs().max()),preparation_and_first_call_seconds=time.monotonic()-start)
            torch.testing.assert_close(out.float(),ref.float(),atol=.006,rtol=.015)
            torch.testing.assert_close(lse.float(),ref_lse.float(),atol=.0003,rtol=.0001)
            row['tuning_us']={}
            for regime,reps in [('hot',20),('cold',1)]:
                graph=graph_of(fn,reps)
                row['tuning_us'][regime]=statistics.median(elapsed(graph,reps,flush if regime=='cold' else None) for _ in range(3))
            row.update(status='passed',payload_copy_bytes_per_call=getattr(op,'payload_copy_bytes_per_call',0))
            for regime in best:
                if name not in best[regime] or row['tuning_us'][regime]<best[regime][name][0]['tuning_us'][regime]:best[regime][name]=(row.copy(),op,fn)
        except Exception as exc:
            row.update(status='failed',error=repr(exc))
            if 'illegal memory' in str(exc).lower():tuning.append(row);save();raise
        tuning.append(row);save();print(json.dumps(row),flush=True)
    result['selected']={r:{m:x[0] for m,x in arms.items()} for r,arms in best.items()}
    required={'fa4_source_mask','fa4_gather_e0','fi_decode','fi_union','own_union'}
    if any(required-set(arms) for arms in best.values()):
        result.update(status='failed_qualification',reason='Required arm has no passing exact-mask configuration');save();return
    measurements=[]
    for regime,reps in [('hot',20),('cold',1)]:
        graphs={m:graph_of(item[2],reps) for m,item in best[regime].items()}
        graphs['fi_union_AA']=graph_of(best[regime]['fi_union'][2],reps)
        for block in range(a.blocks):
            order=list(graphs);rng.shuffle(order)
            for direction,methods in [('random',order),('reverse',list(reversed(order)))]:
                measurements.append({'cache':regime,'block':block,'direction':direction,'order':methods,
                    'us':{m:elapsed(graphs[m],reps,flush if regime=='cold' else None) for m in methods}})
    summary={};brng=np.random.default_rng(81)
    for regime in best:
        rows=[r for r in measurements if r['cache']==regime]
        summary[regime]={'median_us':{m:statistics.median(r['us'][m] for r in rows) for m in rows[0]['us']},
            'fi_union_AA_median_ratio':statistics.median(r['us']['fi_union']/r['us']['fi_union_AA'] for r in rows)}
        for method in required-{'fi_union'}:
            ratios=np.array([[next(r for r in rows if r['block']==i and r['direction']==d)['us'][method]/next(r for r in rows if r['block']==i and r['direction']==d)['us']['fi_union'] for d in ('random','reverse')] for i in range(a.blocks)])
            boot=np.median(ratios[brng.integers(0,a.blocks,size=(10000,a.blocks))],axis=(1,2))
            summary[regime][method+'_over_fi_union']={'median':float(np.median(ratios)),'paired_block_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    result.update(status='completed',measurements=measurements,summary=summary);save();print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
