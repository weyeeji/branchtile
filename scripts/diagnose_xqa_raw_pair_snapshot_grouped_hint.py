"""Replay a captured same-Q counterexample and localize prefix state divergence.

No timing or qualification claim. Uses frozen production CUDA modules. Native
public PDL-on output, unshared checkpoint and paired checkpoint are all retained.
"""
import argparse, hashlib, json, os
from pathlib import Path
import sys, traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--scan',action='store_true');args=ap.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    r={'status':'initializing','args':vars(args),'snapshot_sha256':sha(args.snapshot),
       'source_sha256':{f:sha(ROOT/f) for f in ['scripts/diagnose_xqa_raw_pair_snapshot_grouped_hint.py','src/xqa_raw_pair_padded_runtime.py','src/xqa_raw_pair_build.py','src/xqa_raw_grouped_hint_build.py','src/xqa_raw_continuation_build.py']}}
    def save():out.write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_grouped_hint_build import jit_specs
        from src.xqa_raw_pair_padded_runtime import RawPairPaddedPlan
        torch.set_num_threads(2)
        x=torch.load(args.snapshot,map_location='cpu',weights_only=True)
        assert x['schema']=='native_raw_pair_same_state_v1' and x['k_scale']==x['v_scale']==1.
        specs,receipt=jit_specs();mods={g:s.build_and_load() for g,s in specs.items()};r['build_identity']=receipt['identity']
        q=x['q'].contiguous().cuda();table=x['block_tables'].to(device='cuda',dtype=torch.int32).contiguous()
        cache=torch.empty_strided(x['cache'].shape,x['original_cache_stride'],device='cuda',dtype=x['cache'].dtype)
        cache.copy_(x['cache']);lens=x['seq_lens'].tolist();b=len(q)
        p=RawPairPaddedPlan(mods,q,cache,table,lens,x['prefix_tokens']);assert p.splits==x['S'] and p.sm==x['SM']
        r['contract']={'B':b,'S':p.splits,'P':p.prefix_tokens,'lengths':lens,'cache_stride':list(cache.stride()),'physical_compaction':True,'full_table_width':table.shape[1]}
        native=torch.empty_like(q);native2=torch.empty_like(q)
        def native_run(o):
            xqa(q.unsqueeze(1),p.k,p.v,table,p.lengths,o.unsqueeze(1),p.scratch,p.sem,2,16,q_scale=1.,kv_scale=1.,kv_layout='HND',sm_count=p.sm,enable_pdl=True)
        def compare(a,b):
            mask=a.view(torch.int16)!=b.view(torch.int16)
            return {'bits_equal':not bool(mask.any()),'mismatch_count':int(mask.sum()),'max_abs':float((a.float()-b.float()).abs().max()),'first_coordinates':mask.nonzero()[:10].cpu().tolist()}
        native_run(native);native_run(native2);pair=p.run().clone()
        r['native_AA']=compare(native,native2);r['native_vs_snapshot']=compare(native,x['native_o'].cuda());r['pair_vs_old_failed_candidate']=compare(pair,x['candidate_o'].cuda());r['pair_vs_native']=compare(pair,native);save()
        raw=torch.empty(b*2*p.splits*4*32*35,device='cuda',dtype=torch.float32);unused=torch.empty_like(q)
        def raw_call(cut,phase):
            mods[8].xqa_wrapper(False,p.tail_sm,2,0,1.,None,unused.unsqueeze(1),1.,q.unsqueeze(1),None,p.wrapper_k,p.wrapper_v,None,None,table,p.max_len,p.lengths,b,1.,None,1,None,None,p.sem,p.scratch,False,raw,cut,phase)
        def pair_prefix(cut):
            mods[16].xqa_wrapper(False,p.prefix_sm,2,0,1.,None,p.unused_prefix_o,1.,p.packed,None,p.wrapper_k,p.wrapper_v,None,None,p.pair_table,p.max_len,p.pair_lengths,p.groups,1.,None,1,None,None,p.sem,p.scratch,False,p.state,cut,1)
        def map_raw(a):
            a=a.reshape(p.groups,2,p.splits,4,32,35);n=torch.zeros((b,2,p.splits,4,32,35),device=a.device,dtype=a.dtype)
            for row in range(b):
                for i in range(32):
                    if i%4<2:n[row,...,i]=a[row//2,...,i+2*(row%2)]
                n[row,...,:8,32:]=a[row//2,...,8*(row%2):8*(row%2)+8,32:]
            return n
        valid=torch.zeros((b,2,p.splits,4,32,35),device=q.device,dtype=torch.bool)
        for i in range(32):
            if i%4<2:valid[...,i]=True
        valid[...,:8,32:]=True
        r['prefix_scan']=[];saved_first=False
        for cut in (range(1,p.prefix_tokens//256+1) if args.scan else [p.prefix_tokens//256]):
            pair_prefix(cut);raw_call(cut,1)
            mapped=map_raw(p.state);nr=raw.view_as(mapped)
            mask=(mapped.view(torch.int32)!=nr.view(torch.int32))&valid
            count=int(mask.sum())
            if count:
                coords=mask.nonzero();entry={'cut_tiles':cut,'mismatch_count':count,'first_coordinates':coords[:10].cpu().tolist(),'fields':{name:int(mask[...,lo:hi].sum()) for name,lo,hi in [('U',0,32),('m',32,33),('z',33,34),('hint',34,35)]}}
                r['prefix_scan'].append(entry)
                if not saved_first:
                    path=out.with_suffix('.first_raw.pt');torch.save({'cut_tiles':cut,'mapped_pair':mapped.cpu(),'native':nr.cpu(),'coordinates':coords.cpu(),'snapshot':args.snapshot},path)
                    r['first_raw']={'path':str(path),'sha256':sha(path)};saved_first=True;save()
        raw_call(p.prefix_tokens//256,1);raw_call(p.prefix_tokens//256,2);r['unshared_vs_native']=compare(unused,native)
        # Swap only the two members of each pair; retain original request S.
        perm=torch.arange(b,device=q.device)
        for i in range(0,b-1,2):perm[i]=i+1;perm[i+1]=i
        qp=q.index_select(0,perm).contiguous();tp=table.index_select(0,perm).contiguous();lp=[lens[int(i)] for i in perm.cpu()]
        pp=RawPairPaddedPlan(mods,qp,cache,tp,lp,x['prefix_tokens'])
        swapped=pp.run().index_select(0,perm)
        r['swapped_pair_vs_native']=compare(swapped,native)
        assert r['native_AA']['bits_equal'] and r['native_vs_snapshot']['bits_equal'] and r['pair_vs_native']['bits_equal'] and r['unshared_vs_native']['bits_equal'] and r['swapped_pair_vs_native']['bits_equal'], 'Grouped-hint candidate fails snapshot exactness'
        assert not r['prefix_scan'], 'Grouped-hint raw state still differs'
        r['status']='snapshot_qualified';save();print(json.dumps({k:r[k] for k in ['status','pair_vs_native','swapped_pair_vs_native']}))
    except Exception:r['status']='failed';r['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
