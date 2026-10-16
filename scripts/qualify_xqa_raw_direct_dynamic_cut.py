"""Same captured graph across device cuts, physical mappings, Q and live lengths.

Standalone synthetic correctness gate only, not a model or performance claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',default='6,7,8,9,14,19,21')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    files=['src/xqa_raw_direct_input_build.py','src/xqa_raw_direct_input_runtime.py','src/xqa_raw_direct_dynamic_cut_runtime.py','src/xqa_raw_grouped_hint_build.py','src/xqa_raw_pair_build.py','src/xqa_raw_continuation_build.py','scripts/qualify_xqa_raw_direct_dynamic_cut.py','src/xqa_raw_direct_dynamic_cut_build.py','src/xqa_raw_pair_dynamic_cut_runtime.py','src/xqa_raw_pair_padded_runtime.py','src/xqa_raw_pair_general_runtime.py']
    r={'status':'initializing','scope':__doc__,'args':vars(a),'cases':[], 'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    def save():out.write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_direct_dynamic_cut_build import jit_specs
        from src.xqa_raw_direct_dynamic_cut_runtime import RawPairDirectDynamicCutPlan
        torch.set_num_threads(2);torch.manual_seed(2026101711)
        specs,receipt=jit_specs();modules={g:s.build_and_load() for g,s in specs.items()};r['build_identity']=receipt['identity']
        cuts=[256,272,65520,65536,81920,113280,120528,153552,65536,113280]
        mp=max(cuts)//16;tp=514
        for b in map(int,a.batches.split(',')):
            np=mp+b*tp
            cache=torch.randn((np,16,2,256),device='cuda',dtype=torch.bfloat16).permute(0,2,1,3)
            q=torch.randn((b,16,128),device='cuda',dtype=torch.bfloat16)
            table=torch.zeros((b,mp+tp),device='cuda',dtype=torch.int32)
            lengths=[cuts[0]+17+i*31 for i in range(b)]
            def remap(cut):
                pp=cut//16;perm=torch.randperm(np,device='cuda',dtype=torch.int32)
                table.zero_();table[:,:pp]=perm[:pp]
                for row in range(b):table[row,pp:pp+tp]=perm[mp+row*tp:mp+(row+1)*tp]
            remap(cuts[0]);plan=RawPairDirectDynamicCutPlan(modules,q,cache,table,lengths,cuts[0])
            wrong_cut=torch.tensor([999],device='cuda',dtype=torch.int32)
            try:
                RawPairDirectDynamicCutPlan(modules,q,cache,table,lengths,cuts[0],device_cut=wrong_cut)
            except AssertionError:pass
            else:raise AssertionError('Wrong external device cut accepted')
            old_cut=plan.device_cut.clone();old_lengths=plan.lengths.clone();old_prefix=plan.prefix_tokens
            for bad_prefix,bad_lengths in [(512,[plan.max_len+1]*b),(plan.max_len+16,[plan.max_len+32]*b)]:
                try:plan.change_prefix(bad_prefix,bad_lengths)
                except AssertionError:pass
                else:raise AssertionError('Out-of-extent cut or length accepted')
                assert plan.prefix_tokens==old_prefix and torch.equal(plan.device_cut,old_cut)
                assert torch.equal(plan.lengths.view(torch.int32),old_lengths.view(torch.int32))
            saved_page=table[1,0].clone();table[1,0]=(table[0,0]+1)%np
            try:plan.change_prefix(cuts[0],lengths)
            except AssertionError:pass
            else:raise AssertionError('Foreign physical prefix accepted')
            assert plan.prefix_tokens==old_prefix and torch.equal(plan.device_cut,old_cut)
            assert torch.equal(plan.lengths.view(torch.int32),old_lengths.view(torch.int32))
            table[1,0].copy_(saved_page)
            r.setdefault('negative_controls',[]).append({'B':b,'wrong_external_cut_rejected':True,'invalid_update_unchanged':True,'foreign_physical_prefix_rejected':True})
            native_lengths=torch.ones(b,device='cuda',dtype=torch.int32)
            plan.lengths=native_lengths.view(torch.uint32).reshape(b,1)
            nw=torch.zeros_like(plan.workspace);no=torch.empty_like(q);aa=torch.empty_like(q)
            def native(o):xqa(q.unsqueeze(1),plan.k,plan.v,table,plan.lengths,o.unsqueeze(1),nw[8*1024*1024:],nw[:8*1024*1024],2,16,q_scale=1.,kv_scale=1.,kv_layout='HND',sm_count=plan.sm,enable_pdl=True)
            plan.run();native(no)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):plan.run()
            for index,cut in enumerate(cuts):
                remap(cut);q.normal_();cache.mul_(.99)
                lengths=[cut+8192-row*7 if index%2 else cut+17+row*31 for row in range(b)]
                plan.change_prefix(cut,lengths);plan.state.fill_(float('nan'))
                graph.replay();native(no);native(aa);torch.cuda.synchronize()
                bits=torch.equal(no.view(torch.int16),plan.output.view(torch.int16));aabits=torch.equal(no.view(torch.int16),aa.view(torch.int16))
                c={'B':b,'S':plan.splits,'cut':cut,'index':index,'same_graph':True,'native_AA_bits':aabits,'native_candidate_bits':bits,'max_abs':float((no.float()-plan.output.float()).abs().max())}
                r['cases'].append(c);save()
                assert bits and aabits,c
                if index in (0,5,7):
                    row=b-1;token=torch.arange(lengths[row],device='cuda');kv=cache[table[row,token//16].long(),:,token%16].double()
                    score=torch.einsum('hgd,thd->hgt',q[row].double().reshape(2,8,128),kv[...,:128])/128**.5
                    ref=torch.einsum('hgt,thd->hgd',score.softmax(-1),kv[...,128:]).reshape(16,128)
                    c['FP64_row']=row;c['FP64_pass']=torch.allclose(plan.output[row].double(),ref,atol=.006,rtol=.015);save();assert c['FP64_pass']
            del graph,plan,cache,q,table,nw,no,aa,native_lengths
        r['status']='qualified'
        r['limitations']=['Synthetic standalone gate; no natural output, heldout, whole-model or speed claim.', 'Graph has fixed B and addresses, cut/table/content/lengths change on the same stream outside replay.', 'Short cuts are correctness stress outside frozen production dispatch.']
    except Exception:
        r['status']='failed';r['error']=traceback.format_exc();save();raise
    save()
    print(json.dumps({'status':r['status'],'output':str(out)}))


if __name__=='__main__':main()
