"""Minimal GQA4->8 pair qualification, Qwen3-4B TP2 geometry; synthetic only.

Original B/Hkv/S and every native sequence tile remain unchanged. Even B only.
This is a correctness gate, not an extrapolated model or performance result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = Path(a.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'scope': __doc__, 'cases': [],
              'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                ['scripts/qualify_xqa_raw_gqa4.py', 'src/xqa_raw_pair_gqa4_build.py']}}
    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_pair_gqa4_build import jit_specs
        torch.set_num_threads(2)
        torch.manual_seed(2026101705)
        assert torch.cuda.get_device_capability() == (12, 0)
        specs, receipt = jit_specs()
        modules = {g: s.build_and_load() for g, s in specs.items()}
        report['build_identity'] = receipt['identity']
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        workspace = torch.zeros(64*1024*1024, device='cuda', dtype=torch.uint8)
        nw = torch.zeros_like(workspace)
        sem, scratch = workspace[:8*1024*1024], workspace[8*1024*1024:]
        for b, prefix in [(8, 256), (8, 768), (8, 120528), (6, 120528)]:
            groups, hkv, pp, tp = b//2, 4, prefix//16, 520
            npages = pp + b*tp
            cache = torch.randn((npages,16,4,256), device='cuda', dtype=torch.bfloat16).permute(0,2,1,3)
            perm = torch.randperm(npages, device='cuda', dtype=torch.int32)
            table = torch.empty((b,pp+tp),device='cuda',dtype=torch.int32)
            table[:,:pp] = perm[:pp]
            for i in range(b):
                table[i,pp:] = perm[pp+i*tp:pp+(i+1)*tp]
            assert bool((table[::2,:pp] == table[1::2,:pp]).all())
            q = torch.randn((b,16,128),device='cuda',dtype=torch.bfloat16)
            lengths = torch.tensor([prefix+17+37*i for i in range(b)], device='cuda',dtype=torch.int32)
            lens = lengths.view(torch.uint32).reshape(b,1)
            maxlen = table.shape[1]*16
            splits = min(max(1, sm//(b*hkv)), (maxlen+255)//256)
            state = torch.empty(groups*hkv*splits*4*32*35,device='cuda',dtype=torch.float32)
            packed = torch.empty((groups,1,32,128),device='cuda',dtype=torch.bfloat16)
            pairtable = torch.empty_like(table[::2])
            pairlens = torch.empty_like(lens[::2])
            po = torch.empty_like(packed)
            candidate_o, native_o, aa_o = torch.empty_like(q),torch.empty_like(q),torch.empty_like(q)
            k, v = cache[...,:128], cache[...,128:]
            wk,wv = k.transpose(1,2),v.transpose(1,2)
            def call(group, query, output, bt, ln, batch, phase, sms):
                modules[group].xqa_wrapper(False,sms,hkv,0,1.,None,output,1.,query,
                    None,wk,wv,None,None,bt,maxlen,ln,batch,1.,None,1,None,None,
                    sem,scratch,False,state,prefix//256,phase)
            def candidate():
                packed.view(groups,4,2,4,128).copy_(q.view(groups,2,4,4,128).permute(0,2,1,3,4))
                pairtable.copy_(table[::2]); pairlens.copy_(lens[::2])
                call(8,packed,po,pairtable,pairlens,groups,1,splits*groups*hkv)
                call(4,q.unsqueeze(1),candidate_o.unsqueeze(1),table,lens,b,3,splits*b*hkv)
            def native(o, pdl=True):
                xqa(q.unsqueeze(1),k,v,table,lens,o.unsqueeze(1),nw[8*1024*1024:],nw[:8*1024*1024],
                    hkv,16,q_scale=1.,kv_scale=1.,kv_layout='HND',sm_count=sm,enable_pdl=pdl)
            def bits(a,b):
                return torch.equal(a.view(torch.int16),b.view(torch.int16))
            case = {'batch':b,'prefix':prefix,'Hkv':hkv,'GQA':4,'S':splits,'checks':{}}
            report['cases'].append(case)
            checks=case['checks'];save()
            native(native_o);candidate()
            checks['paired_bits']=bits(native_o,candidate_o)
            native(aa_o,False)
            checks['native_PDL_on_off_bits']=bits(native_o,aa_o)
            checks['AA_5_bits']=True
            for _ in range(5):
                native(aa_o);checks['AA_5_bits'] &= bits(native_o,aa_o)
            for row in [0,b-1]:
                length=int(lengths[row])
                token=torch.arange(length,device='cuda')
                kv=cache[table[row,token//16].long(),:,token%16].double()
                scores=torch.einsum('hgd,thd->hgt',q[row].double().reshape(4,4,128),kv[...,:128])/128**.5
                ref=torch.einsum('hgt,thd->hgd',scores.softmax(-1),kv[...,128:]).reshape(16,128)
                checks[f'native_FP64_row{row}']=torch.allclose(native_o[row].double(),ref,atol=.006,rtol=.015)
                checks[f'candidate_FP64_row{row}']=torch.allclose(candidate_o[row].double(),ref,atol=.006,rtol=.015)
            save();assert all(checks.values()),case
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): candidate()
            for label,newlens in [('tail_growth',[prefix+8192-i*7 for i in range(b)]),
                                  ('dummy_one',[1]*b),('unequal_again',[prefix+7+i*29 for i in range(b)])]:
                q.mul_(.875);lengths.copy_(torch.tensor(newlens,device='cuda',dtype=torch.int32));state.fill_(float('nan'))
                graph.replay();native(native_o);torch.cuda.synchronize()
                checks[label+'_graph_bits']=bits(native_o,candidate_o)
                save();assert checks[label+'_graph_bits'],case
            # Physical relocation and different cache values under the same graph pointers.
            cache.copy_(cache.flip(0));table.copy_((npages-1-table).int());cache.mul_(.875)
            state.fill_(float('nan'));graph.replay();native(native_o);torch.cuda.synchronize()
            checks['remapped_changed_KV_graph_bits']=bits(native_o,candidate_o)
            save();assert all(checks.values()),case
            del graph,cache,table,q,packed,state,kv,scores,ref
        report['status']='qualified'
        report['limitations']=['Synthetic even batches only; no second-model actual capture yet.',
                               'No performance claim. Four-request GQA4->16 grouping is not implemented.']
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();save();raise
    save()
    print(json.dumps({'status':report['status'],'output':str(out)}))


if __name__=='__main__':
    main()
