"""Actual page-16 XQA O-only versus exact shared-prefix cascade qualification.

Synthetic cache values, real fused/interleaved storage ABI and explicit live
page tables. No model quality or speedup claim. Root allocates the GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--dtype',choices=['bf16','fp8','fp8_uint8'],required=True)
    p.add_argument('--prefix-tokens',type=int,default=128)
    p.add_argument('--batches',default='8,4,3,2,1')
    p.add_argument('--nonunit-scales',action='store_true')
    args=p.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert args.prefix_tokens>0 and args.prefix_tokens%16==0
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','args':vars(args),'checks':[],'prepare_receipts':[],
            'rejections':[],'recaptures':[],'scope':__doc__,'uid':os.geteuid(),'python':sys.executable,
            'O_tolerance':{'atol':.006,'rtol':.015},'output_contract':'BF16 O only; cascade internal LSE is charged, native XQA does not request LSE'}
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        from src.vllm_dense_cascade_plan import LayerSharedDenseCascadePlan
        torch.set_num_threads(4);torch.manual_seed(20261017)
        assert torch.cuda.get_device_capability()==(12,0)
        max_batch=8;page=16;layers=2;heads=2;dim=128;tail_capacity=160
        prefix_pages=args.prefix_tokens//page;private_pages=(tail_capacity+page-1)//page
        pages=prefix_pages+max_batch*private_pages
        storage=torch.randn((pages,layers,heads,page,2*dim),device='cuda',dtype=torch.bfloat16)
        if args.dtype!='bf16':storage=storage.to(torch.float8_e4m3fn)
        if args.dtype=='fp8_uint8':storage=storage.view(torch.uint8)
        caches=[storage[:,l] for l in range(layers)]
        ks=[.5,1.25] if args.nonunit_scales else [1.,1.]
        vs=[1.25,.5] if args.nonunit_scales else [1.,1.]
        full_tables=torch.tensor([list(range(prefix_pages))+list(range(prefix_pages+r*private_pages,prefix_pages+(r+1)*private_pages)) for r in range(max_batch)],device='cuda',dtype=torch.int32)
        native_workspace=torch.zeros(256*1024*1024,device='cuda',dtype=torch.uint8)
        report['cache']={'shape':list(caches[0].shape),'stride':list(caches[0].stride()),'dtype':str(caches[0].dtype),'k_scales':ks,'v_scales':vs,'page_size':page}
        actualdtype=torch.bfloat16 if args.dtype=='bf16' else torch.float8_e4m3fn
        def kv(layer):
            view=caches[layer].view(actualdtype) if caches[layer].dtype==torch.uint8 else caches[layer]
            return view[...,:dim],view[...,dim:]
        def reference(query,tables,seq_lens,layer):
            # Independent actual-page indexing, no candidate flattened indices.
            cache=caches[layer].view(actualdtype) if caches[layer].dtype==torch.uint8 else caches[layer]
            cache=cache.double();answer=torch.empty_like(query,dtype=torch.float64);lse=torch.empty(query.shape[:2],device=query.device,dtype=torch.float64)
            for r,length in enumerate(seq_lens):
                positions=torch.arange(length,device='cuda');physical=tables[r,positions//page].long()
                for h in range(heads):
                    payload=cache[physical,h,positions%page]
                    q=query[r,h*8:(h+1)*8].double()
                    scores=(q@(payload[:,:dim]*ks[layer]).T)*(dim**-.5)
                    answer[r,h*8:(h+1)*8]=torch.softmax(scores,dim=-1)@(payload[:,dim:]*vs[layer])
                    lse[r,h*8:(h+1)*8]=torch.logsumexp(scores,dim=-1)
            return answer,lse
        def graph_of(fn):
            for _ in range(3):fn()
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):fn()
            return g
        for batch in map(int,args.batches.split(',')):
            assert 1<=batch<=max_batch
            # Survivor order is reversed and non-prefix when possible.
            survivors=list(reversed(range(max_batch-batch,max_batch)))
            tables=full_tables[survivors].clone()
            query=torch.randn((batch,heads*8,dim),device='cuda',dtype=torch.bfloat16)
            lens_tensor=torch.empty(batch,device='cuda',dtype=torch.uint32)
            native_outputs=[torch.empty_like(query) for _ in range(layers)]
            candidate_outputs=[torch.empty_like(query) for _ in range(layers)]
            plan=LayerSharedDenseCascadePlan(caches,batch,args.prefix_tokens,max_tail_tokens=tail_capacity,k_scales=ks,v_scales=vs)
            candidate_graph=None;native_graph=None
            for index,tail in enumerate((1,15,16,17,127,129)):
                seq_lens=[args.prefix_tokens+tail+(r%3) for r in range(batch)]
                tables.copy_(full_tables[survivors])
                if tail==129:
                    tables[:,:prefix_pages]=torch.flip(tables[:,:prefix_pages],dims=[1])
                    tables[:,prefix_pages:]=torch.roll(tables[:,prefix_pages:],1,1)
                    changed=torch.randn(storage.shape,device=storage.device,dtype=torch.bfloat16).to(actualdtype)
                    storage.copy_(changed.view(torch.uint8) if storage.dtype==torch.uint8 else changed)
                query.normal_();lens_tensor.copy_(torch.tensor(seq_lens,dtype=torch.uint32))
                try:
                    receipt=plan.prepare(tables,seq_lens,require_stable_topology=candidate_graph is not None)
                except RuntimeError as exc:
                    if 'topolog' not in str(exc).lower():raise
                    report['recaptures'].append({'batch':batch,'tail':tail,'reason':str(exc)})
                    receipt=plan.prepare(tables,seq_lens,require_stable_topology=False);candidate_graph=None
                report['prepare_receipts'].append({'batch':batch,'tail':tail,'receipt':receipt})
                def native():
                    for l in range(layers):
                        xqa_batch_decode_with_kv_cache(query,kv(l),native_workspace,tables,lens_tensor,
                            max_seq_len=args.prefix_tokens+tail_capacity,bmm1_scale=dim**-.5*ks[l],bmm2_scale=vs[l],
                            kv_layout='HND',out=native_outputs[l])
                def candidate():
                    for l in range(layers):plan.run_layer(l,query,output=candidate_outputs[l])
                if candidate_graph is None:candidate_graph=graph_of(candidate)
                if native_graph is None:native_graph=graph_of(native)
                # Both graphs see original addresses and updated values.
                candidate_graph.replay();native_graph.replay();torch.cuda.synchronize()
                for l in range(layers):
                    expected,expected_lse=reference(query,tables,seq_lens,l)
                    natural_lse=plan.prefix_lse[l].double()*__import__("math").log(2)
                    lse_pass=bool(torch.isclose(natural_lse,expected_lse,atol=.0003,rtol=.0001).all())
                    report.setdefault("internal_LSE_checks",[]).append({"batch":batch,"tail":tail,"layer":l,"passed":lse_pass,"max_abs":float((natural_lse-expected_lse).abs().max())})
                    save();assert lse_pass,report["internal_LSE_checks"][-1]
                    for name,value in (('native_xqa_O_only',native_outputs[l]),('shared_prefix_cascade',candidate_outputs[l])):
                        passed=bool(torch.isclose(value.double(),expected,atol=.006,rtol=.015).all())
                        report['checks'].append({'method':name,'batch':batch,'survivors':survivors,'tail':tail,'layer':l,
                            'passed':passed,'max_abs':float((value.double()-expected).abs().max()),
                            'relative_l2':float((value.double()-expected).norm()/expected.norm())})
                        save();assert passed,report['checks'][-1]
            if batch>1:
                tables[1,0]=(tables[0,0]+1)%pages
                try:plan.prepare(tables,seq_lens)
                except (ValueError,RuntimeError,AssertionError) as exc:
                    report['rejections'].append({'batch':batch,'case':'divergent_shared_physical_page','message':str(exc)})
                else:raise AssertionError('Shared page identity divergence accepted')
            save()
        sources=['scripts/qualify_dense_cascade_xqa.py','src/vllm_dense_cascade_plan.py']
        report['sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
        report['status']='qualified';report['all_passed']=True;save()
        print(json.dumps({'status':report['status'],'checks':len(report['checks']),'recaptures':len(report['recaptures'])}),flush=True)
    except Exception as exc:
        report['status']='failed';report['error_type']=type(exc).__name__;report['error']=str(exc);report['traceback']=traceback.format_exc();save();raise

if __name__=='__main__':main()
