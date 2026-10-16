"""Extended device-metadata qualification: native cache strides, long tails, and 48-layer reuse.

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
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--batches',default='8,4,3,2,1')
    p.add_argument('--nonunit-scales',action='store_true')
    p.add_argument('--tail-capacity',type=int,default=8448)
    p.add_argument('--tail-cases',default='1,15,16,17,4095,4096,4097,8191,8192,8193,8446')
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
        from src.vllm_dense_cascade_device_plan import LayerSharedDeviceDenseCascadePlan
        torch.set_num_threads(4);torch.manual_seed(20261017)
        assert torch.cuda.get_device_capability()==(12,0)
        max_batch=8;page=16;layers=args.layers;heads=2;dim=128;tail_capacity=args.tail_capacity
        tail_cases=list(map(int,args.tail_cases.split(',')))
        assert tail_capacity>=3 and all(1<=t<=tail_capacity-2 for t in tail_cases)
        prefix_pages=args.prefix_tokens//page;private_pages=(tail_capacity+page-1)//page
        pages=prefix_pages+max_batch*private_pages
        storage=torch.randn((layers,pages,page,heads,2*dim),device='cuda',dtype=torch.bfloat16)
        if args.dtype!='bf16':storage=storage.to(torch.float8_e4m3fn)
        if args.dtype=='fp8_uint8':storage=storage.view(torch.uint8)
        caches=[storage[l].permute(0,2,1,3) for l in range(layers)]
        assert caches[0].stride()==(8192,256,512,1)
        ks=[(.5,1.25)[l%2] if args.nonunit_scales else 1. for l in range(layers)]
        vs=[(1.25,.5)[l%2] if args.nonunit_scales else 1. for l in range(layers)]
        full_tables=torch.tensor([list(range(prefix_pages))+list(range(prefix_pages+r*private_pages,prefix_pages+(r+1)*private_pages)) for r in range(max_batch)],device='cuda',dtype=torch.int32)
        native_workspace=torch.zeros(256*1024*1024,device='cuda',dtype=torch.uint8)
        report['coverage']={'native_cache_stride_asserted':True,'native_block_table_width_10240_or_larger':True,'metadata_CTAs_per_request':(private_pages+255)//256,'tail_cases':tail_cases,'layers':layers,'no_host_plan_after_initialization_enforced':True}
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
            tables=torch.zeros((batch,max(10240,full_tables.shape[1])),device='cuda',dtype=torch.int32)
            tables[:,:full_tables.shape[1]].copy_(full_tables[survivors])
            query=torch.randn((batch,heads*8,dim),device='cuda',dtype=torch.bfloat16)
            lens_tensor=torch.empty(batch,device='cuda',dtype=torch.uint32)
            native_outputs=[torch.empty_like(query) for _ in range(layers)]
            candidate_outputs=[torch.empty_like(query) for _ in range(layers)]
            positions_tensor=torch.empty(batch,device="cuda",dtype=torch.int64)
            plan=LayerSharedDeviceDenseCascadePlan(caches,batch,args.prefix_tokens,max_tail_tokens=tail_capacity,k_scales=ks,v_scales=vs)
            def forbidden_plan(*args,**kwargs):
                raise AssertionError("Host plan called after one-time initialization")
            plan.prefix_wrapper.plan=forbidden_plan
            plan.tail_wrapper.plan=forbidden_plan
            candidate_graph=None;native_graph=None
            for index,tail in enumerate(tail_cases):
                seq_lens=[args.prefix_tokens+tail+(r%3) for r in range(batch)]
                tables[:,:full_tables.shape[1]].copy_(full_tables[survivors])
                if index==len(tail_cases)-1:
                    tables[:,:prefix_pages]=torch.flip(tables[:,:prefix_pages],dims=[1])
                    tables[:,prefix_pages:full_tables.shape[1]]=torch.roll(tables[:,prefix_pages:full_tables.shape[1]],1,1)
                    changed=torch.randn(storage.shape,device=storage.device,dtype=torch.bfloat16).to(actualdtype)
                    storage.copy_(changed.view(torch.uint8) if storage.dtype==torch.uint8 else changed)
                query.normal_();lens_tensor.copy_(torch.tensor(seq_lens,dtype=torch.uint32))
                positions_tensor.copy_(torch.tensor([n-1 for n in seq_lens],dtype=torch.int64))
                if index==0 or index==len(tail_cases)-1:
                    receipt=plan.bind_prefix(tables)
                    report['prepare_receipts'].append({'batch':batch,'tail':tail,'receipt':receipt})
                def native():
                    for l in range(layers):
                        xqa_batch_decode_with_kv_cache(query,kv(l),native_workspace,tables,lens_tensor,
                            max_seq_len=args.prefix_tokens+tail_capacity,bmm1_scale=dim**-.5*ks[l],bmm2_scale=vs[l],
                            kv_layout='HND',out=native_outputs[l])
                def candidate():
                    plan.prepare_device(tables,positions_tensor,verify_prefix=True)
                    for l in range(layers):plan.run_layer(l,query,output=candidate_outputs[l])
                if candidate_graph is None:candidate_graph=graph_of(candidate)
                if native_graph is None:native_graph=graph_of(native)
                # Both graphs see original addresses and updated values.
                candidate_graph.replay();native_graph.replay();torch.cuda.synchronize()
                error_receipt=plan.device_error_receipt()
                report.setdefault("device_error_receipts",[]).append(error_receipt)
                assert error_receipt["valid"] and error_receipt["tail_plan_info_unchanged"]
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
            # Change every row consistently without rebinding: relative row
            # identity still holds, but the previously bound physical epoch differs.
            saved_prefix=tables[:,0].clone()
            tables[:,0]=(tables[:,0]+1)%pages
            candidate_graph.replay();torch.cuda.synchronize()
            receipt=plan.device_error_receipt()
            assert receipt['prefix_identity_error'],receipt
            report['rejections'].append({'batch':batch,'case':'uniform_changed_prefix_without_rebind','receipt':receipt})
            tables[:,0]=saved_prefix
            # A restored valid table cannot silently clear a historical failure.
            candidate_graph.replay();torch.cuda.synchronize()
            assert plan.device_error_receipt()['prefix_identity_error']
            report.setdefault('sticky_flag_checks',[]).append({'batch':batch,'passed':True})
            plan.bind_prefix(tables)
            if batch>1:
                tables[1,0]=(tables[0,0]+1)%pages
                try:
                    plan.bind_prefix(tables)
                except ValueError:
                    report.setdefault('host_bind_rejections',[]).append({'batch':batch,'passed':True})
                else:
                    raise AssertionError('Divergent prefix binding was accepted')
                candidate_graph.replay();torch.cuda.synchronize()
                receipt=plan.device_error_receipt()
                assert receipt['prefix_identity_error'],receipt
                report['rejections'].append({'batch':batch,'case':'divergent_shared_physical_page','receipt':receipt})
                tables[1,0]=tables[0,0]
            for bad_tail in (0,tail_capacity+1):
                plan.device_flags.zero_()
                positions_tensor[0]=args.prefix_tokens+bad_tail-1
                candidate_graph.replay();torch.cuda.synchronize()
                receipt=plan.device_error_receipt()
                assert receipt['tail_bounds_error'],receipt
                report['rejections'].append({'batch':batch,'case':f'invalid_tail_{bad_tail}','receipt':receipt})
            save()
        sources=['scripts/qualify_dense_cascade_device_plan_extended.py','src/vllm_dense_cascade_device_plan.py','src/vllm_dense_cascade_plan.py']
        report['sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
        report['status']='qualified';report['all_passed']=True;save()
        print(json.dumps({'status':report['status'],'checks':len(report['checks']),'recaptures':len(report['recaptures'])}),flush=True)
    except Exception as exc:
        report['status']='failed';report['error_type']=type(exc).__name__;report['error']=str(exc);report['traceback']=traceback.format_exc();save();raise

if __name__=='__main__':main()
