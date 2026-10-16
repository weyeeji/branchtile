"""FP64 actual-address qualification of new paged owner ABI and live metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    from src.vllm_paged_owner_plan import LayerSharedPagedOwnerPlan
    torch.manual_seed(20261016)
    # Interleaved layers reproduce non-contiguous physical page pitch.
    backing=torch.randn((64,2,2,16,256),device='cuda',dtype=torch.bfloat16)
    caches=[backing[:,l] for l in range(2)]
    prompts=[83,87,91]
    histories=[[[torch.tensor(sorted(set(range(q+h+l,64,3))|set(range(64,prompts[q])))) for h in range(2)] for q in range(3)] for l in range(2)]
    base_tables=torch.tensor([[0,1,2,3]+list(range(4+q*20,24+q*20)) for q in range(3)],dtype=torch.int32,device='cuda')
    checks=[]
    for b in (1,2,3):
        plan=LayerSharedPagedOwnerPlan(caches,histories,prompts,'gpu-proof',b,64,max_generated=129,chunk_tokens=7)
        query=torch.randn((b,20,128),device='cuda',dtype=torch.bfloat16)[:,:16]
        first=tuple(range(b));plan.prepare(base_tables[:b],first,[prompts[q]-1 for q in first],'gpu-proof')
        for layer in range(2):plan.run_layer(layer,query)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for layer in range(2):plan.run_layer(layer,query)
        for subset in plan.subsets:
            for tail in (0,1,127,129):
                tables=base_tables[list(subset)].clone()
                if tail==129:
                    tables[:,:4]=tables[:,[1,0,3,2]]
                    tables[:,4:]=torch.roll(tables[:,4:],1,1)
                positions=[prompts[q]-1+tail for q in subset]
                plan.prepare(tables,list(subset),positions,'gpu-proof')
                query.normal_();graph.replay();torch.cuda.synchronize()
                for l in range(2):
                    for r,q in enumerate(subset):
                        logical=torch.cat((histories[l][q][0][:0],torch.arange(prompts[q],prompts[q]+tail)))
                        for h in range(2):
                            ids=torch.cat((histories[l][q][h],logical)).cuda()
                            page=tables[r,ids//16].long();off=ids%16
                            kv=caches[l][page,h,off]
                            actualq=query[r,h*8:(h+1)*8].double()
                            scores=actualq@kv[:,:128].double().T/(128**.5)
                            expected=torch.softmax(scores,dim=-1)@kv[:,128:].double()
                            expected_lse=torch.logsumexp(scores,dim=-1)
                            actual=plan.outputs[l,r,h*8:(h+1)*8].double();lse=plan.lse[l,h,r*8:(r+1)*8].double()
                            ok=bool(torch.isclose(actual,expected,atol=.006,rtol=.015).all())
                            lok=bool(torch.isclose(lse,expected_lse,atol=.0001,rtol=.0001).all())
                            checks.append({'batch':b,'subset':list(subset),'tail':tail,'layer':l,'request':r,'head':h,
                                'output_pass':ok,'lse_pass':lok,'max_abs':float((actual-expected).abs().max()),
                                'max_lse_abs':float((lse-expected_lse).abs().max())})
                            assert ok and lok,checks[-1]
        if b>1:
            broken=base_tables[:b].clone();broken[1,0]=31
            try:plan.prepare(broken,list(range(b)),[prompts[q] for q in range(b)],'gpu-proof')
            except ValueError:pass
            else:raise AssertionError('Divergent shared cache pages were accepted')
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'gpu_paged_owner_qualified','checks':checks,'all_passed':True,'actual_strided_source':True,
        'graph_metadata_updates_exercised':['all_request_subsets_and_permutations','tails_0_1_127_129','physical_page_remap','noncontiguous_Q','changed_query_values'],
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('src/vllm_paged_owner_plan.py','src/branchtile/triton_kernel_paged_owner.py','scripts/qualify_paged_owner_gpu.py')}}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'checks':len(checks)}),flush=True)
if __name__=='__main__':main()
