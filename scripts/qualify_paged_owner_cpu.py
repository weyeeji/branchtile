"""CPU constructor/support audit and offline SM120 compile; no CUDA context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    from types import ModuleType
    stub=ModuleType("src.vllm_sparse_executor");stub.cache_views=lambda cache:(cache,cache)
    sys.modules["src.vllm_sparse_executor"]=stub
    import src.vllm_paged_owner_plan as mod
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    import triton
    assert not torch.cuda.is_initialized()
    # Only replace the CUDA-only storage alias guard, not the actual planner.
    mod.cache_views=lambda cache:(cache,cache)
    caches=[torch.empty((16,2,16,256),dtype=torch.bfloat16) for _ in range(2)]
    prompts=[83,87,91];histories=[]
    for l in range(2):
        histories.append([[torch.tensor(sorted(set(range(q+h+l,64,3))|set(range(64,prompts[q])))) for h in range(2)] for q in range(3)])
    checks=[]
    for b in (1,2,3):
        plan=mod.LayerSharedPagedOwnerPlan(caches,histories,prompts,'cpu-proof',b,64,max_generated=17,chunk_tokens=7)
        for subset,(tasks,rowmap,indices,counts) in plan.subsets.items():
            for l in range(2):
                for h in range(2):
                    for r,q in enumerate(subset):
                        observed=[];bases=[]
                        for task in tasks[l]:
                            hh,region,start,end,active,packed=map(int,task)
                            if hh!=h or active==0:continue
                            live=list(map(int,rowmap[l,h,region,:active]))
                            if r*8 not in live:continue
                            bases.append(packed+live.index(r*8))
                            if region<len(plan.labels):
                                observed.extend(map(int,plan.logicals[l,h,start:end]))
                            else:
                                assert region==len(plan.labels)+r and start==plan.history+r*17 and end==start
                        assert sorted(observed)==list(map(int,histories[l][q][h]))
                        assert len(observed)==len(set(observed))
                        assert bases==indices[l,h,r,:int(counts[l,h,r])].tolist()
                        checks.append([b,list(subset),l,h,r,len(observed)])
    # Compile actual GPU kernels against target; no device or driver execution.
    from src.branchtile.triton_kernel_paged_owner import producer,merge
    specs=[(producer,dict(ROWS=24,TOKENS=8192,REGIONS=10,HEADS=2,QS0=2048,QS1=128,N=64,D=128),
            ['*bf16','*bf16','*bf16','*i32','*i32','*i32','*fp32','*fp32','*fp32']),
           (merge,dict(BATCH=3,GQA=8,ROWS=24,MAX_CONTRIB=128,D=128,HEADS=2,GROUP=16),
            ['*fp32','*fp32','*fp32','*i32','*i32','*bf16','*fp32']),
           (mod._map,dict(HISTORY=2048,CAP=8192,HEADS=2,BATCH=3,TAIL_CAP=2048,TABLE_STRIDE=8192,PAGE=16,BLOCK=256),['*i32']*9),
           (mod._tail_ends,dict(TASKS=128,HEADS=2,BATCH=3),['*i32']*3)]
    compiled=[]
    for fn,constants,types in specs:
        signature={name:typ for name,typ in zip(fn.arg_names,types)}
        signature.update({name:'constexpr' for name in constants})
        result=triton.compile(ASTSource(fn,signature,constants),target=GPUTarget('cuda',120,32),options={'num_warps':4,'num_stages':1})
        compiled.append({'kernel':fn.__name__,'n_regs':result.metadata.n_regs if hasattr(result.metadata,'n_regs') else None,
                         'cubin_sha256':hashlib.sha256(result.asm['cubin']).hexdigest()})
    assert not torch.cuda.is_initialized()
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'cpu_support_and_offline_compile_passed','support_rows_checked':len(checks),'checks':checks,
            'compiled':compiled,'cuda_initialized':False,'not_gpu_qualified':True,
            'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('src/vllm_paged_owner_plan.py','src/branchtile/triton_kernel_paged_owner.py','scripts/qualify_paged_owner_cpu.py')}}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':main()
