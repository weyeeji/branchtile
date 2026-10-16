"""Bounded GPU-only metadata lowering check; no attention or performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','checks':[],'scope':__doc__,'uid':os.geteuid(),'python':sys.executable}
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        import triton
        from src.vllm_dense_cascade_device_plan import _update_tail,_verify_prefix
        assert torch.cuda.get_device_capability()==(12,0)
        torch.set_num_threads(2)
        prefix=120528;prefix_pages=prefix//16;capacity=8448;max_pages=(capacity+15)//16;cols=10240
        sentinel=-193728465;guard=31;physical_capacity=34193
        for batch in (1,2,3,4,7,8):
            table=torch.zeros((batch,cols),device='cuda',dtype=torch.int32)
            positions=torch.full((batch,),prefix+15,device='cuda',dtype=torch.int64)
            ptr=torch.full((batch+1+guard,),sentinel,device='cuda',dtype=torch.int32)
            last=torch.full((batch+guard,),sentinel,device='cuda',dtype=torch.int32)
            indices=torch.full((batch*max_pages+guard,),sentinel,device='cuda',dtype=torch.int32)
            bound=torch.zeros(prefix_pages,device='cuda',dtype=torch.int32)
            flags=torch.zeros(1,device='cuda',dtype=torch.int32)
            def update():
                tiles=triton.cdiv(max_pages,256)
                _update_tail[(batch*tiles,)](table,positions,ptr,indices,last,flags,
                    TABLE_STRIDE=table.stride(0),TABLE_COLS=cols,B=batch,PREFIX=prefix,
                    MAX_TOKENS=capacity,MAX_PAGES=max_pages,TILES=tiles,R=triton.next_power_of_2(batch),BLOCK=256)
                prefix_tiles=triton.cdiv(prefix_pages,256)
                _verify_prefix[(batch*prefix_tiles,)](table,bound,flags,TABLE_STRIDE=table.stride(0),
                    PREFIX_PAGES=prefix_pages,TILES=prefix_tiles,BLOCK=256)
            for _ in range(3):update()
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):update()
            for case,tail in enumerate((16,17,4096,4097,8192)):
                cpu_table=(torch.arange(cols,dtype=torch.int32)[None,:]*17+torch.arange(batch,dtype=torch.int32)[:,None]*max_pages+case*97)%physical_capacity
                cpu_table[:,:prefix_pages]=cpu_table[:1,:prefix_pages].clone()
                tails=[tail+r%3 for r in range(batch)]
                counts=[(n+15)//16 for n in tails];expected_ptr=[0]
                for count in counts:expected_ptr.append(expected_ptr[-1]+count)
                expected_indices=torch.cat([cpu_table[r,prefix_pages:prefix_pages+count] for r,count in enumerate(counts)])
                table.copy_(cpu_table);bound.copy_(cpu_table[0,:prefix_pages]);positions.copy_(torch.tensor([prefix+n-1 for n in tails],dtype=torch.int64))
                ptr.fill_(sentinel);last.fill_(sentinel);indices.fill_(sentinel);flags.zero_()
                graph.replay();torch.cuda.synchronize()
                hp,hl,hi=ptr.cpu(),last.cpu(),indices.cpu()
                checks={'indptr_exact':torch.equal(hp[:batch+1],torch.tensor(expected_ptr,dtype=torch.int32)),
                    'last_exact':torch.equal(hl[:batch],torch.tensor([(n-1)%16+1 for n in tails],dtype=torch.int32)),
                    'indices_exact':torch.equal(hi[:expected_ptr[-1]],expected_indices),
                    'indptr_guard_untouched':bool((hp[batch+1:]==sentinel).all()),
                    'last_guard_untouched':bool((hl[batch:]==sentinel).all()),
                    'unused_indices_and_guard_untouched':bool((hi[expected_ptr[-1]:]==sentinel).all()),
                    'flags_clear':int(flags.item())==0}
                report['checks'].append({'batch':batch,'tail':tail,'tails':tails,'live_pages':expected_ptr[-1],
                    'metadata_CTAs_per_request':triton.cdiv(max_pages,256),**checks})
                save();assert all(checks.values()),report['checks'][-1]
        report['sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'scripts/qualify_device_cascade_metadata_exact.py','src/vllm_dense_cascade_device_plan.py')}
        report.update(status='qualified',all_passed=True,cases=len(report['checks']),prefix_tokens=prefix,tail_capacity=capacity,
            positions_dtype='torch.int64',positions_contiguous=True,table_stride=[cols,1],
            limitations=['Only metadata kernels and graph replay; no attention or model execution.',
                         'Physical page-ID range is an allocator precondition, not checked by these kernels.'])
        save();print(json.dumps({'status':report['status'],'cases':report['cases']}))
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
