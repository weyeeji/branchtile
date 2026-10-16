"""CPU Triton-interpreter checks of compact metadata, remap and error flags."""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['TRITON_INTERPRET']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    import triton
    from src.vllm_dense_cascade_device_plan import _update_tail,_verify_prefix
    assert not torch.cuda.is_initialized()
    rows=[]
    for batch in (1,3,8):
        prefix=128;prefix_pages=prefix//16;capacity=8256;max_pages=(capacity+15)//16
        tables=torch.arange(batch*(prefix_pages+max_pages),dtype=torch.int32).reshape(batch,-1)
        tables[:,:prefix_pages]=torch.arange(prefix_pages,dtype=torch.int32)
        tables=tables.flip(0).contiguous()
        for tail in (1,15,16,17,127,129,8192):
            tails=[tail+r%3 for r in range(batch)]
            positions=torch.tensor([prefix+t-1 for t in tails],dtype=torch.int64)
            indptr=torch.zeros(batch+1,dtype=torch.int32);indices=torch.full((batch*max_pages,),-1,dtype=torch.int32)
            last=torch.zeros(batch,dtype=torch.int32);flags=torch.zeros(1,dtype=torch.int32)
            _update_tail[(batch*triton.cdiv(max_pages,256),)](tables,positions,indptr,indices,last,flags,
                TABLE_STRIDE=tables.stride(0),TABLE_COLS=tables.shape[1],B=batch,PREFIX=prefix,
                MAX_TOKENS=capacity,MAX_PAGES=max_pages,TILES=triton.cdiv(max_pages,256),R=triton.next_power_of_2(batch),BLOCK=256)
            expected_ptr=[0];expected_indices=[]
            for row,t in enumerate(tails):
                count=(t+15)//16;expected_ptr.append(expected_ptr[-1]+count)
                expected_indices.extend(tables[row,prefix_pages:prefix_pages+count].tolist())
            assert indptr.tolist()==expected_ptr
            assert indices[:expected_ptr[-1]].tolist()==expected_indices
            assert bool((indices[expected_ptr[-1]:]==-1).all()),'Wrote unused future-page indices'
            assert last.tolist()==[(t-1)%16+1 for t in tails]
            assert flags.item()==0
            rows.append({'B':batch,'tails':tails,'compact_pages':len(expected_indices),'future_indices_untouched':True})
        for bad in (0,capacity+1):
            positions[0]=prefix+bad-1;flags.zero_()
            _update_tail[(batch*triton.cdiv(max_pages,256),)](tables,positions,indptr,indices,last,flags,
                TABLE_STRIDE=tables.stride(0),TABLE_COLS=tables.shape[1],B=batch,PREFIX=prefix,
                MAX_TOKENS=capacity,MAX_PAGES=max_pages,TILES=triton.cdiv(max_pages,256),R=triton.next_power_of_2(batch),BLOCK=256)
            assert flags.item()&1
        flags.zero_();bound=tables[0,:prefix_pages].clone()
        _verify_prefix[(batch,)](tables,bound,flags,TABLE_STRIDE=tables.stride(0),PREFIX_PAGES=prefix_pages,TILES=1,BLOCK=256)
        assert flags.item()==0
        tables[-1,0]+=1
        _verify_prefix[(batch,)](tables,bound,flags,TABLE_STRIDE=tables.stride(0),PREFIX_PAGES=prefix_pages,TILES=1,BLOCK=256)
        assert flags.item()&2
    assert not torch.cuda.is_initialized()
    report={'status':'CPU_interpreter_passed','cuda_initialized':False,'cases':rows,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/verify_dense_cascade_metadata_cpu.py','src/vllm_dense_cascade_device_plan.py']},
        'limits':['CPU interpreter validates indexing and errors only, not GPU scheduling or attention arithmetic.']}
    out=ROOT/'results/platform/dense_cascade_metadata_cpu_verification_20261017.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'path':str(out),'cases':len(rows),'cuda_initialized':False}))

if __name__=='__main__':main()
