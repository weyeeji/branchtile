"""Compile device metadata kernels for SM120 without CUDA initialization/execution."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from src.vllm_dense_cascade_device_plan import _update_tail,_verify_prefix
    assert not torch.cuda.is_initialized()
    rows=[]
    for b in (1,3,8):
        constants={'TABLE_STRIDE':10000,'TABLE_COLS':10000,'B':b,'PREFIX':120528,
            'MAX_TOKENS':8256,'MAX_PAGES':516,'TILES':3,'R':triton.next_power_of_2(b),'BLOCK':256}
        signature=dict(zip(_update_tail.arg_names[:6],('*i32','*i64','*i32','*i32','*i32','*i32')))
        signature.update({k:'constexpr' for k in constants})
        compiled=triton.compile(ASTSource(_update_tail,signature,constants),target=GPUTarget('cuda',120,32),options={'num_warps':4,'num_stages':1})
        rows.append({'kernel':'update_tail','B':b,'metadata':compiled.metadata._asdict(),'cubin_sha256':hashlib.sha256(compiled.asm['cubin']).hexdigest()})
    constants={'TABLE_STRIDE':10000,'PREFIX_PAGES':7533,'TILES':30,'BLOCK':256}
    signature=dict(zip(_verify_prefix.arg_names[:3],('*i32','*i32','*i32')))
    signature.update({k:'constexpr' for k in constants})
    compiled=triton.compile(ASTSource(_verify_prefix,signature,constants),target=GPUTarget('cuda',120,32),options={'num_warps':4,'num_stages':1})
    rows.append({'kernel':'verify_prefix','metadata':compiled.metadata._asdict(),'cubin_sha256':hashlib.sha256(compiled.asm['cubin']).hexdigest()})
    assert not torch.cuda.is_initialized()
    report={'status':'offline_compile_passed','cuda_initialized':False,'compiled':rows,
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/compile_dense_cascade_metadata_cpu.py','src/vllm_dense_cascade_device_plan.py']},
        'scope':'Existing metadata preparation optimization; no numerical/runtime/speedup or novelty claim.'}
    output=ROOT/'results/platform/dense_cascade_metadata_compile_20261017.json'
    output.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({'output':str(output),'compiled':len(rows),'cuda_initialized':False}))

if __name__=='__main__':main()
