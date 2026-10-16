"""Independent four-request GQA4 -> GQA16 raw continuation experiment.

Preserves each original request's GQA4 tail and split count. Raw fragment
coordinates cross both lane quads and accumulator row halves.
"""
import hashlib
import json
import shutil
from src.xqa_raw_continuation_build import ROOT, UPSTREAM, digest, replace
from src.xqa_raw_pair_gqa4_build import generate as pair_generate


def coordinate_gate():
    checks=0
    for member in range(4):
        for lane in range(16):
            for reg in range(32):
                if reg%4>=2:continue
                source_lane=lane+16*(member%2)
                source_reg=reg+2*(member//2)
                native_row=lane//4+8*((reg%4)//2)
                packed_row=source_lane//4+8*((source_reg%4)//2)
                native_col=8*(reg//4)+2*(lane%4)+reg%2
                packed_col=8*(source_reg//4)+2*(source_lane%4)+source_reg%2
                assert packed_row==native_row+4*member and packed_col==native_col
                checks+=1
    return checks


def generate():
    checks=coordinate_gate()
    base,original=pair_generate()
    identity=hashlib.sha256((original['identity']+digest(__file__)).encode()).hexdigest()[:16]
    dest=ROOT/'build/xqa_raw_quad_gqa4'/identity
    receipt_path=dest/'receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        for path,expected in receipt['generated_sha256'].items():assert digest(dest/path)==expected
        return dest,receipt
    shutil.copytree(base,dest,ignore=shutil.ignore_patterns('receipt.json'))
    path=dest/'xqa/mha.cu';text=path.read_text()
    text=replace(text,'(HEAD_GRP_SIZE == 4 || HEAD_GRP_SIZE == 8)','(HEAD_GRP_SIZE == 4 || HEAD_GRP_SIZE == 16)')
    text=replace(text,'size_t(phase == 3 ? idxReq / 2 : idxReq) * nbKHeads','size_t(phase == 3 ? idxReq / 4 : idxReq) * nbKHeads')
    for offset in (32,33,34):
        text=replace(text,f'stateLane[(idxReq % 2) * 4 * 35 + {offset}]',f'stateLane[(idxReq % 4) * 4 * 35 + {offset}]')
    text=replace(text,'stateLane[(idxReq % 2) * 16 * 35 + i]','stateLane[(idxReq % 4 % 2) * 16 * 35 + i + 2 * (idxReq % 4 / 2)]')
    text=replace(text,'// Paired member 1 lives in quads 4..7: source lane +16, same register.','// Four members: lane +16*(member%2), register +2*(member/2).')
    path.write_text(text)
    receipt={'identity':identity,'base_receipt':original,'patch_sha256':digest(__file__),'coordinate_checks':checks,'contract':'phase1 GQA16 four-request prefix, phase3 original GQA4 tail, native B/S/merge, explicit lane+register mapping; unqualified experiment','generated_sha256':{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()}}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');return dest,receipt


def jit_specs():
    import torch
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    dest,receipt=generate();specs={}
    for group in (4,16):
        original=gen_xqa_module(torch.bfloat16,torch.bfloat16,16,128,group,False,torch.bfloat16)
        specs[group]=gen_jit_spec('iscas_xqa_raw_quad_gqa4_g'+str(group)+'_'+receipt['identity'],[dest/'xqa/mha.cu',dest/'xqa/xqa_wrapper.cu',dest/'flashinfer_xqa_binding.cu'],extra_cuda_cflags=original.extra_cuda_cflags,extra_include_paths=[dest,UPSTREAM],extra_ldflags=['-lcuda'])
    return specs,receipt
