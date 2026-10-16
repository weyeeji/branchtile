"""Versioned native-group backward-hint control for raw shared XQA.

The old whole-M16 rescale branch can refresh a request's backward hint when
only a different packed request needed rescaling. Preserve the original native
GQA group's write predicate. No installed or previously frozen source changes.
"""
import hashlib,json,shutil
from src.xqa_raw_continuation_build import ROOT,UPSTREAM,digest,replace


def generate(native_group=8,fold=2):
    assert (native_group,fold) in [(8,2),(4,2),(4,4)]
    if native_group==8:
        from src.xqa_raw_pair_build import generate as original_generate
    elif fold==2:
        from src.xqa_raw_pair_gqa4_build import generate as original_generate
    else:
        from src.xqa_raw_quad_gqa4_build import generate as original_generate
    base,original=original_generate()
    identity=hashlib.sha256((original['identity']+digest(__file__)+str((native_group,fold))).encode()).hexdigest()[:16]
    dest=ROOT/'build/xqa_raw_grouped_hint'/identity;rp=dest/'receipt.json'
    if rp.exists():
        receipt=json.loads(rp.read_text())
        for path,expected in receipt['generated_sha256'].items():assert digest(dest/path)==expected
        return dest,receipt
    shutil.copytree(base,dest,ignore=shutil.ignore_patterns('receipt.json'))
    p=dest/'xqa/mha.cu';s=p.read_text()
    anchor='''                  smem.ctaRowMax[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, globalRowMax);'''
    patch=f'''                  // A packed request must preserve its ORIGINAL native-group
                  // branch, including deliberately stale backward hints. A different
                  // request's changed row max must not cause this group's hint store.
                  ThrdRegRowMax hintToStore = globalRowMax;
                  if constexpr (HEAD_GRP_SIZE == {native_group*fold}) {{
                    if (phase == 1) {{
                      constexpr uint32_t nativeRows = {native_group};
                      uint32_t const nativeGroupMask = ((1U << nativeRows) - 1U)
                          << (laneId() / nativeRows * nativeRows);
                      if ((needRescaleMask[0] & nativeGroupMask) == 0U) {{
                        hintToStore[0] = smem.ctaRowMax[warpIdx.y][warpIdx.x][laneId()];
                      }}
                    }}
                  }}
                  // Keep the store collective/convergence structure unchanged.
                  smem.ctaRowMax[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, hintToStore);'''
    s=replace(s,anchor,patch);p.write_text(s)
    receipt={'identity':identity,'native_group':native_group,'fold':fold,'base_receipt':original,'patch_sha256':digest(__file__),
             'contract':'Preserve native per-request GQA-group backward-hint update predicate; other arithmetic unchanged; qualification required',
             'generated_sha256':{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()}}
    rp.write_text(json.dumps(receipt,indent=2)+'\n');return dest,receipt


def jit_specs(native_group=8,fold=2):
    import torch
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    dest,receipt=generate(native_group,fold);specs={}
    for group in (native_group,native_group*fold):
        old=gen_xqa_module(torch.bfloat16,torch.bfloat16,16,128,group,False,torch.bfloat16)
        specs[group]=gen_jit_spec('iscas_xqa_grouped_hint_g'+str(group)+'_'+receipt['identity'],
            [dest/'xqa/mha.cu',dest/'xqa/xqa_wrapper.cu',dest/'flashinfer_xqa_binding.cu'],
            extra_cuda_cflags=old.extra_cuda_cflags,extra_include_paths=[dest,UPSTREAM],extra_ldflags=['-lcuda'])
    return specs,receipt
