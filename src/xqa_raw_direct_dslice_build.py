"""Bounded experiment: two prefix CTAs own disjoint D64 PV warp fragments.

Both CTAs retain all original QK, scalar recurrence and backward-hint work.
Only partition zero exports scalars. No new reduction or changed native S.
"""
import hashlib
import json
import shutil
from src.xqa_raw_continuation_build import ROOT, UPSTREAM, digest, replace
from src.xqa_raw_direct_input_build import generate as base_generate


def generate():
    base, original = base_generate()
    identity = hashlib.sha256((original['identity'] + digest(__file__)).encode()).hexdigest()[:16]
    dest = ROOT / 'build/xqa_raw_direct_dslice' / identity
    rp = dest / 'receipt.json'
    if rp.exists():
        receipt = json.loads(rp.read_text())
        for path, expected in receipt['generated_sha256'].items():
            assert digest(dest / path) == expected
        return dest, receipt
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns('receipt.json'))
    p = dest / 'xqa/mha.cu'
    s = p.read_text()
    s = replace(s, '  assert(gridDim.z == batchSize && gridDim.y == nbKHeads);',
        '  bool const splitOutput = phase == 1 && HEAD_GRP_SIZE == 16;\n'
        '  uint32_t const outputPartition = splitOutput ? blockIdx.y % 2 : 0;\n'
        '  assert(gridDim.z == batchSize && gridDim.y == nbKHeads * (splitOutput ? 2 : 1));')
    s = replace(s, '  uint32_t const idxHeadGrp = blockIdx.y;  // inside one request',
        '  uint32_t const idxHeadGrp = splitOutput ? blockIdx.y / 2 : blockIdx.y;')
    s = replace(s, '  uint3 const warpIdx = getWarpIdx(warp);  // @fixme: use BoundedVal',
        '  uint3 const warpIdx = getWarpIdx(warp);  // @fixme: use BoundedVal\n'
        '  bool const ownsPV = !splitOutput || gemm1WarpIdxInGrp(warpIdx.x) == outputPartition;\n'
        '  static_assert(!GRP_LOAD_V, "D-slice experiment requires per-warp V buffers");')
    s = replace(s, '      stateLane[34] = smem.ctaRowMax[warpIdx.y][warpIdx.x][laneId()];',
        '      if (!splitOutput || outputPartition == 0)\n'
        '        stateLane[34] = smem.ctaRowMax[warpIdx.y][warpIdx.x][laneId()];')
    # Retain all page-index progression and empty cp.async commit/wait groups.
    # GRP_LOAD_V=false guarantees the skipped V payload has no shared barrier.
    old = '''      bool const isFullTile = (seqIter + 1 < nbSeqIters);
      if (isFullTile) {'''
    vstart = s.index('    auto loadVTilePart = [&]')
    s = s[:vstart] + replace(s[vstart:], old, '''      bool const isFullTile = (seqIter + 1 < nbSeqIters);
      if (ownsPV) {
      if (isFullTile) {''')
    old = '''            warp, dst, dstHeadOffset, src, warpIdxInGrp, mha::min(nbHeadsAvail, cacheVTileSeqLen));
      }
#if ENABLE_4BIT_KV_CACHE'''
    s = replace(s, old, '''            warp, dst, dstHeadOffset, src, warpIdxInGrp, mha::min(nbHeadsAvail, cacheVTileSeqLen));
      }
      }  // ownsPV: scalar/X feedback and V page progression remain unconditional.
#if ENABLE_4BIT_KV_CACHE''')
    old = '''#if BEAM_WIDTH == 1
              smemXVPartGemm<CacheElem>(warp, accs[hs], skipXRowRescale, xRowNeedRescaleMask,'''
    s = replace(s, old, '''#if BEAM_WIDTH == 1
              if (ownsPV)
              smemXVPartGemm<CacheElem>(warp, accs[hs], skipXRowRescale, xRowNeedRescaleMask,''')
    old = '''      for (uint32_t i = 0; i < 32; ++i) {
        stateLane[i] = reinterpret_cast<float const*>(&accs)[i];
      }
      stateLane[32] = globalRowMax[0];
      stateLane[33] = globalRowSum[0];'''
    new = '''      for (uint32_t i = 0; i < 32; ++i) {
        if (ownsPV) stateLane[i] = reinterpret_cast<float const*>(&accs)[i];
      }
      if (!splitOutput || outputPartition == 0) {
        stateLane[32] = globalRowMax[0];
        stateLane[33] = globalRowSum[0];
      }'''
    s = replace(s, old, new)
    s = replace(s, '  dim3 const dimGrid{nbSubSeqPerSeq, nbKHeads, batchSize};',
        '  dim3 const dimGrid{nbSubSeqPerSeq, nbKHeads * ((phase == 1 && HEAD_GRP_SIZE == 16) ? 2 : 1), batchSize};')
    p.write_text(s)
    receipt = {'identity': identity, 'base_receipt': original, 'patch_sha256': digest(__file__),
        'contract': 'Experimental BF16 GQA8 pair D128 output split into two D64 prefix CTAs; duplicate unchanged QK/scalars, disjoint U warp fragments, partition0 unique scalar writer; native S/tail unchanged. GPU qualification required.',
        'generated_sha256': {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}}
    rp.write_text(json.dumps(receipt, indent=2) + '\n')
    return dest, receipt


def jit_specs():
    import torch
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    dest, receipt = generate()
    specs = {}
    for group in (8, 16):
        upstream = gen_xqa_module(torch.bfloat16, torch.bfloat16, 16, 128, group, False, torch.bfloat16)
        specs[group] = gen_jit_spec('iscas_xqa_direct_dslice_g' + str(group) + '_' + receipt['identity'],
            [dest / 'xqa/mha.cu', dest / 'xqa/xqa_wrapper.cu', dest / 'flashinfer_xqa_binding.cu'],
            extra_cuda_cflags=upstream.extra_cuda_cflags, extra_include_paths=[dest, UPSTREAM], extra_ldflags=['-lcuda'])
    return specs, receipt
