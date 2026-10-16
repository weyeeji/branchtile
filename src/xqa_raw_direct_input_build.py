"""Grouped-hint XQA sharing reading native Q, page tables and lengths directly.

Only the packed prefix uses a request/head address map. The tail and arithmetic
remain native. Odd batches zero-fill the missing request during async Q loads.
"""
import hashlib
import json
import shutil
from src.xqa_raw_continuation_build import ROOT, UPSTREAM, digest, replace
from src.xqa_raw_grouped_hint_build import generate as base_generate


def generate():
    base, original = base_generate()
    identity = hashlib.sha256((original['identity'] + digest(__file__)).encode()).hexdigest()[:16]
    dest = ROOT / 'build/xqa_raw_direct_input' / identity
    receipt_path = dest / 'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for path, expected in receipt['generated_sha256'].items():
            assert digest(dest / path) == expected
        return dest, receipt
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns('receipt.json'))
    p = dest / 'xqa/mha.cu'
    s = p.read_text()
    s = replace(s, 'uint32_t phase = 0)', 'uint32_t phase = 0, uint32_t originalBatch = 0)', 2)
    s = replace(s, 'uint32_t prefixTiles, uint32_t phase)', 'uint32_t prefixTiles, uint32_t phase, uint32_t originalBatch)')
    s = replace(s, 'rawState, prefixTiles, phase);', 'rawState, prefixTiles, phase, originalBatch);', 2)
    s = replace(s, '  uint32_t const idxReq = blockIdx.z;',
                '  uint32_t const idxReq = blockIdx.z;\n'
                '  bool const directPrefix = phase == 1 && HEAD_GRP_SIZE == 16;\n'
                '  uint32_t const idxCacheReq = directPrefix ? idxReq * 2 : idxReq;')
    s = replace(s, 'auto const localQHeadIdxMap = [nbQHeads, idxReq,',
                'auto const localQHeadIdxMap = [nbQHeads, idxReq, directPrefix,')
    s = replace(s, '        return idxHeadLocal;',
                '        return directPrefix ? (idxHeadLocal / 8) * (nbQHeads / 2) + idxHeadLocal % 8\n'
                '                            : idxHeadLocal;')
    s = replace(s, 'uint32_t const idxHeadBeg = nbQHeads * beamWidth * idxReq + headGrpSize * idxHeadGrp;',
                'uint32_t const idxHeadBeg = nbQHeads * beamWidth * idxReq +\n'
                '        (directPrefix ? headGrpSize / 2 : headGrpSize) * idxHeadGrp;')
    old = '''    copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, grainBytes, grainBytes,
                   qkSwizzle, isFullTile, warpTile.y>(warpIdx.x, smem.q[warpIdx.y][0], src,
                                                      nbValidRows, localQHeadIdxMap);'''
    new = '''    if (directPrefix && idxCacheReq + 1 >= originalBatch) {
      // copyPartialHeadsAsync zero-fills rows beyond nbAvailHeads. No dummy Q
      // storage or preceding packing kernel is required for the odd request.
      copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, grainBytes, grainBytes,
                     qkSwizzle, false, warpTile.y>(warpIdx.x, smem.q[warpIdx.y][0], src,
                                                  8, localQHeadIdxMap);
    } else {
''' + old + '''
    }'''
    s = replace(s, old, new)
    for old, new, count in [
        ('getCacheSeqLen<usePagedKVCache>(cacheList, idxReq)', 'getCacheSeqLen<usePagedKVCache>(cacheList, idxCacheReq)', 1),
        ('cacheList, true, idxReq,', 'cacheList, true, idxCacheReq,', 2),
        ('cacheList, false, idxReq,', 'cacheList, false, idxCacheReq,', 2),
    ]:
        s = replace(s, old, new, count)
    p.write_text(s)
    p = dest / 'xqa/mha.h'
    p.write_text(replace(p.read_text(), 'uint32_t prefixTiles, uint32_t phase)',
                        'uint32_t prefixTiles, uint32_t phase, uint32_t originalBatch)'))
    p = dest / 'xqa/xqa_wrapper.cu'
    p.write_text(replace(p.read_text(), 'raw_state.data_ptr()), prefix_tiles, phase);',
                        'raw_state.data_ptr()), prefix_tiles, phase, static_cast<uint32_t>(seqLen.size(0)));'))
    receipt = {'identity': identity, 'base_receipt': original, 'patch_sha256': digest(__file__),
               'contract': 'GQA8 pair, phase1 uses native Q address permutation and every-second native metadata row; odd member zero-fill; original native split and per-group backward-hint control preserved. Qualification required.',
               'generated_sha256': {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return dest, receipt


def jit_specs():
    import torch
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    dest, receipt = generate()
    specs = {}
    for group in (8, 16):
        upstream = gen_xqa_module(torch.bfloat16, torch.bfloat16, 16, 128, group, False, torch.bfloat16)
        specs[group] = gen_jit_spec('iscas_xqa_direct_input_g' + str(group) + '_' + receipt['identity'],
            [dest / 'xqa/mha.cu', dest / 'xqa/xqa_wrapper.cu', dest / 'flashinfer_xqa_binding.cu'],
            extra_cuda_cflags=upstream.extra_cuda_cflags, extra_include_paths=[dest, UPSTREAM], extra_ldflags=['-lcuda'])
    return specs, receipt
