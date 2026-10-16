"""Native-order raw pairing with a device-resident prefix cut for graph reuse."""
import hashlib
import json
import shutil

from src.xqa_raw_continuation_build import ROOT, UPSTREAM, digest, replace
from src.xqa_raw_pair_build import generate as pair_generate


def generate():
    base, original = pair_generate()
    identity = hashlib.sha256((original['identity'] + digest(__file__)).encode()).hexdigest()[:16]
    dest = ROOT / 'build/xqa_raw_pair_dynamic_cut' / identity
    receipt_path = dest / 'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for path, expected in receipt['generated_sha256'].items():
            assert digest(dest / path) == expected
        return dest, receipt
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns('receipt.json'))
    path = dest / 'xqa/mha.cu'
    source = path.read_text()
    source = replace(source, 'uint32_t prefixTiles = 0', 'uint32_t const* prefixTilesPtr = nullptr', 2)
    source = replace(source, 'uint32_t prefixTiles, uint32_t phase', 'uint32_t const* prefixTilesPtr, uint32_t phase')
    source = replace(source, 'rawState, prefixTiles, phase);', 'rawState, prefixTilesPtr, phase);', 2)
    source = replace(source, '  uint32_t const originalSeqIterInit =',
                     '  uint32_t const prefixTiles = prefixTilesPtr == nullptr ? 0u : *prefixTilesPtr;\n'
                     '  uint32_t const originalSeqIterInit =')
    path.write_text(source)
    path = dest / 'xqa/mha.h'
    path.write_text(replace(path.read_text(), 'uint32_t prefixTiles, uint32_t phase',
                            'uint32_t const* prefixTilesPtr, uint32_t phase'))
    for relative in ('xqa/xqa_wrapper.cu', 'flashinfer_xqa_binding.cu'):
        path = dest / relative
        source = replace(path.read_text(), 'int64_t prefix_tiles, int64_t phase',
                         'TensorView prefix_tiles, int64_t phase')
        if relative.endswith('xqa_wrapper.cu'):
            source = replace(source, 'raw_state.data_ptr()), prefix_tiles, phase);',
                             'raw_state.data_ptr()), reinterpret_cast<uint32_t const*>(prefix_tiles.data_ptr()), phase);')
        path.write_text(source)
    receipt = {'identity': identity, 'base_receipt': original, 'patch_sha256': digest(__file__),
               'contract': 'Only prefix cut argument changes from a captured host scalar to an immutable-during-replay device uint32 scalar; native tile stride, arithmetic, raw state and merge unchanged.',
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
        specs[group] = gen_jit_spec('iscas_xqa_raw_pair_dynamic_cut_g' + str(group) + '_' + receipt['identity'],
            [dest / 'xqa/mha.cu', dest / 'xqa/xqa_wrapper.cu', dest / 'flashinfer_xqa_binding.cu'],
            extra_cuda_cflags=upstream.extra_cuda_cflags, extra_include_paths=[dest, UPSTREAM], extra_ldflags=['-lcuda'])
    return specs, receipt
