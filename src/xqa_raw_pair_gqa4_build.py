"""Independent GQA4->8 raw-state continuation build for Qwen3-4B TP2.

This does not reuse the GQA8->16 fragment mapping: the paired second request
moves four rows, hence sixteen lanes of the MMA accumulator fragment.
"""
from pathlib import Path
import hashlib
import json
import shutil
from src.xqa_raw_continuation_build import ROOT, UPSTREAM, digest, replace
from src.xqa_raw_pair_build import generate as pair_generate


def generate():
    base, original = pair_generate()
    identity = hashlib.sha256((original['identity'] + digest(__file__)).encode()).hexdigest()[:16]
    dest = ROOT / 'build/xqa_raw_pair_gqa4' / identity
    receipt_path = dest / 'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for path, expected in receipt['generated_sha256'].items():
            assert digest(dest / path) == expected
        return dest, receipt
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns('receipt.json'))
    path = dest / 'xqa/mha.cu'
    text = path.read_text()
    text = replace(text, '(HEAD_GRP_SIZE == 8 || HEAD_GRP_SIZE == 16)',
                   '(HEAD_GRP_SIZE == 4 || HEAD_GRP_SIZE == 8)')
    text = replace(text, 'laneId() < 8 ? stateLane[(idxReq % 2) * 8 * 35 + 34]',
                   'laneId() < 4 ? stateLane[(idxReq % 2) * 4 * 35 + 34]')
    text = replace(text, 'laneId() < 8 ? stateLane[(idxReq % 2) * 8 * 35 + 32]',
                   'laneId() < 4 ? stateLane[(idxReq % 2) * 4 * 35 + 32]')
    text = replace(text, 'laneId() < 8 ? stateLane[(idxReq % 2) * 8 * 35 + 33]',
                   'laneId() < 4 ? stateLane[(idxReq % 2) * 4 * 35 + 33]')
    text = replace(text, '// m16n8 accumulator rows occupy (4*n+0/1) and (4*n+2/3).',
                   '// Native GQA4 uses lane quads 0..3 and the first MMA row half.\n'
                   '        // Paired member 1 lives in quads 4..7: source lane +16, same register.')
    text = replace(text, '(i % 4 < 2 ? stateLane[i + 2 * (idxReq % 2)] : 0.f)',
                   '(laneId() < 16 && i % 4 < 2 ? stateLane[(idxReq % 2) * 16 * 35 + i] : 0.f)')
    path.write_text(text)
    receipt = {'identity': identity, 'base_receipt': original, 'patch_sha256': digest(__file__),
               'contract': 'phase1 GQA8 paired prefix; phase3 GQA4 native tail; original S; cross-lane raw fragments',
               'generated_sha256': {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return dest, receipt


def jit_specs():
    import torch
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    dest, receipt = generate()
    specs = {}
    for group in (4, 8):
        upstream = gen_xqa_module(torch.bfloat16, torch.bfloat16, 16, 128, group, False, torch.bfloat16)
        specs[group] = gen_jit_spec('iscas_xqa_raw_pair_gqa4_g' + str(group) + '_' + receipt['identity'],
            [dest / 'xqa/mha.cu', dest / 'xqa/xqa_wrapper.cu', dest / 'flashinfer_xqa_binding.cu'],
            extra_cuda_cflags=upstream.extra_cuda_cflags, extra_include_paths=[dest, UPSTREAM], extra_ldflags=['-lcuda'])
    return specs, receipt
