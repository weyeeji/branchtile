"""Independent GQA16-prefix -> native GQA8-tail raw-state continuation fork."""
from pathlib import Path
import hashlib
import json
import shutil

from src.xqa_raw_continuation_build import ROOT, UPSTREAM, digest, generate as base_generate, replace


def generate():
    base, original = base_generate()
    identity = hashlib.sha256((original["identity"] + digest(__file__)).encode()).hexdigest()[:16]
    dest = ROOT / "build/xqa_raw_pair" / identity
    receipt_path = dest / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for path, expected in receipt["generated_sha256"].items():
            assert digest(dest / path) == expected
        return dest, receipt
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns("receipt.json"))
    path = dest / "xqa/mha.cu"
    text = path.read_text()
    text = replace(text, "HEAD_GRP_SIZE == 8 && BEAM_WIDTH == 1",
                   "(HEAD_GRP_SIZE == 8 || HEAD_GRP_SIZE == 16) && BEAM_WIDTH == 1")
    text = replace(text, "size_t(idxReq) * nbKHeads", "size_t(phase == 3 ? idxReq / 2 : idxReq) * nbKHeads")
    text = replace(text, "phase == 2 ? stateLane[34] : safeInitRowMax;", """phase == 3 ? (laneId() < 8 ? stateLane[(idxReq % 2) * 8 * 35 + 34] : safeInitRowMax)
                   : (phase == 2 ? stateLane[34] : safeInitRowMax);""")
    text = replace(text, "(phase == 2 ? resumeOffset : 0)", "(phase >= 2 ? resumeOffset : 0)")
    old = """    if (phase == 2) {
      globalRowMax[0] = stateLane[32];
      globalRowSum[0] = stateLane[33];
#pragma unroll
      for (uint32_t i = 0; i < 32; ++i) {
        reinterpret_cast<float*>(&accs)[i] = stateLane[i];
      }
    }"""
    new = """    if (phase == 2 || phase == 3) {
      if (phase == 3) {
        globalRowMax[0] = laneId() < 8 ? stateLane[(idxReq % 2) * 8 * 35 + 32] : safeInitRowMax;
        globalRowSum[0] = laneId() < 8 ? stateLane[(idxReq % 2) * 8 * 35 + 33] : 0.f;
      } else {
        globalRowMax[0] = stateLane[32];
        globalRowSum[0] = stateLane[33];
      }
#pragma unroll
      for (uint32_t i = 0; i < 32; ++i) {
        // m16n8 accumulator rows occupy (4*n+0/1) and (4*n+2/3).
        reinterpret_cast<float*>(&accs)[i] = phase == 3
            ? (i % 4 < 2 ? stateLane[i + 2 * (idxReq % 2)] : 0.f)
            : stateLane[i];
      }
    }"""
    text = replace(text, old, new)
    path.write_text(text)
    receipt = {
        "identity": identity, "base_receipt": original, "patch_sha256": digest(__file__),
        "contract": "phase1 GQA16 packed prefix; phase3 GQA8 native tail; original S; original raw warp states",
        "generated_sha256": {str(p.relative_to(dest)): digest(p) for p in dest.rglob("*") if p.is_file()},
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return dest, receipt


def jit_specs():
    import torch
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    dest, receipt = generate()
    specs = {}
    for group in (8, 16):
        upstream = gen_xqa_module(torch.bfloat16, torch.bfloat16, 16, 128, group, False, torch.bfloat16)
        specs[group] = gen_jit_spec(
            f"iscas_xqa_raw_pair_g{group}_" + receipt["identity"],
            [dest / "xqa/mha.cu", dest / "xqa/xqa_wrapper.cu", dest / "flashinfer_xqa_binding.cu"],
            extra_cuda_cflags=upstream.extra_cuda_cflags,
            extra_include_paths=[dest, UPSTREAM], extra_ldflags=["-lcuda"],
        )
    return specs, receipt
