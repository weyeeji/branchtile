"""Private, SHA-guarded native XQA checkpoint/resume prototype.

No installed source is changed. This first version is GQA8 -> GQA8 only and
tests the raw-state boundary, not shared-prefix performance. Generated code
retains NVIDIA/FlashInfer licenses. Root owns all GPU execution.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".venv-engine/lib/python3.12/site-packages/flashinfer/data/csrc"
STATE_FLOATS_PER_CTA = 4 * 32 * 35


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    found = text.count(old)
    if found != count:
        raise ValueError(f"source anchor expected {count}, found {found}: {old[:100]!r}")
    return text.replace(old, new)


def generate():
    source = UPSTREAM / "xqa/mha.cu"
    origin_sha = digest(source)
    patch_sha = digest(__file__)
    identity = hashlib.sha256((origin_sha + patch_sha).encode()).hexdigest()[:16]
    dest = ROOT / "build/xqa_raw_continuation" / identity
    receipt_path = dest / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for relative, expected in receipt["generated_sha256"].items():
            assert digest(dest / relative) == expected
        return dest, receipt
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copytree(UPSTREAM / "xqa", dest / "xqa")
    shutil.copy2(UPSTREAM / "tvm_ffi_utils.h", dest / "tvm_ffi_utils.h")
    shutil.copy2(UPSTREAM / "flashinfer_xqa_binding.cu", dest / "flashinfer_xqa_binding.cu")
    text = source.read_text()
    # The private module exposes only the FlashInfer host launcher, never TRT's launcher.
    start, stop = text.index("void launchMHA(\n"), text.index("static uint32_t configureKernel()")
    # The removed legacy launcher closes GENERATE_CUBIN; retain native SMEM setup.
    text = text[:start] + "#endif\n\n" + text[stop:]
    text = replace(text,
        "uint32_t* __restrict__ semaphores = nullptr, void* __restrict__ scratch = nullptr) {",
        "uint32_t* __restrict__ semaphores = nullptr, void* __restrict__ scratch = nullptr,\n"
        "        float* rawState = nullptr, uint32_t prefixTiles = 0, uint32_t phase = 0) {", 2)
    text = replace(text, "semaphores, scratch);", "semaphores, scratch, rawState, prefixTiles, phase);", 2)
    # Add host-launch parameters after stream, keeping the upstream stride ABI.
    text = replace(text, "cudaStream_t stream) {", "cudaStream_t stream, float* rawState, uint32_t prefixTiles, uint32_t phase) {")
    needle = "  // initialize shared memory\n"
    text = replace(text, needle, r'''
  static_assert(HEAD_ELEMS == 128 && HEAD_GRP_SIZE == 8 && BEAM_WIDTH == 1);
  static_assert(!SPEC_DEC && !SLIDING_WINDOW && !LOW_PREC_OUTPUT);
  static_assert(CTA_ROW_MAX_BACKWARD_METHOD == 1);
  static_assert(gemm1NbWarpGrps == 2 && gemm1WarpsPerGrp == 2);
  // Raw register layout, including padded rows: [request, head, split, warpX, lane, 35].
  // 0:32 = original PV WarpAcc, 32=max, 33=sum, 34=QK backward hint.
  size_t const stateCTA = ((size_t(idxReq) * nbKHeads + idxHeadGrp) *
                          nbSubSeqPerSeq + idxSubSeqInSeq) * (4 * 32 * 35);
  float* stateLane = rawState == nullptr ? nullptr :
      rawState + stateCTA + (warpIdx.x * 32 + laneId()) * 35;

''' + needle)
    old = '''  if (ctaThrdId < sizeof(smem.ctaRowMax) / sizeof(float)) {
    reinterpret_cast<float*>(&smem.ctaRowMax[0])[ctaThrdId] = safeInitRowMax;
  }'''
    text = replace(text, old, '''  if (warpIdx.z == 0) {
    smem.ctaRowMax[warpIdx.y][warpIdx.x][laneId()] =
        phase == 2 ? stateLane[34] : safeInitRowMax;
  }''')
    text = replace(text,
        "  uint32_t const seqIterInit = nbSkipLeadingTiles + idxSubSeqInSeq;",
        '''  uint32_t const originalSeqIterInit = nbSkipLeadingTiles + idxSubSeqInSeq;
  uint32_t const resumeOffset = prefixTiles > originalSeqIterInit
      ? divUp(prefixTiles - originalSeqIterInit, seqStrideIters) * seqStrideIters : 0;
  uint32_t const seqIterInit = originalSeqIterInit + (phase == 2 ? resumeOffset : 0);
  uint32_t const loopEnd = phase == 1 ? min(prefixTiles, nbSeqIters) : nbSeqIters;''')
    text = replace(text,
        "for (uint32_t seqIter = seqIterInit; seqIter < nbSeqIters; seqIter += seqStrideIters)",
        "for (uint32_t seqIter = seqIterInit; seqIter < loopEnd; seqIter += seqStrideIters)", 2)
    # Producer drains its last X tile before checkpointing the backward max hint.
    text = replace(text,
        '''      unused(xBar.produced.arrive());
    }
  } else {''',
        '''      unused(xBar.produced.arrive());
    }
    if (phase == 1) {
      if (seqIterInit < loopEnd) {
        xBar.consumed.wait_parity(getAndFlip(xBarConsumedParityNext));
      }
      stateLane[34] = smem.ctaRowMax[warpIdx.y][warpIdx.x][laneId()];
      ldgsts::waitGroup<0>();
    }
  } else {''')
    text = replace(text,
        "    Vec<WarpAcc, nbHeadSplits> accs{};",
        '''    Vec<WarpAcc, nbHeadSplits> accs{};
    static_assert(sizeof(accs) == 32 * sizeof(float));
    static_assert(ThrdRegRowMax::size == 1);
    if (phase == 2) {
      globalRowMax[0] = stateLane[32];
      globalRowSum[0] = stateLane[33];
#pragma unroll
      for (uint32_t i = 0; i < 32; ++i) {
        reinterpret_cast<float*>(&accs)[i] = stateLane[i];
      }
    }''')
    text = replace(text,
        "    auto const fullRescaleMask = UniformRescaleMask::filled(~0U);",
        '''    if (phase == 1) {
#pragma unroll
      for (uint32_t i = 0; i < 32; ++i) {
        stateLane[i] = reinterpret_cast<float const*>(&accs)[i];
      }
      stateLane[32] = globalRowMax[0];
      stateLane[33] = globalRowSum[0];
      ldgsts::waitGroup<0>();
      return;
    }
    auto const fullRescaleMask = UniformRescaleMask::filled(~0U);''')
    text = replace(text, "if (seqIterInit < nbSeqIters) {  // otherwise rcpRowSum will be NAN.",
                   "if (originalSeqIterInit < nbSeqIters) {  // Include checkpoint-only splits.")
    (dest / "xqa/mha.cu").write_text(text)
    header = (dest / "xqa/mha.h").read_text()
    pos = header.index("void launchMHAFlashInfer(")
    end = header.index("void launchHopperF8MHA(", pos)
    segment = replace(header[pos:end], "cudaStream_t stream);",
                      "cudaStream_t stream, float* rawState, uint32_t prefixTiles, uint32_t phase);")
    (dest / "xqa/mha.h").write_text(header[:pos] + segment + header[end:])
    wrapper = (dest / "xqa/xqa_wrapper.cu").read_text()
    pos = wrapper.index("void xqa_wrapper(")
    old = wrapper[pos:]
    new = replace(old, "TensorView semaphores, TensorView scratch, bool enable_pdl) {",
                  "TensorView semaphores, TensorView scratch, bool enable_pdl, TensorView raw_state, int64_t prefix_tiles, int64_t phase) {")
    new = replace(new, "                      stream);",
                  "                      stream, reinterpret_cast<float*>(raw_state.data_ptr()), prefix_tiles, phase);")
    (dest / "xqa/xqa_wrapper.cu").write_text(wrapper[:pos] + new)
    binding = (dest / "flashinfer_xqa_binding.cu").read_text()
    pos = binding.index("void xqa_wrapper(")
    segment = replace(binding[pos:], "TensorView semaphores, TensorView scratch, bool enable_pdl);",
                      "TensorView semaphores, TensorView scratch, bool enable_pdl, TensorView raw_state, int64_t prefix_tiles, int64_t phase);")
    (dest / "flashinfer_xqa_binding.cu").write_text(binding[:pos] + segment)
    receipt = {
        "identity": identity, "origin_mha_sha256": origin_sha, "patch_sha256": patch_sha,
        "upstream": str(UPSTREAM), "phase": {"0": "private-build unsplit control", "1": "checkpoint", "2": "resume"},
        "contract": "BF16 GQA8, head128 page16, raw per-warp state; no pairing yet; qualification only",
        "state_floats_per_cta": STATE_FLOATS_PER_CTA,
        "generated_sha256": {str(p.relative_to(dest)): digest(p) for p in dest.rglob("*") if p.is_file()},
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return dest, receipt


def jit_spec():
    import torch
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    dest, receipt = generate()
    upstream = gen_xqa_module(torch.bfloat16, torch.bfloat16, 16, 128, 8, False, torch.bfloat16)
    # Original SM120 compile flags, including PDL support. Runtime PDL disabled in gate.
    spec = gen_jit_spec(
        "iscas_xqa_raw_selfresume_" + receipt["identity"],
        [dest / "xqa/mha.cu", dest / "xqa/xqa_wrapper.cu", dest / "flashinfer_xqa_binding.cu"],
        extra_cuda_cflags=upstream.extra_cuda_cflags,
        extra_include_paths=[dest, UPSTREAM], extra_ldflags=["-lcuda"],
    )
    return spec, receipt
