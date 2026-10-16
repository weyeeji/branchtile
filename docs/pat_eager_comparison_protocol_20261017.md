# PAT complete-call comparison protocol

This is a bounded **operator** comparison on the captured real train1, rank0,
layer15 BF16 state. It does not measure model or request latency. The first run
uses all eight actual requests and their unequal complete histories.

## Source and ABI

Official PAT is pinned at `8cb067f0a31dabe800958f657093f17ac5d7366d` in
`build/pat_prior`. The separate parent-owned build receipt records the SM120
compiler target, CUDA13 CCCL include path and removal of one unused obsolete
Thrust include. The runner verifies original and copied source hashes plus the
compiled extension hash. No scheduling or arithmetic source adaptation is made.

The input is the actual BF16 capture, never FP8 converted to BF16. The original
fused cache shape, strides, full-width physical page table, per-row lengths and
unit scales are reconstructed. PAT receives NHD K/V views of that same cache;
there is no gather/repack or method-specific precision change. The original CPU
page table remains alive because PAT tree nodes reference its storage.

At page size 16, PAT's API rejects `(M,N,warps)=(16,128,1)`. The runner passes the
remaining ten default tile choices in their original order, with the default
heuristic (`use_compute_model=False`), Hq/Hkv=8 and Hkv=2. There is no measured
input-specific tile search. Internal selected schedules are not exposed by the
official binding, which is disclosed in the receipt.

## Qualification

All arms must pass the same full-history FP64 reference with fixed
`atol=0.006, rtol=0.015` and finite outputs. Native PDL AA and RawPair must also
match native PDL output bits. PAT's native bit differences are reported but do
**not** disqualify a mathematically accurate implementation. Initial and final
outputs are checked. Native output is additionally compared with captured O.
A failure preserves the report and prevents a performance-success claim.

## Fair timing

`native_pdl`, independent `native_pdl_AA`, official `PAT_default` and
`raw_pair_complete` all execute eager on the legacy default stream. PAT retains
its official stream creation/destruction, temporary allocation and final gather;
RawPair retains Q packing, pair page/length copies, raw-state writes/reads and
native tail/merge. No arm uses a CUDA graph.

The official PAT implementation creates blocking streams and gathers on the
legacy default stream. Every timing sample records default-stream CUDA events
**and** wall time ending in `torch.cuda.synchronize()`. The latter includes all
PAT streams even if an event dependency were misunderstood. The optional trace
allows a separate launch/dependency audit. Hot and 256MiB-flushed cold regimes
use randomized paired blocks with AA and bootstrap ratio intervals. Python and
host launch gaps are part of this eager comparison.

Tree building, CPU scheduling and GPU metadata upload are explicitly timed
outside the prepared-metadata operator call. RawPair initial allocation,
validation and length upload are also reported. Input reconstruction is a saved
fixture operation; an online GPU-to-CPU table readback is neither measured nor
claimed free. These setup numbers are not an engine integration benchmark or an
estimate of per-token production planning cost.

## Frozen command

Run only after the parent assigns an idle GPU and the build receipt is complete:

```bash
.venv-engine/bin/python scripts/benchmark_pat_raw_pair_eager.py \
  --build-receipt results/pat_prior_build_v2.json \
  --capture results/dense_capture_b8_train1_bf16_v1/rank0/dense_capture/layer15.pt \
  --output results/pat_raw_pair_eager_v1/train1_r0_l15.json --profile
```

CPU preparation validation: `py_compile` and `--help` pass. No GPU run was made
by the script author at protocol freeze.
