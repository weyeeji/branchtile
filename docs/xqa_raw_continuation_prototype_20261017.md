# Native raw-state continuation: bounded prototype

This is a hardware implementation hypothesis, not a claim that shared-prefix two-stage attention or raw softmax states are new. First establish GQA8→GQA8 checkpoint/resume equality, then consider using the otherwise padded M16 rows for a GQA16 shared-prefix stage and GQA8 private tails. Whole-model integration is conditional on correct outputs and useful complete-operator latency.

## Prior art and narrow distinction

[ChunkAttention, §3.2 and Algorithms 1–2](https://arxiv.org/html/2402.15220v1) already stores unnormalised O/m/n from shared chunks and loads these in a sequence-first phase before private chunks. It reduces independent chunk states; this directly rules out claiming that two-stage raw-state continuation is new by itself. [PAT, §7](https://arxiv.org/html/2511.22333v1) also stores maximum, sum and value accumulators before an online-softmax merge. [FlashInfer cascade](https://docs.flashinfer.ai/api/cascade.html) exposes O/LSE state merges. [FoldAttention, September 2026](https://arxiv.org/html/2609.33410v1) changes the softmax reference to make shared-prefix/split partials additive. None of those descriptions establishes the exact implementation below; this limited comparison does not establish novelty against all work.

The proposed distinction is preservation of a specific existing production kernel's split assignment, per-row strided tile schedule, per-PV-warp partial registers and QK feedback hints while changing which rows share physical KV loads. Its value must come from hardware execution and end-to-end gains, not bitwise equality alone.

## Actual state boundary

The frozen native SM120 XQA has 256-token CTA tiles, four QK producer warps, and four PV consumer warps. At head dimension 128, two PV warp groups retain separate accumulators. They must not be merged at the prefix boundary: native finalisation converts these partials to BF16 before its later reduction. `CTA_ROW_MAX_BACKWARD_METHOD=1` also retains four `smem.ctaRowMax` feedback buffers. Saving only a logical per-row maximum and sum loses state needed by the next tile's probability rounding.

The self-resume implementation saves each consumer warp/lane's 32 FP32 accumulator registers and FP32 max/sum, and each producer warp/lane's backward max hint. Padded values are kept for a transparent initial layout. State shape is `[B,Hkv,S,4,32,35]` FP32: 2,867,200 bytes at B8/Hkv2/S10, written and read once per layer. A later pair-aware compact layout can reduce padding, but is not in this first qualification.

For common-prefix length P, let C=floor(P/256). Split s handles original tiles s,s+S,… . The prefix phase stops before C without changing the full-history length used by load bounds or masks. The tail begins at `s + ceil(max(0,C-s)/S)*S`, so the first mixed shared/private tile is computed entirely by the tail. If no suffix tile belongs to a split, it still finalises its saved prefix state. Only the tail performs the original normalisation, BF16 scratch write and original split merge.

The producer drains its final consumed-X barrier before recording its hint. The tail recreates synchronization objects, restores the four hint buffers before QK runs, and restores the original PV registers before the first suffix iteration. This is a candidate implementation, not a proof of compiler-level bitwise equivalence; the qualification is decisive.

## Build and qualification

`src/xqa_raw_continuation_build.py` copies the 884 KiB local XQA source tree into `build/xqa_raw_continuation/<source+patch hash>/`, preserving upstream license headers. It patches only that private copy and builds a uniquely named FlashInfer JIT module. Installed files and the native JIT/AOT module names are unchanged. Source anchors fail if the installed implementation differs; every copied/generated file has a SHA256 receipt. There is no root access.

CPU build command (activating the existing venv via PATH also supplies ninja):

```sh
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH CUDA_VISIBLE_DEVICES='' FLASHINFER_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 .venv-engine/bin/python scripts/qualify_xqa_raw_selfresume.py --build-only --output results/xqa_raw_selfresume_build_v5/build.json
```

The single-GPU qualification command, run only by the root scheduler after build, is:

```sh
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=0 .venv-engine/bin/python scripts/qualify_xqa_raw_selfresume.py --output results/xqa_raw_selfresume_gpu_v1/qualification.json
```

Gate: ten native A/A comparisons; private unsplit module against native; native runtime-PDL off against production runtime-PDL on; checkpoint does not write O; active raw state finite; resumed O bitwise equal; both outputs independently within unchanged FP64 tolerance; two-kernel captured graph also bitwise equal. Five cases include empty prefix, floor-to-256 mixed boundary, differing request lengths and splits with no remaining tail work. Outputs are preserved on failure, and existing summaries cannot be overwritten.

No speed comparison is made in this first script. Candidate runtime PDL is off; any later timing must use production native PDL-on as the main baseline as well as a PDL-matched control. The first GPU gate should take seconds once built; allow at most 180 seconds process wall time for diagnostics. No full model, scheduler, general head dimensions, sliding window, attention sinks, pair maps or odd-batch handling are implied by this scope.

The main resource risks are fewer prefix CTAs at fixed native S, raw-state global traffic, an extra launch and compiler register growth. The prefix and tail may not select a larger S just to regain occupancy while claiming preserved reduction order. A profitable result must charge Q packing, state traffic, all launches, tail work and final native merge.
