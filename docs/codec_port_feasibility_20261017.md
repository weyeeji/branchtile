# CoDec fair comparison and port feasibility

CoDec can be a meaningful related-method baseline, but its public CUDA path is
not a drop-in baseline for our BF16, fused, 16-token-page engine state. Qwen3-4B
TP2 is easier than the 30B model **for head geometry only**. Precision, layout,
and long-context metadata/reduction capacity remain separate port tasks.

This audit reads the [public implementation at f70686e](https://github.com/wzbxpy/codec/tree/f70686e1024ea07f559be84d9096604293636149)
in `build/codec_prior`. No dependency was installed, source modified, compilation
started, or GPU used. Source hashes are archived in
`results/codec_port_feasibility_20261017.json`.

## Concrete contracts

- `csrc/tree_attn.cpp:185` explicitly accepts only FP16 Q; `:199` and `:203`
  require separate contiguous NHD K/V. The current native fused-cache views have
  interleaved K/V storage and fail this contiguity contract. Reinterpreting BF16
  as FP16 would be invalid; converting only CoDec inputs changes the experiment.
- `csrc/include/config.hpp:19` selects F16/F16/F16/F16 MMA. Stage1 accumulators,
  softmax maxima/sums/scales, normalized per-page outputs/LSE, and stage2/3 merge
  arithmetic are all half. This is substantially more than an input-type guard.
- The actual exported head dispatch is Hq32/Hkv8 (GQA4). Some old generated
  Hq32/Hkv32 files exist but are not selected by the interface. Qwen3-4B TP1 has
  exactly 32/8; TP2 has 16/4, retaining the same GQA4 rows per KV head. Supporting
  TP2 needs new head dispatch and template instantiations in all three stages,
  but leaves its stage1 GQA tile geometry unchanged. Our 30B TP2 shape16/2 is
  GQA8 and doubles that tile's query rows relative to the public configuration.
- Stage2 supports only 1..128 splits (`tree_attn.cpp:275`). A shared 120528-token
  prefix in 16-token physical pages supplies 7533 per-page results and needs an
  8192-way reduction. Merely extending dispatch is insufficient: the current
  stage2 `half smem[NSplits][16]` would require 256 KiB per CTA. The prefix-only
  stage1 output allocation at Hq16 would already be about 471 MiB, before tail
  pages or extra query groups. This is an algorithm/layout mismatch, not a small
  dispatch omission.
- CoDec's own engine defaults to 1024-token pages. At 120K, the whole shared
  prefix then has roughly118 pages, fitting the reduction limit; at144K, 1024
  pages exceed128 and a larger page such as2048 is needed. Large-page APC must
  privatize the partially shared final macro-page when private suffixes differ;
  it cannot insert padding holes into a logical sequence or count the conversion
  from our original pages as free.
- Its model runner calls `build_prefix_tree` every decode step. The builder
  reads lengths/table back to the CPU, constructs page/node/edge arrays, and
  transfers metadata to the GPU. The standalone operator benchmark builds this
  metadata before timing; its engine includes preparation. Both scopes must be
  reported separately in a comparison.
- The pinned CUTLASS gitlink is `c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752`, absent
  in this local checkout. Build work must use an isolated source/output directory
  and fetch that pin or explicitly document another header version. Installing
  into or modifying the active model environment is unnecessary.

## Lowest-cost fair runnable path

First establish a **separate FP16 comparison track** matching the public method:
Hq32/Hkv8, D128, contiguous K/V, and a supported large page size with <=128
splits. All arms must receive the exact same FP16 arrays, page table, and lengths.
Use the unchanged official CoDec operator, a native FlashInfer/XQA control, and
our FP16 native-schedule variant only after independently qualifying it. A common
one-time BF16-to-FP16 preparation is permissible for a clearly labeled converted
operator dataset supplied identically to every arm; it is not a real BF16-model
benchmark. Prefer true FP16 Q/K/V captures for a model-backed secondary track.
Check both implementations against the same FP64 reference and report accuracy;
CoDec need not be bitwise equal to native to be a legitimate approximate baseline.

Qwen3-4B TP1 gives the public head shape without modifying dispatch. Its entire
model and KV path would need a declared FP16 execution contract. This is a
secondary precision/model setting, not evidence about the BF16 30B main result.
TP2 can be used with a small **head-geometry port**, clearly labeled as such.
The page layout must be common to all compared operator arms. Fixed-state kernel
results cannot be presented as whole-engine results or exclude a required cache
repacking/metadata cost from an end-to-end claim.

## BF16 port cost and preserved method

A BF16 implementation preserving CoDec's shared-page computation and tree-merge
method is feasible, but is a substantive port:

1. Parameterize storage and copies; use BF16 Q/K/V/O and an explicit LSE dtype.
2. Use BF16 tensor-core inputs with FP32 MMA accumulators. Convert FP32 softmax
   probabilities to BF16 explicitly before PV; the existing half score fragment
   cannot simply be reinterpreted as the MMA A operand.
3. Define and implement softmax/merge precision consistently in stage1, stage2,
   stage3, params, and copy helpers. FP32 softmax/LSE/merge is the defensible
   numerically robust choice, but changes register/shared-memory demand and must
   be labeled rather than attributed to the official FP16 kernel's performance.
4. Add Hq16/Hkv4 dispatch and generated stage instantiations. This is the small
   part of the port; 16/2 additionally changes the GQA tile size/resource profile.
5. For our actual original-page engine, add native strided loaders plus either
   a hierarchy handling thousands of physical page results or virtual large-page
   grouping with a physical-page gather loader. Keep all dynamic suffix and
   partial-page behavior valid. Preallocated graph buffers and incremental
   metadata updates need separate engineering and correctness gates.

Thus the shape-only TP2 change is small; the BF16 kernel is medium-to-large
work across at least six arithmetic/type files; the original-page E2E integration
is another substantial layout/planning change. A quick `half` name replacement
cannot preserve the method correctly. We have no compiled or measured BF16
CoDec port and should not claim one exists or infer its speed from these limits.

The recommended order is an official FP16 operator control to establish a
reproducible CoDec comparison, then a labeled BF16 port only if the main result
justifies that work. A directly compatible BF16 related-method baseline (the
parent is investigating PAT separately) can provide an earlier same-contract
comparison; it does not erase the need to discuss CoDec's overlapping mechanism.
