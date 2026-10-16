# Paged owner integration: qualification before performance

This is an independent integration of the owner-task idea into the actual BF16 vLLM cache ABI. It does not reuse the contiguous microbenchmark's ABI or establish novelty. Frozen prior runners and kernels are untouched.

## Contract

- Q is actual `[B, H*8, 128]` BF16 with explicit request/head strides; K/V remain original fused-cache aliases, including cross-layer physical-page pitch. The kernel returns actual vLLM O; an internal LSE is retained for numerical qualification but no FlashInfer LSE conversion is required in the E0 production arm.
- Frozen prompt supports are grouped by exact owner bits once. Shared complete-page physical identity is checked before every replay. Divergence rejects replay; it is not silently treated as shared data.
- Every decode step maps physical addresses for all layers on GPU and updates private generated-tail endpoints. Each live request/head has one private-tail CTA whose loop grows with the actual tail. Long tails may therefore be slow.
- Each captured B has prebuilt metadata for every ordered request subset. Request exit/reordering copies that subset's row maps, task active counts and merge lists into persistent buffers. No obsolete request is included in a live output.
- Graph topology is fixed through capacities. New support identities, unsupported capacity or divergent shared pages require rebuilding and graph recapture. Those costs must be included in any reuse argument. Current runner fails rather than attempting an unmeasured recapture.
- Pure-decode graphs execute owner. Mixed/eager batches use an exact same-support E0 fallback, explicitly tagged in records and included in wall time.
- Layer output mirroring remains one charged copy to support graph shadow inspection. CPU metadata, mapping kernels, identity validation/synchronization and output copy are all inside complete generation wall time. Constructor and model graph capture are separately reported and must be charged for amortization.

## Validation stages

1. `CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/qualify_paged_owner_cpu.py --output results/<new>/qualification.json`: actual CPU constructor, 132 exact support/merge rows across all B1/B2/B3 subsets/permutations, four offline SM120 kernel compilations. No CUDA initialization in passing v2.
2. Root schedules `scripts/qualify_paged_owner_gpu.py --output results/<new>/qualification.json` on one GPU. Actual strided cache plus noncontiguous Q, FP64 O/LSE, graph replay after B-specific request permutations, tails 0/1/127/129 and physical page remapping. A shared-page divergence must reject. These are synthetic correctness checks, not application results.
3. Root schedules TP2 `.venv-engine/bin/python scripts/benchmark_vllm_paged_owner_natural.py --output results/<new> --methods e0,owner --budget 4096 --steps 4 --qualification`. Every requested method/rank must cover all 48 layer shadows and pass frozen O tolerance. E0 token paths teacher-force the candidate; actual pre-forcing logits must be finite, have matching key sets, include decode steps beyond prefill and cosine >=0.999. Raw top-1 differences are reported. Any failed/missing qualification rejects completion. This mode is not timing evidence.
4. Only after qualification run the same runner without `--qualification`, with a natural-EOS cap that fits context, e.g. 8192. Report actual output lengths, stop reasons, JCT, anchor-plus-JCT, tokens/sec and setup costs; divergent method lengths cannot be interpreted as pure kernel speedup.

Default developer input is the original exploratory B3 DocFinQA document, not formal held-out evaluation. Native E0 consumes exactly the same sparse supports and actual physical aliases. Optional `union` is the existing FlashInfer masked-union baseline, not our FixedUnion kernel. No FixedUnion deployment result is claimed.

## Current status

CPU support checks and offline compilation pass. Root-scheduled `paged_owner_gpu_v1` then completed with exit 0: 528 FP64 output/LSE checks passed, maximum output absolute error 0.0040671800 and maximum LSE absolute error 6.7349742e-7. The allocation lasted 9.029 seconds (0.00250806 GPU-hours). Real-model qualification and performance remain pending; root owns all GPU allocation and experiment accounting. The first attempted CPU import initialized CUDA through FlashInfer and was rejected before any explicit kernel/qualification; its incident record is retained separately.

## First real-model qualification outcome

`paged_owner_model_qualification_v1` completed both four-token arms but exited 1 at the frozen joint gate. All 576 layer-output shadows passed across both TP ranks and all 48 layers. All 12 same-path logit vectors were finite and had matching top-1 tokens, but q0_s3 cosine was 0.9975648314 below the predeclared 0.999 threshold. The failed result remains failed; its timed arms are not performance evidence. The original runner's `qualification_all_shadow_checks_passed` label incorrectly represented the joint output-plus-logit gate; the independent audit separates these fields without changing raw results.

Next diagnostic is an independent script `scripts/diagnose_vllm_paged_owner_logits.py` with E0 natural → E0 forced AA → owner forced, default 16 tokens. It preserves the 0.999 threshold, collects layer-by-layer shadow errors and cross-arm query/output norm differences, and reports E0-AA and owner logit gates separately. Norm differences can localize divergence but cannot prove a specific MoE-routing or probability-rounding cause. Runtime kernels remain unchanged.
