# Method, closest prior work, and proof obligations

This draft describes a candidate contribution: shared-prefix decoding that preserves the arithmetic schedule and control state of one pinned production XQA implementation, while removing duplicate prefix reads and input-packing launches. It is a concrete implementation hypothesis with bounded numerical evidence, not a proof of universal bitwise equivalence or a claim of first deterministic attention. Whole-model value must be established separately with natural output lengths, deployment costs and new evaluation documents.

## Method and contract

Keep the actual native batch's split count S, its 256-token tile sequence `s, s+S, s+2S, ...` for each split s, original QK/PV casts and accumulation, and final native split merge. Group two GQA8 requests inside the existing M16 tensor-core tile. Only physically identical, complete prefix tiles are read jointly. At cut `C=floor(P/256)`, export each split's unreduced per-PV-warp accumulator fragments, maximum, denominator and backward maximum hints. The private phase restores the native request's fragments and continues at the first tile of that same sequence at or beyond C. The shared/private mixed tile remains in the private phase; there is no independently normalized prefix/tail softmax merge.

Preserving a reduction tree was insufficient in the original implementation. A real late decode state exposed a cross-request branch: one member needing rescaling caused a backward hint update in another member whose native group would have skipped it. The hint changed first; denominator/accumulator changes appeared one tile later. The repair preserves hint-update decisions for each **original GQA group**, not each packed CTA or individual attention row. Group8 and group4 are separate implementations requiring separate qualification. Existing extra rescaling by exp(0) is only justified over the checked finite path, not every exceptional floating-point input.

The direct-input variant moves the two-request permutation into Q load addresses. It reads every second native metadata row rather than copying pair tables/lengths, and zero-fills a missing odd member during asynchronous loads. This removes three prefix-preparation kernels. It does not change the arithmetic or support arbitrary Q strides: this version still requires contiguous Q. Model-side Q/O adaptation, shared-state traffic, prefix validation, setup and graph-bank management remain costs to measure.

The proposed relational contract is replacement equivalence for the **same engine state**: same original batch and native split policy, Q/K/V values and scalar scales, physical page identity, lengths/masks, hardware/compiler/kernel versions and ordered native operations. It is not equality across arbitrary batching, GPUs, releases, prefill/decode routes, or different attention algorithms. Counterexamples invalidate a claimed scope; they are retained and become development regressions after repair.

## What prior work already covers

| Primary source | Established overlap | Candidate distinction and unresolved burden |
|---|---|---|
| [ChunkAttention, §3.2](https://arxiv.org/html/2402.15220v2) | Shared chunks, query grouping, saved softmax numerator/max/denominator, two-phase processing. | Raw-state continuation alone is not new. Distinguish native unreduced warp fragments, original split schedule and hidden control state. |
| [PAT v3, §5–7](https://arxiv.org/html/2511.22333v3) | Prefix-aware packing, multiple tile shapes, long-KV splitting, multistream forwarding and output merging. | Show useful complete-call gains against this mechanism, not merely memory reuse versus naive decoding. Our repaired PAT graph baseline supplies a bounded operator comparison, not its whole-engine result. |
| [CoDec v2, §4–7](https://arxiv.org/html/2505.17694v2) | Prefix-tree partial attention, tree reduction and cost-aware division/scheduling. Its evaluation includes document QA and a 120K shared root. | Neither the long-document scenario nor shared-state scheduling distinguishes us. Its public implementation's dtype constraints limit current fair porting; no measured CoDec win is claimed. |
| [Vosti, §5 and Appendix A](https://arxiv.org/html/2609.38981v1) | Relational bitwise contracts, fixed-operation attention, explicit preservation of masked-out state and dependency reasoning. | Reproduce an existing optimized native schedule instead of claiming general batch invariance. Our tests are not Vosti's verification result. |
| [Taming Bitwise Behavior, §3–6 and Appendix F](https://arxiv.org/html/2609.11356v1) | Arithmetic descriptors, reduction-order-preserving optimization and static equivalence checking; theoretical attention treatment. | Native-schedule optimization is already a general principle. The prospective systems contribution is the XQA sharing/state/control interface and its deployment benefit. We have not implemented the paper's static checker or proved compiled-kernel equivalence. |

These primary versions were rechecked on 2026-10-17. Their descriptions do not establish this exact implementation, but that absence is not an exhaustive novelty proof. Input address mapping and stream/metadata engineering are supporting optimizations, not independent claims of a new attention algorithm.

## Evidence currently available

| Evidence | Independently checked result | Boundary |
|---|---|---|
| Minimal real-state counterexample | Old implementation fails: hint first at cut38, z/U at39. Grouped-hint and direct-input versions pass all49 cuts and final native output. | Extracted split/S1 regression, not a held-out task or all-value proof. |
| Direct input: two docs × two ranks × four layers × B8/7/6 | All48 cells pass native bits, FP64, changed graph inputs and shared-workspace revisits. Native/direct 1.28982–1.49854; copied/direct 1.03117–1.05469. Lowest paired lower95 bound for the latter is1.02963. | Hot complete-call graphs; B7/6 are subsets of captured B8 Q, not later natural survivor queries. No model projection from these ratios. |
| PAT matched graphs, two actual B8 states | Six runs/448 four-arm checkpoints pass. Multistream PAT/raw 1.510 and1.544; native/raw1.342 and1.385. CUPTI confirms every arm contains only full-call graph kernels. | Raw here is the original three-copy grouped-hint version. PAT is locally repaired; comparisons cannot be multiplied with direct-input ratios to infer a new matched result. |
| GQA4 four-way whole-model qualification | Both ranks cover B4..8 with40 states/rank and all36 attention layers; hidden/logits/currentKV and metadata checks pass. | First-eight-states-per-B qualification only; does not establish all later trajectory states or direct-quad performance. |

Artifacts: `results/xqa_raw_direct_input_counterexample_v1.json`, `results/xqa_direct_input_actual_independent_audit_v1.json`, `results/pat_fixed_graph_independent_audit_v1.json`, and `results/gqa4_quad_model_qualification_independent_audit_v1.json`. New unseen-document and multi-document deployment experiments remain necessary; do not fill this table with inferred end-to-end numbers.

## Remaining proof and evaluation obligations

1. **Addresses and ownership:** prove each live logical row reads the same native Q/K/V, mixed-boundary tiles remain private, dummy rows cannot access memory or influence live state, and physical identity remains valid through slot/epoch changes.
2. **Operations and control:** account for every original split tile, rounding point, lane/register fragment, max/sum reduction, hidden hint and group predicate. Restoring final m/z/U alone is insufficient. Include empty splits, unequal tails and original final merge ordering.
3. **Compiled semantics:** compare emitted instructions or use a suitable equivalence argument for altered layouts, exceptional values, zero signs/denormals and additional identity operations. Current source reasoning plus differential tests do not discharge this obligation universally.
4. **Runtime composition:** same-state all-layer checks must include native lengths/table aliases, cache updates and other model outputs; offline source/fingerprint gates must reject old incompatible builds. Natural trajectories and previously unseen documents test behavior beyond the small qualification window.
5. **Useful acceleration:** retain native production and PAT full-call comparisons, the unshared checkpoint/resume ablation, copied/direct ablation, AA noise and cold cases. Report natural complete-task time, token identity/quality, cap rates, all setup/anchor/switch costs and dispatch coverage. A kernel speedup alone cannot decide publication value.

A defensible paper would combine a causally explained native-control hazard, a validated schedule-preserving sharing implementation, and robust practical benefit. If new valid same-state counterexamples remain, narrow or withdraw the exactness claim and re-evaluate whether an approximate implementation contributes enough beyond the existing methods. Neither a generic shared-prefix story nor an ISCAS acceptance guarantee follows from the present evidence.
