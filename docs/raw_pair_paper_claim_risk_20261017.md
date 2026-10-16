# Paper claim risk after the held-out trajectory failure

**Current judgment: retain a promising implementation hypothesis, but withdraw
any unconditional native-bitwise or paper-ready claim.** The reported eval2 token
divergence is a counterexample to the deployed replacement's present contract.
It is not explained away by natural output-length variation. Its cause is still
under investigation; a token mismatch alone does not localize the defect to the
attention arithmetic.

## What is already prior art

- [ChunkAttention](https://arxiv.org/html/2402.15220v2), §3.2, already batches
  queries over shared chunks, saves unnormalized output/max/denominator state and
  processes private chunks in a second phase. Neither prefix sharing nor saving
  raw softmax state is new here.
- [PAT v3, March2026](https://arxiv.org/abs/2511.22333v3) already packs shared-prefix
  queries into multiple tile shapes, splits KV work, forwards with multiple
  streams and merges attention. Broad resource-efficient shared decoding claims
  overlap directly.
- [CoDec v2, March2026](https://arxiv.org/html/2505.17694v2), §4–5, combines a
  prefix-tree representation, partial attention/reduction primitives and a
  cost-aware executor. A new name for shared-prefix scheduling does not separate
  this work.
- [Vosti, September2026](https://arxiv.org/html/2609.38981v1), §5 and §7, treats
  bitwise dependency preservation as an explicit contract and distinguishes it
  from functional correctness. Its cross-kernel and distributed extensions are
  open boundaries, not permission to claim that our empirical tests prove them.

The plausible remaining distinction is a **concrete state interface for reusing
an existing production kernel's data while preserving that kernel's native
floating-point schedule**: unreduced PV-warp fragments, split-local tile order,
backward row-max hints, original casts and final split merge. Fixed engine/model/
TP/layout conditions must be stated. Device-resident cut parameters and graph
reuse make this implementable; they are not independently a novel shared-attention
algorithm.

## How the eval2 failure changes the decision

If the first-divergent same-state intervention finds a metadata, ownership or
adapter bug while the native attention-state transformation remains exact, a
repair may preserve the core hypothesis. Demonstrate that causal repair, retain
the failure, expand long-trajectory qualification and evaluate new untouched
inputs. Eval2 has now become a development counterexample; it must not be reused
as the sole supposedly independent validation of its own fix.

If valid same-state attention inputs still produce different bits because the
transformation changes arithmetic or omitted state, the promised distinction is
unproven or false. A proof obligation or causal state fix is needed. Merely calling
it “almost exact,” silently excluding this document, or relying on first-eight-
state qualification does not recover the contribution. A dispatch exclusion is
credible only when it follows a predeclared, independently validated structural
condition and its lost coverage/cost is reported.

If exactness is abandoned, an approximate/tolerance-equivalent shared kernel can
still be useful, but it must establish a distinct dataflow/resource mechanism and
robust natural-task end-to-end gains at comparable quality against PAT, mature
cascade and native XQA. Present evidence does not yet establish that alternate
paper. One-layer PAT speedup, especially while the stronger multistream path is
being repaired, is insufficient. Ordinary softmax state continuation is already
covered by the earlier work above.

The immediate decision test is therefore causal diagnosis of the first divergence
plus an effective complete-call baseline. Keep whole-model natural latency,
quality, caps and failed runs visible. Neither a fixed speedup percentage nor an
ISCAS acceptance threshold is implied by this assessment.
