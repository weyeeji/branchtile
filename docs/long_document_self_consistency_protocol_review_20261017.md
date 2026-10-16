# Long-document self-consistency: task justification and integration gate

Status: CPU/source/primary-literature review only. No new evaluation results,
GPU job, production worker change or mechanism novelty claim.

## Is this a real workload?

Yes, conditionally: several independently seeded reasoning samples can support
a numerical answer vote and expose disagreement for the **same original
financial QA questions**. Every extra request must contribute to a reported
vote or uncertainty decision. It is not legitimate to add requests merely to
maintain batch size and call the original single-sample task faster.

This use case is established prior art. Hydragen explicitly motivates
self-consistency/multiple completions, supports hierarchical shared prompts,
and evaluates both long-document QA and multi-sample competitive programming.
It therefore prevents claiming long-document repeated sampling or keeping a
large shared-prefix batch as a new application principle.
[Hydragen paper](https://arxiv.org/html/2402.05099),
[authors' implementation](https://github.com/ScalingIntelligence/hydragen).

The 2026 ICML paper Optimal Self-Consistency studies fixed and adaptive sample
allocation and introduces Blend-ASC. Its analysis distinguishes convergence
to the model's modal answer from convergence to the correct answer. A fixed
four/eight-sample task is a clean systems control but cannot be called the
best use of the inference budget.
[Official ICML/PMLR publication](https://proceedings.mlr.press/v306/feng26a.html),
[June 2026 full version](https://arxiv.org/html/2511.12309v2).

Recent work also reports majority-vote failures on small instruction models
in hard science; that does not establish failure on our financial reasoning
model, but it makes a measured accuracy benefit essential rather than assumed.
[When Self-Consistency Backfires](https://arxiv.org/abs/2608.11403).

## Smallest defensible task protocol

1. Use all original questions of one already declared development document.
   First use N=4; N=8 is a separately declared extension, with the first four
   samples nested. Do not create paraphrases, synthetic questions, forced
   tokens, padding requests or repeated identical seeds. Retain the existing
   prompt, numerical answer format, natural EOS and 8192 cap. A deterministic
   hash of `(document_id, question_id, sample_index, study_seed)` assigns each
   request's seed and is stored before any model output is read. Keep existing
   temperature0.6/top_p0.95 and other sampling settings fixed.
2. Enqueue the finite manifest in a fixed round-robin question/sample order
   with the existing concurrency cap. Allow ordinary completion-triggered
   refill until the manifest is exhausted. Native and candidate receive the
   **same complete manifest**, seed per request, admission order and engine
   settings. No wall-clock-driven new arrivals or answer-dependent extra
   samples in the first experiment. Charge every request, including invalid
   parses, capped outputs, abandoned answers and final queue drain.
3. Use existing `financial_answer_metric.parse_value` to canonicalize each
   valid final answer without access to gold. Group exact Decimal value,
   dimension and answer kind; do not use gold precision to cluster answers.
   Prediction is the plurality answer, with ties broken by the earliest
   predeclared sample index. If no sample parses, return an invalid answer.
   Invalid samples remain in the N denominator and never create a valid
   "agreement" class. Score the final prediction with the existing frozen
   exact-match and rounded-match sensitivity metric, explicitly **not**
   DocFinQA program execution accuracy.
4. Report single-sample seed0 accuracy and average individual-sample accuracy,
   vote@4 (and separately vote@8), oracle any-correct@N only as a diagnostic,
   parse failures, cap rates and all sample lengths. Report vote count/N and
   top-two vote margin as uncalibrated agreement signals, not probabilities
   of correctness. A separate strict-majority acceptance view requires >N/2
   matching valid samples and reports both coverage and accepted-answer
   accuracy; abstaining on difficult questions is not free accuracy gain.
5. Compare native-SC@N against candidate-SC@N for system acceleration. Also
   compare native-SC@1 against native-SC@N for task utility and cost. Report
   total generation/prefill/setup/voting walltime, GPU-seconds, valid energy
   if measured, all generated tokens and questions answered correctly. The
   useful plot is quality versus total cost, not tokens/s alone. Same-session
   ABBA is repetition control; independent documents/seeds, not individual
   highly related samples, define the generalization unit.

A one-document pilot can show whether four samples produce diversity and any
useful changed vote, plus whether refill sustains candidate coverage. It
cannot establish a quality improvement on a population. Stop expanding this
scenario if samples are effectively identical, vote quality offers no useful
benefit/uncertainty distinction at its extra cost, or refill does not improve
actual candidate coverage. A speedup for an inferior fixed allocation is
still a backend result, but not a claim of best task-level cost efficiency.

## Why refill might help, and what it cannot fix

With Q questions and N samples, C active slots can stay full until roughly
QN-C samples have completed. This amortizes the final shrinking batch over
more **useful** work. It does not eliminate the final C-request drain and
does not bound its walltime: long reasoning samples are variable, the maximum
length can grow with more samples, and correlated difficult questions can
still dominate the end. No gain estimate is justified from request count.
The first actual native pilot must report per-step active batch, eligible
batch, prefix/private lengths, prefill/decode mix, preemptions and the fraction
of walltime after the last refill. Preserve all long samples and cap hits.

At fixed C, maximum simultaneously active KV need not grow with total QN,
but private completed-prefix cache entries can occupy capacity until eviction.
The long document must remain physically shared. Prefix eviction/recomputation
and APC hit rate are part of the result. The TP1 memory concern remains:
Thinking4B's 120K prefix alone is 16.55GiB; more requests do not create memory.

## Current source does not safely support this production claim yet

The frozen direct worker supports one initial cohort and its surviving
subset. At `scripts/vllm_native_raw_pair_direct_worker.py`, `prepare` rejects
any current `(request_id, slot, epoch)` absent from the first cohort with
`cohort_or_prefix_epoch_changed`. A replenished request therefore makes the
current implementation fall back to native. Removing this guard without a
replacement is incorrect. Existing shrinking-cohort receipts do not qualify
slot reuse or membership additions.

A new independent adapter would need these explicit contracts:

- A manifest identity `(question_id, sample_id, seed)` per request. Current
  precision admission maps prompt hash using `.index`, which identifies the
  original question but cannot distinguish its repeated samples. Keep that
  question identity and add the sample/seed registry; do not modify prompt
  tokens to encode seeds and lose the common-prefix workload.
- Admission/rebinding for every newly used slot/epoch. After its prompt is
  available, verify its full physical shared-prefix page sequence against
  the document's live reference before enabling candidate graphs. Existing
  survivors retain validated identities. Failed/evicted/mismatching pages
  cause native fallback and fresh verification. Charge checks and any GPU
  synchronization; do not inherit free one-time binding across arbitrary
  requests. New pages from an overwritten slot cannot inherit old trust.
- Mixed prefill/decode remains native; only actual pure single-token decode
  with supported actual=padded B uses the candidate. Do not force all mixed
  steps into a decode-only graph. Native GPU table/length buffers must remain
  aliases, and the bank must choose original native S for the actual B.
- Fresh same-state qualification across first admission, one/multiple slot
  replacements, prefix last-partial tiles, new short tails coexisting with
  long tails, natural completion and any preemption. Check all Q/O, hidden,
  logits and current KV on both ranks; test final native tokens per sample,
  not only aggregate vote equality. New receipts must reject the old cohort
  source fingerprint.
- Refill order must be audited. Per-request RNG seeds prevent accidental
  reuse of one global stream, but arithmetic may depend on batch membership.
  Record request/sample identities, positions and admission events to show
  native/candidate execute the same logical schedules. Differences in output
  trajectory must be diagnosed rather than hidden by a coincidentally equal
  final vote.

The native baseline must get the identical queue/refill policy; otherwise a
gain could be ordinary continuous batching rather than the shared-attention
mechanism. A useful decomposition is native-SC queued refill versus native-SC
fixed waves, then candidate versus native **both queued refill**. Fixed-wave
results are optional diagnostics, not a weaker headline baseline.

Recommendation: a justified scenario-validation branch, conditional on a
small native utility/coverage pilot and new admission correctness work.
It is not a replacement mechanism or a novel scheduler claim, and it should
leave the original single-sample natural-EOS evaluation unchanged.
