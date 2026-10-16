# Device metadata control: full-model qualification and first natural seed

This is an implementation control for the existing exact dense prefix/private-tail cascade. GPU metadata construction and avoiding repeated CPU planning are established techniques; neither is presented as a new paper mechanism. The prior four-seed host-plan control and all negative results remain intact.

## Implementation and correctness contract

Frozen runtime sources are `src/vllm_dense_cascade_device_plan.py`, `scripts/vllm_dense_cascade_device_worker.py`, and `scripts/benchmark_dense_cascade_device.py`. Source hashes are recorded in every summary and verified by the independent CPU audit.

FA2 private-tail planning uses `disable_split_kv=True`. Its single-chunk work queue is stable for fixed batch size and one query per request. Each captured model graph constructs compact tail indptr, indices, and last-page lengths on the GPU from actual positions and page tables. Attention still reads every real private-history token; it does not pad the attention workload with future tokens. All 48 layers share these metadata arrays, with separate stable plans for batches 2 through 8.

The first eligible cohort in each phase performs one physical-prefix identity check and binds its prefix. Later calls admit only a subset of those request identities with unchanged CPU slot/prefix epochs. A GPU prefix-identity check also executes on every replay. Prefix overwrites or new requests cause native fallback, and metadata bounds or identity violations accumulate device error flags that reject the phase. Batch 1 and nonpure-decode work use the existing native engine path. Native attention preparation remains included in the candidate wall time.

Both wrapper `.plan` methods are replaced by guards that raise if called after initialization. Tail planning and metadata `.item()` calls are absent from ordinary decode steps. Flags are read at phase end; raw generation time excludes that RPC, so the audit also reports generation plus phase validation and record export. Setup separates configuration and production/checked graph capture. A conservative first-use ratio charges all candidate setup, including experimental checked graphs.

Wave 40 full-model qualification passed for both BF16 and FP8 caches: 15 decode steps × 48 layers × 2 ranks = 1,440 checks per dtype. BF16 retained one failed candidate/native pairwise screen; both outputs passed the original `.006/.015` tolerances against an independent full-history FP64 reference. FP8 had no pairwise failures. Original pairwise-pass layers are screened, not claimed independently FP64-certified. Independent native-layout small qualifications cover long tails, page crossings, reorder, nonunit FP8 scales, 48 layers, and invalid metadata rejection; those are separate from the 16-token model qualification.

## Wave 42 results, seed 20261016

All four jobs cleanly exited. Every phase and every request produced exactly the same token sequence as its corresponding old implementation, for native and cascade separately. This verifies that this implementation change preserved the observed trajectories; it does not mean native and cascade produce identical tokens. All active candidate batch sizes 2–8 had zero device error flags, unchanged plans, a single initial binding, and no exceptional fallback. Natural production preparation medians were 16.7–19.6 microseconds per rank. All 64 natural outputs stopped without the length cap.

Ratios below are native wall divided by candidate wall; greater than one favors the candidate. Fixed rows are full generation with a fixed output length of 128 per request, not strictly identical work, because native and candidate tokens and MoE routing differ.

| Condition | Fixed generation | Natural generation | Natural cold including validation/export | Conservative first-use cold including all candidate setup |
|---|---:|---:|---:|---:|
| train1 BF16 | 1.2374 | 1.3894 | 1.2133 | 1.1494 |
| train1 FP8 | 1.1991 | 1.0670 | 1.0394 | 0.9931 |
| train7 BF16 | 1.2801 | 0.9723 | 0.9829 | 0.9438 |
| train7 FP8 | 1.2287 | 0.3602 | 0.5723 | 0.5569 |

Cold means one uncached anchor prefill plus the complete ready group. Engine initialization is shared and reported separately. The conservative column includes both production and experimental checked graph setup, rather than silently assuming free setup. The phase-end RPC includes record export, which is instrumentation as well as validation.

The implementation helps fixed-length generation, but the negative natural conditions remain. For example, train7 FP8 generates 7,317 native versus 13,915 candidate tokens. Train7 BF16 generates 10,003 versus 9,774 tokens yet candidate task wall is still slower because the distribution of surviving batch sizes differs. Output length alone is not a sufficient explanation or performance metric.

Strict FINAL quality is also insufficient for a task-success claim: train7 BF16 rounded accuracy is 3/8 native versus 1/8 candidate; other conditions range from 2/8 to 2/8. These are eight-question internal metrics with parsing and unit limitations, not official program-execution accuracy. All parse failures remain in the denominator. Historical old/new wall timings are not a contemporaneous paired test, even though the token trajectories match exactly.

## Reproduction and next evidence

The complete first-seed audit is `results/dense_cascade_device_natural_audit_v1/audit.json`, produced by `scripts/summarize_dense_cascade_device_natural.py`. It records runtime source hashes, clean-exit ledger receipts, 48-layer checks, dynamic metadata flags, costs, per-question quality, batch survival, and exact historical token comparisons. Per-session correctness/cost auditing is in `scripts/audit_dense_cascade_device_model.py`.

The additional seeds 20261017, 20261018, and 20261019 retain the previously declared BA/AB/BA orders and sampling settings in `configs/dense_cascade_multiseed_natural_v1.json`. The new independent runner `scripts/benchmark_dense_cascade_device_multiseed.py` changes the phase list and seed only, reusing the frozen device worker/plan and inclusive timing contract. `scripts/audit_dense_cascade_device_four_seeds.py` will combine all four seeds without filtering positive/negative timing, caps, or parsing failures. No multi-seed conclusion is asserted before those jobs finish.

## Completed four-seed audit

All eight actual sessions (initial and additional seeds for four conditions) completed with clean exits. The unified audit is `results/dense_cascade_device_four_seed_audit_v1/audit.json`. It verifies the prescribed order and unchanged sampling settings, source hashes, all-layer qualification, device metadata, complete histories, and ledger termination. All 256 natural outputs across both methods stopped without the cap. Every observed natural token sequence remains exactly equal to the corresponding old control with that seed and method.

| Condition | Four-seed generation ratio, including phase validation/export | Four-seed cold ratio, including phase validation/export | Rounded accuracy, native → candidate |
|---|---:|---:|---:|
| train1 BF16 | 0.96683 | 0.97889 | 8/32 → 8/32 |
| train1 FP8 | 0.96680 | 0.97880 | 8/32 → 8/32 |
| train7 BF16 | 0.97383 | 0.98323 | 13/32 → 10/32 |
| train7 FP8 | 0.65678 | 0.78319 | 9/32 → 8/32 |

Ratios use summed wall times across the four seeds. Candidate cold wall is approximately 2.16%, 2.17%, 1.71%, and 27.68% higher respectively before separately charging candidate setup. The smaller overhead therefore did not make any of these aggregate natural-task conditions faster. The matching historical trajectories confirm that the observed changes from the old implementation do not arise from altered outputs, but historical timing comparisons still cannot isolate causal speedup from run-to-run system variation.

This closes the hypothesis that removing the prototype's repeated host planning alone is sufficient to produce a natural-task win on these two documents. It does not deny the measured graph execution opportunity or fixed-length generation improvement, and it does not rule out a different mechanism or workload with stronger reuse and better task accuracy. No novelty or ISCAS-ready result is claimed for this engineering control.
