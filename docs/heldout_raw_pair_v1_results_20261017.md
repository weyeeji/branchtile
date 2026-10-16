# Formal heldout v1: modest conditional speed, numerical contract falsified

The frozen primary evaluation is complete, but the implementation does **not** satisfy its bitwise-preservation claim across the selected documents. Evaluation document 2 changes generated tokens. A separate same-state diagnostic subsequently confirms an actual attention-output difference, with identical Q and repeatable native controls. Passing documents do not cancel this counterexample. A repaired mechanism needs a new version and complete reevaluation; filtering this document is not an admissible repair.

The selection and production-only accounting were frozen before heldout execution (`e5b58c1`). All eight documents and 101 questions remain: five paired completions covering 57 questions, one output-divergence failure covering 14 questions, and two context failures covering 30 questions. No document was truncated or replaced. The summary is `results/native_raw_pair_heldout_production_final_audit_v1.json`.

| Evaluation index | Natural questions | Outcome | Ready generation ratio | Anchor + generation ratio | Primary ratio including per-document setup |
|---|---:|---|---:|---:|---:|
| 2 | 14 | Output divergence; no paired speed claim | — | — | — |
| 3 | 8 | Paired completion | 1.0539 | 1.0329 | 1.0229 |
| 5 | 16 | Paired completion | 1.1210 | 1.1010 | 1.0777 |
| 6 | 10 | Paired completion | 1.1309 | 1.0720 | 1.0589 |
| 7 | 14 | Paired completion | 1.1470 | 1.0812 | 1.0537 |
| 8 | 21 | Context failure retained | — | — | — |
| 9 | 9 | Paired completion; two output caps per arm | 1.0789 | 1.0636 | 1.0576 |
| 10 | 9 | Context + reserved-output failure retained | — | — | — |

Ratios are native time divided by candidate time. Each primary denominator charges measured offline-receipt preflight, candidate configuration, actual production graph capture, and setup-receipt export once for that document, with no amortization. Among the **five paired completed documents only**, the predeclared primary geometric mean is 1.054001 and the ratio of summed times is 1.056278. These are conditional timing observations, not a claim that the mechanism passed evaluation. Common input/model initialization is excluded from warm-service task latency; separate first-use component estimates are not independently timed cold-service measurements.

## Numerical counterexample

Document 2 first visibly diverges on question 6 at output token index 1939 (zero-based): native token 3405 versus candidate token 8350. Eight of its fourteen answers ultimately differ; candidate question 10 reaches the 8192-token cap. Neither arm records preemption. Batch/question/position metadata remains equal until a later exit changes the surviving cohort.

The retained same-state diagnostic snapshot at graph index 1855, rank 1, layer 0 contains identical native and candidate Q but two differing BF16 O entries, both in row 3, corresponding to original question 6. The maximum absolute difference is 2.384185791015625e-7. Native AA, PDL-off, witness, and an unshared-prefix checkpoint agree bitwise; candidate poison replay reproduces its own differing output. Thus a small numerical difference falsifies the declared exact contract. Shared question identity and earlier occurrence support a propagation hypothesis, but this snapshot alone is not a complete counterfactual proof of every later token transition. The initial diagnostic process also encountered a receipt-writing error; that failed process and the usable snapshot are both retained.

Independent CPU evidence is `results/raw_pair_eval2_late_same_state_v1/independent_snapshot_audit.json`, with the snapshot hash and exact coordinates. The later diagnostic must establish the complete reference trajectory and propagation separately. The official evaluation result remains failed regardless of diagnosis.

## Quality and timing limits

For paired completions, both methods produce the same 19 exact and 28 rounded-correct answers out of 57. Across all 101 selected questions these are verified lower bounds of 19/101 and 28/101, not a claim that unavailable results were evaluated as ordinary answers. Document 9 hits the output cap on two questions in both arms, and those answers remain in the denominator. Parse failures and reference-answer parse failures are retained.

Native AA generation time ratios are approximately 1.0001, 1.0829, 1.1021, 1.0041, and 1.0367 for indices 3, 5, 6, 7, and 9. The 8.3% and 10.2% drifts limit precision of those single-session means; two ABBA repeats do not establish a confidence interval. Follow-up repeats must be reported separately and cannot replace the frozen first observations with whichever result is faster.

The six executed formal jobs have no intersection with another recorded allocation on the same GPUs. This ledger check does not rule out CPU contention, unrecorded activity, clock/thermal variation, or other timing interference. Earlier PAT CUDA-visibility exposure is separately documented and does not overlap these formal jobs.

An additional Thinking-4B development run has strict accuracy 0/8, but CPU case review finds a mixture of missing FINAL units, wrong-column reasoning, missing original tables, and targets inferred from rounded ratios. Its performance is not evidence of a high-quality application. The explanatory review preserves the original score and all inputs: `results/gqa4_thinking_task_quality_cpu_v1.json`.
