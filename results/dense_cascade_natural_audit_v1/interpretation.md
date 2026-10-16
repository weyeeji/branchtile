# Dense shared-prefix natural-outcome audit, 2026-10-17

Frozen metric: `strict-final-financial-answer-v1-20261016`; source and artifact hashes, complete question scores, output lengths, and both ranks' replay transitions are in `audit.json`. All 32 answers ended by stop with closed reasoning; no length caps or excluded questions. These are two training documents, one seed, not a quality noninferiority study.

| Document | Arm | Group JCT (s) | Anchor + JCT (s) | First-use anchor + setup + JCT (s) | Output tokens | Exact | Rounded |
|---|---|---:|---:|---:|---:|---:|---:|
| train1, prefix120528 | native | 33.702 | 54.805 | 54.805 | 12999 | 0/8 | 2/8 |
| train1 | cascade | 31.986 | 53.074 | 55.544 | 10750 | 1/8 | 2/8 |
| train7, prefix144576 | native | 20.841 | 49.764 | 49.764 | 7317 | 1/8 | 2/8 |
| train7 | cascade | 59.570 | 88.476 | 90.865 | 13915 | 1/8 | 2/8 |

The candidate improves train1 observed generation JCT by 5.1% and anchor-inclusive wall by 3.2%, but its 2.469 s initial plan/capture cost removes that single-use gain. Train7 candidate JCT is 2.858 times native, with 1.902 times as many output tokens. Its 2.389 s setup is additional. Input preparation and model initialization are separately preserved in JSON, not hidden in these task-latency columns. The same engine is shared across arms, so independently launching a candidate-only production process was not measured.

These are valid descriptions of completed task outcomes despite different trajectories. They do not isolate implementation speed. Conversely, per-token normalization would erase the additional tokens that the user actually waits for. The fixed128 ABBA measurements are the separate controlled-work performance evidence; natural multi-seed distributions remain required.

## Per-question interpretation

Indices below are zero-based. These annotations describe model outputs relative to frozen labels, not a new scoring metric or a complete manual validation of the dataset.

| Document/question | Native final → cascade final | Interpretation |
|---|---|---|
| train1 q0 | 63.07% → 63.1%, gold64.85% | Both use total Commercial membership rather than the question's Other Commercial subgroup; numeric/selection mismatch, not missing unit or unfinished reasoning. |
| train1 q1 | 49.58% → 49.58%, gold49.6% | Both correct only under the frozen rounding sensitivity. |
| train1 q2 | 1.00 → 1.0, gold1.73 | Both infer cash equals net tangible assets from “primarily consisted”; substantive value mismatch. |
| train1 q3 | 11.07% → 11.1%, gold11.1% | Cascade gains exact match by reporting reference precision; both already pass rounded match. |
| train1 q4 | 4395509 → 4395509, gold66649 | Both select consolidated total liabilities; same large value mismatch. No evidence of candidate-specific quality loss. |
| train1 q5 | 3170800 → 3171000, gold3171794.87 | Both differ from label. Native reads the displayed total; cascade rounds approximately. This is not caused by missing units. |
| train1 q6 | 14.26 billion → 14.26 billion, gold12.525 | Same selected year/value mismatch. Gold is bare, so unit verification is unavailable; the frozen metric compares written coefficients. |
| train1 q7 | 6869614 → 6869614, gold119941 | Both select consolidated assets; same large value mismatch. |
| train7 q0 | 17 → 17, gold-17 | Same sign convention mismatch: both describe decrease magnitude rather than signed change. |
| train7 q1 | no → 8.00%, goldno | Native correct; candidate prose explicitly says rate was not higher, but FINAL gives a rate instead of yes/no. Answer-type/format failure, not evidence candidate reversed the comparison. |
| train7 q2 | 53,50,51 → 53,50,51, gold27 | Both return a list instead of one quantity and select total stock compensation rather than performance-unit award expense. Candidate takes5643 vs1854 tokens for the same wrong final. |
| train7 q3 | 4.70% → 4.70%, goldyes | Both prose correctly says greater, but FINAL gives the rate instead of yes/no. |
| train7 q4 | 11.53% → 11.53%, gold11.5% | Both pass rounded sensitivity only. |
| train7 q5 | 1.83 → 1.83, gold1.4 | Both average the displayed three yearly yields; mismatch to frozen label, not a candidate-specific change. |
| train7 q6 | 3.30 → 3.30%, gold3.30% | Candidate gains exact and rounded credit solely by including required percent unit. Native prose already describes the correct percentage difference. |
| train7 q7 | 83.6 → 68%, gold83.6% | Native has correct numeric coefficient but omits required unit. Candidate selects a different extensions/additions passage/year and produces wrong percentage. Both fail strict metric for different reasons. |

The original prompt in `scripts/capture_model.py:151` requests a numerical answer and unit, even for yes/no questions. This is a protocol limitation shared by both arms. The audit does not repair answers or change prompts after seeing results. A separately preregistered quality experiment should use answer-type-neutral instructions in both arms and retain these original failures.

## Why B8 operator gains do not determine task JCT

Train1 native has594 B1 replay steps, whereas cascade has2063. Train7 native has309 B2 and550 B1 steps, whereas cascade has2803 B2 and1840 B1 steps. Request exit and longer tails therefore change the amount of time spent where sharing is least effective. B1 uses native fallback.

On rank0 the observed same-batch step-interval median at B8 is14.86→12.40ms (train1) and15.62→12.52ms (train7); at B2 it is8.32→10.55ms and8.61→11.80ms respectively. These are descriptive host preparation-start intervals, not controlled matched-state kernel timings. Reported cascade preparation intervals around8–10ms include synchronization with in-flight GPU work; do not label their entire duration independent CPU overhead or sum them on top of JCT.

The expanded fixed-snapshot operator audit (`../dense_shared_capture_bench_audit_v2/audit.json`) contains B1/B2/B3 on train1 layer15: hot native/cascade ratios0.872/1.200/0.879, cold0.968/0.938/1.000. B4 on train7 is1.184 hot and1.211 cold; B8 two documents/two selected layers is2.20–2.36 hot and1.91–2.05 cold. This is not a monotonic cross-document scaling law: B4 uses a different prefix and kernel configurations can change discretely. Cold event quantization produces some degenerate confidence intervals. Shared-prefix computation is a known mechanism, not the new contribution.
