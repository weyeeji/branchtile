# Native-equivalent RawPair: B8 full-model end-to-end audit

The two training documents passed strict same-state qualification and same-session fixed-length and natural-EOS ABBA comparisons. Independent audits verified runtime/build source hashes, clean ledger exits, all 48 attention-layer Q/O bits, hidden/logits/current KV bits, native AA, unchanged inputs, 128 actual native states on both ranks, and all compared output tokens/APC hits/finish reasons/graph trajectories. Natural outputs reached EOS without caps or preemption. This is one seed and two repeats per arm, not a population confidence interval or a heldout result.

The candidate currently dispatches only at actual B8 with matching graph padding, using native fallback as requests finish. Ratios below are mean native wall divided by mean candidate wall; values above one favor the candidate.

| Document | Fixed128 generation | Fixed128 anchor + generation | Natural generation | Natural anchor + generation | Natural complete phase control | Candidate natural graph steps |
|---|---:|---:|---:|---:|---:|---:|
| train1 | 1.09516 | 1.00995 | 1.01535 | 1.01003 | 1.00982 | 291/2612 (11.14%) |
| train7 | 1.11242 | 1.00985 | 1.02857 | 1.01704 | 1.01677 | 506/3726 (13.58%) |

Complete phase control includes control RPCs and receipt export. Generation includes synchronized physical-prefix checks and metadata copies. Their measured host intervals can wait for previous GPU work and must not be treated as an independent additive CPU cost. Graph step fractions are counts, not GPU-time fractions; non-full-graph prefill is not counted. Every phase begins with a cold prefix-cache anchor. Local post-phase quality scoring and report hashing are experimental analysis outside the phase-control wall.

Native AA generation max/min ratios are 1.00159/1.00165 for fixed128 and 1.00086/1.00081 for natural outputs. Strict exact accuracy is 12.5%/25%, rounded accuracy 25%/37.5%; both methods produce identical answers and scores. The low absolute task quality is retained, not hidden by excluding failed answers.

This prototype requires session qualification before enabling the candidate and captures witness, checked and production graphs. Including all candidate setup and mandatory qualification, conservative first-activation natural cold ratios are 0.60666/0.62566, meaning the first task is slower. Even the counterfactual estimate that assumes qualification already happened offline, while retaining all graph setup, gives only 0.95053/0.97242. Do not advertise the steady-session 1–1.7% cold improvement as first-use end-to-end improvement.

The current B8-only policy establishes exact full-model numerical compatibility and a small positive steady-session signal. It does not establish an ISCAS-level benefit. The next training experiment should qualify a wider set of naturally shrinking batch sizes and remove synchronized per-step metadata validation without weakening identity guards. All new setup, qualification, fallback and natural-output costs must remain visible. No additional B8-only seeds are needed before that engineering change.

Sources are frozen driver `38fb4b0` and adapter `cb57a72`. Independent auditor: `scripts/audit_native_raw_pair_e2e.py`. Results: `results/native_raw_pair_e2e_v1/train1_bf16/analysis.json` and `train7_bf16/analysis.json`. The outcome-blind heldout selection and its two context failures are documented separately; no heldout model inference has run.
