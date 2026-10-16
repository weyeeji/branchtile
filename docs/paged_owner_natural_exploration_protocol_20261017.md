# Exploratory natural execution after a failed logit screen

The original cosine >= 0.999 screen remains failed. It is an internal numerical screen, not a conference requirement or a validated task-quality metric. Local attention FP64 qualification and all real-model layer-output shadows passed; this does not make final task quality automatic.

## Why the cosine screen and generated quality are different

Logit cosine changes under a common additive shift even though softmax is invariant to that shift. Near a tie, extremely high cosine can still switch the top prediction. Matching top-1 on a forced path does not prove matching stochastic samples or final answers.

The actual 16-token BF16-PV trace demonstrates the distinction. The lowest-cosine vector q1_s9 (0.93813) has a top-p distribution TV distance of zero under the configured temperature 0.6 / top-p 0.95, with top-1 probability essentially one. Conversely, q0_s14 passes cosine (0.999767) but has the largest top-p TV distance, 0.10906. Across 48 vectors, top-p TV has median zero and mean 0.00693. These CPU probability calculations are descriptive and do not replicate vLLM RNG or establish answer quality. Raw logits and source hashes are retained in the distribution audit.

## Frozen exploratory experiment

Use the independently named `scripts/explore_vllm_paged_owner_natural.py`. It loads and hashes the prior failed gate receipt, refuses qualification mode, and labels every summary `EXPLORATORY_ONLY_UNQUALIFIED_NATURAL_EXECUTION`, `performance_qualification_passed=false`, and `paper_speedup_claim_allowed=false`. Original kernels, planning, sparse supports and generation settings remain intact. No shadow/forcing/logit copies run in measured natural arms.

First run the existing B3 development document with E0 and owner, one seed, cap 8192 and two order-reversed repeats (E0→owner, owner→E0). This is a four-arm deployment smoke test, not four independent documents. A second already selected development document and independent seed can follow if no correctness/runtime issue occurs. Do not choose documents by a favorable result.

Mandatory outputs:

- Actual natural EOS/stop versus cap-hit and complete final text; cap-hits remain failures/incomplete outcomes in the denominator.
- The unchanged strict FINAL financial exact metric and its mandatory rounded sensitivity, including unit/parse failures. This is not official executable-program accuracy.
- Per-question actual output lengths, whole-group JCT, anchor-plus-JCT and descriptive output tokens/sec. Different natural lengths must remain visible; a shorter answer is not an isolated kernel-speedup claim.
- One-time selector/capture/model setup, per-step preparation and actual B3→B2→B1 survivor transitions. Larger/new supports or divergent physical sharing must fail rather than replay stale plans.
- Source hashes and the failed prior gate status in the report. Do not retrospectively relax that gate because a natural answer looks correct.

The existing `scripts/summarize_vllm_sparse_natural.py` can score the raw outputs using the frozen metric; its `status=completed` denotes data completeness only. The exploratory warning propagates through its limitations. Formal publication claims require separately frozen workloads, multiple independent documents/seeds, actual quality evaluation/noninferiority and replicated timing; this pilot cannot establish them.

## FP32-PV localization result

The independent IEEE FP32-PV branch passed 528 synthetic FP64 checks. Compared with BF16-PV, maximum local output error fell from 0.00406718 to 0.00388986 and mean per-check maximum error fell from 0.00168527 to 0.00152547. However, in the real 16-token model diagnostic, all 4,320 local output shadows passed while the final-logit screen still failed: 13/48 owner vectors were below 0.999, versus 10/48 for BF16-PV. Minimum cosine was 0.927485 and maximum logit difference was 4.21094. E0 AA again matched exactly and all owner top-1 predictions matched.

FP32-PV top-p TV had median zero, mean 0.009429 and maximum 0.094374, compared with BF16-PV mean 0.006933 and maximum 0.109056. Thus higher PV precision did not uniformly reduce differences from the E0 deployment path. The evidence does not isolate BF16 probability rounding as the cause. This numerical-precision branch is now closed; no performance conclusion is drawn from its instrumented runs.

The separate exploratory natural run scheduled by root uses the original frozen `benchmark_vllm_paged_owner_natural.py` plus `configs/paged_owner_natural_exploratory_v1_interpretation.json` to retain the failed gate labels. The newly added `explore_vllm_paged_owner_natural.py` was not used in that run.
