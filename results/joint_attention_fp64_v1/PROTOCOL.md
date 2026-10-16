# Frozen held-out FP64 attention screen

This protocol is committed before numerical evaluation. It extends the locked
CPU support-exchange probe without changing its masks, epsilon grid, request
orders, or two-round update rule. It is a descriptive attention proxy, not a
new selector claim or an end-task quality test.

- Sources: the three corrected `suffix_v2` official 30B FP8 TP2 captures,
  rank 0, layer 15, both local KV heads and all eight query heads per KV head.
- Budgets: 4,096 and 16,384 document positions per request.
- Variants: independent support, then epsilon 0, 0.001, 0.005, 0.01 in both
  fixed ascending and descending request orders. No order or epsilon is chosen
  using held-out queries. Every reconstructed selected-index hash must match
  the prior CPU result before held-out queries are opened for that document.
- Queries: actual co-ready decode groups anchored at steps 16, 64, and 256.
  The frozen JSON retains each request's actual prediction step and position;
  a group anchor does not mean all requests have the same prediction step.
- Context: original system prefix plus common document KV plus the complete
  request-private suffix, including generated tokens and the two protected
  document-tail tokens. The latter are included exactly once. Total context
  length must equal the actual query position plus one. Captured cache-match
  and physical-page-match flags must hold.
- Arithmetic: original captured BF16 Q/K/V converted to FP64, scaled dot
  product by `1/sqrt(128)`, FP64 softmax, output reduction and natural-log LSE.
  Full-context logits may be shared between support variants because they
  depend only on the fixed Q/K, not on support membership. All V are original.
- Reporting: output relative L2, maximum absolute error and cosine against
  dense and independent attention; changes in dense-reference errors; LSE
  differences; actual retained dense softmax mass; document-conditional
  retained mass; and the mass/LSE identity residual. Metrics are reported for
  each request/KV-head group and its individual query heads. A relative-L2
  denominator floor of `1e-12` only defines reporting near zero.
- Validation: an analytic uniform-logit scattered-subset check runs first.
  Epsilon 0 must reproduce independent output and LSE bit for bit in every
  natural cell. Input SHA256, schema, actual step metadata, support hashes,
  environment, CPU/GPU memory, and timing are recorded. Expected coverage is
  1,080 rows, each with eight individual query-head records.

Proposal section 5.7 concerns same-mask BF16 kernel correctness against an
FP64 oracle and calibrated native tolerances. It supplies no threshold for
different-support attention quality. Therefore this screen has **no success
threshold for joint-mask quality**, and it does not reuse a kernel tolerance
as a task-quality criterion. Increases and decreases in errors are both kept.
No natural generation, production change, mask selection using future Q,
physical capacity release, or latency/speedup claim is part of this stage.
The queries and private generated histories come from the captured dense
trajectory; they do not model an on-policy sparsified generation trajectory.

The first GPU allocation is two P2 jobs on GPUs 2 and 3 with 330 seconds per
job and at most 15 seconds of termination grace each. The total worst-case
allocation including grace is 0.192 GPUh, below the approved 0.2 GPUh. The
shared `run_jobs.py` ledger supplies authoritative allocated GPU time. Timing
inside this script describes resource use, not method performance.

Reproduction from the repository root, using the existing virtual environment:

```bash
.venv/bin/python scripts/run_jobs.py results/joint_attention_fp64_v1/jobs.json
```

Output files refuse overwrite. Re-runs require a separately named job/output
and additional budget authorization. The CPU constructor is imported without
modification; its import clears CUDA visibility, which this wrapper restores
to the single GPU assigned by `run_jobs.py` before CUDA initialization.
