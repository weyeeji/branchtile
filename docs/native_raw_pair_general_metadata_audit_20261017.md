# Independent whole-model metadata audit

The first three completed general-policy qualification jobs passed the independent
metadata/state audit: train1 (B8), train26 (B21), and train33 (B19), each on both
TP ranks. There are 486 checked native states across the six receipts, with all
48 attention Q/O comparisons, hidden/logits/current-KV checks, and GPU length
alias checks true. These are bounded empirical checks, not exhaustive proof.

Per rank, train1 checked 24 states covering B6/7/8; train26 checked 125 covering
all B6..21; train33 checked 94 covering all B6..19. Each naturally occurring B
has at most eight checks, and fewer where that batch existed only briefly. The
qualification retains native sampled outputs and does not constitute a
candidate end-to-end timing experiment.

Every receipt has exactly one full physical prefix binding. Subsequent checks
use native GPU sequence-length aliases and CPU request/slot/overwrite-epoch
guards, with no host length transfer. No metadata alias or cohort/epoch rejection
occurred. The only fallback reasons were the expected small/non-pure batch gate
and completed per-batch qualification quota.

The audit independently recomputed S from actual B, SM count, and full native
table width. All admitted steps use that original S, including odd batches.
All batch sizes and all48 layers retain one 64 MiB workspace and one raw-state
arena. Recorded source hashes match the frozen current files, and every
captured witness/checked/production batch visited all48 layers.

Subsequent host prepare timestamps have medians around 31–35 microseconds in
these qualification sessions. Qualification synchronizes to compare outputs;
these figures **must not** be interpreted as production latency or added to GPU
time. The code and binding records establish removal of the prior per-token
physical-prefix host readback. Actual speedup awaits unchanged-source production
ABBA runs and exact natural-output comparisons.

Artifacts: `results/native_raw_pair_general_metadata_audit_v1.json`, regenerated
by `scripts/audit_raw_pair_model_metadata.py` with the three completed summaries
under `results/native_raw_pair_general_policy_qual_v1/`.
