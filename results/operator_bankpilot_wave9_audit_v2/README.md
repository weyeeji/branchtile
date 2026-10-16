# Wave9 register-bank pilot: complete independent audit v2

2026-10-16. **All eight original wave9 states are complete**: four B3 states
and four B4 states. This new v2 supplements the historical
[B3-only partial v1](../operator_bankpilot_wave9_audit_v1/README.md); v1 files,
cutoff and completed-B3 input hashes are preserved and checked. It does not
replace that snapshot or merge later expanded-grid results into wave9.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_bank_snapshots_v2.py --wave 9
```

The [JSON evidence](audit.json) records all eight clean ledger terminals and
their unchanged source/capture identities. Independent checks passed:

- **480/480 configurations**, zero numerical/configuration failures: four B3 ×
  44 and four B4 × 76. Each config retained the original frozen FP64 O/LSE
  assertions; this CPU auditor verifies their recorded success, not a fresh
  attention computation.
- The complete earlier baseline/policy/persistent grids remain present. V4 is
  restricted to **S64 × N32/N64**. The other dynamic controls retain S32/S64/S128
  × N32/N64, with v3 retaining its original three N64 configurations. B3/B4
  fixed partitions and binary trees are independently enumerated and matched.
- Protocol and every per-candidate source hash match; progress JSONL equals
  final tuning JSON. **52 capture-file hashes** match prior frozen source or
  overlay receipts. Rank0, document, measured readiness, prefix and full
  private-suffix provenance agree.
- Every selected winner is independently recovered as its eligible tuning
  minimum; `best_bank_control` includes all non-v4 families. All 16 hot/cold
  cells retain 20 random/reverse blocks and every measured method. Scalar
  medians and **240 paired ratios/95% intervals** reproduce with 10,000 block
  bootstrap draws; preparation/amortization arithmetic also reproduces.
- All eight completed allocation costs are included: **0.0456135594076477 P2
  GPU-hours**, including initialization/JIT/planning. Audit cost is zero GPU-hours.

No v4 winner beats the best non-v4 control in any of the 16 cache cells: the
paired speed ratio `control time / v4 time` ranges **0.383–0.703× hot** and
**0.476–0.833× cold**, with every interval upper endpoint below 1.

The newly completed B4 portion shows the remaining boundaries explicitly:

| B4 document budget | FullCol/v4 hot | Cold | Best non-v4/v4 hot | Cold |
| --- | ---: | ---: | ---: | ---: |
| 4K | 1.131× | 1.023× | 0.383× | 0.476× |
| 32K | 0.984× | 0.872× | 0.479× | 0.632× |
| 64K | 0.846× | 0.823× | 0.509× | 0.738× |
| 128K | 0.858× | 0.798× | 0.703× | 0.702× |

Above 1 favors v4. Small-budget local gains relative to FullCol do not establish
a gain relative to the strong baseline. The original two-point pilot also
cannot establish an optimally tuned v4 result: other families can select
different split/N. Equal-S64/N search readings are retained separately and
remain descriptive five-reading tuning results, not final paired inference.
Persistent changes fixed-M32 masked MMA/softmax work, so comparisons with it
are not pure state-transport ablations.

This audit is CPU-only, initializes no CUDA, modifies no original experiment
file, and preserves the prior v1 hashes. GPU `run()` timings exclude one-time
CPU preparation and host synchronized wall; reuse scenarios are hypothetical.
Single-rank fixed-state results, artificial hot/cold conditions and repeated
layers/blocks do not establish application, quality, energy, traffic or TP2
critical-path gains. The ongoing lane-select and wide-CTA qualifications supply
no performance result to this report.
