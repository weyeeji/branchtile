# Wave12 B3 four-point bank grid: separate independent audit

2026-10-16. This is a **new experiment on four B3 states**, distinct from the
[original wave9 two-point pilot](../operator_bankpilot_wave9_audit_v2/README.md).
V4 searches **S64/S128 × N32/N64 only**. S32 is absent, and no B4 expanded-grid
result is included. Neither this audit nor its measurements replace earlier
wave9 snapshots.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_bank_snapshots_v2.py --wave 12
```

[JSON evidence](audit.json) preserves all four clean terminal records, all
configuration/source hashes, matched-parameter tuning readings, selected
parameters and final paired statistics. Independent checks passed:

- **184/184 configurations**, zero numerical/configuration failures: four × 46.
  All 42 prior non-v4 configurations per state remain present; four v4 points
  are added. The original numerical tolerances are unchanged. This CPU audit
  verifies recorded FP64 assertions and does not rerun attention.
- Frozen protocol and every candidate source hash match. Progress JSONL equals
  final tuning JSON. **24 capture files** match prior frozen hashes; document,
  rank0, same-event readiness, system prefix and private suffix provenance agree.
- Winner minima reproduce over their complete eligible grids. All eight cache
  cells retain 20 random/reverse blocks with verified reversal and all methods.
  Every median and **120 paired ratio/95% interval results** reproduce using
  10,000 block-bootstrap draws. Preparation/amortization arithmetic reproduces.
- All four allocation ledger costs are included: **0.02089508956168882 P2
  GPU-hours**, including initialization/JIT/planning. Audit cost is zero GPU-hours.

All four states select **S128/N32** for v4 in both cache regimes. Compared
within this new experiment, paired ratios are:

| Comparison time / v4 time | Hot range | Cold range |
| --- | ---: | ---: |
| FullCol | 1.142–1.211× | 1.083–1.087× |
| Fixed-M32 persistent | 1.058–1.095× | 1.000–1.001× |
| Best non-v4 control | 0.598–0.806× | 0.750–0.914× |

Above 1 favors v4. The hot local improvement relative to FullCol and persistent
is measurable, but **none of eight cache cells beats the best non-v4 control**;
all corresponding 95% upper endpoints are below 1. Cold persistent comparisons
include ties and visibly quantized readings; the raw intervals remain in JSON.

For the original document at 32K, hot strong control / FullCol / persistent /
v4 medians are approximately **19.92 / 28.57 / 27.06 / 24.72 µs**. This is a
within-wave comparison. A direct division of wave9 and wave12 times would mix
different runs/devices and cannot isolate the causal effect of adding S128.
The wave9 two-point results remain reported separately.

The four-point v4 grid is still bounded and narrower than controls that include
S32. Independently selected winners can also differ in split/N. Parameter-matched
five-reading search summaries are descriptive, not the randomized paired final
experiment. Fixed-M32 persistent changes masked MMA and softmax work, so its
comparison is not a pure state-transport ablation.

No GPU work, compilation or original-result modification occurs in this audit.
GPU `run()` excludes one-time CPU preparation and host synchronized wall. Reuse
scenarios, repeated layers and artificial cache conditions do not establish
application quality/speed, energy, traffic or TP2 critical-path gains. No
lane-select or wide-CTA performance conclusion is inferred from concurrent
qualification jobs.
