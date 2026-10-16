# Wave7 fixed-M32 persistent control: independent CPU audit

2026-10-16. All eight jobs are terminal with clean exits. This audit reads saved
results and frozen captures; no attention execution, GPU initialization,
compilation, benchmark modification, or new numerical/performance sample occurs.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_overhead_wave7.py
```

[Machine-readable evidence](audit.json) retains every job, all 48 split/N
matched tuning triples, and all 16 final paired hot/cold cells.

## Counts, provenance and accounting

- The seven-GPU main manifest and deferred GPU0 manifest together cover GPUs
  0–7 and the eight predeclared states from three development documents.
  GPU0 started after its extended v4 qualification ended normally. Each GPU
  state remains an independent rank0 capture replay, not a TP2 application run.
- **464/464 configurations** completed their unchanged numerical O/LSE checks,
  with zero configuration failures: four B3 states × 42 and four B4 states × 74.
  Independent partition/tree enumeration reproduces the full configuration
  set. The original baseline 30/62 and wave5 policy 36/68 sets remain present;
  persistent adds exactly six split/N configurations. There is no v4 candidate.
- Protocol, source and per-configuration source hashes match. Progress JSONL
  equals the final tuning list. All **52 relevant capture file SHA256 values**
  match earlier frozen source/overlay receipts; document identity, rank0,
  measured readiness, system prefix and complete private suffix metadata agree.
- Selected family/control winners reproduce as tuning minima over every
  eligible configuration. Each cache cell contains 20 random/reverse blocks;
  method retention and exact reversed order are verified. All scalar medians
  and **192 reported paired ratios/95% intervals** (16 cells × 12 comparisons)
  reproduce using 10,000 block-bootstrap draws. Preparation medians and
  amortization arithmetic also reproduce.
- All eight allocation ledger entries are included: **0.09406569849751475 P2
  GPU-hours**, including initialization/JIT/planning. Independent audit cost is
  zero GPU-hours. Numerical qualification is the recorded success of the
  benchmark's frozen FP64 assertions; this CPU audit does not recompute attention.

## Local improvement does not yet beat the strong baseline

A ratio above 1 indicates persistent is faster:

| Final paired ratio | Hot range | Cold range |
| --- | ---: | ---: |
| Strong control time / persistent time | 0.408–0.863× | 0.550–0.957× |
| FullCol time / persistent time | 1.042–1.225× | 1.000–1.101× |

Persistent improves the hot execution time relative to the compact FullCol
control in all eight measured states. It still **does not win against the
strong control in any of 16 hot/cold cells**; every corresponding 95% interval
upper endpoint remains below 1. Cold FullCol comparisons include ties and
quantized readings; per-cell intervals are retained rather than treating a
degenerate interval as arbitrary precision.

For example, B4 L15 at 4K has hot strong control / FullCol / persistent medians
**10.01 / 30.06 / 24.53 µs**. At B4 L31, 128K they are
**99.31 / 122.36 / 114.99 µs**. These establish a measurable improvement of one
control implementation while leaving the main acceleration objective unmet.

## The persistent/FullCol comparison is not a pure transport ablation

FullCol dynamically uses M16 or M32 according to the active Q-row count, and
gathers/scatters canonical shared U through RowMap. Persistent fixes every
region to M32, keeps output/softmax fragments across regions, stages Q once,
and applies runtime owner masks. For regions with one or two GQA8 requests,
FullCol uses M16 while persistent still executes M32: masked MMA and associated
softmax work differ. Consequently the measured change cannot be attributed
solely to eliminating canonical-state transport.

Of 16 final family-winner comparisons, 11 use the same split/N; the other five
also change selected launch parameters. **Even those 11 are not pure
state-transport comparisons**, because the M dimension and work above differ.
The JSON preserves this flag for every matched tuning pair and final cell.

At identical split/N, all 48 FullCol/persistent tuning pairs are retained, along
with the previous emptyguard/FullCol comparison. Their five-reading tuning
medians produce descriptive FullCol/persistent ratios from **0.875 to 1.637**.
These occur during search, are not the final randomized paired measurements,
and cannot substitute for the final independently selected winner statistics.

The grid remains bounded and its per-capture winner is a hindsight oracle.
GPU `run()` includes producer/merge and required packing/restoration, excluding
one-time CPU preparation or host synchronized complete-call wall time. Plan
reuse scenarios are hypothetical; artificial hot/cold states do not reproduce
whole-model interference. Repeated layers/budgets/blocks are not independent
documents. No application, answer-quality, energy or hardware-traffic conclusion
is established by this screen.
