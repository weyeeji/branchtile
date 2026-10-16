# Wave5 FullCol policy screen: independent CPU audit

2026-10-16. All eight jobs reached clean terminal exits. This audit reads frozen
results and capture files; it does not run attention, initialize CUDA, change
benchmark source, or generate new numerical/performance samples.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_overhead_wave5.py
```

[Machine-readable evidence](audit.json) retains every job, all 48 matched
split/N tuning pairs, and all 16 final hot/cold paired cells.

## Protocol and provenance checks

- Eight distinct GPUs, three development documents, B3/B4, step64, two KV heads.
  All wave4 jobs were terminal before wave5 started. No register-bank v4
  candidate appears in this policy screen.
- All **416 configurations** completed their unchanged FP64 O/LSE checks, with
  zero configuration failures: four B3 cells × 36 and four B4 cells × 68.
  Independent restricted-growth partitions and binary-bipartition enumeration
  recover the full expected grid. The original baseline 30/62 configurations
  remain present, with exactly six FullCol additions per cell.
- Frozen protocol/source hashes, per-candidate source hashes, and tuning JSONL
  versus final JSON match. All **52 relevant capture files** match previously
  frozen source or overlay SHA256 values; system prefix, full private suffix,
  rank, document identity and measured ready metadata remain present.
- Selected family/control winners were independently recovered as minima over
  all their eligible tuning configurations. Every final scalar median and
  reported pair's 10,000-draw block-bootstrap interval was recomputed from the
  raw measurements. Each cache cell has 20 random/reverse blocks, with reverse
  order verified and all strong-control families retained.
- All preparation medians and 1/16/64/256 reuse arithmetic reproduce. These
  remain hypothetical plan-amortization scenarios, separate from the GPU
  graph execution timings.
- Allocated GPU wall cost from **all eight terminal ledger entries** is
  **0.17233429570087336 P2 GPU-hours**, including initialization/JIT/planning.
  The independent audit itself consumes zero GPU-hours.

FP64 qualification here means the frozen benchmark recorded its numerical
assertions as successful; the independent CPU audit does not rerun FP64
attention. This is distinct from the earlier [FullCol compiled-artifact and
mutable-graph audit](../operator_fullcol_guard_gpu_audit_v1/README.md).

## Matched tuning and final paired measurements answer different questions

**Matched split/N tuning:** 48 parameter-matched FullCol/emptyguard pairs × two
cache regimes yield 96 descriptive ratios, ranging from **0.901 to 1.224** for
`emptyguard time / FullCol time`; 50 exceed 1. Each underlying tuning value is
the median of five readings, acquired during search. These are not the final
randomized paired experiment and do not support a paired confidence claim.
Some parameter choices can improve while the best attainable family result
does not improve.

**Final independently selected winners:** each family selects its own tuning
winner before the 20 random/reverse blocks. FullCol and emptyguard use identical
split/N in 13 of 16 cells; the remaining three are comparisons of different
selected parameters and cannot isolate just the warp policy. The complete
per-cell configurations are retained in JSON.

| Final paired speed ratio | Hot range | Cold range |
| --- | ---: | ---: |
| Strong control time / FullCol time | 0.333–0.814× | 0.435–0.874× |
| Emptyguard time / FullCol time | 0.959–0.993× | 0.929–1.000× |

A ratio above 1 would mean FullCol is faster. **FullCol does not beat the
strong control in any of 16 cache cells**, and every corresponding 95% interval
upper endpoint is below 1. Its independently selected winner also does not
beat emptyguard: no paired median exceeds 1, with several quantized cold
intervals touching or collapsing to 1. Hot A/A medians lie around
0.998986–1.000643; cold A/A medians are all 1, which does not eliminate visible
event quantization or establish arbitrary precision.

For example, long B4 L15 at 64K has hot control / emptyguard / FullCol medians
**46.57 / 73.72 / 76.26 µs**. At L31, 128K they are
**99.05 / 120.33 / 121.68 µs**. This screen supports retaining the original strong
baselines and treating the isolated FullCol warp-policy change as insufficient
for a new net acceleration result.

The measured `run()` includes producer/merge and required query packing/output
restoration. It excludes one-time CPU preparation and host synchronized
complete-call wall time; artificial cold flushing is outside the timing event.
Per-capture tuning is a hindsight oracle, and the split grid remains bounded.
Repeated layers/budgets/blocks do not add independent documents. No application
speed, answer quality, energy or actual hardware-memory-traffic claim follows.
