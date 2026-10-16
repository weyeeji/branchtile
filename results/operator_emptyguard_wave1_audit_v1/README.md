# Independent wave1 empty-shard-guard audit

CPU-only independent audit of the seven completed wave1 baseline jobs on
publicdev6, separate from root's v4 attempts. No benchmark source or raw output
was changed. Audit command:

```bash
CUDA_VISIBLE_DEVICES='' .venv-latest/bin/python scripts/audit_emptyguard_wave1.py
```

All seven source jobs exited cleanly. All 62/62 configurations per job finished,
with zero configuration failures across all families; all rows remain in the
raw outputs and audit. The 42 matching split/block configurations for v2 versus
emptyguard produce 84 successful FP64 qualifications under the original output
atol .006 / rtol .015 and natural-LSE atol .0003 / rtol .0001. Both families'
maximum observed absolute output error is .0009625891 and LSE error .0000102676.
Matching scalar maxima are not a tensor-level bitwise-equivalence check.

## Geometry and the causal boundary

All seven measured states have all 15 owner labels present in each of two KV
heads. The shortest region is 803 tokens. For split32/64/128, reconstructed
per-state shard totals are respectively 960/1920/3840, and **zero shards are
empty**. Thus this wave exercises branch overhead, with no skipped gather or
scatter work. It cannot establish guard benefit in genuinely empty-shard states.
The separate all-empty/empty-head GPU fixtures have not been run by this audit.

## All paired final cells

Each family independently chose its best tuning configuration, but every hot
and cold selection happens to match: split128, block_n32. Final measurements
therefore compare identical shape parameters, with 20 whole random/reverse
blocks and 10000 paired block-bootstrap draws. Ratio is v2 time / guard time;
below one means the guard is slower.

| State | Hot v2 → guard, µs | Hot ratio | Cold v2 → guard, µs | Cold ratio |
|---|---:|---:|---:|---:|
| r0_l15_b65536 | 70.398 → 74.300 | 0.9477 | 100.352 → 104.448 | 0.9608 |
| r0_l15_b131072 | 121.964 → 125.173 | 0.9746 | 141.312 → 143.360 | 0.9857 |
| r0_l31_b65536 | 59.734 → 63.982 | 0.9333 | 92.160 → 94.208 | 0.9783 |
| r0_l31_b131072 | 117.622 → 120.106 | 0.9801 | 133.120 → 135.168 | 0.9848 |
| r1_l15_b65536 | 73.338 → 76.731 | 0.9547 | 104.448 → 108.544 | 0.9623 |
| r1_l15_b131072 | 121.164 → 123.812 | 0.9785 | 139.264 → 141.312 | 0.9853 |
| r1_l31_b131072 | 119.758 → 121.615 | 0.9850 | 135.168 → 137.216 | 0.9851 |

All 14 selected final cells have ratio below one, and all reported bootstrap
95% upper endpoints remain below one. Guard adds median paired 1.821–4.267 µs
in hot cells and 2.048–4.096 µs in cold cells. Many cold intervals are degenerate
because readings are quantized; this does not confer arbitrary precision.

The full 42 configuration pairs are retained in `audit.json`, including the
isolated descriptive tuning wins (2/42 hot and 1/42 cold) rather than selecting
only unfavorable or favorable rows. Those tuning samples were not randomized
final pairs, so they do not overturn the paired selected-cell result.

## Scope and costs

The timed graph executes the complete producer plus merge with prepared
metadata. It excludes one-time CPU planning/JIT and is not a host synchronized
complete-call latency. Artificial cold flushing does not reproduce model-layer
interleaving. This is one development document, with rank/layer/budget states,
not seven independent documents and not application JCT or quality evidence.

Source jobs used 0.168599562040 P2 GPUh including initialization,
tuning and failures (none). The independent audit used zero GPU hours and
asserted CUDA remained uninitialized. Input result hashes, terminal ledger rows,
mask/metadata/overlay receipt hashes, every FP64 configuration record, geometry,
and complete bootstrap cells are retained in `audit.json`.

Decision: no performance win for emptyguard in this measured large-budget
regime. Any later small-region test must predeclare states that actually have
empty shards and keep this zero-skipped-work result in the final record.
