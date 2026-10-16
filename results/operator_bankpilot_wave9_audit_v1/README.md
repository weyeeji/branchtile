# Wave9 register-bank pilot: frozen B3-only partial audit

Snapshot cutoff: **2026-10-16 13:34:03.792837 UTC** (21:34:03 Beijing), the last
of four B3 job terminal records. Expected states: **8; audited complete B3: 4;
pending/unaudited B4: 4**. This is explicitly a partial report. Later B4 starts,
results and subsequent split128 experiments are excluded; they do not rewrite
this snapshot or supply any B4 performance conclusion here.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_bankpilot_wave9.py
```

[Machine-readable audit](audit.json) retains all expected jobs, the four clean
B3 terminals, matched-parameter search readings, every final paired ratio, and
source/capture hashes. The auditor is CPU-only, imports no Torch, and does not
execute attention, compile a kernel, initialize CUDA or modify source results.

## What passed independent checks

- Four B3 jobs completed **176/176 configurations** (44 each), with zero recorded
  numerical/configuration failures and unchanged FP64 O/LSE assertions. This
  CPU audit verifies those recorded successes; it does not recompute attention.
- Every prior baseline30, policy36 and persistent42 configuration is retained.
  V4 adds only **split64 × N32/N64**, two points. Emptyguard, FullCol, persistent
  and v2 retain split32/64/128 × N32/N64; v3 retains its original three N64
  configurations. All five fixed partitions and three B3 hierarchy trees remain.
- Protocol and complete source/per-candidate hashes match. Tuning progress JSONL
  equals each final JSON. **24 relevant capture file hashes** match prior frozen
  inputs; rank0, document identity, same-event readiness, system prefix and full
  private suffix provenance agree.
- Every family and alias winner is recovered as the minimum over its eligible
  tuning configurations. `best_bank_control` includes **all non-v4 families**,
  including persistent, FullCol, old dynamic versions and native/fixed/tree.
- All eight hot/cold cells have 20 random/reverse paired blocks, verified reverse
  order and every measured method. All scalar medians and **120 paired ratio /
  95% interval results** (8 cells × 15 comparisons) reproduce using 10,000 block
  bootstrap draws. Preparation medians and amortization arithmetic reproduce.
- All four completed B3 allocation costs are retained: **0.018850180151768856 P2
  GPU-hours**, including initialization/JIT/planning. The four pending B4 states
  contribute no performance sample or claimed zero allocation cost; their future
  costs belong to later ledger records. Independent audit cost is zero GPU-hours.

## B3 result: local 4K improvement, no strong-control win

Ratios are paired `comparison time / v4 time`, so above 1 favors v4:

| Document / budget | FullCol/v4 hot | FullCol/v4 cold | Best non-v4 control/v4 hot | Cold |
| --- | ---: | ---: | ---: | ---: |
| Original / 4K | 1.192× | 1.083× | 0.593× | 0.833× |
| New / 4K | 1.202× | 1.077× | 0.595× | 0.750× |
| Original / 32K | 0.953× | 0.926× | 0.670× | 0.778× |
| New / 32K | 0.963× | 0.926× | 0.669× | 0.778× |

The 4K hot improvements relative to FullCol are reproducible within this pilot;
their paired 95% lower endpoints are above 1. At 32K the v4 pilot is slower than
FullCol in both regimes. **V4 has no win against the best non-v4 control in any
of the eight audited cells**, with all corresponding interval upper endpoints
below 1. Cold readings remain quantized; an interval touching 1 or collapsing
to a point does not imply unlimited precision.

For context, original 4K hot control / FullCol / v4 medians are
**8.36 / 16.79 / 14.09 µs**; original 32K hot medians are
**20.30 / 28.87 / 30.31 µs**. This demonstrates a local implementation gain at
small budget while the target acceleration relative to strong controls remains
unmet.

## Pilot limitations and attribution

V4 is restricted to split64. The other families search a broader grid, and their
final selected split/N often differs: for both 32K states v4 selects N64/split64
while FullCol selects N32/split128. Those are valid pilot winner comparisons,
but **they do not establish the best possible v4 performance or isolate a pure
register-state effect**. Equal split64/N32 and split64/N64 search comparisons
are retained separately, using five tuning readings per cache; these are
descriptive and do not carry the final paired confidence claim.

The fixed-M32 persistent control also changes masked MMA/softmax work relative
to compact FullCol/v4, so its comparisons are not pure transport ablations.
Actual register/no-shared-U evidence belongs to the independent generated-code
audit, not inference from a speed ratio. Subsequent split128 qualification or
timing is a new frozen experiment and cannot be substituted into these results.

These are same-state single-rank attention replays. GPU `run()` includes required
producer/merge and packing/restoration, excluding one-time CPU preparation and
host synchronized complete-call wall time. Plan reuse is hypothetical. Two
documents, repeated layers and blocks do not establish task-level quality,
application speed, energy, hardware traffic or actual TP2 critical-path gains.
