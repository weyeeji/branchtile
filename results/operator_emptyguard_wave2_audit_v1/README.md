# Independent wave2 empty-shard-guard audit

CPU-only audit of six small-budget states from three development documents and
one full-document state. Every source job is terminal and exited cleanly.
B3 states have 30 planned configurations and B4 states have 62: all 306/306
finished, with zero configuration errors across all methods. All 42 matching
v2/guard parameter pairs passed original FP64 qualification (84 records).
Maximum absolute output error is .0009617090; natural-LSE error .00000955054,
under the unchanged output atol .006 / rtol .015 and LSE atol .0003 / rtol .0001.
Matching scalar error maxima are not a bitwise tensor equivalence check.

Reproduction:

```bash
CUDA_VISIBLE_DEVICES='' .venv-latest/bin/python scripts/audit_emptyguard_wave2.py
```

## Actual empty shards

Unlike wave1, this wave contains genuinely empty shards. Split32 has zero empty
shards in every state. The reconstructed larger splits are:

| State | Smallest region, tokens | Empty/total split64 shards | Empty/total split128 shards |
|---|---:|---:|---:|
| docfinqa256_r0_l15_b4096 | 175 | 0/896 | 0/1792 |
| docfinqa256_r0_l31_b4096 | 58 | 6/896 | 70/1792 |
| publicdev1_256_r0_l15_b4096 | 137 | 0/896 | 0/1792 |
| publicdev1_256_r0_l31_b4096 | 150 | 0/896 | 0/1792 |
| publicdev6_256_r0_l15_b4096 | 94 | 0/1920 | 95/3840 |
| publicdev6_256_r0_l31_b4096 | 65 | 0/1920 | 214/3840 |
| publicdev6_256_r0_l15_b173232 | 87 | 0/640 | 220/1280 |

Counts include both KV heads and all owner regions. A region's split intervals
are counted directly from integer begin/end formulas and independently checked
against max(0, splits-region_length). The full-document state therefore has
220/1280 (17.1875%) empty split128 shards; this is not a hypothetical empty-input
fixture. Its two final selected implementations both use split128/N32.

## All final paired cells

Time is the producer+merge GPU graph call with metadata already prepared.
Ratios are median(v2/guard) over paired samples, not quotients of the displayed
separate latency medians. Each interval resamples 20 complete random/reverse
blocks 10000 times. Below one means guard is slower.

| State | Hot v2 → guard, µs | Hot paired ratio | Cold v2 → guard, µs | Cold paired ratio |
|---|---:|---:|---:|---:|
| docfinqa256_r0_l15_b4096 | 15.582 → 16.548 | 0.9415 | 26.592 → 26.592 | 1.0000 |
| docfinqa256_r0_l31_b4096 | 15.704 → 16.610 | 0.9463 | 24.576 → 26.624 | 0.9242 |
| publicdev1_256_r0_l15_b4096 | 16.170 → 17.134 | 0.9437 | 26.624 → 26.624 | 1.0000 |
| publicdev1_256_r0_l31_b4096 | 15.632 → 16.538 | 0.9459 | 26.624 → 26.592 | 1.0000 |
| publicdev6_256_r0_l15_b4096 | 27.146 → 28.892 | 0.9387 | 43.008 → 45.056 | 0.9545 * |
| publicdev6_256_r0_l31_b4096 | 27.930 → 29.376 | 0.9508 | 42.976 → 43.008 | 0.9542 |
| publicdev6_256_r0_l15_b173232 | 125.942 → 126.754 | 0.9938 | 137.232 → 139.264 | 0.9855 |

The publicdev6 L15 small-budget cold cell marked `*` chose split64 for v2 and
split128 for guard. It is a comparison of separately tuned family winners,
not a pure single-guard ablation. Every other final cell matches split/block
parameters. Small-budget hot winners use split64/N32; most cold winners use
split128/N32; the full-document cell uses split128/N32 in both conditions.

All seven hot paired ratios are below one with 95% upper endpoints below one.
Cold has three approximately tied cells and four paired median ratios below
one; some intervals include one. There is no selected-cell median ratio above
one. The full-document case with 17.19% empty shards still adds median paired
.782 µs hot (.993832 ratio, 95% .992467–.995705) and 2.032 µs cold (.985514,
95% .985294–.999772). Its skipped gather/scatter work did not offset the branch
and generated-code costs in this measured condition.

All 42 matching tuning pairs remain in `audit.json`, including the one hot and
one cold descriptive tuning win. These are not randomized final paired cells.
They are retained rather than being selected into a superiority claim. Cold
readings are visibly quantized and some bootstrap intervals degenerate; this
does not imply arbitrarily precise timing or population generality.

## Scope and cost

No raw benchmark results or running source were modified. The audit records
all terminal ledger rows, every configuration failure (none), all tuning pairs,
selected configs, actual split geometry, input hashes and per-cell intervals.
Original captures without an overlay receipt retain mask/metadata hashes and
an explicit null overlay-receipt field; no receipt is fabricated.

Source jobs consumed 0.149169431725 P2 GPUh, including setup, compilation,
tuning and final measurements. The audit consumed zero GPU hours and asserted
CUDA remained uninitialized. GPU graph calls exclude one-time CPU planning/JIT;
artificial flushing does not reproduce complete-model cache interleaving.
There is no complete-model JCT, task quality, energy or formal-data conclusion.
Multiple layers from one document are not independent documents.

Decision: this guard alone has no positive final paired signal in the tested
small-budget or full-document states, including states where it actually skips
empty shards. Keep it as an explicit ablation/control and compare the independent
FullCol control before attributing any v4 effect to reduced state movement.
