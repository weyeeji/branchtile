# Wave20 expanded owner grid: independent complete-call audit

All eight jobs ended cleanly, and all **680/680 configurations** passed FP64 O/natural-LSE qualification. Each B3 input retains the exact 55 previous controls plus 14 owner configurations (69 total); B4 retains 87 plus 14 (101 total). Owner candidates cover chunk64/128/256/512/1024/2048/4096 × N32/N64 at 128 threads. The six original owner configurations and all prior non-owner controls remain present.

The CPU auditor independently re-enumerates the exact partitions, trees and parameter grids; verifies frozen source/capture/protocol/qualification hashes and eight complete ledger pairs; checks all configuration identities and numerical statuses; and reproduces **all 448 paired medians and 95% block-bootstrap intervals** from the raw twenty-block random/reverse measurements. It retains all 680 candidate status/error/JIT receipts, 112 owner metadata records and 208 planning receipts. There are zero candidate failures and zero timed-out jobs in this wave.

The timed graph calls include both owner producer and merge, returning O and natural-log LSE. Hot graphs contain 20 complete calls and cold graphs one. Complete RegionPlan/operator preparation is recorded separately. Changing owner topology requires rebuilding and recapturing and is outside these steady-state timings; direct graph replay bypasses the host guard.

**One of eight hot cells improves over its strongest retained control:** publicdev6, L15, 4K, chunk128/N64 yields paired speed ratio **1.04795268 [1.04575334, 1.05147596]**. The other seven hot cells are slower; no cold cell has a 95% interval wholly above 1. Across sixteen cache/input cells the audit classifies one improvement, eleven slower comparisons and four intervals including 1. This is a local same-input, search-selected result requiring independent fixed-parameter confirmation.

Each speed ratio below is `best_owner_control_time / owner_time`; above 1 favors owner. CI resampling uses **timing blocks within the same fixed capture**, not documents. The eight states repeat wave18 inputs and are not new documents. The intervals are not adjusted for candidate search or multiple comparisons across cells.

| Capture / layer / budget | Cache | Owner chunk / N | Owner µs | Strong control µs | Speed ratio [95% block CI] |
|---|---|---:|---:|---:|---:|
| docfinqa256_l15_b4096 | hot | 128 / 64 | 9.454 | 8.329 | 0.8811 [0.8804, 0.8819] |
| docfinqa256_l15_b4096 | cold | 128 / 64 | 20.448 | 20.480 | 1.0008 [1.0000, 1.1092] |
| docfinqa256_l31_b32768 | hot | 256 / 64 | 22.410 | 20.003 | 0.8940 [0.8924, 0.8950] |
| docfinqa256_l31_b32768 | cold | 256 / 64 | 47.104 | 43.008 | 0.9130 [0.9130, 0.9130] |
| publicdev1_256_l15_b4096 | hot | 128 / 64 | 9.588 | 8.468 | 0.8839 [0.8825, 0.8847] |
| publicdev1_256_l15_b4096 | cold | 128 / 64 | 20.480 | 18.432 | 0.9015 [0.9000, 1.0000] |
| publicdev1_256_l31_b32768 | hot | 256 / 64 | 23.172 | 20.703 | 0.8929 [0.8911, 0.8940] |
| publicdev1_256_l31_b32768 | cold | 256 / 64 | 47.104 | 43.008 | 0.9130 [0.9130, 0.9130] |
| publicdev6_256_l15_b4096 | hot | 128 / 64 | 9.602 | 10.044 | 1.0480 [1.0458, 1.0515] |
| publicdev6_256_l15_b4096 | cold | 64 / 64 | 22.528 | 20.480 | 0.9091 [0.9091, 0.9091] |
| publicdev6_256_l31_b32768 | hot | 256 / 64 | 25.964 | 21.538 | 0.8299 [0.8268, 0.8307] |
| publicdev6_256_l31_b32768 | cold | 512 / 64 | 53.216 | 51.168 | 0.9615 [0.9242, 0.9618] |
| publicdev6_256_l15_b65536 | hot | 512 / 64 | 62.015 | 46.122 | 0.7439 [0.7408, 0.7455] |
| publicdev6_256_l15_b65536 | cold | 512 / 64 | 92.160 | 92.160 | 1.0000 [1.0000, 1.0222] |
| publicdev6_256_l31_b131072 | hot | 1024 / 64 | 110.642 | 99.251 | 0.8970 [0.8956, 0.8987] |
| publicdev6_256_l31_b131072 | cold | 1024 / 64 | 120.832 | 120.832 | 1.0000 [1.0000, 1.0086] |

## Preparation cost and conditional break-even

For the single favorable hot cell, median complete owner preparation is **13,550.884090 µs**, versus **1,055.395231 µs** for the selected native strong control. The extra preparation is **12,495.488860 µs**; subtracting steady-state medians gives **0.44160001 µs** saved per call.

Under the hypothetical model `P_owner + R*t_owner <= P_control + R*t_control`, the same immutable plan must survive about **28,295.94 calls** (at least **28,296 integer reuses**) to cover the extra preparation. This is a point estimate using three-sample preparation medians and timing medians. It is not a measured lifetime or a break-even confidence interval. Search, new JIT, topology changes and graph recapture would need additional accounting. Hypothetical 1/16/64/256 reuse arithmetic is independently checked; none establishes actual reuse duration.

All sixteen preparation comparisons and point estimates, including cases with no positive steady-state saving, are preserved in `preparation_break_even`. No cold point estimate alone is presented as a statistically supported improvement.

A/A paired intervals and all raw cache readings are retained. Three A/A intervals narrowly exclude 1, and coarse repeated cold readings can collapse empirical intervals; these measurements do not support sub-percent or cross-document certainty. The owner mechanism changes chunk placement and partial packing together; the uniform/chunk × rectangular/packed factorial remains incomplete.

All eight P2 jobs cost **0.0664536811 GPUh**. The CPU audit costs zero GPUh. Wave22 was separately stopped during new-shape compilation and is not included in these complete comparisons; its partial candidates and ledger charges must remain separate. All earlier snapshots remain unchanged.

No natural-EOS latency, application quality, full-model acceleration, energy or actual memory-counter claim is made. `splits=64` is a compatibility argument; `chunk_tokens` defines actual owner tasks.

Reproduce using the project virtual environment with no GPU visible and a fresh output location (the script preserves existing snapshots):

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_operator_owner_wave20.py
```
