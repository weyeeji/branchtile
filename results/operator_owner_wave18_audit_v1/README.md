# Wave18 complete owner-task call: independent audit

All eight jobs ended cleanly. All **616/616 configurations** passed the frozen FP64 O/natural-LSE contract: four B3 cells each contain 61 configurations and four B4 cells each contain 93. The complete previous control sets (55 B3 / 87 B4) are retained exactly; six owner candidates add chunk256/512/1024 × N32/N64 at 128 threads. All candidate identities include both thread count and owner chunk size.

The audit independently re-enumerates partitions/trees/grids, verifies every candidate source SHA and per-job manifest/ledger, verifies eight wave17 prerequisite receipt hashes and 52 immutable capture-file checks, and reproduces all **448 original paired ratio medians and bootstrap intervals** from raw random/reverse timing readings. All 616 candidate statuses/errors and initialization costs are retained, alongside 48 owner metadata records and 208 planning receipts. No failure or timeout was omitted.

Timing calls the bound operator and includes both owner producer and merge, returning O and natural-log LSE. Frozen source line evidence is saved in `complete_call_source_proof`. Hot graphs contain 20 complete calls; cold graphs contain one. Steady-state graph latency excludes preparation and topology recapture. Full RegionPlan + Attention preparation and prepare-plus-one-call host wall are retained separately; hypothetical reuse amortization is checked arithmetically.

Owner-task improved on independently tuned FullCol in all 16 hot/cold cells (paired speed ratios 1.087–2.485), but **did not beat the strongest retained control with a 95% interval entirely above 1 in any cell**. The 128K cold cell is approximately tied; its interval includes 1. The small-budget owner winner sits at the minimum tested chunk, and the largest-budget winner sits at the maximum tested chunk, so this six-point screen does not establish optimal granularity.

Each speed ratio below is `best_owner_control_time / owner_time`; above 1 favors owner. The intervals resample **20 timing blocks within one fixed captured input**. They are not confidence intervals across documents, queries, layers or GPUs. Family winners were selected on separate five-sample tuning readings from the same input; this is a hindsight tuning oracle, not a deployed selector.

| Capture / layer / budget | Cache | Owner chunk / N | Owner µs | Strong control µs | Speed ratio [95% block CI] |
|---|---|---:|---:|---:|---:|
| docfinqa256_l15_b4096 | hot | 256 / 64 | 12.194 | 8.367 | 0.6851 [0.6834, 0.6865] |
| docfinqa256_l15_b4096 | cold | 256 / 64 | 22.528 | 20.480 | 0.9091 [0.9091, 0.9091] |
| docfinqa256_l31_b32768 | hot | 256 / 64 | 22.398 | 20.171 | 0.9004 [0.8990, 0.9028] |
| docfinqa256_l31_b32768 | cold | 256 / 64 | 47.104 | 43.008 | 0.9130 [0.9130, 0.9130] |
| publicdev1_256_l15_b4096 | hot | 256 / 64 | 12.357 | 8.484 | 0.6862 [0.6854, 0.6883] |
| publicdev1_256_l15_b4096 | cold | 256 / 64 | 22.528 | 18.432 | 0.8263 [0.8182, 0.9091] |
| publicdev1_256_l31_b32768 | hot | 256 / 64 | 23.103 | 20.649 | 0.8939 [0.8929, 0.8954] |
| publicdev1_256_l31_b32768 | cold | 256 / 64 | 47.104 | 43.008 | 0.9130 [0.9130, 0.9348] |
| publicdev6_256_l15_b4096 | hot | 256 / 64 | 12.105 | 10.052 | 0.8306 [0.8291, 0.8310] |
| publicdev6_256_l15_b4096 | cold | 256 / 64 | 24.576 | 20.480 | 0.8333 [0.8333, 0.8333] |
| publicdev6_256_l31_b32768 | hot | 256 / 64 | 26.154 | 21.773 | 0.8328 [0.8303, 0.8361] |
| publicdev6_256_l31_b32768 | cold | 512 / 64 | 53.216 | 51.200 | 0.9615 [0.9604, 0.9621] |
| publicdev6_256_l15_b65536 | hot | 512 / 64 | 62.018 | 46.226 | 0.7449 [0.7435, 0.7468] |
| publicdev6_256_l15_b65536 | cold | 512 / 32 | 94.176 | 92.160 | 0.9786 [0.9783, 1.0000] |
| publicdev6_256_l31_b131072 | hot | 1024 / 64 | 110.137 | 99.536 | 0.9038 [0.9021, 0.9050] |
| publicdev6_256_l31_b131072 | cold | 1024 / 64 | 120.832 | 118.800 | 1.0000 [0.9831, 1.0000] |

A/A medians range from 0.99833 to 1.00365; three hot-cell intervals narrowly exclude 1. Cold readings have coarse repeated timing values, so some empirical intervals collapse. These features limit sub-percent interpretation and are preserved rather than suppressed. They do not support a positive owner-versus-strong-control conclusion.

The owner mechanism changes both task partitioning and partial storage. The uniform/chunk × rectangular/packed factorial is incomplete; this is not a pure packing or state-transport ablation. `splits=64` is a compatibility argument: actual segmentation uses `chunk_tokens`. Immutable topology is required for each timed graph. Changing offsets/counts/row_map/labels requires rebuilding and recapturing; direct graph replay bypasses the host guard.

All eight P2 jobs cost **0.0657304619 GPUh**. CPU audit cost: zero GPUh. Source/capture provenance and all prior wave snapshots remain separate. No application quality, natural-EOS latency, energy, or full-model acceleration claim is made.

Reproduce from the project virtual environment with no GPU visible (use a fresh copy/output location because the script preserves existing snapshots):

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_operator_owner_wave18.py
```
