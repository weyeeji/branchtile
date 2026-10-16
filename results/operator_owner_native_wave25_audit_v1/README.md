# Wave25 fixed owner versus complete native grid: independent audit

All eight jobs ended cleanly. All **384/384 configurations** passed the frozen FP64 O/natural-LSE contract and saved original-K/V source-alias check. Each input contains the exact **47 original native/E1-own/E4 controls plus one predeclared owner C128/N64, 128-thread configuration**. Independent enumeration confirms all 15 partitions for both E1 implementations, all 15 binary hierarchies, and both E0 SIMT/tensor-core choices.

The owner configuration is fixed across both caches and every input. Its frozen alias is an extra timed measurement, not another tuned candidate; its selected configuration is identical to the sole owner family configuration. Baseline selection remains a separately tuned, same-input hindsight oracle among all 47 controls. Old slow dynamic engineering families are explicitly excluded in this protocol; earlier full-grid results remain separate.

The CPU audit verifies frozen source commit `f9b4608`, manifest/protocol and all candidate source hashes, eight prerequisite qualification receipts, eight ledger start/end pairs, source-alias flags, task/workspace/preparation accounting and complete producer-plus-merge call-chain evidence. It recomputes **all 208 paired ratio medians/95% intervals** from **1,280 raw timing rows**: 40 random/reverse blocks, 80 rows per cache and 160 per input. All 160 selected-family preparation receipts and all 384 candidate status/error/initialization receipts are retained.

## What was confirmed

These are **eight correlated captured states from one development document**, covering TP ranks0/1 × layers0/15/31/47. Layers0/31 use natural anchor16; layers15/47 use anchor256. All four requests are naturally present in each ready group. Each request retains its own recorded prediction step and position; anchor256 does not imply every request has the identical generation count. This is not eight independent documents, a formal held-out evaluation, or application evidence.

The sole local improvement is **rank1, layer15, anchor256, hot**: paired speed ratio **1.0540000 [1.0529911, 1.0548425]** against the selected native strong control. All seven other hot cells are slower under the within-input intervals. No cold cell improves: two are slower and six have intervals including 1. Overall: one improvement, nine slower cells and six unresolved/tied intervals.

Ratio is `best_native_control_time / fixed_owner_time`; above 1 favors owner. These intervals resample forty random/reverse **timing blocks within the same input**, not documents or decoding lifetimes. They are not adjusted for sixteen-cell multiplicity.

| TP rank | Layer | Natural anchor | Cache | Fixed owner µs | Best native control µs | Speed ratio [95% block CI] |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 0 | 16 | hot | 9.6704 | 8.5224 | 0.8808067 [0.8795342, 0.8821293] |
| 0 | 0 | 16 | cold | 18.4320 | 18.4320 | 1.0000000 [1.0000000, 1.0000000] |
| 0 | 15 | 256 | hot | 10.1896 | 10.1864 | 0.9990601 [0.9976488, 0.9998431] |
| 0 | 15 | 256 | cold | 22.5280 | 20.4800 | 0.9104477 [0.9103840, 1.0000000] |
| 0 | 31 | 16 | hot | 9.7520 | 8.8752 | 0.9108792 [0.9094638, 0.9135843] |
| 0 | 31 | 16 | cold | 20.4800 | 20.4800 | 1.0000000 [1.0000000, 1.0000000] |
| 0 | 47 | 256 | hot | 9.9936 | 8.9192 | 0.8925858 [0.8901048, 0.8943447] |
| 0 | 47 | 256 | cold | 20.4800 | 20.4800 | 1.0000000 [1.0000000, 1.0000000] |
| 1 | 0 | 16 | hot | 9.7216 | 8.4944 | 0.8736407 [0.8727609, 0.8743387] |
| 1 | 0 | 16 | cold | 18.4320 | 18.4320 | 1.0000000 [1.0000000, 1.0000000] |
| 1 | 15 | 256 | hot | 9.5968 | 10.1088 | 1.0540000 [1.0529911, 1.0548425] |
| 1 | 15 | 256 | cold | 22.5280 | 22.5280 | 1.0000000 [1.0000000, 1.0000000] |
| 1 | 31 | 16 | hot | 10.2200 | 10.0960 | 0.9877946 [0.9842948, 0.9891791] |
| 1 | 31 | 16 | cold | 22.5280 | 20.4800 | 0.9090909 [0.9090909, 0.9090909] |
| 1 | 47 | 256 | hot | 9.7288 | 8.7824 | 0.9019991 [0.9009780, 0.9032046] |
| 1 | 47 | 256 | cold | 20.4800 | 18.4000 | 0.8984375 [0.8984375, 0.8992198] |

## Cost and limits

At the sole favorable point, median complete owner preparation is **13,584.387489 µs** and native control preparation is **1,033.844426 µs**. The extra **12,550.543062 µs** and steady-state saving **0.51200017 µs/call** imply about **24,512.77 identical-plan reuses** to break even in a median-based static model. This is not a measured lifetime or confidence interval and excludes new search/JIT/topology recapture.

The timed owner call includes both producer and merge, with BF16 O and natural-log FP32 LSE. Preparation and prepare-plus-one-call host wall are separately visible; hypothetical reuse1/16/64/256 arithmetic is verified. Workspace scopes differ across implementations and do not provide comparable peak memory or actual DRAM traffic. Task partitioning and partial packing change together; this is not a pure packing ablation.

Both full-union A/A and fixed-owner A/A are recomputed. One owner cold A/A interval is a coarse point above 1; every full-union A/A interval includes 1. Every A/A result remains in `audit.json`. Repeated cold timing values can collapse empirical intervals and do not establish exact equality or high precision.

The immutable capture audit checks 56 per-input file hashes covering 50 unique metadata/tensor files against prior evidence; the two tracked export summaries separately match frozen source commit `f9b4608`. All four ready-group request positions/actual prediction steps are preserved in each job provenance.

All eight P2 jobs cost **0.0883591768 GPUh**; the independent CPU audit costs zero GPUh. No original result was edited. The frozen protocol measurement prose accidentally says20 blocks; the manifest, runner default, result arguments and actual raw arrays all establish **40**. This narrative inconsistency is recorded without modifying the protocol.

No cross-document generalization, natural-EOS JCT, quality, energy or full-model acceleration claim follows.

Reproduce with normal uid, project virtual environment, no visible GPU and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_owner_native_wave25.py --output results/operator_owner_native_wave25_audit_recheck
```
