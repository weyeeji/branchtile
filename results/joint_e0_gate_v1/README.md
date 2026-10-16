# Bounded same-native E0 mask mechanism screen

This screen finds a **conditional cold-cache signal**, not a general production speedup. All six document/budget cells pass each mask's own FP64 qualification using the same FlashInfer 0.7.2 FA2 tensor-core/default-split schedule. Under production request-major/head-minor (BH) batching, exchange's O-only complete-call hot ratios range from **0.9981 to 1.0150**; the corresponding cold ratios range from **1.0000 to 1.1538**. All three 16K cold cells have within-run paired-bootstrap intervals above one. Thus it would be inaccurate to conclude that there is no same-kernel signal at all. The evidence is narrow: small mixed hot effects, a forced-cold signal, and no full-engine or natural-quality validation.

The primary arm reproduces the production native batch order **B,H,G**, rather than only permuting the output of the old H,B,G order. Every method aliases identical original K/V payload. Actual per-request support lengths and native `plan_info` are identical across masks within each batching order. The original H,B,G arm remains a separately reported sensitivity control. The replay source is still a contiguous single-layer export, unlike the 48-layer noncontiguous engine cache.

The benchmark label `complete_layout` refers only to the reported output shape. **BH returns a view and does not include the engine's destination `output.copy_`.** HB includes the permutation copy, not complete engine integration. This experiment uses **FlashInfer 0.7.2, contiguous exported KV and external CUDA-graph capture**; the actual production path uses **FlashInfer 0.7.0, strided engine KV and shared-plan integration**. Matching batch order therefore does not make this an end-to-end production-call measurement.

## Primary production-BH results

Ratios are the median of paired independent/method timings, so greater than one favors the changed support. Brackets contain the within-run 95% bootstrap interval over 20 blocks, preserving each random/reverse pair. These are **not ratios of separately aggregated medians**. All points, both cache conditions and both contracts are retained.

O-only, complete output interface `[B,H*G,D]`:

| Document | Budget | Exchange hot ratio [CI] | Exchange cold ratio [CI] | Cover hot ratio [CI] | Cover cold ratio [CI] |
|:---|---:|:---|:---|:---|:---|
| docfinqa256 | 4096 | 1.0011 [0.9993, 1.0050] | 1.0000 [1.0000, 1.0017] | 0.9957 [0.9916, 0.9992] | 1.0000 [1.0000, 1.0017] |
| docfinqa256 | 16384 | 1.0150 [1.0136, 1.0166] | 1.0714 [1.0714, 1.0714] | 0.9833 [0.9824, 0.9855] | 1.0714 [1.0714, 1.0714] |
| publicdev1_256 | 4096 | 1.0006 [0.9989, 1.0020] | 1.1250 [1.0017, 1.1261] | 1.0038 [1.0000, 1.0071] | 1.0017 [1.0000, 1.1250] |
| publicdev1_256 | 16384 | 0.9981 [0.9967, 0.9994] | 1.1538 [1.0736, 1.1538] | 1.0076 [1.0054, 1.0094] | 1.1538 [1.1538, 1.1538] |
| publicdev6_256 | 4096 | 1.0074 [1.0031, 1.0099] | 1.0017 [1.0000, 1.1111] | 0.9994 [0.9965, 1.0000] | 1.0009 [1.0000, 1.1111] |
| publicdev6_256 | 16384 | 1.0021 [1.0002, 1.0044] | 1.1250 [1.1240, 1.1250] | 1.0092 [1.0073, 1.0104] | 1.1250 [1.1250, 1.1250] |

O+raw log2 LSE, complete output interface `[B,H*G,D]` plus `[B,H*G]`. Log conversion occurs only in qualification, never inside this timing contract:

| Document | Budget | Exchange hot ratio [CI] | Exchange cold ratio [CI] | Cover hot ratio [CI] | Cover cold ratio [CI] |
|:---|---:|:---|:---|:---|:---|
| docfinqa256 | 4096 | 1.0019 [0.9998, 1.0059] | 1.0000 [1.0000, 1.0000] | 0.9987 [0.9976, 1.0004] | 1.0000 [1.0000, 1.1228] |
| docfinqa256 | 16384 | 1.0144 [1.0128, 1.0159] | 1.0714 [1.0363, 1.0714] | 0.9851 [0.9831, 0.9868] | 1.0714 [1.0714, 1.0714] |
| publicdev1_256 | 4096 | 1.0004 [0.9994, 1.0032] | 1.0017 [1.0000, 1.0018] | 1.0008 [0.9988, 1.0030] | 1.1250 [1.0623, 1.1250] |
| publicdev1_256 | 16384 | 0.9986 [0.9959, 0.9997] | 1.0782 [1.0757, 1.1538] | 1.0075 [1.0066, 1.0086] | 1.1538 [1.1226, 1.1538] |
| publicdev6_256 | 4096 | 1.0088 [1.0066, 1.0103] | 1.1111 [1.0000, 1.1111] | 0.9973 [0.9959, 0.9983] | 1.1111 [1.0009, 1.1111] |
| publicdev6_256 | 16384 | 1.0037 [1.0018, 1.0045] | 1.1245 [1.1176, 1.1250] | 1.0064 [1.0047, 1.0079] | 1.1865 [1.1250, 1.1875] |

Every primary complete-call A/A interval includes one for both contracts and both cache conditions. Native-kernel-only and HB-sensitivity results remain in `summary.json` and each raw job JSON; no favorable scope or order replaces the primary one.

Cold samples show substantial clustering at recurring times such as 16.384, 18.432, 26.624, 28.672, 30.720 and 32.768 microseconds. Several bootstrap intervals consequently collapse to a single ratio. Histograms are preserved in `summary.json`; this observation does not identify the cause or justify precise repeated-run confidence. Production BH native execution and complete-layout execution differ only by views (no extra GPU copy), yet their separately measured cold medians can differ materially: for example publicdev1/4K/O-only exchange has paired median ratio 1.0000 in the native scope versus 1.1250 in the complete scope. This is a warning against treating a single cold point as a stable deployment gain. The three 16K exchange cold cells remain positive in both scopes, but independent repeated-run/cross-device validation was not part of this allocation.

Cover yields a stronger geometric reduction than exchange in the prior CPU screen, yet it does not consistently improve hot latency here. Logical union reduction alone therefore does not determine native execution time. These comparisons keep each request's token count fixed; no claim of reduced independent E0 FLOPs or freed KV capacity follows.

## Numerical and experimental checks

All **144 method/order/contract/scope qualifications** pass each method's own FP64 reference on its original mask, original Q/K/V and complete system/private/generated context. Output tolerance is atol 0.006/rtol 0.015; natural-log LSE tolerance is atol/rtol 0.00005. Maximum observed absolute output and LSE errors are **0.00234382** and **0.00062156** respectively; the latter passes the explicitly frozen combined absolute-plus-relative tolerance, not a standalone 0.00005 absolute cap. Different masks are not expected to yield equal outputs.

There are **3,840 paired measurement records**, each holding all three masks plus independent A/A. Every random ordering has its exact reverse, all times are positive/finite, and the CPU audit recomputes the reported paired medians. Every method resolves FA2 tensor-core/default-split scheduling, retains exactly equal per-native-batch token counts, and shares the original payload. Index/support construction, selector recomputation, planning and the exported-HB-to-production-BH query preparation are separately reported and excluded from CUDA-event call timing. The kernel/complete scopes and O-only/O+raw-LSE contracts are never pooled. CUDA graphs omit host dispatch overhead; these are not host scalar-call or application latencies.

The changed supports are exactly the already locked 1% ascending exchange and classical full-universe cover supports, verified by per-request index hashes before loading held-out Q. No epsilon or order was chosen from these timings. Cover's exact CPU setup and the exchange preparation cost remain material: they are not silently amortized into a free production selector. Native plan times are initialization measurements that can include first-use work, not symmetric warmed prepare benchmarks.

Preparation records cover selected components only; they are not a complete decomposition of scoring, selection, all metadata/storage work and engine integration. **Do not sum these partial records and label the result total preparation cost.**

## Failure preservation and resource accounting

Initial protocol/script commit: `51ccc73`. The first six jobs failed before numerical qualification because the CPU-helper no-CUDA-initialization assertion was placed after importing `compare_attention`, whose TileLang import initializes the correctly assigned device. The original six failure JSON files, original protocol and ledger events are preserved. Commit `94f9a82` moved that assertion before the import and froze retry1, without changing inputs, methods, tolerances or measurements. Retry timeouts were reduced from 75 to 68 seconds per job; combined failure-plus-retry worst-case allocation remained below 0.15 GPUh. No unrecorded retry occurred.

Actual P2 allocation, including framework imports and the initial failures:

- Initial six failures: **0.010431786413 GPUh**.
- Successful six retries: **0.023144122200 GPUh**.
- Total: **0.033575908614 GPUh**.

All 12 jobs have matching start/end events by name and PID. All successful jobs finished within 17.04 seconds; no timeout was extended. GPUs 2–7 are released. Ordinary UID and `.venv-latest` were used. This stage does not establish actual 48-layer cache reuse, on-policy generation quality, full-model throughput, allocator savings or algorithm novelty. A conditional forced-cold signal is the appropriate conclusion; the evidence does not support an unrestricted pass or a blanket no-signal claim. No additional GPU mechanism stage was launched from these results.

Reproduction of the corrected frozen jobs (new output names required because output overwrite is refused):

```bash
.venv/bin/python scripts/run_jobs.py results/joint_e0_gate_v1/jobs_retry1.json
.venv/bin/python scripts/summarize_joint_e0_gate.py
```

The initial `jobs.json` and protocol are retained specifically as failed-attempt provenance. The CPU summary includes exact result hashes and all corresponding ledger events.
