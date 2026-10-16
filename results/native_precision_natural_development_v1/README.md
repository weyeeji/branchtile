# Native KV precision: complete natural development controls

All four TP2 jobs started together on the eight RTX 5090 GPUs and exited normally. The frozen protocol is `configs/native_precision_natural_development_v1.json` (commit `9370ff4`). This fills a missing mature full-history baseline; it is not a new method and does not reopen the stopped dynamic-row or joint-selection candidates.

Each arm used two cold-APC repetitions after engine/sampler warmup, the official FP8 model weights, BF16 compute/query, native FlashInfer XQA, full model graphs, page16, the same memory fraction and limits, and the same natural sampling seed. FP8 KV uses the official default unit scales, not calibrated scales. The two repetitions have exactly equal tokens, APC hits and scheduled question/position trajectories within every arm. Cross-precision APC hits match. All 24 output records reach EOS and have closed thinking; no output hits the8192 cap. These are **two development documents, six unique questions and one seed**; repeated records are not independent task-quality samples.

| Development report | KV precision | Natural output tokens | Strict /3 | Rounded /3 | Generation, s | Cold anchor + generation, s |
|---|---|---:|---:|---:|---:|---:|
| original `c0f11a67` | BF16 |3066|1|2|12.857|23.604|
| original `c0f11a67` | FP8 unit |2902|1|2|9.215|20.621|
| new `593ec0c3` | BF16 |6732|0|2|31.145|42.486|
| new `593ec0c3` | FP8 unit |4932|0|1|17.267|29.127|

Times are medians of the two identical trajectories within each arm. The cold anchor is q0's complete prompt with a discarded one-token answer, followed by together-admitted natural questions. The displayed sum combines two measured phases; it is not one continuous client latency. Initialization/JIT, warmup, input preparation, prefix reset, audit export and allocated process time remain separately recorded in the raw summaries/ledger.

For the new report's first question, both precisions report the same13.92 numerical value in their answer, but FP8's final line is `FINAL: 13.92` while BF16's is `FINAL: 13.92%`. The frozen unit-aware metric therefore loses one rounded match. We do not repair this unit after observing outputs, and do not describe this single format discrepancy as established arithmetic degradation. The third question's missing segment and earlier label ambiguities remain in the source audit and denominator. Across six unique questions, strict is1/6 in both arms, rounded4/6 versus3/6; neither primary ties nor this tiny difference establish noninferiority or a general quality effect.

The descriptive BF16/FP8 ratios for anchor+generation are1.145 and1.459, but output lengths and precision differ, and each document's arms run on different GPU pairs. Therefore these are **not isolated executor speedups or quality-qualified application gains**. Both pairs per document have within-NUMA NODE topology; dtype assignment is reversed across the two NUMA nodes. No frequency or power settings were changed. `prelaunch_hardware.json` and `board_state.jsonl` preserve device conditions. Concurrent loads contend for CPU/disk, and initialization's compiled-cache differences must not be attributed to KV precision.

Both precisions reserve approximately9.45GiB of KV storage per rank under the same pool fraction. Actual token capacity is206528 for BF16 and413072 for FP8. The process's current allocated bytes are identical,27170969088 per rank. This demonstrates a capacity change in a fixed pool, **not halved process memory**. Higher configured context/batch limits require new qualification and may change the profiled pool.

The new read-only worker hashes prompt tokens from the scheduler's CPU admission data. Before graph execution it asserts all request IDs are already mapped, making the inherited GPU prompt-readback branch unreachable. CPU hashing and step metadata remain charged; their records are retained. Attention/graph math is unchanged, with no selector, teacher forcing or Q/O/logit snapshots. The stochastic warmup uses two tokens to exercise an actual B3 decode before measurement.

The independent audit in `results/native_precision_independent_audit_v1/` checks source/point/phase coverage, all layer/rank precisions and scales, actual XQA metadata, answer denominators, A/A and cross-precision APC, and clean ledger exits. The preceding own-cache numerical qualification passed only in its separately recorded clean-exit confirmation; the first timeout is still a failed job. This natural pilot costs **0.3350587247 P1 GPUh**; both qualification attempts cost another0.1686492912 P1 GPUh. No automatic retries or formal evaluation expansion followed.

`analysis.json` and `quality_cost.png/.pdf` are reproducible with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/summarize_native_precision.py
.venv/bin/python scripts/plot_native_precision.py
```

The raw answers, exact tokens and per-step records remain available. The current paper requirements are still unmet: this is a stronger baseline, not a successful replacement contribution.
