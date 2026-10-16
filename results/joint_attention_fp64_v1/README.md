# Held-out FP64 joint-support attention screen

The exchange masks preserve the frozen pooled-score constraint, but that constraint does not bound held-out decode attention loss by the same fraction. At epsilon 1%, the largest observed drop in retained total dense-softmax mass is **6.50 percentage points** at 4K and **4.92 percentage points** at 16K. Median changes in dense-reference output error are small and often favorable; individual heads also worsen. These descriptive results neither establish task quality nor select a deployable epsilon/order.

The protocol and numerical script were committed before natural computation in `20b13d8`. All three corrected `suffix_v2` official main-model captures were used at rank 0, layer 15, both local KV heads, all GQA query heads, both budgets, and all eight epsilon/order variants plus independent supports. Thirty decode files span the frozen co-ready anchors 16/64/256; actual request prediction steps are preserved, including the smaller steps of later admitted requests. Complete system-prefix, document and request-private/generated KV are included. The two protected document-tail positions reside in private KV and are counted once.

Validation covers **1,080 unique request/KV-head/anchor/config rows and 8,640 individual query-head records**. All 240 epsilon-zero rows exactly reproduce independent masks, FP64 output and natural-log LSE. The maximum absolute discrepancy between retained softmax mass and `exp(LSE_sparse-LSE_dense)` is `4.22e-15`. All numbers are finite and context lengths match actual causal query positions.

## Descriptive output changes

Each row below aggregates 60 request/KV-head/anchor observations (480 query heads) across three documents. These are correlated measurements, not 60 or 480 independent samples. Relative-L2 changes are signed absolute differences from the independent support's error against dense; positive is worse. Union reductions refer to six distinct document/KV-head masks per budget, rather than repeating each mask for every step. Both orders are retained without choosing a winner.

| Budget | Epsilon | Request order | Logical union reduction, median | Group relative-L2 change, median | Group increase, maximum | Query-head increase, maximum | Retained mass change, minimum |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 4096 | 0.001 | ascending | 7.15% | -0.000114 | +0.010276 | +0.077422 | -1.877 pp |
| 4096 | 0.001 | descending | 7.15% | -0.000142 | +0.010999 | +0.082827 | -1.952 pp |
| 4096 | 0.005 | ascending | 14.93% | -0.000322 | +0.010115 | +0.076557 | -4.029 pp |
| 4096 | 0.005 | descending | 15.03% | -0.000390 | +0.010774 | +0.081543 | -4.033 pp |
| 4096 | 0.010 | ascending | 19.59% | -0.000257 | +0.017242 | +0.133470 | -6.499 pp |
| 4096 | 0.010 | descending | 20.18% | -0.000497 | +0.017995 | +0.131564 | -5.311 pp |
| 16384 | 0.001 | ascending | 8.10% | -0.000229 | +0.002061 | +0.015279 | -1.361 pp |
| 16384 | 0.001 | descending | 8.14% | -0.000281 | +0.002110 | +0.014647 | -1.295 pp |
| 16384 | 0.005 | ascending | 15.67% | -0.000636 | +0.003872 | +0.036073 | -2.888 pp |
| 16384 | 0.005 | descending | 15.89% | -0.001106 | +0.003998 | +0.031921 | -2.662 pp |
| 16384 | 0.010 | ascending | 18.73% | -0.001030 | +0.003744 | +0.030399 | -4.256 pp |
| 16384 | 0.010 | descending | 19.92% | -0.001312 | +0.005126 | +0.044762 | -4.923 pp |

Epsilon 0 is identical to independent in every cell, so its changes and union reductions are all zero. Independent 4K attention has group relative L2 against dense of median 0.112749 and maximum 0.288220; independent 16K has median 0.060341 and maximum 0.244148. These existing approximation errors remain part of the context for interpreting small changes.

The largest 4K individual-head increase at epsilon 1% occurs for `publicdev1_256`, question 2, KV head 1/query head 14, anchor 16 (actual prediction step 14), ascending order: relative L2 against dense rises from 0.463124 to 0.596593. With the same epsilon, grouped-error medians decrease by 0.000257/0.000497 for the two orders, while 168/162 of 480 individual heads worsen. At 16K, the largest individual-head increase is 0.044762 for descending order; 137/140 of 480 heads worsen under ascending/descending order. Full signed changes, cosines, maximum absolute output error, LSE changes, actual total and document-conditional mass, and all per-document aggregates are retained in the JSON files.

No different-support success threshold is supplied by proposal section 5.7, which addresses same-mask kernel numerical correctness. No threshold is invented here. This screen evaluates fixed attention on **dense-trajectory captured queries and generated histories**, not an on-policy sparse decode trajectory. Pooled prefill scores remain a proxy; full-context mass can be dominated by private/prefix positions, so document-conditional mass is also recorded. All physical KV remain allocated, so physical capacity released is zero. Novelty and natural generation quality remain unestablished.

## Reproduction and accounting

`protocol.json` freezes every input SHA, actual query-step metadata, numerical conventions, and grid. `scripts/probe_joint_attention_fp64.py` reconstructs the prior CPU masks before opening held-out Q for each document and checks every selected-index hash against the earlier locked results. The only readout of held-out Q is evaluation, not mask construction. Raw results are split into the two job JSONL files; job summaries record ordinary UID, venv, Torch, GPU UUID, memory and script/protocol/result hashes. `summary.json` includes the exact matching ledger start/end events.

```bash
.venv/bin/python scripts/run_jobs.py results/joint_attention_fp64_v1/jobs.json
.venv/bin/python scripts/summarize_joint_attention_fp64.py
```

The first command intentionally refuses existing output; any re-run needs distinct output names and GPU allocation. The second command is CPU-only and validates/summarizes the existing immutable outputs.

Both initial P2 jobs completed successfully on RTX 5090 GPUs 2 and 3. Allocated time was 6.825602 + 6.524148 seconds, totaling **0.003708264 GPUh**. Peak allocated CUDA memory was 513,118,720 and 959,822,336 bytes. These times include framework startup according to `run_jobs.py`; they are resource accounting, not attention-method latency comparisons. No natural text generation or production change was performed.
