# Independent audit of the native BF16/FP8 KV development comparison

All **323 source, receipt, coverage and terminal checks passed**. Four TP2 jobs exited with code 0 and no timeout; exact allocation was **0.335058724745 P1 GPUh**. This is a second-agent audit in the same model family, not human review. No new GPU work was performed.

All 24 answer records are retained: two documents × three questions × two precisions × two cold repeats. There are only **six distinct questions**. Every answer stopped naturally before the 8,192-token cap and had a closed thinking section. Every method/document pair reproduced tokens, APC hits and rank-wise scheduled question/position sequences exactly across its two cold repeats. Cross-precision APC hits matched question by question: `[86848,86816,86816]` for the first document and `[89792,89760,89760]` for the second.

| Document | KV | Strict / rounded FINAL, per repeat | Output tokens per repeat | Median anchor + generation sum (s) |
|---|---|---:|---:|---:|
| `c0f11a67aae850fc` | BF16 | 1/3 · 2/3 | 3,066 | 23.604 |
| `c0f11a67aae850fc` | FP8 unit | 1/3 · 2/3 | 2,902 | 20.621 |
| `593ec0c33f45ea02` | BF16 | 0/3 · 2/3 | 6,732 | 42.486 |
| `593ec0c33f45ea02` | FP8 unit | 0/3 · 1/3 | 4,932 | 29.127 |

The second document's q0 FP8 FINAL omits `%`; the frozen unit-aware metric therefore rejects it, although its written numerical value is 13.92, as in BF16's `13.92%`. This is an observed unit-format scoring difference, not proof of degraded financial reasoning. No unit was inferred or repaired. All other failures, including questions with pre-existing source ambiguity, remain in the denominator. The metric is strict/rounded FINAL matching, not official program execution accuracy.

All 48 layers on both ranks passed dtype/layout and tensor/host Q/K/V scale checks at each of three receipt stages. BF16 stores BF16 KV; FP8 uses uint8 E4M3 storage. Both use BF16 Q and official XQA decode after warmup and generation. The same 0.80 pool fraction allocated approximately the same cache bytes: **10,151,264,256 BF16 versus 10,151,657,472 FP8 bytes per rank**, giving **206,528 versus 413,072 token capacity**. Recorded peak process allocations are approximately 27.286 GB in both arms. This is increased token capacity within a fixed pool, not halved total process VRAM. Native graph pool logs report 0.05 GiB actual versus 0.46 GiB estimated per rank; those values are rounded log figures, distinct from exact receipt counters.

Source inspection confirms no forced measured tokens and no Q/O/logit shadow copies. The worker hashes CPU admission tokens before the original `add_requests` and asserts the mapping exists before the inherited graph hook; the inherited GPU prompt-readback fallback is therefore unreachable on a successful run. Attention forward and CUDA graphs remain unchanged. CPU hashing, metadata and scheduler hooks remain charged. Runtime phase receipts report zero logit copies and no shadow checks.

Timing remains descriptive. The table sums separately measured cold-anchor and natural-generation phases; it is not an integrated client latency measurement. The anchor is the entire q0 prompt plus one discarded token, not a pure-document prefill. Natural output trajectories and lengths differ across precision, and GPU pairs are confounded within document. Concurrent model loading, compilation, tokenizer work, warmup, audit exports and allocation costs are retained separately in `audit.json`. No causal executor speedup, six-question noninferiority claim, calibrated-FP8 claim or joint-selection contribution follows.

Reproduce this CPU audit without altering the experiment:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_native_precision_natural.py
```
