# Independent Thinking4B scenario

Qwen3-4B-Instruct-2507 remains a completed short-output development scenario, including its low baseline task accuracy. Its results are not replaced. The next scenario uses the already local official Qwen3-4B-Thinking-2507 revision `768f209d9ea81521153ed38c47d515654e938aea`, whose three weight shards were previously verified against immutable Hugging Face SHA256 values in `results/data_4b_download_identity.json`. This preparation rechecks shard sizes and separately hashes config,index,tokenizer and generation configuration.

The same complete document and all8 selected training questions are retokenized with the Thinking model's own tokenizer. Their prompt lengths are120571,120563,120576,120574,120560,120570,120565,120560; these happen to match the prior30B Thinking template, but are independently recomputed. Model path and pinned weight identity differ from Instruct; the live execution fingerprint includes model path as well as config,dtype,software,device,TP and exact geometry.

The native mechanism remains the independently qualified GQA4→8 paired prefix, original GQA4/S tails. Code reuse is limited to the frozen geometry-compatible worker/plan. No Instruct model receipt can authorize Thinking production: the strict driver requires fresh same-session Thinking states, while the optional worker offline mode checks the exact live model/source/device fingerprint and36-layer receipts on both ranks. The new driver and protocol have separate source hashes and output locations.

Pipeline:

1. `scripts/capture_gqa4_thinking_native.py` runs native eager attention only and exports actual Q/O and BF16 paged KV at layers0/11/23/35 on both ranks. Third complete B8 state, original physical layout and metadata, no copied single-step Q.
2. The frozen `scripts/benchmark_xqa_raw_gqa4.py --capture ...` checks original native O bits, native AA/PDL controls and FP64, then measures the complete call. Passing Instruct captures do not substitute for this check.
3. `scripts/benchmark_gqa4_thinking_raw_pair_e2e.py --qualify-only` performs strict actual36-layer Q/O,hidden/logits/KV/metadata checks while continuing the native sampler trajectory. Initial candidate scope is actual/padded B8 only; every survivor batch uses native.
4. Only after those gates, a fresh output path without `--qualify-only` runs strict qualification plus fixed128 and naturalABBA. Natural EOS is enabled,cap8192 retained,no minimum output length. Cap hits,unfinished thinking and missingFINAL remain explicit failures under the frozen metric. No artificial extension of thinking/output is introduced.

The Thinking parser requires a closing `</think>` before scoring a completed answer; a length-capped answer is not rescued. The fixed128-token phase remains a separate fixed-work diagnostic. Model initialization,candidate configuration,graph capture,anchor,generation and control/export costs remain reported separately. B6/B7 and four-way grouping are outside this initial scenario.

Root-assigned TP2 CLI examples:

```bash
CUDA_VISIBLE_DEVICES=6,7 .venv-engine/bin/python scripts/capture_gqa4_thinking_native.py --document-id docfinqa-680456070a0a0714 --output results/gqa4_thinking_native_capture_train1_v1
CUDA_VISIBLE_DEVICES=6,7 FLASHINFER_CUDA_ARCH_LIST=12.0 .venv-engine/bin/python scripts/benchmark_gqa4_thinking_raw_pair_e2e.py --document-id docfinqa-680456070a0a0714 --qualify-only --output results/gqa4_thinking_model_qual_v1/train1
```

No GPU was launched during CPU preparation. Shared frozen Instruct worker/runtime files were not changed.
