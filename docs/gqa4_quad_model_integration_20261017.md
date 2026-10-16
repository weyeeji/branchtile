# Thinking4B grouped-hint quad whole-model development experiment

This independent path uses the corrected grouped-hint GQA4→GQA16 prefix module (`28ecb045c76babdb`) and original-GQA4 tails. It preserves the native full-batch split count and final native merge. It does not modify the older pair paths or reinterpret their measurements.

The fixed scope is nominal shared P=120528, actual/padded batch 4–8, and current maximum private length at most 8192. B1–3 and other out-of-scope states use native attention. The cutoff used inside the kernel remains floor(P/256) complete tiles; all remaining prefix tokens and the complete private suffix execute in the original native tail order. Natural generation keeps EOS enabled with the original 8192 cap and no minimum output length. The model, tokenizer, prompt IDs, sampling and quality parser are pinned by `configs/gqa4_thinking_quad_training_v1.json`.

`RawGQA4SharedQuadPlan` adds optional caller-owned workspace and raw state to the qualified quad dataflow. Five fixed batch plans share one 64 MiB workspace and one maximum-sized raw checkpoint arena across 36 layers. Each plan aliases native full-width page tables and GPU int32 lengths (viewed as uint32). Packing uses four request members inside each KV head; the last group receives zero-Q dummy members. Native tail launch B and S do not include those dummy rows. Every layer refreshes the exact KV view, and Q copy, padding, packing, group-table/length copies, state exchange and O copy remain inside the candidate call.

The strict worker captures native witness, checked candidate and production candidate graphs for all five batch shapes. On the original native trajectory it checks the first eight observed eligible states per batch: all 36 Q/O pairs, current KV slots, hidden state, local and gathered logits, metadata values and pointer aliases. Any failing rank rejects performance. An unobserved batch stays on native. The initial scope checks do not prove universal exactness; the full natural ABBA additionally requires identical complete output tokens and EOS trajectories. This is a development integration experiment, not unseen-input validation.

The CPU test constructs the real bank and runtime with recording module stubs and a mocked device-properties query. It verifies each B4–8 packing map and zero dummy member, preserves native S in both launches, checks shared workspace/state and table/length pointers, and verifies switching to layer 35 updates the KV addresses. Scope boundary and offline stale-source/missing-batch guards also pass without initializing CUDA. This is not a substitute for actual model qualification.

Run strict qualification first on a root-assigned TP2 pair:

```
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH \
CUDA_VISIBLE_DEVICES=6,7 FLASHINFER_CUDA_ARCH_LIST=12.0 \
.venv-engine/bin/python scripts/benchmark_gqa4_thinking_quad_e2e.py \
  --document-id docfinqa-680456070a0a0714 --qualify-only \
  --output results/gqa4_thinking_quad_model_qual_v1/train1
```

After qualification, omit `--qualify-only` and choose a new output directory for the fixed128 and natural-EOS ABBA. The full run requalifies its own actual native states. All output paths reject an existing summary. The baseline and candidate use the same prompts, model precision, support, sampling and native prefix cache. Setup, cold-document anchor and complete generation wall time are separately retained. Thinking4B's financial task quality is limited and must be reported with the frozen strict FINAL/rounded metrics; faster decoding is not an accuracy improvement claim.

An additional independent production-only driver, `benchmark_gqa4_thinking_quad_production.py`, accepts strict offline receipts from both ranks and captures only the five production graphs. It rejects old pair-source receipts and requires all B4–8 to have actual strict coverage. This separates research witness/checking graph setup from deployment setup. It retains native/candidate ABBA, exact output gates, natural quality, cold-document and first-use component-sum accounting. It must use a new output directory and cannot run before the new strict qualification completes. Its first-use figures are explicitly component-sum estimates, not independent cold-service measurements.
