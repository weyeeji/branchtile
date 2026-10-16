# Native raw-pair whole-model adapter: implementation contract

Status: CPU-only design and executable admission checks. No model hook is installed and no model performance is claimed. The CUDA pair API is independently maintained in frozen commit `8d0dbf0`; its GPU qualification must pass before adapter activation.

The CPU scaffold is `src/vllm_native_raw_pair_contract.py`. Run `.venv-engine/bin/python scripts/check_vllm_native_raw_pair_contract.py`; it checks 22 explicit fallback conditions and writes `results/native_raw_pair_model_contract_v1.json`. It imports neither Torch nor CUDA.

## Confirmed CUDA API

Owner-confirmed API: `RawPairPlan(modules, query, cache, block_table, seq_lens, prefix_tokens, workspace=None, raw_state=None)` in `src/xqa_raw_pair_runtime.py`. Modules are GQA8/GQA16 from `src/xqa_raw_pair_build.py`. The current plan accepts B8 BF16 contiguous Q, original BF16 fused cache, a full contiguous int32 page table and unit scales. It packs Q, executes GQA16 phase 1, then GQA8 phase 3 reading mapped paired state. Runtime candidate PDL is disabled.

`prepare(seq_lens)` performs a synchronized physical prefix check and copies lengths. That is an explicit host cost; it must never be called per layer or inside CUDA graph capture/replay. The initial whole-model adapter may use it once per actual admitted step outside the model graph, with the complete cost recorded. Device-only preparation is a later independent optimization, not an assumed property of this API.

## Admission and native split preservation

Start with only actual request count = actual decode tokens = padded tokens = query rows = 8, pure one-token decode and FULL CUDA graph. All other batch shapes, including B1/B2 and odd survivors, use the original native graph. A Python hook seeing `query.shape[0]==8` is insufficient: an actual B7 batch can use a padded B8 graph. Admission must happen in `model_state.prepare_attn` using the actual input-batch descriptor, before selecting a candidate graph.

The original public XQA wrapper derives its C++ `maxSeqLen` from **full page-table columns × page size** (`flashinfer/xqa.py:363`). The C++ launcher uses the same value for both split selection and row pitch (`mha.cu`, `maxNbPagesPerSeq`). Preserve the original full contiguous table width. Do not replace the C++ argument with a smaller logical sequence maximum while keeping a wider table.

For the current B8/Hkv2/170-SM native launch with 10,240 table columns, the extent is 163,840 tokens and S=10. The paired prefix uses synthetic SM argument `S * 4 * 2 = 80`; the tail uses original SM argument 170, original B8 and original table. Both have S=10. Pairing must not halve the B used to choose the tail split count.

Only complete native 256-token tiles are shared. For P=120,528, cut at 120,320 (470 tiles); the remaining 208 logically shared tokens are intentionally read again in each private continuation, along with the private suffix. The prefix identity check covers original physical pages, not equal token content. Cohort admission, request mapping and prefix-overwrite/preemption epochs must remain valid. On a mismatch, choose native before graph launch; an invalid graph phase is rejected and never counted as successful candidate work.

## Q stride and resource ownership

Record actual incoming Q shape/stride in the model hook. Existing exported captures saved Q into contiguous diagnostic buffers, so they do not prove the live incoming Q is contiguous. The installed vLLM native XQA path itself calls `decode_query.contiguous()`. Copy noncontiguous incoming Q into a preallocated B8 BF16 buffer before either packing or the tail call; count this copy. Avoid `.view()` on the uncanonicalized query. Preserve the output buffer expected by vLLM; use a preallocated temporary and charged copy only if the caller's output stride requires one.

Allocate one 64 MiB workspace (8 MiB semaphore + 56 MiB scratch) and one 1,433,600-byte paired state for all 48 layers. The stream order is prefix(layer L) → tail(layer L) → next layer. State and scratch may be overwritten only after their consumer completes. No concurrent graph replay, overlapping model microbatches, second stream or separate request graph may share these buffers without synchronization. No state allocation multiplied by 48 is needed.

The scaffold's conservative production-buffer inventory totals 68,837,424 bytes, including Q/packed-Q/O buffers, full pair table and lengths. Original KV and full native table remain aliases; no KV compaction or payload gather. Qualification-only saved Q/O across 48 layers costs another 3,145,728 bytes and must not appear in the production timing graph. A simple first adapter can instantiate 48 plans with shared workspace/state; per-layer smaller buffers remain measurable overhead. A subsequent adapter can share those too, provided captured addresses and serial lifetimes are verified.

Pair page-table/length refresh ideally runs once before layer 0 because all layers share the same logical metadata. The existing small-qualification `run()` repeats its copies each invocation; if reused unchanged, charge all 48 repetitions rather than claiming one preparation. An optimized layer-shared adapter must be qualified independently. Layer-specific cache K/V aliases still differ and must be bound correctly at graph capture.

## Worker hook sequence

Use existing `DenseCascadeDeviceWorker` as a hook-location reference, not as a numerical policy to inherit:

1. Before creating candidate graphs, register the original per-layer `impl.forward`, cache pointers, scales, actual cache strides and attention configuration. Require original FlashInfer XQA, BF16, Hq16/Hkv2/D128/page16, no DCP, sinks, custom/speculative masks, sliding window or quantized output. Keep the original KV-cache update path, which is outside this attention forward hook.
2. Create production, checked and native-witness graphs before fresh real requests. Native-witness calls original attention and records Q/O. Checked candidate records Q/O around the candidate; production omits diagnostic copies. Dummy capture inputs must be valid and safely bounded. Do not derive production cohort identity from dummy page-zero tables.
3. Wrap `runner.prepare_attn` only to retain the actual gathered table/slot aliases. Wrap `model_state.prepare_attn` to inspect actual B, padding, metadata type, request IDs/epochs and native length buffers. Select native for every unqualified condition. Bind or verify physical shared prefix outside timing only if explicitly reported as setup; per-step work belongs in the per-step total.
4. On admitted B8 steps, verify the candidate graph's table/length buffers alias or are freshly copied from the exact actual native metadata. During graph capture the attention wrapper uses candidate code only for the intended B8 graph. Other graph shapes call original attention.
5. At layer 0 refresh shared metadata once if using the independently qualified layer-shared variant. At each layer canonicalize Q, bind that layer's original K/V aliases, run prefix phase 1 and tail phase 3, then return the expected output buffer. Do not allocate, plan, call `.item()` or construct CPU length tensors in this captured path.
6. Select the candidate graph in `cudagraph_manager.run_fullgraph` only from the admitted actual-step record. Capture mode determines what gets recorded in a graph; Python mode changes after capture do not switch operations inside an existing graph.

## First real-model gate: 128 fixed native steps

Start with native-generated tokens and unchanged sampling state. For each selected pure B8 state, replay original native, native-witness, checked candidate and production candidate on the same actual inputs before scheduler advancement. The native result continues to the sampler during qualification; the candidate is a same-state intervention until all requirements pass. This avoids conflating output-length changes with operator qualification.

Compare raw bytes for hidden states, full per-rank vocabulary logits and gathered logits where present; compare all 48 incoming Q and attention O tensors; compare all 48 currently written KV slots using actual slot mappings. Check original input IDs, positions and full page tables remain unchanged. Snapshot/reference current-token KV consistently before repeated arms, and verify native-AA. `torch.equal` on floating-point values is insufficient for a bitwise label: compare integer/byte views, including signed zero.

Collect each rank's pass/fail receipts without rank-local early exceptions around TP collectives. The driver jointly rejects if either rank differs or fails to execute the intended candidate graph. Source, generated module and capture hashes, native/pair launch grids, graph addresses, actual/padded B, copy/prepare costs and fallback reasons belong in the receipt.

Only then time the production graph without checks, using randomized paired blocks and native-AA at frozen real states. A fixed 128-token completion experiment should separately report actual wall-clock prefill/decode totals, setup, preparation and generated-token equality. Finally evaluate natural termination with the qualified dispatch and native fallbacks. Neither same-state graph savings nor fixed-128 timing alone establishes natural end-to-end benefit.
