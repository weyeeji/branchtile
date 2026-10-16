# Calibrated FP8 KV: minimum official-loader interface, read-only audit

**A scale-only derived checkpoint is supported by the installed vLLM 0.31.0 source path. It has not been built or run here.** Calibration is an established baseline, not a proposed contribution or a remedy established by the six-question pilot. The pilot's one rounded-match difference included a missing unit; it does not establish that calibration will improve reasoning or FINAL formatting. This audit installs nothing, changes no checkpoint, executes no calibration, and uses zero GPUh.

The official [vLLM KV quantization guide](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) recommends dataset calibration and distinguishes scalar from per-head scales. Its generic [LLM Compressor example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_kv_cache/README.md) also changes weight/activation quantization; copying that recipe and re-exporting this already-FP8 model would not guarantee the original weight bytes remain unchanged. The interface below instead follows the actual installed loader, whose exact file hashes are in `metadata.json`.

## Minimal derived artifact

Keep the immutable Qwen checkpoint's four original safetensors shards byte-identical, with original SHA256 receipts, and preserve its configuration/tokenizer/template. Use a separate derived directory and identity; do not label the derived artifact as the original immutable HF revision. Read-only links to original shards or exact copies can avoid reserialization. The original checkpoint index has 37,491 tensor entries and no Q/K/V/KV cache scale keys.

Add one safetensors shard with **96 positive finite FP32 scalar tensors**, one K and one V scale for every layer:

```text
model.layers.0.self_attn.attn.k_scale
model.layers.0.self_attn.attn.v_scale
...
model.layers.47.self_attn.attn.k_scale
model.layers.47.self_attn.attn.v_scale
```

Prefer shape `()` or `(1,)`; `KVCacheScaleParameter.weight_loader` rejects `numel()!=1` and reshapes a singleton to scalar. Do not substitute a vector of per-head values on this scalar FLASHINFER path. Both scales should be present for every layer: missing values can fall back to 1 or duplicate K into V, which would silently defeat the intended calibration. Validate finiteness and positivity in the artifact builder rather than relying on the loader's positive-value branch alone.

**FP32 storage does not imply FP32 loading precision on this installed path.** `BaseModelLoader` constructs the model under default dtype BF16, and `KVCacheScaleParameter` constructs `torch.tensor(-1.0)` without an explicit dtype; its load copy therefore rounds into a BF16 scalar before final conversion into FP32 scale buffers. Attention's scale-placeholder creation has no overriding FP32 context. A CPU-only PyTorch fixture reproduced `0.12345679104328156 → 0.12353515625`, with CUDA uninitialized; it did not instantiate a full vLLM model. A future artifact must record this source-predicted conversion and check actual loaded values. Either encode predeclared BF16-representable values in the FP32 file or compare against `FP32(BF16(file_scale))`; do not silently relax checks against arbitrary raw FP32 values. Validate positive finite values after conversion as well.

The derived `model.safetensors.index.json` must preserve every original weight-map entry and append all 96 new names pointing to the new shard. If retaining `metadata.total_size`, increment its tensor payload count by 384 bytes for 96 FP32 scalars; the physical safetensors file also has a header. **An unindexed sidecar is not enough**: `filter_duplicate_safetensors_files` filters discovered shards against the index and rejects referenced missing files. Preserve original 128×128 FP8 weight-block quantization settings; no change to `quant_method`, weight tensors, weight scales or activation policy is necessary for this interface.

Suggested provenance includes original four SHA256s, derived index/shard SHA256, 96 named values, training document and prompt hashes, calibration coverage/counts, estimator definition, TP reduction rule, source version, and an explicit list of unchanged files. Merely adding JSON calibration metadata without safetensors parameters does not load scales through this path.

## Exact installed loading and conversion path

- `Fp8Config.get_quant_method` selects `Fp8KVCacheMethod` for Attention. `BaseKVCacheMethod.create_weights` creates scalar `q_scale/k_scale/v_scale/prob_scale` placeholders before loading.
- Qwen3MoE uses `AutoWeightsLoader`. This loader merges the quantization cache-scale mapper. Canonical `.self_attn.attn.k_scale/v_scale` names above remain canonical; `.self_attn.k_scale/v_scale` also map to them. `Fp8Config` additionally accepts compressed-tensors-style `.k_proj.output_scale/.v_proj.output_scale`. The deprecated `.kv_scale` path should be avoided because it shares a single scale.
- `KVCacheScaleParameter.weight_loader` copies the supplied singleton through the placeholder dtype described above. The original weights remain independently loaded using their existing loaders. Scalars are replicated across TP ranks; this artifact format does not encode rank-specific scales.
- `BaseModelLoader.load_model` calls `process_weights_after_loading` before returning the model. For quantized KV, `BaseKVCacheMethod` copies separate positive K/V scales into device `_k_scale/_v_scale`, host `_k_scale_float/_v_scale_float`, and CPU scale buffers, then deletes temporary load parameters. On this NVIDIA E4M3 platform the AMD FNUZ-specific factor-of-two conversion does not apply. The engine then profiles/allocates cache and warms/captures graphs.
- With K/V-only scale tensors, the current source briefly assigns missing Q scale from K and logs that fallback, **but a later Q-scale branch overwrites the final missing-Q value to 1.0**. Do not infer final Q scale from the earlier warning. For this workload BF16 Q is required and `get_xqa_bmm1_scale` only multiplies Q scale for FP8 Q; it multiplies K scale for quantized KV. Runtime receipts must still confirm actual dtypes and all tensor/host values.
- For `kv_cache_dtype='auto'`, quantized-KV postprocessing is skipped and initialized K/V unit scales are retained. The installed `CacheConfig` and `EngineArgs` contain no `calculate_kv_scales` option; supplying an old runtime-calibration flag is not this version's interface.

These statements are source-path evidence, not a successful end-to-end derived-checkpoint load. Existing unit-scale receipts qualify the current native backend but cannot substitute for a new non-unit-scale qualification.

## Why a fresh engine/cache/graph is required

Cache writes pass device `_k_scale/_v_scale` tensors to `reshape_and_cache_flash`, whereas attention readers use host scales and cached backend values. `FlashInferImpl.forward` initializes `self.bmm2_scale` once from `_v_scale_float`; later XQA calls use that cached value. CUDA graphs also capture host scalar arguments. Changing only a host float, only a device tensor, or both after graph capture does not update every cached value and captured launch. Previously stored FP8 bytes were also encoded under the old scales. The conservative supported experiment is a **fresh process loading the derived artifact before any prefill, cache write, warmup or graph capture**, with empty APC. Reusing old graphs/cache would require separately demonstrated invalidation and recapture and is outside this baseline audit.

## Calibration and later qualification requirements

Calibration statistics can be collected on a separately frozen **training-only** document set using the same original FP8 weights, BF16 compute and actual post-normalization/post-RoPE K and V at the cache-write boundary. This is scale estimation, not weight training. Declare prompt lengths, truncation policy, prefill/decode coverage and generated-tail policy in advance; calibration on short prompts alone does not establish long-context coverage. Do not tune on development/evaluation answers or choose scales to fix the observed unit-format miss.

For a single shared scalar per layer/K-or-V, combine statistics across both TP ranks and all their KV heads using the declared estimator; do not export one rank's range as a full-model statistic. A fixed positive-floor max-absolute/E4M3-range estimator is a possible conventional control, but no estimator, sample count, clipping rule or quality threshold has been selected or run in this audit. There is no basis here to promise max-range, percentile or optimization-based scales improve quality.

A later explicitly frozen qualification must verify all 96 loaded values against the artifact's declared loader conversion on both ranks before capture, tensor/host/cached-backend consistency, BF16 Q, unchanged source shards, non-unit own-cache FP64 correctness, APC partial-page behavior, FULL B1/2/3, cold A/A and clean exit. Unit-scale-hardcoded checks must become expected-artifact-value checks in a **separate** protocol, preserving the original result. Natural output/quality comparisons then require a separate frozen experiment and the unchanged metric; own-cache numerical agreement is not BF16 task-quality equivalence.
