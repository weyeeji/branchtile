# Native FP8 KV unit-scale qualification: numerical receipt complete, job timed out

The frozen protocol's **overall result is `timeout_failed`**, because it explicitly treats any timeout as failure. The raw driver's `qualified_instrumented_native_FP8KV_unit` status describes the completed numerical and trajectory assertions before process teardown; it does not override the job outcome. Neither raw `summary.json` nor the ledger has been rewritten.

`summary.json` was last written at 2026-10-16 10:36:27.044125 UTC. Both final worker phase files were written by 10:36:27.030452 UTC. The log then records process-manager shutdown and EngineCore teardown at 10:36:27. The allocation runner terminated the still-live process at its 201-second deadline, recording exit −15, `timed_out=true`, and 201.165894 allocated seconds at 10:36:30.231241 UTC. Filesystem mtimes are observed timestamps, not fsync receipts. This ordering supports a timeout during teardown; clean shutdown was not demonstrated. All four process IDs were absent and the GPU process query empty after termination.

The single TP2 job used GPU 0/1, uid 1000 and `.venv-engine`. Its exact ledger cost is **0.111758830 P1 GPUh**. There was no automatic retry or timeout extension. Source/protocol freeze: commit `6aa8296`.

The completed evidence is:

- All 48 own-cache FP64 output checks passed the frozen elementwise `atol=0.006, rtol=0.015`: two ranks × four layers × B=1/2/3 × FULL-A/eager-teacher. Maximum absolute error was 0.0321753 and maximum relative L2 was 0.00348171. The absolute maximum exceeds `atol` alone but satisfies the frozen combined relative/absolute elementwise test. All 24 same-input native eager/FULL output comparisons were bitwise equal.
- Each of three phases recorded all 36 finite rank-0 logit keys. Cold A/A and teacher-path pre-forcing logits were bitwise equal; output tokens, APC hits and per-rank scheduled question/position sequences matched. Each phase had 15 pure decode dispatches, including B3→B2→B1. Forced lengths 8/12/16 and `ignore_eos=True` are instrumentation, not natural output quality.
- Official FlashInfer resolved BF16 Q / E4M3 KV XQA decode and FA2 prefill on SM120. Actual Q and metadata Q dtypes are BF16, so the official same-dtype `maybe_quant_query` path does not quantize Q. Tensor and host Q/K/V scales were all 1.0 across 48 layers per rank, and rechecked at numerical points. The reference decodes actual FP8 cache bytes with these scales and real strides, including complete prefix, private suffix and generated tail.
- APC's complete shared pages and the nonaligned request-private partial page were checked. Each layer's cache was uint8 E4M3 storage with shape `[25817,2,16,256]` and strides `[8192,256,512,1]`; allocated token capacity was 413,072 per rank. Loaded native-library paths and SHA256 are preserved in phase receipts.
- Each rank's extra instrumentation graphs allocated 33,790,976 bytes, including 221,184 bytes of persistent selected-layer Q/O/K/V buffers. The original graphs and official attention/cache-update math were retained. Instrumented generation timings do not establish application speed or production memory benefit. Initialization took 141.779 seconds and included first-use compilation.

The `eager_teacher` phase disables FULL graph dispatch while retaining torch.compile mode 3. Native XQA exposes no LSE, so only FP64 reference-LSE finiteness was checked. These are same-quantized-input checks, not FP8-versus-BF16 task-quality equivalence. The unit-scale timeout says nothing about calibrated FP8 validity. This baseline is independent of the stopped joint-selection mechanism.

`audit.json` independently verifies source/artifact hashes, exact numerical-point coverage, logit key sets, phase correspondence and the start/end ledger pair. Reproduce the CPU audit with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_native_fp8_kv_qualification.py \
  --output results/native_fp8_kv_unit_qualification_v1 \
  --job-name native_fp8_kv_unit_qualification_v1
```
