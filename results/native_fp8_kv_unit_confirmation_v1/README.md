# Native FP8 KV confirmation: passed with clean exit

This explicitly authorized confirmation satisfies both frozen requirements: all numerical/trajectory assertions completed and the process exited normally (`exit_code=0`, `timed_out=false`). Protocol commit: `acd9ab5`. The first attempt remains `timeout_failed` in its separate directory and ledger; its completed receipts are not relabeled as a passing job.

The same executable source, immutable model, prompt tokens, engine settings, unit scales, 8/12/16 forced tail lengths and numerical thresholds were retained. Only the predeclared wall-time ceiling changed from 201 to 240 seconds, with at most 15 seconds of termination grace. The new protocol includes the first attempt's protocol, summary, audit and log SHA256. No further retry occurred.

| Receipt | Confirmation result |
|---|---:|
| Own-quantized-cache FP64 output checks | 48/48 passed |
| Same-input native FULL/eager output comparisons | 24/24 bitwise equal |
| Rank-0 finite logits, each of three phases | 36/36 keys |
| Cold A/A and teacher-path pre-forcing logits | Bitwise equal |
| Tokens, APC hits, scheduled question/position trajectories | Equal across all phases |
| Actual query / KV precision | BF16 / E4M3, all scales 1.0 |
| Initialization, compiled cache already available | 40.053611 seconds |
| Allocated TP2 wall time | 102.402830 seconds |
| This job, P1 GPUh | 0.0568904613 |
| Both attempts including first timeout, P1 GPUh | 0.1686492912 |

The job ran on GPU 0/1 from 2026-10-16 10:41:40.214730 UTC to 10:43:22.616203 UTC. All four process IDs were absent and the GPU compute-process query was empty after completion.

Maximum absolute error against own-cache FP64 was 0.0321753; maximum relative L2 was 0.00348171. Every element passed the unchanged combined `atol=0.006, rtol=0.015` test. The actual cache is uint8 E4M3 with real paged strides; the reference includes full prefix, request-private partial page/suffix and generated tail. Native FlashInfer resolves XQA for BF16 Q and FP8 KV. All 48 layers per rank have equal unit tensor/host Q/K/V scales; the per-point checks confirm the actual Q remains BF16. No native LSE is exposed or qualified.

Instrumentation retains official attention/cache-update math but adds persistent selected-layer Q/O/K/V snapshots and separate native graphs. `eager_teacher` disables FULL dispatch while retaining torch.compile mode 3. These timings are not production speed measurements. Fixed tails with `ignore_eos=True` do not measure task quality or natural output length. Own-FP8-cache agreement does not establish BF16-KV quality equivalence, and this native baseline is independent of the stopped joint-selection mechanism.

`audit.json` independently checks frozen hashes, exact point/key coverage and the matching start/end ledger events. `cross_run_receipt.json` records whether Q/output hashes and numerical errors reproduced across attempts; it does not change the first attempt's failure status. Reproduce the CPU audit with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_native_fp8_kv_qualification.py \
  --output results/native_fp8_kv_unit_confirmation_v1 \
  --job-name native_fp8_kv_unit_confirmation_v1
```
