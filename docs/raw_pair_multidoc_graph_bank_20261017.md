# Reuse a dynamic-cut graph bank across two completed document cohorts

The new experimental path captures the native graph shapes once and the three research modes (witness, checked, production) once in one 30B TP2 engine. It then changes documents by uploading one stable device int32 prefix-cut scalar. Original native batch size, split count, page-table width, GPU length aliases and arithmetic are retained. This tests whether document-dependent graph recapture can be removed; no speed or energy result exists yet.

## Frozen workload and controls

`configs/raw_pair_multidoc_training_v1.json` preserves both original development documents, all 21/19 distinct questions, model tokenizer, natural EOS and 8192-token cap. Nominal shared prefixes are 135376 and 123808 tokens. The unchanged development dispatch rule enables B6..21 only when P>=65536 and private/P<=0.125. Unobserved or unqualified batches use native. This is sequential completed document groups, not a mixed-document scheduler or an arrival-distribution claim.

Each document first receives its own original-native natural trajectory with up to eight same-state checks for every observed enabled B. The checked and production graphs must match native AA, every layer Q/O, current KV, hidden state and logits bitwise. Qualification sets are isolated by document. The performance order is native, candidate, candidate, native, each arm running document A then B. All resulting tokens, finish reasons, APC hits and request/position trajectories must agree. Both ranks must have identical graph trajectory hashes. A candidate speed claim additionally requires actual candidate graph execution in both candidate arms.

A document switch requires an empty scheduler, sets native/unarmed mode, clears request/slot/epoch and physical-prefix ownership, and replaces the original shared config dictionary's prompt hashes and P. Candidate/shadow switches synchronize, upload the device cut, then synchronize; native switches do not upload the candidate cut. The same full-width table and native GPU lengths remain untouched. The first actual candidate state checks current physical prefix identity again. Capture is guarded against a second invocation.

Both methods reset APC and run the same cold one-token anchor before each document. Initialization, candidate configuration, one-time three-mode capture, per-document activation/reset/anchor/group-control/generation/export and total phase walls are separately retained. The cold-document metric includes switch control. Research qualification and witness/checked capture costs are explicit, not silently charged to native-only production or treated as production-only cold deployment. This first driver cannot establish production-only cold-start amortization by itself.

## Energy evidence

Read-only NVML cumulative energy samples bracket activation, anchor/reset, group generation and phase export. Actual TP worker CUDA UUIDs select and verify the physical NVML devices; selector-number assumptions alone cannot yield a task total. Unsupported, decreasing or unverified counters give null task Joules. Actual raw counters and errors remain. Initialization/identity RPC and every NVML sampling duration are separately recorded. Counters include every process on the device, so root must ensure exclusive GPU use. No clocks or power controls are modified.

## CPU preparation

`results/raw_pair_multidoc_cpu_v1.json` checks all 16 enabled shapes, shared cut address, original S, live alias stability, native activation without upload/synchronization, document-specific qualification and ownership reset, invalid cut rejection and honest energy null handling. CUDA was not initialized. `results/raw_pair_multidoc_preparation_v2/summary.json` independently reconstructs both documents using the local model tokenizer and verifies prompt hashes, lengths, complete-page prefix and context limit.

## Root-scheduled GPU commands

Use the existing GPU exclusion/ledger scheduler, two idle GPUs and the project virtual environment. No root privileges are needed.

```bash
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH CUDA_VISIBLE_DEVICES=0,1 FLASHINFER_CUDA_ARCH_LIST=12.0 .venv-engine/bin/python scripts/benchmark_raw_pair_multidoc.py --qualify-only --output results/raw_pair_multidoc_model_qual_v1
```

Suggested qualification timeout is 900 seconds: two complete natural groups, with strict shadows only for the first eight states per observed B. After a successful gate, run the same frozen script without `--qualify-only` to a fresh `results/raw_pair_multidoc_e2e_v1` directory (suggest 2400 seconds). It repeats qualification and then all eight document-group performance runs. Existing summaries are rejected, failures retained, and source/model/protocol hashes are saved. These are timeout ceilings, not promised runtimes or authorized GPU launches by this subagent.
