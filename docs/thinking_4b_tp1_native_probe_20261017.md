# Thinking4B on one GPU: native capacity and opportunity probe

The local model configuration is 36 layers, 32 query heads, 8 KV heads and head size 128. TP1 therefore retains GQA4 but doubles per-device query/KV heads relative to TP2. Native BF16 KV payload is **147456 bytes/token**, excluding allocator, graph and workspace overhead. This is an independent native-only probe, not qualification of the TP2 candidate on TP1.

The protocol preserves the full train1 document and all eight original questions, prompt token hashes, sampling seed/temperature, natural EOS and 8192 cap. It fixes max_model_len=131072, GPU utilization=.9, max_num_seqs=8, APC and original native full-decode graphs for B1..8. Maximum prompt plus cap is 128768 tokens, so the context guard does not truncate anything. Initial page-rounded shared-group payload is 120880 token slots (17.82 GB decimal); all eight private tails reaching their caps would need 186416 slots (27.49 GB). The latter is an upper bound, not a natural-length forecast. Available memory after weights and native graph capture is measured, not assumed sufficient.

A new worker asserts actual TP1 Hq32/Hkv8/36-layer/cache shape and reports allocated token capacity, unique cache storage, GPU free/allocated/reserved/peak memory and actual native dispatch metadata. Missing kernel fields in legal eager metadata remain visible instead of being mislabeled XQA. The driver keeps initialization failure/OOM, cap hits and per-request preemptions. It never retries with truncated text, fewer questions, a shorter cap or modified memory utilization.

Default execution is one cold anchor plus the full natural eight-question ready group. It reports complete generation and cold-anchor-plus-generation wall time, all outputs/finish reasons, strict FINAL/rounded financial score, and natural feasibility without cap/preemption. This is one capacity/opportunity observation, not replicated performance. TP1 may follow a different trajectory than TP2; a future TP1 candidate must be compared to its own TP1 native baseline.

A separate cold anchor and fixed 128-token native group enables the PyTorch profiler for 32 iterations after a 64-iteration delay, plus CUDA events around the actual whole-model graph. This forced-output/profile run is excluded from natural time and quality. It can locate attention versus the rest of the model after removing TP communication; graph-event timing excludes host preparation/sampling, while unprofiled task wall time includes them.

CPU preparation was run with CUDA hidden, checked all model artifact and prompt hashes and left CUDA uninitialized. Root owns GPU allocation. One exclusive RTX 5090 and an initial 1200-second process limit are sufficient as an exploratory allocation; timeout remains a failure, not a reason to silently shorten output.

```bash
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/probe_thinking_4b_tp1_native.py --prepare-only --output results/thinking_4b_tp1_native_prepare_v2
# Root assigns exactly one visible GPU for this command:
.venv-engine/bin/python scripts/probe_thinking_4b_tp1_native.py --output results/thinking_4b_tp1_native_v1
```

`--capacity-only` stops after native model/graph initialization and the read-only capacity receipt. It does not demonstrate natural-task feasibility. Both modes preserve the initial failure instead of overwriting any output directory.
