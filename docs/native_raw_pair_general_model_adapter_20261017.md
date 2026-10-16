# General model adapter: implementation pending GPU qualification

This independent adapter leaves the running B8 experiment unchanged. The CPU
contract checks do **not** establish CUDA, whole-model, or performance correctness.

## Dispatch and buffer ownership

`scripts.vllm_native_raw_pair_general_worker.py.NativeRawPairGeneralWorker` uses
`RawPairPaddedPlan`: ceil(B/2) prefix groups, original B tail rows, and the split
count chosen for the original B. The odd dummy contributes no tail row. The
initial production eligibility policy is B >= 6; configuration may restrict the
set further, but cannot enable B <= 5 in this worker. Actual requests, actual
scheduled tokens, and padded tokens must all equal B. Unsupported shapes replay
the unchanged native graph.

All batch plans and all 48 serial layers share one 64 MiB workspace and a single
raw-state arena sized from the maximum of ceil(B/2) * 2 * nativeS(B) * 4480 FP32
values. B6..21 on 170 SMs requires less than 2 MiB of state. Small Q, padding,
output, and paired table buffers are per batch, not per layer. Each layer's Q
canonicalization, odd padding, pair packing, paired metadata copy, and O copy
remain in the measured graph. KV payloads remain original-cache aliases.

## Removing the measured host synchronization

The plan aliases `runner.input_buffers.seq_lens[:B]`, an existing contiguous
int32 GPU buffer, as uint32 for XQA. Every admitted step verifies that the native
attention metadata points to this same buffer and the original full-width page
table. There is no host-to-device length copy or new length-generation kernel.
Every strict model check reads back both views and compares them against the
CPU positions plus one; this readback is qualification-only.

At the first admitted step in each phase, a full physical page comparison checks
that **all** current requests share the declared prefix. This initial sync is
charged. Subsequent steps check on the CPU that request IDs are a subset of the
bound cohort and preserve their slot and prefix-overwrite epoch. Consequently
survivor reordering and odd pairing do not require another device readback.
Any new request, slot/epoch change, or alias mismatch falls back to native. The
proof scope depends on vLLM updating the gathered table through the monitored
block-table path; this is not a general cache-concurrency guarantee.

Graph capture supplies dummy native lengths of one. A separate low-cost CUDA
qualification must check that capture at this length, then original-buffer
updates and real replay, is safe. Production admission always requires a length
strictly greater than the shared prefix.

## Strict versus production-only setup

`setup_mode='strict'` captures witness, checked, and production candidate graphs,
only for enabled exact-B descriptors. Native graphs remain available for all
fallbacks. Strict diagnostics allocate max-B Q/O snapshots for all48 layers.

`arm_raw_pair_qualification(max_steps, per_batch_steps=8)` retains the actual
native token trajectory while comparing checked and production candidates at
identical states. It observes at most eight states for each naturally occurring
eligible B, then continues native generation until other batch sizes appear.
Checks cover Q/O, hidden, local/gathered logits, current KV slots, native AA,
unchanged inputs, and native length/table aliases. Numerical mismatches are
reported after both TP ranks complete the phase. Only batches actually checked
and passed become candidate-eligible; missing sizes remain native.

`setup_mode='production_only'` allocates no diagnostic Q/O snapshots and captures
only production candidate graphs. It requires explicit offline strict receipts
covering every enabled B on **every** TP rank with matching worker/runtime source
hashes and model/device/engine fingerprint. Union across multiple documents is
allowed per rank. The fingerprint excludes document-specific prefix length;
physical prefix and native split/alias contracts are enforced at runtime.
These provenance checks are not receipt authentication or an exhaustive proof.
No production-only performance claim is available until that mode is actually
run and compared. The initial driver uses strict setup.

## RPC and records

RPC names remain configure/capture/set/finish_dense_cascade plus
`arm_raw_pair_qualification`. Configuration adds `enabled_batches`, `setup_mode`,
and (production-only) `offline_qualification_records`.

Finish reports `qualified_batches`, `qualification_states_by_batch`,
`captured_layers_by_batch[mode][str(B)]`, `execution_fingerprint`, source hashes,
and every actual prepare/qualification/fallback record. Production dispatch to
an unobserved B records `batch_not_yet_model_qualified`. Preparation records
separate the initial `prefix_binding` from later CPU checks and record
`native_GPU_lengths_aliased=True` / `host_length_copy=False`.

CPU validation: `scripts/check_native_raw_pair_general_setup.py` checks original
B versus padded B split selection, state capacity, descriptor restrictions, and
21 negative admission/provenance cases. Its output explicitly remains GPU
unqualified. Full-model tests and timing are separate root-scheduled work.
