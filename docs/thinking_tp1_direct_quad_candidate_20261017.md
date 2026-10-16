# TP1 direct-quad: separate Hq32/Hkv8 qualification

The native capacity/natural pilot completed with the whole eight-question input and no preemption or output cap. Its low financial-answer score remains a task-quality limitation. No TP2 speed or numerical qualification transfers to TP1.

`src/xqa_raw_tp1_direct_quad_runtime.py` is an independent adapter using the already built `src/xqa_raw_direct_quad_build.py` modules. These modules take KV head count at runtime. The new adapter supplies Hkv8, native Q stride32 heads, original native S=floor(SM/(B*8)) bounded by native table extent, ceil(B/4) prefix groups, original-B tail and raw state sized by groups×8×S×4480. The grouped backward-hint predicate still uses four rows per original request. It directly aliases native Q/table/length storage and invokes only prefix and native tail/merge. The TP2 adapter and assertions remain untouched.

CPU fake-module checks cover B1..8, head-address mapping, state/ABI sizes, native S, input pointers and physical-prefix rejection. This is not GPU evidence. TP1 benchmark B1..3 is diagnostic only: no TP2 negative or positive dispatch result is imported. Full-model deployment scope will be frozen only after measured native versus complete-call results.

First collect native actual Q/KV/O on layers 0/11/23/35 (and all36 layer Q/O), with own TP1 model, eager native execution and full original page table/strides. The first eight capped inspection tokens use the same natural sampling configuration; this is a capture diagnostic, not task-quality timing. All8 real independent queries and actual native outputs are retained; no copied single Q is substituted.

```bash
# Root assigns one exclusive GPU; suggested timeout300s.
.venv-engine/bin/python scripts/capture_thinking_tp1_native.py --document-id docfinqa-680456070a0a0714 --output results/thinking_tp1_native_capture_v1
```

The capture CPU preflight verified exact prompt hashes and CUDA remained uninitialized. Per-layer captures are approximately 0.5GB each and are raw local artifacts, not git blobs. Qualification next compares fresh same-B native PDL-on, AA, PDL-off, unshared continuation and direct quad bit patterns; original capture O is a full-B8 check only. FP64 rows, dummy-length-one graph capture followed by real lengths, swapped membership, poisoned state, and actual private-tail growth are required. Complete-call timing includes both kernels and native merge; setup and capture are reported separately. No whole-model candidate experiment is eligible until these new Hkv8 gates pass.

The independent GPU qualification and timing runner is now frozen. Suggested first jobs (root allocation only):

```bash
# A short synthetic B8 qualification plus native/AA/PDLoff/unshared controls.
.venv-engine/bin/python scripts/qualify_benchmark_tp1_direct_quad.py --batch 8 --prefix 120528 --blocks 30 --output results/tp1_direct_quad_gate_v1/synthetic_b8.json
# Once TP1 capture completed, use its actual original page layout and O.
.venv-engine/bin/python scripts/qualify_benchmark_tp1_direct_quad.py --capture results/thinking_tp1_native_capture_v1/rank0/dense_capture/layer35.pt --batch 8 --blocks 30 --output results/tp1_direct_quad_gate_v1/actual_l35_b8.json
# Meaningful odd/long-tail stress; this synthetic extension is labeled as such.
.venv-engine/bin/python scripts/qualify_benchmark_tp1_direct_quad.py --capture results/thinking_tp1_native_capture_v1/rank0/dense_capture/layer35.pt --batch 7 --extra-tail 8192 --blocks 30 --output results/tp1_direct_quad_gate_v1/actual_l35_b7_tail8192.json
```

Allow 180 seconds for a normal case, 300 seconds for compute-sanitizer (`--tool memcheck --error-exitcode 99`) on B7. First actual O/FP64 qualification precedes any model integration. The runner verifies capture summary/worker/file SHA provenance, preserves original B8 O parity, recomputes all subset native outputs, checks graph replay after Q/length changes and reversed cohort membership, poisons raw state, rejects foreign physical prefix and retains original native PDL-on as the performance baseline. Thirty randomized paired blocks report hot and eviction-controlled measurements. No timing winner is selected as a per-input policy.

## Independent whole-model qualification

The new TP1 bank/worker uses Hq32/Hkv8, all 36 actual layer views and a single shared arena across B4..8. Each layer aliases the native contiguous Q/O pointers, full-width native page table and int32 device lengths. The fingerprint includes the TP1 native worker, independent runtime/bank and independent qualification scope; actual TP2 direct-quad receipts are rejected. CPU checks exercise all 180 layer/batch pointer selections and reject noncontiguous Q/O. The new B4..8/P120528/private<=8192 scope is a bounded qualification region, not a copied TP2 performance policy.

After TP1 actual-state operator and memcheck gates pass, root can run one GPU (initial timeout900s):

```bash
.venv-engine/bin/python scripts/qualify_thinking_tp1_quad_model.py --document-id docfinqa-680456070a0a0714 --output results/thinking_tp1_quad_model_qual_v1
```

This driver is qualification-only: it retains the original native generation trajectory and checks up to eight naturally observed states for each B4..8 against native AA, a native witness graph, checked candidate and production candidate. It verifies every layer Q/O, full hidden/logit output, current KV slots, original inputs/positions/table and native metadata aliases. It restores native slots before continuing. Missing states remain unqualified; a successful first set is not a proof for all later states. No fixed/natural performance arms are launched by this driver. The whole-model receipt must pass before a separate matched TP1 production experiment can support a speed claim.
