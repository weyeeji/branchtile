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
