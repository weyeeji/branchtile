# GQA4 survivor operator gate: B6/B7/B8

The independent `RawGQA4PaddedPlan` retains original request count and native split selection at each fixed batch. On SM170 the original S is7,6,5 at B6,7,8 respectively. Only prefix request groups use ceil(B/2); the tail remains the original B. B7 pads one zero-query row for prefix work, with the last real request's physical prefix pages. It has no separate orphan decode and no repeated full-history scan. Native GQA4 raw-state mapping and compiled CUDA modules are unchanged.

Actual captures contain eight independent real queries at one autoregressive state. The new runner selects rows `[7,0,6,1,5,2,4,3][:B]`, retaining original cache stride/physical IDs and table width. This is a real-state subset diagnostic, not later natural survivor queries. B6/B7 compute a new same-B native reference: original B8 captured outputs used S5 and must **not** be asserted bitwise equal to the changed native S7/S6. Original O parity is required only for full B8 without appended tail.

All five arms remain: production native PDL-on, native AA, native PDL-off, unshared raw checkpoint and complete paired call. The candidate includes odd padding,query packing,pair metadata D2D,raw state export/resume and final native merge. Both native/candidate are checked against FP64 on first/last request; candidate/native bytes must match for every output. A graph captured with aliased native int32 length1 is replayed after changed Q/unequal lengths and actual request permutation, with old raw state poisoned. The zero dummy remains zero. This checks fixed-address updates and new pair membership without asserting unlimited plan reuse.

`--extra-tail 8192` appends explicitly synthetic private KV to the real prefix,Q and captured private tokens, with new random physical mapping and declared stride. It is a growth stress case, not natural long-tail output. The frozen oldGQA8 dispatch thresholds are not reused. Results must guide a separate GQA4 deployment decision.

Root-assigned single-GPU CLI:

```bash
FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=6 .venv-engine/bin/python scripts/qualify_benchmark_xqa_raw_gqa4_survivors.py --capture results/gqa4_native_capture_train1_v1/rank0/dense_capture/layer11.pt --batch 7 --output results/gqa4_survivor_operator_v1/train1r0l11b7.json
```

Repeat with B6/B8 and distinct output paths, then extra-tail8192 for B6/B7 if initial gates pass. Default30paired blocks×10hot replays and1cold replay; all prepare/capture times separate. CPU compile/help checks passed; no GPU launch was performed while preparing this path. Whole-model survivor integration requires another independent version and actual natural shrinking-batch qualification.
