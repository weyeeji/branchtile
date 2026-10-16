# Independent direct-input plus dynamic-cut combination

The new CUDA build starts from the grouped-hint direct-input implementation and changes the prefix-cut scalar ABI to one device pointer. Prefix Q addresses still map original requests/heads; original table/length rows are read directly and missing odd members are zero-filled. Native-group hint control, original S, raw fragments and original native tail/merge remain unchanged. This combination has a distinct build identity and must pass new gates; neither constituent's numerical qualification or performance results transfer.

`RawPairDirectDynamicCutPlan` retains the device-cut module shim and delegates its two-kernel path to the direct-input runtime. Its explicit candidate scope is B>=5; the model dispatch remains the unchanged stricter B6..21 development policy. The multi-document bank selects native contiguous Q/O addresses per layer during capture, with one shared workspace/state arena and one device cut across all shapes. Noncontiguous Q/O is rejected. Document activation preserves original native table/length aliases, synchronizes candidate cut updates only between empty scheduler groups, clears old physical ownership/request-slot epochs and retains qualification by document. Both original development documents and their natural question counts remain fixed.

Independent strict and production workers/drivers use new fingerprints and reject old dynamic-cut or direct-input qualifications. The production path only accepts a completed new strict two-document summary, including both ranks' original native trajectories and audited receipt bytes. Native and candidate use identical cold anchor/APC controls and ABBA document order; all switch/setup costs and UUID-verified energy counters retain their prior explicit scope. No new full-model or performance claim exists yet.

CPU evidence is `results/xqa_raw_direct_dynamic_cut_cpu_build_v1.json` (both modules compiled, CUDA not initialized), `results/raw_pair_direct_multidoc_cpu_v1.json` (all 16 B shapes/48 layers, one cut pointer, native Q/O/table/length aliases, physical ownership and invalid-cut/stride rejection, actual old qualification rejected), `results/raw_pair_direct_multidoc_offline_cpu_v1.json` (document-indexed provenance negative cases), and `results/raw_pair_direct_multidoc_preparation_v2/summary.json` (actual two-document tokenizer input preparation).

## Required root-scheduled gate order

All GPU calls must use the root's exclusive allocation and ledger. The following are CLIs, not launches by this subagent. Set the project virtual environment in PATH, `FLASHINFER_CUDA_ARCH_LIST=12.0`, and assign only idle GPU IDs.

1. Single-GPU graph gate, suggested timeout 180 seconds:

```bash
.venv-engine/bin/python scripts/qualify_xqa_raw_direct_dynamic_cut.py --output results/xqa_raw_direct_dynamic_cut_gate_v1.json
```

This uses seven B values (6/7/8/9/14/19/21), ten device cuts including non-256-aligned prefixes and repeated earlier cuts, changed actual Q/KV/page maps/live lengths, poisoned state and original native PDL-on AA. It checks 70 same-graph outputs, 21 FP64 row references, wrong external cut, invalid atomic updates and foreign physical-prefix rejection. Initial capture also executes with native lengths 1. Failed outputs remain preserved.

2. Independent actual-value regression artifact, single GPU, suggested timeout 120 seconds:

```bash
.venv-engine/bin/python scripts/replay_minimal_raw_group_direct_dynamic_cut.py --output results/xqa_raw_direct_dynamic_cut_counterexample_v1.json
```

The artifact is the preserved real two-query/single-split reduction. It must still reproduce the old grouped-hint failure, while the fixed and new direct/device-cut raw states match native at all 49 cut points and final O. This is an arithmetic regression artifact, not a natural application workload.

3. Single-GPU memcheck, suggested timeout 300 seconds:

```bash
/usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 99 --log-file logs/raw_direct_dynamic_cut_memcheck_v1.memcheck.log .venv-engine/bin/python scripts/qualify_xqa_raw_direct_dynamic_cut.py --batches 7,19 --output results/xqa_raw_direct_dynamic_cut_memcheck_v1.json
```

4. Only after all three gates pass, actual TP2 two-document strict model qualification (suggested timeout 900 seconds):

```bash
.venv-engine/bin/python scripts/benchmark_raw_pair_direct_multidoc.py --qualify-only --output results/raw_pair_direct_multidoc_model_qual_v1
```

5. Only after a completed strict summary, CPU real-receipt preflight followed by root-scheduled TP2 production natural ABBA (suggested timeout 1800 seconds):

```bash
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/benchmark_raw_pair_direct_multidoc_production.py --qualification-summary results/raw_pair_direct_multidoc_model_qual_v1/summary.json --prepare-only --output results/raw_pair_direct_multidoc_production_cpu_v1
.venv-engine/bin/python scripts/benchmark_raw_pair_direct_multidoc_production.py --qualification-summary results/raw_pair_direct_multidoc_model_qual_v1/summary.json --output results/raw_pair_direct_multidoc_production_v1
```

Do not use the old non-direct multi-document summary as qualification. A matched original native comparison remains inside the new ABBA session. Comparing separately obtained old copied-input and new direct-input speed ratios is descriptive until a matched experiment attributes the difference. The objective is to remove both repeated input copies and repeated document graph capture; benefit is not assumed.
