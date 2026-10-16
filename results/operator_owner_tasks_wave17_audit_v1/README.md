# Wave17 owner-task qualification: independent audit

The eight qualification jobs passed all **40 cases × 6 phases = 240 FP64 O/natural-LSE checks**. The CPU audit independently checks frozen source commit `aac3e85`, protocol commit `d7ace33`, all eight start/end ledger pairs, the complete phase sequence, mutation hashes, topology rejection/rebuild receipts, and immutable capture provenance. Original qualification receipts remain unchanged: `resource_review_pending` denotes their original review state, not a numerical failure.

All **144 launched compiled-kernel roles** have zero Triton spills, cubin LOCAL/STACK, and SASS LDL/STL. All 576 saved PTX/cubin/resources/SASS hashes match. Fresh CPU `cuobjdump` of all **38 unique cubins** exactly reproduces each corresponding saved resource and SASS dump. The 16 empty producer roles correctly report `not_launched`, with no fabricated artifacts; their merge kernels run and produce empty O/LSE semantics.

| Role | Compiled roles | Registers | Static shared | Dynamic shared | Total shared |
|---|---:|---:|---:|---:|---:|
| Producer, N32 | included in 64 | 123 | 1,024 B | 18,432 B | 19,456 B |
| Producer, N64 | included in 64 | 128 | 1,024 B | 28,672 B | 29,696 B |
| Merge | 80 | 40 or 42 | 1,024 B | 2,048 B | 3,072 B |

The maximum recorded O absolute error is 0.00763512, and maximum finite natural-LSE error is 4.92699e-6. Qualification uses elementwise `atol + rtol × abs(reference)` (`O`: 0.006/0.015; `LSE`: 0.0003/0.0001); an O maximum above absolute tolerance alone does not mean the combined contract failed. These are the preserved GPU FP64 comparisons, not a new CPU recomputation of attention.

All 40 cases reject `validate_plan()` on the stale operator after offsets mutation; none replays that stale graph. Q/K/V and original Indices values are tested in place under the existing graph. Changed owners are checked only after constructing a new plan/operator and capturing a new graph. `graph.replay()` bypasses host guards, so this is **not** the earlier v4 same-graph offset-update contract. Eight empty cases intentionally increment offset version even when offset values remain equal.

The eight clean terminal jobs cost **0.0477035164 P1 GPUh** (171.732659 allocated GPU seconds); the audit costs zero GPUh. The 52 per-job capture-file checks cover 19 unique files and agree with the earlier immutable-capture audit. Later source changes are reported separately from the frozen run-time source hashes.

This is numerical/topology/resource qualification. Zero spills does not establish occupancy or performance. Preparation and rebuild/recapture diagnostic wall costs are preserved in `audit.json`, including JIT warmup, without treating them as calibrated latency results. `chunk_tokens` defines tasks; `splits=64` is a compatibility parameter. No application quality, natural-EOS latency, or strong-baseline acceleration claim follows.

Reproduce with a fresh output directory, normal uid, and the project virtual environment:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/audit_owner_tasks_wave17.py --output results/operator_owner_tasks_wave17_audit_recheck
```
