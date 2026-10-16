# Wide FullCol (256 threads): independent artifact audit

This CPU-only audit reconstructed resources and launch parameters for all **23 saved cubins** from qualification waves 13 and 15. All fresh `cuobjdump` resource and SASS outputs exactly match their saved counterparts. Hashes of CUDA, cubin, host C, TVM metadata and SASS also match the numerical receipts. The audit launches no GPU work and imports neither Torch nor TileLang.

**Seven jobs completed successfully; one remains failed.** The wave13 `fullcol_w13_b3_n64_s3_t256` process exited with **-11 (SIGSEGV)**. Its raw JSON still says `running`, but the terminal ledger is decisive: only four synthetic fixtures completed, and `real_capture` was never qualified. The 24 saved checks from this process remain partial evidence. They do not turn its failed terminal status into a pass.

The other three wave13 jobs completed five fixtures each. The four wave15 jobs completed one real capture each, using four distinct capture/split/batch targets. None is the failed B3/S3 configuration, so wave15 does not repair or retry that failure. In total, **138 saved numerical/graph checks passed**, comprising 114 checks in clean-exit jobs and 24 partial checks in the failed job.

| Batch | Actual launch block | Dynamic SMEM | Static SMEM | Total SMEM | Registers/thread | LOCAL / STACK / LDL / STL |
|---|---:|---:|---:|---:|---:|---|
| B3 | 256 × 1 × 1 | 61,632 B | 1,024 B | 62,656 B | 114–124 | All zero |
| B4 | 256 × 1 × 1 | 65,792 B | 1,024 B | 66,816 B | 114–124 | All zero |

The actual packed host-call slots independently establish the launch block and dynamic SMEM; TVM metadata establishes the order of those slots. The generated CUDA also declares `__launch_bounds__(256, 1)`. All grid dimensions agree with each case's head count and split count. These are saved-compiler-artifact facts, without a new kernel launch.

The generated CUDA retains two `RowMap`-indexed canonical-U gathers from shared memory, two scatters back into shared memory, and a final shared-to-global U store. Canonical U begins at float offset 48 (B3) or 64 (B4), with a payload of 12,288 or 16,384 bytes. **This remains the FullCol shared-U control, not the register-bank mechanism.** LDS/STS are present in SASS; their static site counts also include Q/KV/P and reduction traffic and are not physical traffic measurements.

All eight runs used the same currently frozen FullCol kernel and wrapper hashes. Wave13 records older hashes for the unrelated lane-select candidate, while wave15 records its current hashes; the report preserves these differences rather than asserting all source snapshots are identical.

Each fixture passed eager FP64 comparison and five graph phases: baseline, changed Q, changed KV, changed indices, and changed offsets plus indices. Eager and baseline graph O hashes agree. The comparison used `O: atol=0.006, rtol=0.015` and natural-log `LSE: atol=0.0003, rtol=0.0001`. The maximum recorded O absolute error is 0.00763513, which must be interpreted under the combined absolute-plus-relative contract, not as a standalone 0.006 cutoff. The maximum finite LSE error is 4.92699e-6. The checks are repeated correctness phases, not independent task-quality samples.

The eight qualification jobs consumed **0.07868751 GPU-hours**, including the failed process; this audit consumed **0 GPU-hours**. Initialization-and-first-call and whole-job durations are retained as receipts, but include work beyond JIT and do not isolate compilation cost. Spill-free cubins alone establish neither throughput nor occupancy. There is no speed, application-quality, energy or multi-rank critical-path claim here.

Reproduce with the project's non-root virtual environment:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_wide_cta_gpu_artifacts.py
```

The machine-readable evidence, selected terminal ledger records, complete source-hash review, launch reconstruction, generated-CUDA line evidence, resource fields and artifact hashes are in `audit.json`. Original qualification result statuses are unchanged.
