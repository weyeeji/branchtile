# Lane-select split-bank compiled artifact audit

Wave14 passes independent CPU review of all 4 jobs, 20 compiled cases and
120 saved FP64-reference / mutable-graph checks. Every job has a clean exit 0.
The raw qualification status `resource_review_pending` is preserved; this
separate audit supplies the resource review. No GPU was launched by this audit.

| Batch | Tile N / splits | Cases | Registers/thread | Total SMEM/CTA | LOCAL / STACK / LDL-STL sites |
|---|---|---:|---:|---:|---:|
| 3 | 32 / 3 | 5 | 120–164 | 29,952 B | 0 / 0 / 0 |
| 3 | 64 / 64 | 5 | 150–200 | 48,384 B | 0 / 0 / 0 |
| 4 | 32 / 3 | 5 | 122–174 | 29,952 B | 0 / 0 / 0 |
| 4 | 64 / 64 | 5 | 152–208 | 48,384 B | 0 / 0 / 0 |

Fresh `cuobjdump --dump-resource-usage` and `--dump-sass` exactly match all
saved artifacts. Host C and CUDA metadata independently reproduce every
128-thread launch, grid dimension and dynamic shared-memory allocation.
Dynamic SMEM is 28,928 B for N32 and 47,360 B for N64; each cubin also has
1,024 B static shared memory. All qualification source hashes agree across
the four jobs and match the reviewed files.

Generated C++ declares the four persistent canonical banks `cu0..3[8]`
before the region loop, separately from the M16/M32 MMA output arrays
`out[16]` / `out_1[32]`. Each producer has 24 local canonical-bank read sites,
six local MMA-output-to-slot copy sites, two sets of B fixed-bank scatter
sites and B final global U stores directly from canonical banks. No CU,
OUT-slot or MMA-output reference shares a source line with a shared-memory
address. Data-dependent request/label variables select buffers through
branches; they do not index the canonical arrays. Cubin LOCAL/STACK zero and
no SASS LDL/STL support the absence of observed spills for these cases.

CM/CL still gather/scatter through shared memory using RowMap. Q/K/V/P
storage, cross-warp reduction scratch and synchronization also remain.
This removes canonical shared-U transport; it does not remove all shared
traffic, or establish that removed traffic was the performance bottleneck.
Static instruction-site counts are not executed traffic or stall measurements.

The saved numerical receipts include eager, graph baseline and replay after
Q, KV, indices, and offsets/indices mutation. Eager/graph baseline outputs
are bitwise equal in all 20 cases. This audit checks the saved checks and
their input/output hashes; it does not recompute GPU outputs. The original
combined absolute/relative tolerance remains the acceptance rule.

Whole qualification wall time is 66.9–75.8 s per job. Cold-signature
`initialization_and_first_call_seconds` is 14.8–17.9 s and includes allocation,
compilation and first execution. The unequal fixture reuses the compiled
signature and reports about 0.18–0.21 ms. Neither quantity is isolated JIT
duration or steady-state operator latency; do not compare the warm fixture
with cold signatures as evidence of a compiler improvement.

The earlier wave11 attempt remains four failed jobs, each exiting 1 before
any numerical check or compiled resource receipt. Its single-buffer code
hit TileLang's same-buffer/different-index restriction (`cu: [i + 8, d] and
[i, d]`). CPU mathematical routing had passed that source but did not prove
compiler legality. Wave14 qualifies the changed, separate-buffer source
`6c3a98f`; it does not retroactively qualify `b7c7666`. Wave11's
0.007849 GPU-hours and wave14's 0.083482 GPU-hours remain charged separately.

This is an initial grid: N32/S3 and N64/S64 at B3/B4, each with all_masks,
unequal, empty, one_token and one real capture. Other tuning points and
runtime Labels changes are outside this qualification. No speedup,
full-model latency, task-quality, energy or novelty conclusion follows.

Reproduce in the project virtual environment, without a GPU:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_lane_select_gpu_artifacts.py
```

`audit.json` records per-case source evidence, launch/resource re-extraction,
artifact SHA256s, complete terminal receipts and retained failures. Candidate
source files and raw qualification results are not changed.
