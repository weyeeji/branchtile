# Independent persistent-fixed wave6 compiled-artifact audit

All seven wave6 jobs exited cleanly. Their 35 cases (all masks, unequal lengths,
empty, one-token, real capture) passed six saved numerical/graph phases each:
eager, graph baseline, changed Q, changed KV, changed indices, changed offsets
and indices. This audit preserves the original resource-review-pending status
and supplies a separate completed review; no qualification result was edited.

Reproduce using CPU only:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_persistent_fixed_gpu_artifacts.py
```

The audit checked all 35 generated CUDA, cubin, resource-text and SASS hashes,
then independently ran CPU `cuobjdump` on each cubin. Every fresh resource and
SASS dump matched its saved artifact exactly. All 13 qualification source hashes
are identical across seven jobs and match the reviewed current files. Input
hashes, original receipts/terminal rows, source lines and static instruction-site
counts are retained in `audit.json`; no Torch or TileLang was imported.

| Compiled group | Registers/thread across fixtures | Dynamic SMEM | Static SMEM | Total SMEM |
|---|---:|---:|---:|---:|
| B3, N32 | 112/114/116 | 27648 B | 1024 B | 28672 B |
| B4, N32 | 112/116 | 27648 B | 1024 B | 28672 B |
| B3, N64 | 150/154 | 46080 B | 1024 B | 47104 B |
| B4, N64 | 142/146 | 46080 B | 1024 B | 47104 B |

Every cubin reports LOCAL=STACK=0, and every independently decoded SASS has
zero LDL/STL sites. These shapes have no explicit local spill traffic. This
does not establish low register pressure, occupancy or speed for other shapes.

Generated CUDA confirms the intended persistent state path: per-thread
`out[32]`, `maximum[4]`, `denominator[4]` arrays are initialized before owner
execution. Within owner execution, out is only rescaled or passed as the PV
MMA accumulator; canonical state never appears in a shared-address expression.
Final U writes directly from out after region execution. Q loads occur before
owner execution. There is no RowMap/canonical shared gather/scatter. In the
seven all-empty cases there is only one region and the compiler elides the
extent-one region loop; the runtime empty-shard guard and state path remain.
The other 28 cases retain the region loop and fixed state across its iterations.

Shared Q/K/V/P, two cross-warp reduction scratch paths and synchronization still
exist. SASS LDS/STS are therefore expected. This review does not attribute each
SASS site to a tensor or interpret static sites as executed bytes or stalls.

The 210 saved checks passed the frozen combined output atol=.006, rtol=.015 and
natural-LSE atol=.0003, rtol=.0001. Largest saved absolute output error is
.007635121; individual elements pass the combined absolute-plus-relative
tolerance, not a uniform .006 absolute ceiling. Largest finite LSE error is
.00000538548. Eager and graph-baseline O hashes agree in all cases; changed
Q/K/V hashes establish real mutation. Labels-vector mutation was not tested.
These receipt checks do not recompute GPU outputs in the CPU audit.

Source qualification jobs consumed .071414270047 GPUh including initialization
and compilation. This audit consumed zero GPU hours. Qualification alone gives
no timing, full-model/task-quality, energy or novelty result. Persistent M32
uses additional masked MMA work relative to compact dynamic FullCol, so the
comparison is a strong fixed-row control, not a pure state-transport-only change.
