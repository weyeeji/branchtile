# Native E0 physical KV alias control

All eight frozen cells completed cleanly. Numerical O/raw-LSE bitwise checks: **48/48**; graph address-sensitive V mutations: **384/384**. GPU allocation (including imports/initialization/qualification): **0.018620 GPUh**.

Each pair uses one FlashInfer 0.7.2 native page-one sparse E0 wrapper, plan, workspace, query, output/copy destination and index buffer. This is not the vLLM 0.31 native dense page16 XQA configuration. All B physical replicas remain allocated. Only prepared index values change outside the timing interval. Private mappings and shared-copy base addresses both rotate; A/A covers shared0 and private0. The output copy is timed.

| Document | Budget | Shared/private unique payload MiB | Hot private/shared ratio (3 rotations) | Cold ratio (3 rotations) |
| --- | ---: | ---: | ---: | ---: |
| docfinqa256 | 4K | 6.26 / 12.35 | 0.999–1.007 | 1.222–1.222 |
| docfinqa256 | 8K | 11.66 / 24.35 | 0.992–1.008 | 1.364–1.364 |
| docfinqa256 | 16K | 21.67 / 48.35 | 0.995–1.005 | 1.533–1.533 |
| docfinqa256 | 32K | 39.71 / 96.35 | 1.717–1.798 | 1.696–1.773 |
| publicdev1_256 | 4K | 6.37 / 12.37 | 0.998–1.004 | 1.222–1.222 |
| publicdev1_256 | 8K | 11.81 / 24.37 | 0.984–1.017 | 1.364–1.364 |
| publicdev1_256 | 16K | 22.18 / 48.37 | 0.988–1.004 | 1.533–1.600 |
| publicdev1_256 | 32K | 40.80 / 96.37 | 1.725–1.803 | 1.665–1.696 |

The cold result shows that the existing native E0 already benefits from shared physical addresses. At 32K the same is evident in hot repeated calls. Below that hot point, small effects change sign across address rotations; these are not a robust new speedup. The logical private working set at 32K is about 96.4 MiB versus about 40 MiB shared, near this GPU’s 96 MiB L2 capacity, but that coincidence does **not** identify L2 hit rates or isolate cache capacity from other address effects.

This explains why a comparison against private-copy execution can exaggerate the incremental opportunity for a new sharing kernel. It does not prove that all possible fusion gains are exhausted. The earlier dynamic/joint stop decisions remain unchanged.

## Limits

- One real layer/step, two documents; eight devices are eight different cells, not independent replication of every cell.
- Hot repeats one working set. Cold is one call after flush; event timings have prominent 2.048us quantization. Bootstrap intervals omit systematic/quantization uncertainty.
- Different active footprints and address mappings may affect TLB, partitioning and multiple cache levels. No DRAM bytes or L2 hit rate were measured.
- Index installation is outside timing; full source replicas stay allocated in every arm. Not a capacity, energy, end-to-end, or new-method gain.

Raw per-call measurements, source hashes, rotations, A/A, and all confidence intervals are in `analysis.json` and the eight cell files. `job_logs/*.txt` preserves every terminal log. Protocol/code were committed before execution as `6c9c7da`. No source outputs, supports, tolerances or frozen evaluation data were changed.
