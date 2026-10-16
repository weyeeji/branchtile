# Classical reverse-delete/refill control, CPU v1

Protocol and code were committed as `dafb210` before natural score computation. All 96 frozen cells completed; no decode Q, KV or answer text was consumed. CUDA stayed uninitialized (0 GPUh). This is a classical optimization control, not a new algorithm or task-quality result.

Each deletion refills every affected request with its best remaining optional token, preserving the original per-request cardinality, all protected positions and the original per-request score floor. Original-union and full-universe starts, one fixed normalized-score order and all four epsilon values remain separate. A final independent `fsum` of selected values enforces each original inequality. This is a one-pass heuristic, not an exact solver or an approximation guarantee.

| Loss allowance | Starting candidates | Union reduction min / median / max | Selection wall seconds, 12 cells |
| --- | --- | --- | --- |
| 0 | original_union | 0.00% / 0.00% / 0.00% | 0.084 |
| 0 | all_positions | 0.00% / 0.00% / 0.00% | 0.538 |
| 0.001 | original_union | 3.52% / 4.13% / 5.28% | 0.747 |
| 0.001 | all_positions | 2.22% / 2.71% / 4.04% | 4.146 |
| 0.005 | original_union | 10.50% / 11.45% / 14.77% | 0.776 |
| 0.005 | all_positions | 6.99% / 8.82% / 14.36% | 4.171 |
| 0.01 | original_union | 15.59% / 17.70% / 20.44% | 0.794 |
| 0.01 | all_positions | 11.28% / 15.62% / 20.44% | 4.212 |

At 1%, the original-union control reduces common-source union by 15.59–20.44% and the full-universe control by 11.28–20.44%. The original exchange screen has 17.65–22.99% (ascending) or 18.12–22.87% (descending). These ranges do not imply per-cell dominance; the frozen paired raw cells are retained. Full-universe reverse-delete can perform worse because irrevocable deletion order is a heuristic. No result-selected order or extra pass was added.

The two distinct-score tiny fixtures demonstrate exchange stagnation: a triangle of two-owner supports has union 3 while feasible optimum 2; an initially unselected common token gives feasible optimum 1 while exchange remains at 2. Reverse-delete reaches those cardinality lower bounds, including the exact 1% boundary. This does not establish optimality on natural data.

Data/run process CPU time was 16.028s and wall time 16.030s, excluding imports/self-tests. Peak process RSS was 519.29MiB. Source hashes are inherited through the exact frozen exchange protocol hash. Full source cache remains retained; physical capacity released is zero.

Zero epsilon uses an explicitly reported unique-top-k identity shortcut; its CPU time is not a general zero-loss optimization benchmark. The first pre-freeze fixture attempt exposed an overly conservative numerical guard at the exact boundary; it was corrected and both fixtures passed before any natural score computation. Natural execution completed without retry or implementation changes.
