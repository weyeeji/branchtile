# Classical full-universe saturated greedy cover control

The classical cover control explains the logical-sharing opportunity without an algorithmic novelty claim for singleton exchange. With the same original scores, mandatory/private positions, exact per-request budgets and epsilon 1%, **cover produces a smaller actual-common-source logical union in all 12 paired cells than ascending exchange, and in all 12 than descending exchange**. This is a geometric proxy comparison; cover's attention and natural generation quality have not been evaluated.

The protocol/script were committed first as `bf1cfef`. All 48 cells completed: 12 exact independent identity controls and 36 positive-epsilon cover cells. The full candidate universe contains every nonmandatory actual-common document position, including positions outside independent masks. Normalized weighted-rank saturated marginals use exact integer arithmetic on the original binary scores and frozen decimal-rational epsilon, with only original token ID breaking ties. Final stable top-k refill preserves exact budgets and mandatory/private positions; every request additionally passes the unchanged original `math.fsum` loss and loss-ratio checks.

## Geometry and constraints

| Epsilon | Cells | Union reduction: min / median / max | Maximum request score-loss fraction | Selection CPU seconds per cell: min / median / max |
|---:|---:|---:|---:|---:|
| 0.000 | 12 | 0.00% / 0.00% / 0.00% | 0.000000000000 | 0.000000 / 0.000000 / 0.001001 |
| 0.001 | 12 | 6.73% / 7.54% / 8.46% | 0.000999996004 | 0.750677 / 1.144808 / 2.863355 |
| 0.005 | 12 | 14.76% / 16.05% / 17.57% | 0.004999923415 | 0.734198 / 0.920190 / 2.722606 |
| 0.010 | 12 | 19.91% / 21.08% / 23.83% | 0.009999977006 | 0.654128 / 0.788033 / 2.476112 |

At epsilon 1%, the 12 cells contain 40 request/head constraints, and all 40 satisfy the original strict bound. Across all positive-epsilon cells, all 120 request/head constraints pass. All 12 epsilon-zero rows retain identical selected-index hashes to the original independent supports. Full-document private-tail positions are never counted as interchangeable shared backing: the primary union below is restricted to the actual common source.

| Fixed control | Epsilon 1%: cover smaller / equal / larger union | Additional union tokens removed by cover: min / median / max |
|:---|---:|---:|
| exchange_ascending | 12 / 0 / 0 | 36 / 166.5 / 763 |
| exchange_descending | 12 / 0 / 0 | 32 / 130.0 / 551 |
| reverse_delete_all_positions | 12 / 0 / 0 | 149 / 341.0 / 2410 |
| reverse_delete_original_union | 12 / 0 / 0 | 149 / 276.0 / 1532 |

Across all three positive epsilon values, cover is smaller in 35/36 cells than each fixed exchange order and larger in 1/36 by seven tokens; there are no ties. It is smaller in all 36/36 than either previously frozen reverse-delete arm (`original_union` and `all_positions`). These are the recorded implementations under their frozen priorities, not proofs that greedy dominates every reverse-delete or exchange algorithm. No per-document best method or order is selected.

## All 12 paired cells at epsilon 1%

| Document | Budget | KV head | Cover union | Ascending exchange union | Descending exchange union | Cover selection CPU seconds | Ascending CPU seconds | Descending CPU seconds |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| docfinqa256 | 4096 | 0 | 4900 | 5009 | 4964 | 0.692379 | 0.007096 | 0.007038 |
| docfinqa256 | 4096 | 1 | 4872 | 4929 | 4931 | 0.683341 | 0.007119 | 0.007188 |
| docfinqa256 | 16384 | 0 | 17291 | 17999 | 17806 | 0.722767 | 0.010931 | 0.010919 |
| docfinqa256 | 16384 | 1 | 17298 | 17952 | 17849 | 0.822176 | 0.010666 | 0.010647 |
| publicdev1_256 | 4096 | 0 | 4906 | 4987 | 4954 | 0.753890 | 0.007244 | 0.007204 |
| publicdev1_256 | 4096 | 1 | 4875 | 4961 | 4921 | 0.654128 | 0.007172 | 0.007171 |
| publicdev1_256 | 16384 | 0 | 17737 | 18378 | 18136 | 0.917475 | 0.011163 | 0.011197 |
| publicdev1_256 | 16384 | 1 | 17300 | 18063 | 17688 | 0.726435 | 0.010797 | 0.010824 |
| publicdev6_256 | 4096 | 0 | 7672 | 7708 | 7704 | 1.725388 | 0.019889 | 0.019797 |
| publicdev6_256 | 4096 | 1 | 6894 | 6938 | 6939 | 1.708064 | 0.019032 | 0.018993 |
| publicdev6_256 | 16384 | 0 | 26180 | 26404 | 26376 | 2.476112 | 0.030825 | 0.030712 |
| publicdev6_256 | 16384 | 1 | 24442 | 24711 | 24749 | 2.465099 | 0.029024 | 0.029082 |

The matched 1% per-cell CPU medians are **0.788033 s for cover**, **0.010864 s for ascending exchange**, and **0.010871 s for descending exchange**. The cover timer includes exact integer/rational setup, the complete candidate heap, marginal reevaluations and top-k refill; each exchange timer includes its frozen exchange call. Both exclude source loading and subsequent geometry reporting. These are separate, unrepeated runs with different system activity, so this is evidence of an implementation cost tradeoff, not a controlled latency benchmark or a claim that the exact-arithmetic implementation is optimally engineered. Whole-run 47.46 s versus the earlier 2.465 s must not be treated as a per-cell speed comparison: they cover different grids and loading/reporting work.

## Qualification, accounting and limits

Sixty-four small fixtures passed before natural execution, including the independent review's triangle and outside-initial-union examples, mandatory common/private tokens, zero-score cardinality, and 60 deterministic random cases. Exhaustive feasible-U enumeration gives optima 2 and 1 for the two counterexamples, both attained. Lazy greedy matches full greedy recomputation's complete chosen sequence and exact marginal numerators in all fixtures. This is implementation qualification, not proof of greedy global optimality or any approximation constant for the natural real-valued cover.

The natural run used **47.428872 CPU seconds and 47.457775 wall seconds after import**, peak RSS **605,952 KiB (591.75 MiB)**, ordinary UID 1000, and the project virtual environment. CUDA remained uninitialized and GPU use was **0 GPUh**. All cells completed within the frozen 600 CPU-second limit; no partial cells or silent retries occurred. `run_summary.json` records the actual scope of these whole-run timers. `results.jsonl` retains every request's original/retained score and selected-index hash, union/owner/N64 geometry, selected-entry count, exact-cover state, CPU/wall time and peak memory. `summary.json` preserves all pairwise cells and the hashes of every joined result source.

Pooled-score feasibility is not actual attention or generation quality. Only one rank and one layer from three documents are examined; heads, requests and epsilon configurations are correlated. The full source KV remain allocated and **physical capacity released is zero**. No decode Q/K/V or model output was read, no GPU was used, and no production selector was modified. More union reduction with a higher preparation cost does not supply a new optimization problem or establish a deployable advantage.

```bash
.venv/bin/python scripts/probe_saturated_greedy_cover.py --self-test
.venv/bin/python scripts/probe_saturated_greedy_cover.py
.venv/bin/python scripts/summarize_saturated_greedy_cover.py
```

The natural run refuses existing output. The summary command is CPU-only and can validate/join existing results without rerunning selection.
