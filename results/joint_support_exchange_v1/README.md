# CPU joint-support exchange screen v1 results

Protocol and implementation were committed before computation as `e771346`. The frozen 132-row run completed without a retry or implementation change. All 10 input selector files are corrected `suffix_v2`, rank 0, layer 15; all three documents, 4096/16384 budgets, natural requests (3/3/4) and both KV heads are included.

**This is a positive logical feasibility signal, not a task-quality, attention-fidelity, novelty, GPU-speed or capacity result.** At 1% allowed pooled-score loss, both fixed request orders reduce common-source logical union by at least 17% in all 12 document/budget/head cells. Epsilon zero produces exactly unchanged masks in all cells. The two orders remain separate; no best-order deployment rule is selected.

| Epsilon | Fixed request order | Union reduction min / median / max | Owner N64 tile reduction min / median / max | Cells with union reduction ≥15% |
| --- | --- | --- | --- | --- |
| 0 | ascending | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 0/12 |
| 0 | descending | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 0/12 |
| 0.001 | ascending | 6.80% / 7.48% / 8.29% | 5.70% / 7.73% / 8.12% | 0/12 |
| 0.001 | descending | 6.80% / 7.48% / 8.32% | 5.70% / 7.60% / 8.27% | 0/12 |
| 0.005 | ascending | 14.29% / 15.08% / 17.20% | 12.66% / 15.04% / 16.73% | 7/12 |
| 0.005 | descending | 14.29% / 15.48% / 17.20% | 13.13% / 14.89% / 16.54% | 8/12 |
| 0.01 | ascending | 17.65% / 19.48% / 22.99% | 17.10% / 19.10% / 22.64% | 12/12 |
| 0.01 | descending | 18.12% / 20.18% / 22.87% | 18.18% / 19.16% / 22.24% | 12/12 |

The percentages above count only the actual shared common-source interval. The protected partial-page document tail remains request-private and unchanged. The full source cache is still retained; **physical KV capacity released is zero**. Owner-partition tile counts and fixed-union tile counts are different counters, both present in every raw row. Neither is measured kernel time.

| Document | Budget | Head | Original common-source union | 1% ascending union | 1% descending union |
| --- | ---: | ---: | ---: | ---: | ---: |
| docfinqa256 | 4096 | 0 | 6133 | 5009 | 4964 |
| docfinqa256 | 4096 | 1 | 6083 | 4929 | 4931 |
| docfinqa256 | 16384 | 0 | 21968 | 17999 | 17806 |
| docfinqa256 | 16384 | 1 | 21800 | 17952 | 17849 |
| publicdev1_256 | 4096 | 0 | 6200 | 4987 | 4954 |
| publicdev1_256 | 4096 | 1 | 6171 | 4961 | 4921 |
| publicdev1_256 | 16384 | 0 | 22802 | 18378 | 18136 |
| publicdev1_256 | 16384 | 1 | 21940 | 18063 | 17688 |
| publicdev6_256 | 4096 | 0 | 9705 | 7708 | 7704 |
| publicdev6_256 | 4096 | 1 | 8867 | 6938 | 6939 |
| publicdev6_256 | 16384 | 0 | 33708 | 26404 | 26376 |
| publicdev6_256 | 16384 | 1 | 32089 | 24711 | 24749 |

The common-mask formula controls reduce union by 24.85–57.82%, but their worst individual pooled-score losses are 23.18% for max and 18.66% for mean. They therefore cannot be described as satisfying the same 1% constraint. Their geometry matches because each uses one shared mask with the same cardinality and protected tail; their selected identities and per-request scores differ. They are not a QuoKA reproduction.

Measured post-import data/run phase: 2.465 process CPU seconds and 2.465 wall seconds. Process-wide peak RSS was 530,012 KiB (517.59 MiB), including interpreter/Torch overhead. CPU/wall timers start after imports and the initial self-test. Per-variant selection and selection-plus-metrics times, shared input-load time, and partial NumPy state bytes are in the raw rows. This is one CPU execution, not a timing distribution. CUDA stayed uninitialized; GPU usage was 0 GPUh.

Validation independently rechecked unique coverage, all input/output hashes, every per-request loss fraction, zero-epsilon mask hashes, owner histograms, N64 arithmetic, private-tail invariance and equality of accepted swaps to union reduction. Budget/protected-position assertions and protected-top-k checks passed inside the run. `validation.json` records these checks.

Only pooled scores, original supports and suffix-position provenance were consumed from selector files. No step 16/64/256 decode Q, source-KV tensors, task output, reference answer or future query was used. No O/LSE evaluation was performed. Pooled-score loss is not actual attention-mass loss, nor a semantic-quality guarantee; the current 4K task-quality concern is not resolved by this result.

The screen justifies considering a separately frozen attention-fidelity and natural-generation experiment after the main quality grid is assessed. It does not select epsilon or request order for deployment, waive quality gates, claim generality across layers/ranks/documents, or establish algorithmic novelty.

Files: `protocol.json` and `PROTOCOL.md` freeze the procedure; `results.jsonl` contains all 132 rows and per-request scores/index hashes/exchange traces; `summary.json` gives complete provenance, runtime and separately aggregated orders; `validation.json` records the independent arithmetic audit; `run.log` records the three completed documents. Reproduce with the command in `PROTOCOL.md`; the script refuses to overwrite existing results.
