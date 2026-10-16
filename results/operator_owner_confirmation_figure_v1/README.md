# Wave25 fixed-owner confirmation figure

`fixed_owner_paired_speedup.png` and `.pdf` show the eight completed wave25 capture cells, with separate hot and artificial-cold panels. Horizontal position is the median of paired `best_owner_control / owner_frozen_128_64` latency ratios. Above 1 favors owner. Bars are the original 95% percentile block-bootstrap intervals, independently recomputed from raw random/reverse timings, not a ratio of marginal timing medians.

The owner C128/N64/threads128 configuration is predeclared and fixed in both caches and every cell. Its comparator is the same-cell, separately hot/cold-selected oracle over all 47 bounded E0 native/E1 native/E1-own/E4 controls. No earlier wave or most favorable cross-wave point is substituted. Other dynamic engineering controls are not part of this reduced native-confirmation protocol.

These eight inputs come from one development document: TP ranks0/1 × layers0/15/31/47; layers0/31 use natural anchor16 and layers15/47 use anchor256. All four requests are present, each retaining its own actual prediction step/position. These are not eight independent documents or a formal held-out evaluation. Forty paired random/reverse timing blocks and 10,000 bootstrap draws give within-cell timing intervals, without multiplicity correction or cross-document inference.

Cold single-call event timings are coarse. A dagger marks a degenerate empirical interval, including [1,1]; its zero width is not evidence of exact equality or equivalence. The raw bounds are neither widened nor replaced. Both A/A controls remain in the raw JSON and independent audit; they are not extra document samples.

Both producer and merge, BF16 output and natural-log FP32 LSE are included in each owner call. Preparation, search/JIT and topology recapture are not included in the GPU event statistic. They cannot be inferred from this figure. No application latency, quality, lifecycle, traffic or energy claim follows.

`metadata.json` retains exact plotted numbers, every input JSON SHA256, the plot script SHA256, frozen benchmark/operator source hashes, audit/protocol-erratum hashes and output hashes. The plotter validates all 384 configuration receipts, the fixed arm, the unique one-document identity, cell coverage and raw block statistics before drawing. Original JSON files are read only.

Reproduce with a fresh output directory, normal uid and the project virtual environment:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/plot_owner_fixed_confirmation.py --output-directory results/operator_owner_confirmation_figure_recheck
```
