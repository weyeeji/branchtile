# Natural large-working-set boundary: CPU audit

168 geometry rows reconstruct unchanged suffix-only QC supports across three documents, two ranks, four layers and seven requested budgets. All 96 existing mask restorations pass; no GPU was initialized. Full source and private tails remain charged.

| Document | Budget | BF16 unique payload MiB | Hypothetical FP8 maximum MiB | Best fixed / dynamic logical MMA |
| --- | ---: | ---: | ---: | ---: |
| docfinqa256 | 65536 | 66.31–70.52 | 35.26 | 1.0046–1.0297 |
| docfinqa256 | 131072 | 85.04–85.04 | 42.52 | 0.9993–0.9993 |
| docfinqa256 | full | 85.04–85.04 | 42.52 | 0.9993–0.9993 |
| publicdev1_256 | 65536 | 66.22–71.82 | 35.91 | 1.0063–1.0304 |
| publicdev1_256 | 131072 | 87.96–87.96 | 43.98 | 0.9996–0.9996 |
| publicdev1_256 | full | 87.96–87.96 | 43.98 | 0.9996–0.9996 |
| publicdev6_256 | 65536 | 72.65–100.97 | 50.48 | 1.0127–1.0654 |
| publicdev6_256 | 131072 | 134.88–153.93 | 76.97 | 1.0053–1.0329 |
| publicdev6_256 | full | 169.60–169.60 | 84.80 | 0.9996–0.9996 |

The long B4 report has an untested BF16 single-call footprint above 96 MiB. This is a real boundary of prior small-budget measurements, but capacity alone does not imply cache misses or a dynamic-kernel advantage. At 128K the optimistic logical MMA difference against the best fixed partition is only about 0.5–3.3%; this is not a runtime upper bound. FP8 byte-volume estimates remain below 96 MiB even at full budget (84.8 MiB); actual FP8 operands and numerical behavior were not tested here. Full-budget public context has a single owner set, removing the core variable-row opportunity.

These results do not authorize an automatic reopening of the stopped dynamic application branch. Exact hierarchy work was not recomputed in this CPU audit; that stronger control remains required before a new performance claim. No new acceleration mechanism has qualified.

Reproduce with `CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_natural_capacity_boundary.py`. Input SHA and all fixed-partition Pareto points are retained in audit.json.
