# GQA4 follow-up operator audit

All14cells pass source/capture provenance,recorded numerical and graph-update checks,and independent paired-CI recomputation. No GQA8 policy was reused.

| Kind/cell | B | S | Maximum private tokens | Hot speedup [95% CI] | Cold speedup [95% CI] |
|---|---:|---:|---:|---|---|
| survivor/[6, 0] | 6 | 7 | 49 | 1.390 [1.388,1.437] | 1.279 [1.277,1.287] |
| survivor/[6, 8192] | 6 | 7 | 8246 | 1.134 [1.132,1.139] | 1.089 [1.088,1.095] |
| survivor/[7, 0] | 7 | 6 | 49 | 1.389 [1.383,1.430] | 1.307 [1.299,1.307] |
| survivor/[7, 8192] | 7 | 6 | 8246 | 1.240 [1.229,1.261] | 1.177 [1.171,1.183] |
| survivor/[8, 0] | 8 | 5 | 49 | 1.403 [1.399,1.428] | 1.338 [1.338,1.345] |
| survivor/[8, 8192] | 8 | 5 | 8246 | 1.260 [1.240,1.272] | 1.209 [1.205,1.210] |
| thinking/[0, 0] | 8 | 5 | 51 | 1.414 [1.389,1.449] | 1.338 [1.331,1.345] |
| thinking/[0, 11] | 8 | 5 | 51 | 1.444 [1.407,1.449] | 1.339 [1.331,1.345] |
| thinking/[0, 23] | 8 | 5 | 51 | 1.409 [1.387,1.443] | 1.338 [1.332,1.341] |
| thinking/[0, 35] | 8 | 5 | 51 | 1.347 [1.342,1.388] | 1.345 [1.338,1.345] |
| thinking/[1, 0] | 8 | 5 | 51 | 1.395 [1.389,1.431] | 1.338 [1.336,1.339] |
| thinking/[1, 11] | 8 | 5 | 51 | 1.454 [1.414,1.458] | 1.342 [1.338,1.345] |
| thinking/[1, 23] | 8 | 5 | 51 | 1.421 [1.391,1.445] | 1.338 [1.333,1.343] |
| thinking/[1, 35] | 8 | 5 | 51 | 1.401 [1.356,1.416] | 1.345 [1.338,1.350] |

The weakest long-tail cell is B6 cold:1.089×,95% paired lower bound1.088×. All tested endpoints remain positive. This supports a bounded development model experiment at P120528 and private≤8192; it does not establish performance at other prefixes,models,batches or every intermediate length. Natural survivor states still require full-model qualification.
