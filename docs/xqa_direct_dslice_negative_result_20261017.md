# D64 prefix partition: qualified numerics, negative performance

Decision: close the frozen D-slice variant `f397e8e360618405`; do not extend
it to GQA4 quad, sweep tile shapes, or integrate it into the model. Both real
cases fail the predeclared performance gate by a large margin. This negative
result does not invalidate the existing direct-input mechanism.

Independent CPU audit: `scripts/audit_xqa_raw_direct_dslice.py` and
`results/xqa_raw_direct_dslice_independent_audit_v1.json`. The audit recomputes
the randomized paired medians/bootstrap intervals, verifies frozen source
hashes, all stored numerical checks and the actual CUPTI kernel grid/counts.
The raw result JSON files are in `results/xqa_raw_direct_dslice_v1/`.

| Actual capture, B8 | Native hot us | Current direct hot us | D-slice hot us | Direct / D-slice, paired 95% CI | Cold direct / D-slice |
|---|---:|---:|---:|---|---:|
| train1/r0/L15 | 208.589 | 139.674 | 211.354 | 0.6604 [0.6596, 0.6615] | 0.7642 |
| train7/r1/L47 | 230.912 | 166.912 | 235.725 | 0.7082 [0.7074, 0.7089] | 0.7966 |

All numbers are complete standalone O-only calls in CUDA graphs, not model
or request latency. Thirty randomized blocks per regime compare native PDL,
native AA, current direct, and D-slice on the same inputs. Native AA paired
median is 1.000 in both hot cases and its confidence interval contains one.
Cold single-replay CUDA event samples show visible quantization, sometimes
giving degenerate bootstrap bounds; the hot multi-replay conclusion is much
larger than this resolution and agrees in direction with cold measurements.

Both cases passed every full-state U/m/z/hint bit comparison against current
direct, all finite checks after NaN poisoning, six declared cut boundaries,
native AA and native-output bits, independent FP64 tolerance, changed-Q and
unequal-length graph replay, restoration and poisoned replay. This is bounded
empirical correctness, not a universal proof or a memcheck result. Following
the stopping rule, this performance-negative candidate receives no further
memcheck or full-model expansion.

CUPTI confirms the intended mechanism actually executed. Candidate prefix
grid is `(10,4,4)` = 160 CTAs, control prefix is `(10,2,4)` = 80, and both
tails are `(10,2,8)` = 160. Candidate prefix has 152 registers/thread versus
control 168; both allocate 79,488B shared memory. Every arm's kernels belong to
captured graph nodes; native/AA each have one XQA kernel, both candidates have
two and no packing or metadata-copy kernel. The sampled prefix durations are
203.360 vs 131.136us and 224.543 vs 159.487us; tail samples remain about
13–14us. These profiled samples only locate the regression in prefix work and
are not substituted for the paired timing estimates.

The result is consistent with the predicted cost of duplicated QK/K work and
additional scalar/barrier work outweighing the extra CTA parallelism. No
hardware traffic, achieved-occupancy or stall counters were collected, so the
logical `2K+V` traffic estimate is **not a measured bandwidth diagnosis**. The
counterexample to the original performance hypothesis is simpler: doubling
grid coverage and reducing registers did not improve complete-call latency.

Avoiding duplicated QK while increasing PV parallelism would require sharing
the narrowed P tiles plus all original rescale/control metadata between CTAs,
or a separate materialized-P pass. That is a different and substantially more
complex experiment: a B8/Hq16/P120K BF16 P plane alone is about 30.9MB, with
additional writes/reads, state scheduling and another launch. XQA's backward
hint feedback also prevents assuming that generic QK materialization is an
equivalent implementation. We do not recommend building this without a new
measured bottleneck and a complete cost estimate. Existing native async
prefetch should likewise be the control for any proposed pipeline change.

The most defensible next action is to use the already queued full-model trace
and the separate TP/model geometry evidence to select the next bottleneck,
rather than interpreting idle grid slots as an automatic optimization target.
