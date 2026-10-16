# Direct-input whole-model trace: launch and interval audit

The short train26 B21 trace passes the independent launch audit on both TP
ranks. Each arm generated 16 fixed output tokens on the same original task;
15 decode model graphs per rank were profiled. The remaining prefill and
sampling activity is not included in the decode-graph accounting below.
The driver checked token and graph-trajectory agreement. This is an
instrumented diagnostic, not an E2E performance experiment.

Every native model graph has 48 XQA nodes; every candidate graph has 96,
exactly two per layer. All candidate XQA kernels have nonzero graph/node IDs
and correlate with the actual `cudaGraphLaunch`. There are no mixed missing
native/candidate layers. Whole graphs contain 1300 native versus 1348 candidate
kernels: the difference is exactly the 48 additional XQA calls. Both retain
48 normal `reshape_and_cache_flash_kernel` updates and one `memcpy32_post`
per model graph. The direct bank's source/capture-address contract removes
its Q/O and prefix metadata copies; this does not remove framework KV updates
or general copies from the model.

The two original audit outputs are
`results/raw_direct_model_trace_independent_audit_v1.json` and
`results/raw_direct_model_trace_interval_analysis_v2.json`. The latter uses
interval unions as well as sums and records per-graph data. Its category
assignment is based on symbol names, not semantic attribution.

Median per-decode-graph observations, milliseconds:

| Method/rank | First-to-last kernel span | Sum of kernel durations | Union of all kernel intervals | XQA union | XQA/quant interval overlap |
|---|---:|---:|---:|---:|---:|
| Native/r0 | 34.901 | 60.836 | 34.295 | 26.059 | 26.003 |
| Native/r1 | 34.854 | 61.146 | 34.265 | 26.385 | 26.329 |
| Direct/r0 | 25.550 | 25.998 | 24.936 | 16.882 | 0.519 |
| Direct/r1 | 25.549 | 26.004 | 24.929 | 17.099 | 0.523 |

Native standalone token-quant kernel sums are 26.230/26.554ms per graph,
while direct gives 0.719/0.724ms. Nearly the entire native attention interval
overlaps those quant intervals. This is consistent with native PDL launching
dependent quant kernels early and their elapsed duration including dependency
waiting; it does **not** establish that the candidate optimized quantization
by 25ms. That interpretation would double-count the long attention interval.

Other named components are much less changed: GEMM/reduce duration sums are
about 5.06–5.15ms, with interval unions about 4.52–4.60ms; NCCL sums about
1.16–1.60ms; other quant/fused-quant sums about 0.87–0.88ms. Normal cache/copy
kernels are approximately 0.08ms. These categories overlap in time and their
individual medians need not sum to the median graph span. Uncovered intervals
inside the graph span are about 0.60–0.62ms; they are not identified as host
overhead by this measurement. Profiled CPU `cudaGraphLaunch` duration itself
is about 1.60–1.79ms and is heavily affected by instrumentation.

The trace substantiates that attention remains a large part of an early,
full B21 decode graph and that the integrated candidate really uses its
intended two-kernel path. It does not quantify the natural-EOS cohort's tail,
the complete generation walltime, or a causal removable fraction. There is
no dependency-graph critical-path reconstruction or counter-based compute
occupancy analysis. Production ABBA results and natural batch distributions
remain the source of E2E claims.
