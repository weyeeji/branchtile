# Remaining schedule-preserving operator headroom

Status: CPU analysis and compiled bounded prototype only, 2026-10-17. No GPU
performance or correctness result is claimed for the new D-slice prototype.

## Measured evidence versus estimates

The existing matched-graph trace
`results/pat_fixed_graph_performance_v1/train1_r0_l15_multi.trace.json` records
native grid `(10,2,8)` = 160 CTAs, shared prefix `(10,2,4)` = 80 CTAs, and native
tail 160 CTAs. All use 256 threads and 79,488 bytes shared memory; native/tail
have 168 registers/thread and prefix 167. The actual board has 170 SMs. These
are CUPTI launch/resource observations, not inferred from latency.

Register and shared-memory footprints each prevent two of these CTAs from
residing on one SM. NVIDIA's current guide lists 64K 32-bit registers, 48
resident warps, 128KB shared memory per SM and 99KB per block for SM120. Thus
one such CTA means eight resident warps on that SM. The trace's zero achieved
occupancy estimate is unsupported and is **not** used as a measurement.
[NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html).

Keeping the actual native split count `S=floor(170/(B*Hkv))` in these long
contexts gives the following grid coverage upper bounds; this is neither a
measurement of active SMs nor a throughput prediction.

| Mapping | B | S | Native CTAs | Prefix CTAs | Prefix CTAs / 170 |
|---|---:|---:|---:|---:|---:|
| GQA8 pair | 6 | 14 | 168 | 84 | 49.4% |
| GQA8 pair | 7 | 12 | 168 | 96 | 56.5% |
| GQA8 pair | 8 | 10 | 160 | 80 | 47.1% |
| GQA8 pair | 19 | 4 | 152 | 80 | 47.1% |
| GQA8 pair | 21 | 4 | 168 | 88 | 51.8% |
| GQA4 quad | 4 | 10 | 160 | 40 | 23.5% |
| GQA4 quad | 5 | 8 | 160 | 64 | 37.6% |
| GQA4 quad | 6 | 7 | 168 | 56 | 32.9% |
| GQA4 quad | 7 | 6 | 168 | 48 | 28.2% |
| GQA4 quad | 8 | 5 | 160 | 40 | 23.5% |

The trace's sampled native/prefix/tail durations are 201.952/125.088/11.999us.
These profiled samples identify a long prefix and much shorter tail; they are
not the randomized performance estimates or causal whole-model fractions.
Existing paired measurements in that experiment are approximately
194.970us native and 145.037us complete copied-input raw continuation. The
newer direct-address mechanism has separate qualified measurements. The JSON
`results/xqa_schedule_preserving_headroom_v1.json` retains the trace hash,
resource records and computed grids.

## One bounded hypothesis: partition independent output columns

Split each original shared-prefix CTA into two CTAs, each owning one existing
D64 PV warp fragment. This doubles the GQA8 B8 prefix grid from 80 to 160 while
keeping the original sequence split S=10. The head dimension remains 128 for
QK. No sequence reduction is split further, and the native tail and final
merge remain unchanged.

Both CTAs execute the complete original QK producer path, scalar max/sum,
per-native-group backward-hint update and synchronization. Partition p alone
loads V and executes PV MMA for consumer warp `warpX % 2 == p`. Unowned warps
still execute scalar feedback and arrive at the original barriers; they skip
only their V payload and PV MMA. BF16/head128/beam1 has `GRP_LOAD_V=false`, so
those V buffers and async-copy groups are warp-local.

Global state has its existing `[pair, Hkv, S, warpX, lane, 35]` layout. Fields
0:32 are U fragments and have the unique writer `partition=warpX%2`.
Partition zero alone writes fields 32/33/34 (m/z/backward hint) for every warp
and lane. A CPU enumeration verifies exactly one writer for every state word.
The kernel boundary assembles the state before the unchanged tail reads it;
there is no arithmetic cross-partition merge.

The tradeoff is material: relative to current shared prefix, K payload and QK
work double, V payload and useful PV work remain the same. Idealized logical
KV traffic becomes `2K+V` instead of `K+V`, i.e. 1.5x when K and V are equal
size. Actual L2/GDDR traffic can differ. More CTAs do not establish a benefit;
the hypothesis is that additional memory/concurrent-warp capacity outweighs
this duplication. Dynamic branch and unchanged shared-memory allocation can
also erase the gain. It is acceptable for the experiment to close this idea.

### Frozen falsification protocol

The first implementation supports BF16, head128, GQA8, unit cache scales and
actual B8 only. Two original real captures are train1/rank0/layer15 and
train7/rank1/layer47, with original physical page IDs and fused-cache strides.
No FP8 cast or synthesized larger cohort is used. The optional synthetic
case and appended synthetic tail are explicitly labeled.

Before timing: native PDL-on and native AA outputs must match bitwise; both
current direct and D-slice must match native and pass the existing independent
FP64 tolerance. Every raw state field must equal the direct control bitwise
and be finite after NaN poisoning. Declared cuts include 1, S-1, S, S+1, 339
where in range, and the full shared tile prefix, so empty splits and the
previous backward-hint trigger are not hidden by a final reconvergence.
Changed Q and unequal tail lengths are replayed through unchanged graphs,
then restored and poisoned/revisited. Any numerical or ownership failure
stops that job before performance claims.

Four complete-call arms share the same state: native PDL-on, native AA,
current direct, and D-slice. Randomized paired 30-block CUDA graph timings
include both prefix and tail; hot samples contain ten replays and cold samples
one replay after a 256MiB flush. All setup and Python prepare is outside these
operator timings and reported separately where measured. CUPTI captures each
arm outside the performance window. No kernel-sum result is called E2E.

Stop if either real case fails numerical qualification or if D-slice does not
beat current direct with paired lower 95% bound above 1.02 in both hot cases
without a material cold regression (lower bound below 0.98). This is a
development triage rule, not a paper acceptance threshold. Only a survivor
gets memcheck, the minimal hidden-hint regression, larger coverage and actual
whole-model testing. No quad extension or tile sweep follows a negative gate.

Prototype: `src/xqa_raw_direct_dslice_{build,runtime}.py`; build receipt
`results/xqa_raw_direct_dslice_build_v1.json`; runner
`scripts/qualify_benchmark_xqa_raw_direct_dslice.py`; locked-job manifest
`configs/xqa_raw_direct_dslice_v1_jobs.json`. CPU compilation initialized no
CUDA context. Failed preliminary source-anchor/PATH attempts did not execute
GPU work; the final build receipt covers the successfully compiled sources.
CPU-only `cuobjdump` of those compiled binaries reports 152 registers/thread
for D-slice G16 versus 168 for current direct G16, both stack64/local0. This
does not permit two 256-thread CTAs in the 64K register file; original dynamic
shared allocation also remains. Exact binary hashes and output are retained
in `results/xqa_raw_direct_dslice_binary_resources_v1.json`.

## Deferred alternative: on-chip prefix-to-tail continuation

Fusing private tail into a shared-prefix CTA could remove one launch and raw
state traffic, but still must compute both private rows, retain native hint
control and native split merging. Sequential private rows would also reduce
tail parallelism and could increase live registers. Even removing all of the
sampled 12us tail from a roughly 137us prefix+tail kernel span would be less
than 10% operator improvement; that unrealistic deletion is not an E2E bound.
The real removable part is only launch/state traffic. Defer this larger patch
until full-model traces show enough residual handoff cost.

## Prior-art boundary

Output-column partitioning is not new. Taming explicitly separates output
tiling from accumulator reduction order and discusses attention's additional
softmax, narrowing and mask contracts. The implementation hypothesis here is
limited to preserving XQA's concrete backward feedback and raw fragment
ownership while exposing additional SM parallelism. Successful tests would
not constitute a universal bitwise proof.
[Taming Bitwise Behavior](https://arxiv.org/html/2609.11356v1).

GraniKV's HOT contiguous prefix/COLD token-level suffix storage, dual backend
and per-step dispatch already cover asymmetric layouts and heterogeneous
prefix dispatch. Its authors distinguish kernel reuse from their storage
contribution; the fixed-output high-concurrency evaluations are not directly
our natural shrinking-cohort task. This experiment retains the existing
paged cache and has no HOT migration or claimed new prefix-sharing principle.
[GraniKV](https://arxiv.org/html/2608.15584v1).

LLM-42 verifies a fast path against a deterministic schedule with rollback;
our target is the currently selected native implementation and its schedule,
without verification during production decode. That distinction does not
establish novelty by itself. ChunkAttention/PAT/CoDec already establish shared
prefix operator reuse; the project must retain actual native-output and
end-to-end evidence in addition to any small new dataflow improvement.
[LLM-42](https://arxiv.org/html/2601.17768v2),
[ChunkAttention](https://arxiv.org/html/2402.15220v2),
[PAT](https://arxiv.org/html/2511.22333v3),
[CoDec](https://arxiv.org/html/2505.17694v2).
