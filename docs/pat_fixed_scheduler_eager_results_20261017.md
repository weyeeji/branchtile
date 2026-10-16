# Repaired PAT versus corrected native-state sharing: bounded eager evidence

Independent audit `scripts/audit_pat_fixed_perf_results_v1.py` verifies13 completed runs,647 numerical checkpoints across four arms, and three memcheck logs with zero errors. This comprises six allocation-lifetime qualifications (two real captures × multistream/default caller, single-stream/default caller, multistream/nondefault caller), four matched-eager measurements, and three sanitizer checks. Every run performs30 repeated complete calls before admission. All FP64 checks pass; native AA and grouped-hint RawPair additionally match native int16 outputs exactly. PAT has normal numerical differences and is not required to match native bits.

The baseline is **PAT compat fixed scheduler**, a local compatibility repair of pinned upstream `8cb067f`. Original failed schedules and stream-patch experiments remain preserved. The repair changes split-slot numbering and caller/child stream dependencies while preserving packed query sets, KV token partitions, tile choices, CTA ordering and arithmetic CUDA objects. The timed build removes diagnostic NaN initialization and retained temporary tensors. It must not be described as unmodified upstream PAT.

## Complete eager calls, hot repeated regime

All figures below use one fixed mode across inputs, the repaired multistream PAT path. Ratios are medians of paired randomized-block ratios, not ratios of the displayed separate medians. Brackets are paired-bootstrap95% intervals. No per-input mode oracle is used.

| Real capture | Shared prefix | Native / PAT / corrected RawPair, μs | RawPair speedup over native | RawPair speedup over PAT |
|---|---:|---:|---:|---:|
| train1, rank0, layer15 |120528|198.77 /225.35 /158.02|1.257[1.252,1.262]×|1.427[1.423,1.434]×|
| train7, rank1, layer47 |144576|240.55 /267.30 /183.32|1.313[1.310,1.319]×|1.459[1.457,1.462]×|

Synchronized-wall ratios corroborate these event results: RawPair/PAT advantages1.421× and1.453×, and native advantages1.254× and1.310×. Single-stream PAT is retained as a separate control: RawPair/PAT ratios1.535× and1.517×. The headline uses the faster multistream implementation, not the weaker single-stream choice. The repaired PAT is numerically valid but slower than native on these two measured shapes; this does not characterize all PAT workloads or platforms.

Each measurement has30 randomized paired blocks,5 eager calls per hot sample, equal warmup before each arm, and a numerical gate outside timing after every block. Included PAT costs are allocation, forward kernels, stream/event setup and joins, and gather. Included RawPair costs are Q packing, group page/length copies, prefix state production and native tail/merge. Native PDL and native AA use matching inputs, page identities, table width, precision and caller stream. No arm uses a CUDA graph.

## Cold regime and profiler evidence

A separate256MiB flush precedes each cold single call. RawPair/PAT ratios are1.214× and1.253×. However, native-AA cold confidence intervals are wide (approximately0.90–1.14 and0.90–1.12), so precise cold native speedup and small differences between runs should not be emphasized. Hot native-AA intervals contain1 and are narrow.

CUPTI correlation maps every kernel launch to its annotated arm. Native/AA each contain one XQA kernel. RawPair contains three copy kernels plus the prefix and tail XQA kernels. PAT train1 contains four forward tile kernels and one gather; train7 contains three forward tile kernels and one gather. Multistream forward kernels use separate child streams; gather starts after all complete. Single-stream runs place all PAT kernels on the caller stream. Every recorded kernel has graph ID0. Profiler durations and questionable SM120 occupancy estimates are not used as speed or occupancy evidence.

## Scope and setup costs

The two inputs are real captured BF16 B8 states, not synthetic queries or fake replicated requests. Their Q, original native cache strides/physical pages and full table width are preserved. They differ in document, TP rank and layer, but do not constitute broad model coverage or an end-to-end result. Current evidence cannot establish natural output latency or universal bit equivalence; model validation remains separate.

Prepared-metadata timing excludes tree construction and metadata upload. In the multistream runs, PAT CPU tree construction is approximately56–62μs, scheduling146–155μs and synchronized upload212–232μs. Raw allocation/initial physical validation/upload is approximately31ms. These are setup components, not costs silently amortized into a first-use claim. An online GPU-to-host page-table transfer is not measured or claimed free. Repeated eager timings also include host launch gaps and should not be mixed with graph timings.

The next predeclared expansion uses the remaining14 cells of the two documents × two ranks × four captured layers, always the multistream mode and the same protocol. No GPU jobs are launched by the manifest-preparation step.
