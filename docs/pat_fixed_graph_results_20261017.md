# Independent matched CUDA Graph comparison

The two qualification runs and four performance runs pass the independent CPU audit in `results/pat_fixed_graph_independent_audit_v1.json`: 448 four-arm numerical checkpoints, including 30 repeated qualifications per run, three input-update cases, and every measured block's outside-timing check. All arms pass the same FP64 tolerance; native AA and grouped-hint raw continuation additionally equal native bits. PAT is allowed its own rounding and is not required to match native bits.

Each arm captures its complete operator call once and then measures only `graph.replay`. Explicit eager calls occur during setup and outside-timing changed-input qualification. Changed Q, one-token truncation within an existing private CTA, and restoration all reuse the same graphs and pointers, and each graph equals its own eager reference bit-for-bit. This does not demonstrate arbitrary PAT tree replanning or sequence growth beyond its existing assigned topology.

The primary comparison uses the repaired PAT multistream mode, without selecting a winner separately for each input. Hot-cache caller-stream event medians and paired bootstrap ratios are:

| Actual capture | Native XQA (us) | PAT (us) | Raw continuation (us) | Native/raw, paired 95% CI | PAT/raw, paired 95% CI |
|---|---:|---:|---:|---|---|
| train1, rank0, layer15, P120528 | 194.970 | 218.938 | 145.037 | 1.342 [1.334, 1.354] | 1.510 [1.504, 1.513] |
| train7, rank1, layer47, P144576 | 234.157 | 261.370 | 169.408 | 1.385 [1.380, 1.390] | 1.544 [1.540, 1.547] |

The corresponding synchronized-wall PAT/raw ratios are 1.503 and 1.540. Single-stream PAT remains a separately reported control (hot PAT/raw 1.590 and 1.591). Cold-cache multistream PAT/raw is 1.313 [1.301, 1.321] and 1.368 [1.357, 1.375]; native/raw is 1.198 [1.183, 1.236] and 1.241 [1.228, 1.277]. Cold single-call AA uncertainty is wider than hot AA, so small cold differences deserve caution.

All four sampled CUPTI traces associate every GPU kernel with one of four `cudaGraphLaunch` calls. Each arm has one graph launch and no eager kernel launch, memcpy, or memset inside its annotated call. Every kernel has nonzero graph and node IDs. Native and AA each contain one XQA kernel. Raw contains the original three copy kernels plus two XQA kernels. PAT contains four forward kernels plus gather for train1, or three forward kernels plus gather for train7. Multistream execution uses distinct child streams, and gather starts after every forward finishes. Single-stream uses one stream. Profiler durations and SM120 occupancy estimates are not used as performance evidence.

Thus the observed operator advantage survives removal of PAT's per-call eager host overhead. This is not a whole-model result, a proof over every input, or a measurement of the newer direct-input two-launch candidate. It covers two actual B8 states, whereas the earlier eager audit covers sixteen. Prepared tree/metadata upload is separately recorded, and capture/setup costs are not silently included in or inferred from replay time. The artifact has no graph-specific memcheck; earlier repaired eager single/multistream/nondefault-stream memchecks passed.

PAT here is explicitly a **local compatibility implementation**: stable split-slot numbering and caller/child stream dependencies were repaired; arithmetic, tile choices, physical KV partitions, and CTA order were preserved. Original upstream failures and unsuccessful intermediate fixes remain archived. Do not label these timings as an unchanged upstream PAT implementation.
