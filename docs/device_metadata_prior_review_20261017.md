# Device metadata is a necessary baseline repair, not an established new mechanism

The planned FA2 device-metadata experiment addresses a real implementation problem: our first cascade baseline checks the entire shared page table with a synchronous GPU-to-CPU result, copies prefix IDs every step, and rebuilds private-tail plans. Removing those costs is worthwhile before judging cascade end-to-end performance. However, direct primary sources already cover GPU index construction, synchronization-free plan preparation, stable graph buffers, and even fully GPU-generated attention schedules. A claim based only on moving metadata to GPU would be too broad.

## Direct sources, dated and pinned

**SGLang, source commit October 14, 2026.** Its [`fast_prefill_plan`](https://github.com/sgl-project/sglang/blob/df1a089cd39e02ad56eb5dfe5e561a77cfb93cbe/python/sglang/srt/layers/attention/flashinfer_backend.py#L164) accepts already-known host scheduling inputs, updates persistent graph buffers on device, and calls the cached native planner without reading those inputs back from GPU. The backend applies fast planning to decode and specified draft/verification paths. Elsewhere it constructs cumulative indices on device and writes directly to a graph wrapper's index buffer. This is direct precedent for eliminating redundant transfers and per-replay synchronization; it does not prove support for our particular shared-prefix cascade contract.

**FlashInfer GPU planner, source commit August 27, 2025.** The official [`blackwell/plan.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/87e669de13d94a95b06780abbf6df711ff29a74c/include/flashinfer/attention/blackwell/plan.cuh) contains a GPU planner that derives work counts and query-tile/head/batch task mappings from sequence lengths, including a PDL launch option. This is stronger precedent than merely storing CPU-computed plans on GPU. Its existence is not proof that the planner is directly compatible with SM120 FA2; porting or specialization can still be necessary engineering.

**FlashInfer discussion, September 2025–March 2026.** In [discussion #1743](https://github.com/orgs/flashinfer-ai/discussions/1743), a maintainer points to that GPU planner. A speculative-decoding implementation later describes preparing multiple steps' metadata together, avoiding repeated readbacks and using persistent device buffers. Thus both GPU planning and amortizing repeated speculative-step planning have public precedents.

**FlashInfer MLSys 2025.** Its [paper](https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf) already separates scheduling from execution, places plans in stable device workspaces, and uses fixed-grid kernels for CUDA graphs. Merely reusing a work queue across layers or retaining graph-compatible pointers is not a new principle.

**vLLM, March 2026.** The official [Triton backend discussion](https://vllm-project.github.io/2026/03/04/vllm-triton-backend-deep-dive.html) describes fixed-grid persistent kernels reading GPU metadata to assign dynamic work. It also explains wasted waves under captured grids. Generic persistent scheduling or avoiding inactive padded tiles is therefore covered.

**PersistentKV, June 2026.** The [paper](https://arxiv.org/html/2606.26666v1) already explores direct native block-table consumption, row-local effective lengths, and compact nonempty work queues. Its benchmark uses synthetic traces on an RTX 3060 and older baselines, so it is precedent rather than proof of performance on this workload. A new claim cannot rest solely on avoiding a CSR repack or compacting ragged tasks.

## Installed implementation evidence

The installed vLLM FlashInfer backend already has `_compute_flashinfer_kv_metadata`: host cumulative page counts, GPU page-index copying, and last-page updates. Installed FlashInfer `fast_decode_plan` removes several redundant buffer operations but still obtains `last_page_len` on CPU when page size exceeds one. Our page size is 16. This explains why a specialization may still help without implying that upstream lacks metadata optimizations altogether.

Source hashes and dates are recorded in `results/platform/device_metadata_prior_20261017.json`; `scripts/audit_device_metadata_prior_cpu.py` reproduces the source snapshot check. This audit uses no GPU and changes no installed library.

## What remains worth measuring

First qualify the device-resident metadata path through page boundaries, batch shrinkage, and real request replacement. Replay correctness requires more than stable pointer addresses: live page ownership, request identity, lengths, masks, and the captured scheduler topology must remain valid. Any state that invalidates the specialization must select an appropriate graph or rebuild before execution. A stale page table is not an acceptable way to remove overhead.

Then compare the same mature FA2 arithmetic with and without the redundant host preparation, counting all device update kernels in full model wall time. If most of the gap disappears, the contribution is a stronger baseline and an integration result. Kernel timing alone cannot demonstrate the benefit of eliminating a host synchronization.

A further operator contribution would need a different measured dataflow or work decomposition that remains beneficial after these repairs. The cohort-by-speculation-depth study is a reasonable opportunity test because it changes useful query reuse, but packed prefix queries already have PAT and multi-level cascade precedents. It must show a specific effect that existing sharing, ordinary batching, and standard speculative verification do not already obtain. No such new effect is established by this metadata audit. Do not launch another kernel variant solely to replace one small metadata kernel with a differently named one.
