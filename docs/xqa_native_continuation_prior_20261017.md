# Native attention continuation: prior art and state contract

The useful hypothesis is narrow: reuse shared KV across two requests while preserving the actual selected native XQA execution's per-row arithmetic and final merge. It is not generic deterministic inference, a new online-softmax recurrence, or the first two-stage shared-prefix attention.

[ChunkAttention, ACL 2024](https://arxiv.org/html/2402.15220v2), §3.2, Equation 1 and Algorithms 1–2, already saves **unnormalized** output numerators with row maxima and denominators for shared chunks, then combines these with sequence-specific chunks. Its partial chunks are computed independently and merged. Therefore merely exporting unnormalized prefix state and processing private suffixes is established prior art. The candidate distinction would be preserving a specific production kernel's unreduced live fragments and arithmetic schedule instead of recombining independent chunk results.

[AWS Neuron Attention CTE](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/attention-cte.html), checked 2026-10-17, documents `skip_output_normalization` with cached softmax state for cross-step reduction. Deferring normalization and exposing raw state are existing interfaces, not sufficient novelty.

[Taming Bitwise Behavior, September 2026](https://arxiv.org/html/2609.11356v1) already studies preserving arithmetic descriptors while optimizing GPU kernels. [Vosti, September 2026](https://arxiv.org/html/2609.38981v1), §5 and Appendix A, gives row-level equivalence reasoning, fixed-operation attention and explicit preservation of inactive row state. A bounded XQA comparison cannot claim its formal guarantees, arbitrary batch invariance or prefill/decode equivalence. Our prospective contract is replacement equivalence for the same native engine state and native launch policy.

## Installed SM120 XQA state that must survive a kernel boundary

Source: `.venv-engine/lib/python3.12/site-packages/flashinfer/data/csrc/xqa/mha.cu`; frozen source hash is recorded by the real-capture runner.

1. Preserve original `S=nbSubSeqPerSeq`, split identity and tile sequence `s,s+S,...`, 256-token tile boundaries, masking, original QK/PV instruction shapes, probability casts and final split merge.
2. `gemm1WarpsPerGrp=2` and `gemm1NbWarpGrps=2` for head dimension 128. Preserve the consumer warp/group accumulator fragments (`accs`, lines around 2404 and 2547), not a prematurely reduced single output vector. Preserve corresponding live `globalRowMax` and `globalRowSum` values before normalization or final merge.
3. `CTA_ROW_MAX_BACKWARD_METHOD=1`. Each QK producer loads its particular `smem.ctaRowMax[y][warpIdx.x]` hint (around 2056–2060). Consumers update a slot only when `idxXTile==warpIdx.x` (around 2500–2503). These four hints need not equal the final global maximum. Replacing them with a final maximum can change the next tile's probability rounding. Save and restore exact hint values after draining the original producer/consumer handoff.
4. Reinitialize synchronization machinery for the new launch, but restore value-carrying state. A barrier count is not an arithmetic value; pending K/V/X work must be drained consistently before the checkpoint. The next tile must be neither skipped nor processed twice.
5. For a prefix ending inside a 256-token native tile, stop the shared phase before that tile. Resume the mixed tile using each request's full original logical addresses and masks. Retain each request's actual length.

The first falsifiable gate is native GQA8 → state export → native GQA8 continuation against uninterrupted native GQA8. This isolates state completeness before adding pair packing. Only after exact equivalence should GQA16 shared-prefix → GQA8 private-tail continuation be tested. Include state IO and extra launches in graph timing. Neither stage is a performance result until run.
