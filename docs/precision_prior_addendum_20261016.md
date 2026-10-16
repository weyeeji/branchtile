# Precision-based alternatives: primary-source check, 2026-10-16

This is a literature constraint, not a local reproduction. No GPU was used and
neither method is adopted as a new contribution of this project.

**ReadKV (8 October 2026).** Progressive stored codes permit query-dependent
key-channel and value-token reads. Its objectives already combine the importance
of query heads sharing a physical KV head; ordinary shared precision allocation
is therefore not a new idea here. Crucially, flexible-reader quality and restricted
GPU-reader performance are different configurations. Appendix F.8 identifies a
substantial low-rate quality cost from imposing equal K/V rates. In Appendix G's
tested A10G layer matrix, Dense is faster than all ReadKV configurations; the
reported advantage is against the tested TurboQuant codec. Cache construction
and the rest of the model are outside that timing. These results justify checking
precision-aware baselines but do not establish that replacing this project's
sparse masks with a progressive reader improves complete application cost.
[Primary paper, Sections 3, 5 and Appendices F.8/G](https://arxiv.org/html/2610.11245v1)

**Minima-KV (24 August 2026).** Retaining every logical page while using FP8 for
recent/anchor pages and packed low precision for stale pages is prior art, as is
merging their attention contributions with one global normalization. Its page
transition protocol already accounts for reader and conversion references. The
paper separates materializing quality tests, physical fused decode, and deployment
accounting: they cannot be assembled into one verified configuration. In particular,
the fused canary is one throughput pair, with no repeated-run uncertainty and an
unbound dense-control KV dtype. Attention scoring is disabled in the reported
Qwen profiles. Consequently neither “retain an approximate background” nor
“ownership-safe shared compressed pages” alone distinguishes a new proposal.
[Primary paper, Sections 3–6 and 8](https://arxiv.org/html/2608.23834v1)

For this project, a precision change would require its own source/cache identity,
calibration cost, physical layout, strong native quantized control and natural
quality evaluation. The present exact-mask FP64 qualification cannot certify an
approximate codec, and reduced payload bits do not by themselves prove lower
latency. This check authorizes no GPU experiment or change to the frozen evaluation
set. PQ-HSA and HiSparse are reviewed separately against their public source code.
