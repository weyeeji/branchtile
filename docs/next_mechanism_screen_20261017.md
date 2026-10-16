# Beyond prefix sharing: current screening does not yet establish a new mechanism

A causal copy-draft diagnostic finds useful sequential-step headroom in the current workload, but the obvious mechanisms already have close precedents. This report does not promote a new paper idea or claim an end-to-end speedup.

## Measured workload opportunity

The two completed native-FP8 natural-generation runs contain 20,316 tokens over 16 questions. A fixed four-token lookup key proposes the next four tokens from the most recent matching location in the full prompt or the request's own already committed replay history. The proposer never receives future output or sibling answers. Replaying observed token strings takes 12,437 rounds, or 1.634 output tokens per replay round. Using only the request's past output gives 1.460.

These are sampled native trajectories. Exact string matches are not rejection-sampling acceptance probabilities: a real speculative sampler can follow a different trajectory, and its correction distribution matters. The replay also excludes verification cost, rollback, CPU lookup, scheduler behavior, and request-tail imbalance. Consequently 1.634 is not a measured speedup and not a general speculative-decoding upper bound. It identifies a concrete low-cost test worth considering only after identifying a distinct mechanism: native prompt lookup versus ordinary decoding with identical sampling contracts, then measure actual accepted length and full verification-round latency. A four-token verify pass plus all overhead must cost less than the observed committed-token count times the decode-step cost to help.

Reproduce with `CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/screen_copy_draft_opportunity_cpu.py`. The JSON receipt records original summary hashes and individual questions; prompt token hashes are verified against the real generation records. CUDA remains uninitialized.

## Prior-art boundaries

- [SuffixDecoding, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf) already uses suffix-tree history reuse for agent workloads. History-based drafting alone is not a new contribution.
- [ASPIRE, COLM 2026 author page](https://amir-zsh.github.io/ASPIRE/) describes independent per-request draft/verify scheduling, mixed forwards, and a full-attention refresh layer. Its reported evaluation is decode-only on H100. Generic asynchronous sparse self-speculation is therefore not an unexplored mechanism, and its speedups do not transfer automatically to this B8 MoE/SM120 setup.
- [SpecStream, September 2026](https://arxiv.org/html/2609.33184v1) combines committed-history KV tiering, streamed verification, and target-priority drafting on shared GPUs. Simple overlap or commit-aware cache management is already covered.
- [AgentSpec, August 2026](https://arxiv.org/html/2608.24004v1) addresses agent batch speculation using semantic structure and redundancy-aware budgets. [VeriCache, May 2026](https://arxiv.org/abs/2605.17613) uses lossy caches for drafting with full verification. These further narrow any claim based only on agent structure or compressed-cache drafts.
- [SplitZip, May 2026](https://arxiv.org/abs/2605.01708) already studies lossless KV transfer compression, including FP8. [ZipServ, ASPLOS 2026](https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf) provides hardware-aware lossless compression with fused decompression. A proposed lossless FP8 layout would need measured entropy, attention consumption cost, and a genuine difference from these mechanisms before GPU implementation.
- [Runtime-Certified Bounded-Error Quantized Attention](https://arxiv.org/abs/2605.20868), May 2026, already targets online error bounds and precision fallback. Replacing fixed sparse thresholds with an error certificate is not, by itself, a new mechanism.

The existing native MoE audit also shows that batch-aware grouping and split-K routing are already active. No evidence currently supports relabeling those optimizations as a new proposal. The immediate independent experiment remains the bounded probability-components test, whose CPU resource increase is explicitly a warning against assuming fewer casts means faster execution.
