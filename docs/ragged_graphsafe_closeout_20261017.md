# G0 graph-safe qualification and closeout

The two real layer captures (15 and 47) passed all 28 O/FP64 and future-KV isolation checks, plus both growth checks per layer. This is causal attention replay over actual autoregressive Q/KV states, not an actual whole-model speculative verifier.

The same K=32 allocation step survives internal CUDA-graph mode. For layer 15, shared attention is 195.17 / 269.39 / 269.37 us across the three frozen query partitions; layer 47 is 194.94 / 269.42 / 269.07 us. Nonuniform/uniform is about 1.38, versus 1.40 with the finite shape graph pool. Thus conservative graph planning does not remove the effect.

The separately collected layer-15 trace shows the same prefix kernel, grid [170,1,2], 250 registers and 48 KiB shared memory. Its time rises from 179.84 to 255.91 / 257.19 us. The private-tail kernel remains about 11.3–11.5 us. The plan's 170 padded split workitems correspond to 340 grid blocks with two KV heads; they are not 170 CTAs. This is a kernel trace, not a hardware memory-traffic measurement.

Each layer has eight ordinary captures: two dynamic graph arms and six finite-shape arms. Dynamic graphs remain valid for the three allocations and the two tested length increments. Finite graphs safely reuse within a page; a page crossing rejects reuse and explicitly rebuilds. That fallback costs 16.06 / 18.28 ms preparation plus 1.02 / 1.16 ms capture for layers 15 / 47. Dynamic workspace is 512 MiB; three finite plans use 1.5 GiB, with another temporary 512 MiB on fallback. There is no unlimited-plan-reuse claim.

The measured prepare+switch+replay wall times are 6.3–7.3 ms. This conservative Python diagnostic builds/checks large page lists and updates all three wrappers in each arm. It is not evidence that a production device metadata path must incur that cost. Nor should the slower dynamically padded full-attention control become a weak baseline for a new scheduler.

**Decision:** close the complex cohort-depth scheduler's whole-model integration on current evidence. The step is real, but the enhanced prefix-aware group cost table already predicts the observed cases within 0.18%; see `results/ragged_group_cost_audit_v1/audit.json`. That fit has only two distinct feature values and two layers, so it is not a general cost-model validation. More importantly, there is no evidence of a policy advantage over cost-table marginal greedy and no acceptance trace or verifier end-to-end gain. A 40% allocation step alone is not a scheduling contribution. DScale's verification tile/depth work and existing cascade/grouped attention remain required prior art.

All failed workspace/capture runs are retained. Closing this branch does not close the overall search for an effective inference mechanism. The independent XQA row-pair hardware experiment is a separate, bounded hypothesis.
