# Direct model alias and GQA4 grid independent review

The CPU review of frozen model integration `675a124` found no launch blocker. `LayerSharedRawPairDirectBank.run_layer` selects the current layer's native contiguous Q and O objects and all four K/V views immediately before invoking the wrapper. Kernel arguments are passed by value at capture; the mutable Python plan is not a pointer table consulted later by graph replay. Rebinding it for the next layer therefore does not retarget an earlier captured node.

The original model computes Q inside the captured forward and consumes its original O buffer afterward. The installed vLLM manager retains each CUDA graph and its graph pool, and copies final model output into its own persistent hidden-state buffer. Ordinary intermediate lifetimes and serial graph replay remain the relevant memory contract. There is no new raw external allocator or asynchronous Q/O lifetime extension. The bank retains a reference only to its most recently selected Q/O; it does not pin all48 Python intermediates, which would inhibit normal graph-pool reuse. Actual same-state checked and production model graphs must validate this reasoning; CPU fake tensors alone cannot establish captured lifetime correctness.

The source manifest includes direct bank/runtime/build and every inherited arithmetic generator. Independently feeding the two actual old grouped-hint train26 receipts into the current offline validator rejects them with `Offline source fingerprint mismatch`. Live production validation also compares the actual model/device/engine fingerprint and both-rank batch coverage. Unsupported Q/O strides are rejected before launch. Shared raw state and scratch remain serial across layers and batches; concurrent graph replay is out of scope.

This changes the **model bank** from Q canonicalization plus O copy-back to direct native aliases, in addition to the standalone removal of prefix Q/table/length preparation. These are different statements: the standalone two-launch result is already qualified, while the new full-model alias integration requires its own qualification and trace. Normal framework KV writes, persistent model-output staging and other model copies remain. No claim of a copy-free model is justified.

The independent GQA4 grid audit in `results/direct_model_quad_independent_audit_v1.json` verifies16 terminal runs, all source hashes, generated source identity `bc26510388b35577`, all numerical flags and every paired bootstrap interval. **Every grid input is synthetic**, at nominal P120528 with initial private length17+31*row, optionally adding8192 tokens. FP64 checks cover first and last requests; all request outputs compare native bits. Tests include dummy-length capture, changed inputs and membership permutation. Hot samples use10 graph replays; cold samples use1 after a256MiB flush.

| B | Hot native/direct, initial | Hot native/direct, +8192 | Interpretation |
|---:|---:|---:|---|
| 1 | 0.923 | 0.942 | Slower; scope extension is a negative control |
| 2 | 0.945 | 0.966 | Slower |
| 3 | 0.983 | 0.987 | Slower |
| 4 | 1.064 | 1.048 | Modest positive in this synthetic grid |
| 5 | 1.221 | 1.121 | Positive |
| 6 | 1.459 | 1.186 | Positive; longer private work dilutes reuse |
| 7 | 1.500 | 1.283 | Positive |
| 8 | 1.545 | 1.428 | Positive |

All hot direct/copied ratios exceed1, but that improvement does not rescue B1–3 relative to native. Cold B2/B3 results are near parity or modestly positive, rather than universally slower, so the negative statement should explicitly name hot-cache behavior. Do not change a frozen whole-model policy or infer real-task gains from this grid.
