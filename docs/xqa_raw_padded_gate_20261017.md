# General-batch v2: dummy prefix row, original tail batch

The v1 odd-orphan path passed all correctness gates but was substantially slower. Its results and sources remain frozen. The new `RawPairPaddedPlan` instead packs ceil(B/2) prefix groups; the final group contains one real query and one permanently zero dummy query. Tail uses exactly the original B requests, split count and Q. The dummy's state is never selected by a real tail row. No separate orphan kernel scans the long prefix.

The CUDA modules are unchanged from the qualified B8 implementation. For odd B, the extra Q-padding copy and ordinary packing copy are both included in graph timing. Prefix SM parameter is S*ceil(B/2)*Hkv; tail parameter is S*B*Hkv. S is still calculated from the original dispatched batch B, never B+1. B<=4 falls back to native following the measured negative v1 B4 results.

`RawPairPaddedPlan` keeps the general adapter API and fields, with optional shared workspace/raw state. State capacity must cover ceil(B/2)*Hkv*S*4480 FP32 values. The gate's 2 MiB allocation covers declared B1–32 on the current GPU. Batch changes still require distinct plans/graphs and all setup/capture costs are recorded.

Run the same frozen v1 commands with `scripts/qualify_benchmark_xqa_raw_padded.py` and new outputs under `results/xqa_raw_padded_v2/`. Suggested four initial jobs: train1 initial, train7 initial, train1 plus8192 synthetic tail, and synthetic B21/20/19/18 with long tails. Use the same batch subsets and private KV inputs for the v1/v2 comparison. New output directories preserve the negative v1 results.

The v2 gate retains int16 bit comparison, native A/A, FP64, same-address Q/length graph updates, NaN-poisoned raw state and revisiting graphs after other B shapes used shared workspace. It additionally checks that the dummy query remains zero. This is operator correctness and complete-call cost, not actual model requests or a claim of unrestricted topology reuse.
