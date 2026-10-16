# Raw XQA pair gate v1

The independent pair fork builds on the frozen raw self-resume implementation. Prefix uses GQA16 and the original split count; tail uses GQA8 at the original batch size. Each tail request gathers its own raw accumulator row from the paired prefix state. This is not an independent prefix/tail softmax merge. Existing ChunkAttention/PAT comparisons and the novelty limitation in `xqa_raw_continuation_prototype_20261017.md` apply.

The first self-resume gate passed five cases, FP64 and same-input graph replay. Its `torch.equal` checks establish numerical element equality, but cannot distinguish signed zeros. The new pair gate explicitly compares `view(torch.int16)` to establish BF16 bit patterns, including signed zero.

Prefix raw state is `[B/2,Hkv,S,4,32,35]` FP32. Tail request r maps to pair r//2; member r%2 selects MMA register positions `4*n+2*member+j` into native active positions `4*n+j`. Scalar m/z/hint use lane row `r_local+8*member`, a distinct mapping. Inactive tail padding is initialised separately. At B8/Hkv2/S10 the state is 1,433,600 bytes; initial implementation writes and reads it once. Both private JIT modules share one 64 MiB workspace and state can be shared across serial layers. Installed source files remain untouched.

`RawPairPlan` requires B8, Hq16/Hkv2, head128, BF16, page16, contiguous Q, int32 page table, native fused `[pages,2,16,256]` cache with preserved strides, unit scales and no window/sinks. `prepare` explicitly checks physical prefix-page identity before updating length buffers. `run` captures Q packing, pair-table/length updates, prefix and tail. The Python host checks are transparent diagnostic preparation, not a production device metadata implementation. Whole-model use must account for them or implement and qualify a new device path.

CPU checks cover 67,673 split partitions and 1,024 active register mappings. The C++ code compiles for SM120 with both GQA8 and GQA16. GPU correctness is still a separate gate.

```sh
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH CUDA_VISIBLE_DEVICES='' FLASHINFER_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 .venv-engine/bin/python scripts/qualify_xqa_raw_pair.py --build-only --output results/xqa_raw_pair_build_v2/build.json

# Root scheduler only: synthetic four cases; then actual capture on another free GPU.
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=4 .venv-engine/bin/python scripts/qualify_xqa_raw_pair.py --output results/xqa_raw_pair_synthetic_v1/qualification.json
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=5 .venv-engine/bin/python scripts/qualify_xqa_raw_pair.py --capture results/dense_capture_b8_train1_bf16_v1/rank0/dense_capture/layer15.pt --output results/xqa_raw_pair_train1_l15_v1/qualification.json
```

Allow 180 seconds per process after CPU build. Synthetic prefix lengths 256, 768, 4096 and 4368 cover partial-prefix split participation, full and mixed tile boundaries, equal-length different private KV in the first pair, and unequal tails in the other pairs. Actual captures reconstruct full logical KV with original physical page IDs and cache strides, then check the original captured O and independent FP64. No fake repeated single-step queries are introduced.

Gates require production native PDL-on versus PDL-off equality, native A/A ten times, paired bit equality, FP64 qualification and no input mutation. Captured graphs are replayed after in-place changes to Q, actual KV values, all physical page IDs and tail lengths; old state is poisoned with NaNs before replay. Pair members are then swapped. An invalid physical-prefix claim must fail before length updates. Only B8 is supported in this gate; odd batches and exits remain native fallback in any preliminary integration.

No performance benefit is declared by this qualification. A later complete-operator timing must include packing, device metadata copies, both kernels and state traffic, and use production native PDL-on as the primary baseline. The PDL-matched native control is auxiliary. Do not turn the self-resume baseline's extra checkpoint work into the performance comparator.
