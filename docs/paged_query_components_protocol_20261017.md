# Original-page query-components qualification

This bounded follow-up tests whether the approximately 2–3 microsecond increment of two-component FP8 QK over its BF16-QK control survives the original fused physical KV layout. It is an implementation hypothesis, not an established novel method or an end-to-end result.

The new independent kernel takes native Q strides, a four-dimensional fused cache, and shared-prefix physical page IDs. Both arithmetic arms traverse identical pages; the only switch is BF16 QK versus independently scaled high/low FP8 QK. Q decomposition remains inside each producer CTA and is fully charged. Neither arm packs Q or compacts KV during a call. The merge can emit base-two LSE for a later private-tail merge. Unit FP8 KV scales, D=128, GQA=8, page size 16, and the captured B8 case are the initial supported contract.

The new runner compares native FlashInfer XQA, mature FlashInfer FA2 shared-prefix attention, the paged BF16-QK control, and the paged components arm. All four read the same Q and fused physical storage. Reconstruction from an offline capture restores its original page IDs and cache strides before timing; it is not a claimed free online transform. FP64 complete-prefix output and base-two LSE checks gate timing. A separate span-wide QK sample checks the arithmetic. The latter probe uses compact reference data and is not itself proof of paged address correctness; complete-prefix output is the corresponding paged check.

Timing uses 20 randomized paired blocks with 10 CUDA-graph replays per arm. XQA emits O only; FA2 and both new arms emit O plus LSE because their intended role is a split-attention producer. This difference is disclosed rather than attributed to arithmetic. No private-tail cost, live page-metadata maintenance, full-model output quality, or end-to-end speedup is established here. The experiment does not authorize a broad tile search. Stop the components hypothesis if the increment disappears against the same paged control; if it persists, its model relevance still requires measurement against the mature cascade.

Offline SM120 compilation passed for both producer arms and both LSE modes with CUDA uninitialized. The compilation receipt is `results/platform/paged_query_components_compile_20261017.json`. No GPU performance result exists at the time of this protocol commit.

Bounded synthetic invocation (GPU assignment belongs to the root scheduler):

```bash
CUDA_VISIBLE_DEVICES=4 .venv-engine/bin/python scripts/qualify_paged_query_components_gpu.py \
  --output results/paged_query_components_gpu_v1/synthetic_m64.json \
  --tokens 8192 --batch 8 --block-m 64 --blocks 20 --replays 10
```

For a real case, replace the synthetic arguments with `--capture results/dense_capture_b8_train1_fp8_v2/rank0/dense_capture/layer15.pt`. The four captured layers are 0, 15, 31, and 47; train7 provides a second document. Initial paired comparison keeps M=64 fixed.
