# General-batch raw continuation gate

This independent runtime keeps the frozen CUDA modules unchanged. It supports fixed B with paired requests followed by a native single-request orphan for odd B. B<4 falls back to native. Prefix, paired tail, and orphan all explicitly retain the original full batch's split count; deriving S from the smaller sub-batch would silently change numerical order. B changes construct a new plan and graph, whose setup/capture costs are recorded. There is no claim that topology can change without preparation.

`RawPairGeneralPlan` retains the B8 prototype's BF16/head128/GQA8/page16/unit-scale restrictions. Every plan can share a 64 MiB workspace and 2 MiB raw-state allocation across serial calls/layers. Request membership is expressed by actual Q/table row selection. The initial script checks native A/A, exact int16-view O equality, unchanged FP64 thresholds, graph replay after Q/length updates and poisoning raw state, and revisits old graphs after different batch sizes have overwritten the same shared buffers.

The cost experiment includes the complete candidate call. In odd batches the extra orphan kernel reads its entire context and may erase gains; this is deliberately measured, not hidden. Native PDL-on remains the comparator. B<4 is labelled fallback, not a contribution. Repeated samples are randomly interleaved with native A/A.

```sh
# Root schedules GPUs. Actual capture: no copied queries to manufacture B>8.
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=0 .venv-engine/bin/python scripts/qualify_benchmark_xqa_raw_general.py --capture results/dense_capture_b8_train1_bf16_v1/rank0/dense_capture/layer15.pt --batches 8,7,6,5,4,3,2,1 --output results/xqa_raw_general_v1/train1_initial.json

# Explicit stress augmentation: real captured Q/prefix and original private KV,
# then seeded synthetic private KV appended; not a natural-generation result.
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=1 .venv-engine/bin/python scripts/qualify_benchmark_xqa_raw_general.py --capture results/dense_capture_b8_train1_bf16_v1/rank0/dense_capture/layer15.pt --extra-tail 8192 --batches 8,7,6,5,4,3,2,1 --output results/xqa_raw_general_v1/train1_tail8192.json

# Larger batches use independent random Q/KV until real B19/B21 captures exist.
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=2 .venv-engine/bin/python scripts/qualify_benchmark_xqa_raw_general.py --prefix 120528 --extra-tail 8192 --batches 21,20,19,18 --output results/xqa_raw_general_v1/synthetic_large_tail8192.json
```

Use a 180-second process limit initially. The same commands support train7 or other frozen layers. The loader preserves native physical IDs/stride for unmodified captures. Growth stress uses remapped shared prefix plus independent private pages and explicitly records this layout change. The captured Q is from the initial AR step, not the future Q corresponding to an 8192-token suffix; this stress qualifies the operator's changing lengths and cost, not model-level quality or speed.

If odd-orphan cost is poor, a separate candidate may pack a dummy row for the final prefix pair and let the tail operate at original B. That alternative must receive its own same-S/state-mapping qualification. It is not implemented or silently substituted in this version.
