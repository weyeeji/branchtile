# Original-page result: stop the Q-only components hypothesis

The real-page follow-up does **not** preserve the small improvement found on compact contiguous KV. All four cases pass output and LSE gates, but the two-component QK path is no faster than its matched BF16-QK control. Mature FlashInfer FA2 shared-prefix attention is substantially faster than both custom paths on these same inputs.

| Case | XQA O-only, us | FA2 shared prefix, us | Paged BF16-QK, us | Paged two-FP8-QK, us |
|---|---:|---:|---:|---:|
| Synthetic 8K, M32 | 16.90 | 14.73 | 35.05 | 35.19 |
| Synthetic 8K, M64 | 17.32 | 14.72 | 50.54 | 51.47 |
| Train1 120,528 prefix tokens, M64 | 127.79 | 53.13 | 76.22 | 76.32 |
| Train7 144,576 prefix tokens, M64 | 150.91 | 60.05 | 78.95 | 79.94 |

The paired median control-minus-components deltas on the two real documents are -0.101 us (95% block-bootstrap interval [-0.229, +0.075]) and -0.907 us ([-1.227, -0.774]). These intervals resample timing blocks, not documents. Only 7/20 and 1/20 paired blocks favor the components path. Therefore the prior contiguous-layout increment is not evidence of a deployable improvement. No further Q-only tile sweep is warranted.

Both custom implementations and FA2 produce O plus base-two LSE; XQA produces O only. All read the same fused physical KV storage and Q. Capture reconstruction occurs before timing, while the timed kernel directly traverses the captured page IDs. Complete-prefix outputs are qualified against FP64. These are prefix-only microbenchmarks, not a live private-tail cascade, model quality test, or end-to-end result.

Audit script: `scripts/analyze_paged_query_components_gpu.py`. Machine-readable paired audit: `results/platform/paged_query_components_gpu_analysis_20261017.json`. Source raw results remain in `results/paged_query_components_gpu_v1/`.
