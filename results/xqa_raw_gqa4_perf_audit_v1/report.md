# Actual GQA4 complete-call audit

All8cells pass provenance,original-native-output bit checks and independently recomputed timing statistics. The actual second model is Qwen3-4B-Instruct-2507 TP2,BF16 KV,Hq16/Hkv4,D128. Both ranks used original native XQA with S5. The reconstructed cache retains original physical IDs and stride `[16384,256,1024,1]`.

| Rank/layer | Native hot µs | Pair hot µs | Hot speedup [95% CI] | Cold speedup | Native/AA | Native/unshared checkpoint |
|---|---:|---:|---|---:|---:|---:|
| 0/0 | 373.15 | 263.58 | 1.414 [1.394,1.454] | 1.338 | 0.9997 | 0.976 |
| 0/11 | 373.25 | 261.12 | 1.453 [1.402,1.462] | 1.343 | 0.9995 | 0.975 |
| 0/23 | 375.35 | 260.71 | 1.454 [1.401,1.462] | 1.346 | 1.0000 | 0.976 |
| 0/35 | 368.23 | 270.18 | 1.363 [1.355,1.410] | 1.348 | 1.0000 | 0.975 |
| 1/0 | 372.32 | 265.42 | 1.395 [1.389,1.446] | 1.338 | 1.0000 | 0.976 |
| 1/11 | 371.92 | 264.19 | 1.402 [1.393,1.458] | 1.341 | 1.0007 | 0.976 |
| 1/23 | 373.56 | 261.73 | 1.450 [1.395,1.462] | 1.341 | 1.0004 | 0.976 |
| 1/35 | 368.44 | 271.05 | 1.355 [1.349,1.394] | 1.339 | 1.0004 | 0.974 |

The complete pair call is1.355–1.454× faster hot and1.338–1.348× faster cold. Native AA stays near1; unshared checkpoint is slower than native. This supports KV-read sharing as the source of the gain, while checkpointing itself adds cost. Pairing GQA4→8 leaves half of the M16 tile unused; no four-way variant was used.

This is one development document and operator evidence only. Timing CIs concern within-cell paired blocks, not independent documents or seeds. Per-layer setup and graph capture are reported separately. The running36-layer model qualification must pass before reporting model performance.
