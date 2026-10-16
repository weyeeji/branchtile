# Second-model B8 integration gate

The independent driver `scripts/benchmark_gqa4_raw_pair_e2e.py` targets local Qwen3-4B-Instruct-2507, TP2,BF16 KV,36 layers,Hq16/Hkv4,D128 and page16. It enables only actual B8 with exactly8 padded tokens. Every survivor batch uses the unchanged native graph. No GQA8 development dispatch parameters or performance evidence are inherited.

The implementation copies the frozen general adapter's qualified integration controls into separately named GQA4 files. A single raw state/workspace serves36 layers serially; each capture binds the original fused cache views and full-width gathered native page table. Sequence lengths alias the native int32 device buffer as uint32. The first admitted cohort verifies shared physical prefix pages; each subsequent admission checks request,slot and prefix-overwrite epoch identity, plus original native XQA metadata and pointer aliases. All query canonicalization/packing, metadata D2D, state export/resume and output copies remain charged inside the graph.

Strict setup captures witness,checked and production B8 graphs before real requests. During qualification the original native hidden state continues to the sampler; candidate replays are checked against that same state. Required fields cover every36-layer Q/O,all current KV slots,hidden state,local/gathered logits,IDs/positions/tables,device length values and native metadata addresses. Both ranks complete all collective calls even if a numerical check fails; the driver reads both receipts and refuses the performance phase on any failure. One passing synthetic operator gate is insufficient to enable the model path.

The initial driver always performs strict same-session qualification. The worker retains a separate production-only setup option, but it requires offline receipts matching the exact new source set,model/device/engine fingerprint,B8 and all36 layers on both ranks. It cannot accept prior48-layer receipts. CPU negative checks cover stale source,wrong geometry,extra layer,failed O and missing rank. No production-only driver or bypass is introduced in this initial integration.

The protocol uses newly computed4B tokenizer hashes for the same two development document subsets. Each phase refreshes the common-prefix anchor. Natural generation permits EOS up to the existing8192-token cap; fixed128-token work is separately labeled. ABBA compares complete generation,anchor+generation and phase-control wall time,records setup/capture cost and requires identical output tokens,finish reasons,cache hits and graph input trajectories. Instruct outputs use the existing nonthinking parser,then the same frozen strict numerical answer metric.

Do not launch model qualification before actual-native capture and complete-call checks pass. Root assigns GPUs. Qualification-only command:

```bash
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=6,7 .venv-engine/bin/python scripts/benchmark_gqa4_raw_pair_e2e.py --document-id docfinqa-680456070a0a0714 --qualify-only --output results/gqa4_raw_pair_model_qual_v1/train1
```

Remove `--qualify-only` for strict qualification followed by fixed and natural ABBA; use a fresh output path. Current qualification observes up to8 actual B8 states. This is a development gate, not exhaustive numerical proof or population confidence. If B8 never appears or no candidate graph executes, the driver rejects a speed claim rather than timing a silent native fallback.
