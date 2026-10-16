# Direct native-address GQA4 quad model integration

This independent path uses the root's qualified `xqa_raw_direct_quad_build.py` CUDA mapping and `RawGQA4DirectQuadPlan.run` launch sequence, with the existing shared workspace/state arena. The bank binds each layer's actual contiguous native Q and O addresses during graph capture. It also refreshes that layer's four K/V views. The prefix kernel reads original Q/page-table/length addresses and zero-fills absent group members internally; the tail retains original B, S and arithmetic. There are no per-layer candidate Q normalization, padding/packing, group metadata copy or O copy launches. State export, continuation and native split merge remain charged.

The previous copied-input quad worker, scripts, protocol and results are unchanged. Scope remains nominal P120528, B4..8, every L>P and max private<=8192. B1..3 remain native: standalone evidence did not justify extending the policy. The model is the same local Qwen3-4B-Thinking-2507 with 36 layers and TP2 Hq16/Hkv4. Limited task quality remains an explicit boundary, not an accuracy improvement claim.

The new strict worker checks original native AA, all 36 layers' Q/O, current KV, hidden and local/full logits on the same states. Natural output identity is mandatory. The separate production driver accepts only this new source/model fingerprint with all five B values qualified on both ranks. Old copied-input quad receipts fail closed. GPU full-model qualification is still required; standalone CUDA qualification and pointer tests alone do not prove full-model correctness.

`results/gqa4_direct_quad_cpu_v1.json` covers all 36 layers × five batch sizes, both original Q/table/length pointer paths, original S, actual native O destination, shared state/workspace and rejection of noncontiguous Q/O. It also rejects an actual old quad receipt and keeps B1..3 native. CUDA was not initialized. Previous actual Thinking quad capture receipts on both ranks record Q and O strides `[2048,128,1]`; this supports trying the strict contiguous contract, not assuming every future model layout is valid.

Root-scheduled qualification, using an idle exclusive TP2 pair:

```bash
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH CUDA_VISIBLE_DEVICES=6,7 FLASHINFER_CUDA_ARCH_LIST=12.0 .venv-engine/bin/python scripts/benchmark_gqa4_thinking_direct_quad_e2e.py --document-id docfinqa-680456070a0a0714 --qualify-only --output results/gqa4_thinking_direct_quad_model_qual_v1/train1
```

Suggested timeout is 600 seconds. No GPU run was launched by this subagent. After success, the production path is:

```bash
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH CUDA_VISIBLE_DEVICES=6,7 FLASHINFER_CUDA_ARCH_LIST=12.0 .venv-engine/bin/python scripts/benchmark_gqa4_thinking_direct_quad_production.py --document-id docfinqa-680456070a0a0714 --offline-qualification-records results/gqa4_thinking_direct_quad_model_qual_v1/train1/rank0/qualification_raw_pair.json results/gqa4_thinking_direct_quad_model_qual_v1/train1/rank1/qualification_raw_pair.json --phase-kind natural --output results/gqa4_thinking_direct_quad_production_v1/train1
```

This captures only five production graphs and runs natural ABBA; suggested timeout 900 seconds. Setup and cold-anchor costs remain explicit. Existing result summaries are never overwritten. Candidate numerical and end-to-end results cannot be inferred from the old copied-input path's timings.
