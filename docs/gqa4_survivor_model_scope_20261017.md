# B6/B7/B8 Thinking4B model experiment

Independent audit `results/gqa4_followup_perf_audit_v1` covers8actual Thinking4B B8 cells and6survivor cells. Every numerical check and recomputed paired CI passes. At long synthetic tails,the worst B6 cold ratio is1.089× (paired lower bound1.088×). These endpoints justify a bounded development experiment,not a broad dispatch policy.

Frozen scope is nominal shared prefix **exactly120528**,actual/padded batch **6,7 or8**,and **0<max(current native length)-120528≤8192**. It uses current lengths only,not future EOS or online timing. Original native S remains7,6,5 for B6,7,8 on SM170. Kernel prefix boundary remains floor(P/256); the rule uses full nominal P,matching the measurements. Other shapes/lengths fall back to native. Intermediate lengths are a declared experiment between tested endpoints,not a guaranteed performance region; no GQA8 policy was reused.

The new shared runtime adds optional caller-owned state/workspace to the qualified padded runtime. Allocation changes only: same Q packing,zero dummy row,metadata refresh and original prefix/tail calls. One arena serves3fixed batch graphs and36layers serially. Each graph binds its original B native lengths/table and layer-specific cache strides; B7 has no orphan full-history scan. Direct native int32 lengths remain aliased as uint32. Dummy length1 capture was separately qualified at these shapes.

The new worker/driver use new source fingerprints and include the scope JSON in the execution fingerprint. Strict qualification follows actual natural Thinking outputs while returning original native hidden states. Up to8eligible states per observed B are checked across both TP ranks: all36layer Q/O,KV slots,hidden/logits,input metadata and device aliases. Unobserved batches remain unqualified and use native. Offline setup requires all enabled batches on both ranks,so B8-only receipts cannot enable survivors. CPU guards test private8192/8193,P mismatch,B5 and missing rank1B7 receipt.

The default driver is strict same-session only. Qualification-only example,root assigns GPUs:

```bash
CUDA_VISIBLE_DEVICES=6,7 FLASHINFER_CUDA_ARCH_LIST=12.0 .venv-engine/bin/python scripts/benchmark_gqa4_thinking_survivor_e2e.py --document-id docfinqa-680456070a0a0714 --qualify-only --output results/gqa4_thinking_survivor_qual_v1/train1
```

Natural EOS and8192output cap are unchanged. There is no artificial request removal to manufacture B6/B7. The full driver subsequently runs fixed128-token ABBA separately from naturalABBA,checks complete token/cache/trajectory identity and reports setup,capture,anchor and generation costs. A long answer that exceeds the current-private8192 bound simply returns to native for its remaining tokens. No original frozen B8/Instruct/Thinking files were modified.
