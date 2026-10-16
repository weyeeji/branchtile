# 原生 B3/B8 MoE 时间线与 2026 排重

本轮只用 CPU 复算已有 16 份 rank trace，没有启动 GPU。脚本为
[`analyze_native_moe_opportunity.py`](../scripts/analyze_native_moe_opportunity.py)，
逐文件 SHA 与详细结果为
[`native_moe_opportunity_20261017.json`](../results/platform/native_moe_opportunity_20261017.json)。
每份是固定 B 的 32 步模型图，每步 48 个 router、96 个 MoE grouped GEMM。

| 原生配置 | 48层 gate GEMM＋必要 split-K reduce | 96个 grouped GEMM | scatter/gather/count | 从 gate 到 expert gather 的区间总长 |
|---|---:|---:|---:|---:|
| B3 BF16 KV | 607–614 µs | 2073–2080 µs | 441–450 µs | 3.64–3.82 ms |
| B3 FP8 KV | 624–636 µs | 1954–2149 µs | 439–454 µs | 3.70–3.94 ms |
| B8 BF16 KV | 183–191 µs | 3706–3818 µs | 460–467 µs | 5.02–5.36 ms |
| B8 FP8 KV | 191–192 µs | 3705–3760 µs | 454–461 µs | 5.22–5.25 ms |

关键修正是 B8 gate 已切换到 cuBLAS split-K：每层先执行 BF16 GEMM，随后 reduce，再进入实际 topk。B3 则没有这个 reduce。这里按 topk 的真实前驱归因，不能仅检索全部 BF16 GEMM 名称，也不能把 B3 的 600 µs 外推 B8。MoE 链不含前置 RMSNorm、后置 NCCL。

这些是带 profile 的区间与 kernel 求和。脚本也给出 interval union 和只由该类 kernel 占用的区间，但它们**都不是可直接消除的关键路径上界**。PDL 把部分 producer 等待记到 consumer 内，NCCL 还包含 rank 到达等待；固定 B3/B8 不能代表自然停止后的 B1/B2。不同配置文档、长度亦有混杂。父任务的新 cached-O 全图诊断单独处理 attention 的因果空间，不能被本表替代。

## 已排除的泛化主张

- [MonoMoE](https://arxiv.org/html/2609.04244v1) 已将 router/topk/quant/up/activation/down/reduce 纳入 weight-major persistent kernel，明确面向 B1–8，包含执行就绪控制。直接提出“融合小 batch MoE”或“按权重组织”不足以构成新机制。
- [Cursor Warp Decode，2026-04-06](https://cursor.com/blog/warp-decode) 已针对 Blackwell 小 batch 按输出神经元分配 warp，在两个 kernel 内完成专家运算及加权归并。其 B200/内部模型结果不能作为本机收益预测。
- [Decoding the Skew](https://arxiv.org/html/2607.23099v1) 已按专家分布选 kernel；[BASE](https://arxiv.org/html/2609.36222v1) 已用预测移除误差协调 batch 内专家选择。仅把原联合 selector 改名用于专家不是独立创新。
- [FoldMoE](https://aclanthology.org/2025.acl-long.186/) 已研究 attention/MoE 的 token 级流水，范围为长序列训练；[TAPER](https://arxiv.org/html/2605.06914v1#S2) 已研究 agent 分支并发宽度。decode/SM120 适配仍须给出新机制与实际收益，不能仅凭应用范围变化宣称首次。

当前不把任何 MoE 候选提升为论文方案。已有 TinyGEMM 是成熟基线覆盖检查，见[原协议](tinygemm_baseline_gate_20261016.md)。优先推进真实 B8 长前缀 attention 的成熟 cascade 对照和独立 FP8 数据流假设。
