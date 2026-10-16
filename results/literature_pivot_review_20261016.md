# 第二轮候选与强对照核查（2026-10-16）

这份记录约束后续实验的新颖性主张。源码存在某优化不等于本平台已经可用；
找不到作者实现也不等于可以把自己的慢版本称作强对照。第一轮文献记录仍有效。

## 选择器：已有方法，但值得作为强基线实际测量

[MiniKV, ACL Findings 2025 §3.2](https://aclanthology.org/2025.findings-acl.952.pdf)
已经用 FlashAttention 输出的 LSE 再计算 KV 列重要性。因此“复用 prefill LSE”
不能作为本文独立创新。完整成本必须分别报告 LSE 产生、评分、pool、选择、
mask/索引输出以及运行时准备；只给 known-LSE 增量时间会高估应用改善。

[SnapKV 作者实现](https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py)
和 [NVIDIA KVPress SnapKV](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/snapkv_press.py)
提供 QK、softmax、pool、top-k 的语义参考；某个文件未使用 LSE 并不能证明新颖性。
[SnapStream](https://arxiv.org/abs/2511.03092) 也讨论融合评分、选择与 gather。

[FlashInfer top-k API](https://docs.flashinfer.ai/api/topk.html) 提供原生选择与确定性
tie-break。当前实验采用可在 SM120 上使用的路径；需要更大 shared memory 的
配置不可作为有效实现。原生 CUB 首次编译很重，超时及 CPU-only 后续编译均记录。
先 pool 后保护：protected token 的原始分数仍应参与邻居窗口，不能提前置负无穷。
原生 BF16 attention 的 LSE 与完整 FP32 分母可能略有差别，必须报告 top-k 边界
间隙和实际 mask 变化，不能只用输出误差容差掩盖 token 替换。

## 共享注意力与低精度：需要越过的已有边界

[LiteTopK/LiteDSA](https://github.com/Heisenberg-Yin/LiteTopK) 已有 query-major
成员关系、split/merge、FP8 概率缩放与 LSE 补偿；union 路径返回索引和 bitmap。
[MSA 论文](https://arxiv.org/abs/2606.13392) 与
[作者实现](https://github.com/MiniMax-AI/MSA) 覆盖 KV 外层调度和 query gather，
公开 SM100/CuTe DSL 实现的存在不证明 SM120 直接兼容。

[VC-Attention](https://arxiv.org/abs/2609.15810)（2026-09）涉及 V 聚类/平滑及
FP8 attention；[MpFA](https://arxiv.org/abs/2609.33135)（2026-09）涉及混合精度
QK/PV、平滑和 prefill/decode 分工。若试验缩小 inactive partial 的存储精度，
必须给予固定基线相同格式并验证真实 occupancy 台阶，不能仅凭字节数声称收益。

[Broken Symmetry in BF16](https://arxiv.org/abs/2609.34272) 主要涉及训练/反向质量
守恒，并非本文 decode 直接证据；
[FlashAttention rounding note](https://severinvisionary.github.io/flashattention-rounding-error/)
（2026-10-06）指出若干 merge 重归一化已存在于成熟实现。
[vLLM train/inference consistency](https://vllm-project.github.io/2026/10/14/train-inference-consistency.html)
（2026-10-14）也提示不能把 prefill/decode 的数值路径一致性当作未经核查的新贡献。

## 当前决策

1. 先量化强选择器和实际引擎时间占比。LSE 复用与融合评分目前只作为控制实现。
2. E1 全 15 固定划分和 E4 精确二叉层次已执行，记录见
   `results/partition_cascade/summary.json`。层次机制移植不是完整 QUILT 复现。
3. 固定文档支持、私有 suffix 增长的增量 metadata/plan 是待查工程假设。
   先与成熟 native cached/incremental plan 比较；若常规缓存已消除主要开销，停止
   把它当作新机制。必须用官方主模型真实共同 ready 组验证，不能拼接不同时间的 Q。
4. 目前没有通过强对照、质量和完整应用检验的新候选；完整论文目标仍在进行。

检索基线使用 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
和 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) 的官方
输入规则、固定权重 revision 和校验 hash。这只是开发网格起点，不宣称是2026年
所有检索系统中的最优配置；正式结论需冻结同质量的可用强对照。
