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
最初仅核查了作者SM100/CuTe DSL。后续FlashInfer0.7.2源码已确认SM120/121、D128、GQA≤16支持；但当前接口以128-token块选择，无逐token mask。我们的自然mask有大量混合块，直接使用会改变语义。需要计费gather重排的E0适配，不能把它标成原位同mask基线。详见 `results/platform/flashinfer_msa_applicability.json` 与 [官方接口](https://docs.flashinfer.ai/generated/flashinfer.msa_ops.msa_sparse_decode_attention.html)。

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


## 2026-10-16 15:43 增补：当前原生能力与共享压缩

[FlashAttention4 b35](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta35)
于10月14日发布。独立venv安装并按immutable tag核对六个核心源码后，实际SM120
任意token mask、GQA4/8、B1/B3/B4及不等长strided KV资格通过；split>1及paged
仍被接口拒绝。FA4同源mask、独立gather和compact-union+gather都独立调优。
两自然decode点热/冷完整call仍慢于FI0.7.2，连不计gather的乐观下界也未改变
结论。只限这些小batch长decode形状，不能外推训练/prefill或FA4整体性能。
未过滤sanitizer的68条CUDA API版本查询错误保留，单独设备内存检查0错误；
不能将其写成无条件clean memcheck。完整证据 `results/fa4/summary.json`。

[FastKVzip官方实现](https://github.com/Janghyun1230/FastKVzip/tree/e04afaa6007c58d18500d0698d2d8ae592ff22c2)
发布了与本实验4B-Instruct-2507对应的模型专用gate。HF revision
`67ce1265f510a946bf22486a30269f99066b3cbb`中没有主模型30B-A3B gate，不可跨模型套用。
[NVIDIA实现](https://github.com/NVIDIA/kvpress/blob/7136c224105ba7b046a6e538742c3589698680b8/kvpress/presses/fastkvzip_press.py)
与作者gate公式相同，但保护窗、全局预算和threshold/tie语义不同；其默认使用
mask模拟压缩，源码明确不代表优化物理压缩性能。当前拟做准确命名的NVIDIA
全层全头mask质量参考，成本与作者优化chunked-prefill分开，绝不将fake mask
时间当作实际KV压缩加速。

[EchoPress](https://arxiv.org/abs/2610.00412)于9月30日提交，利用prefill Q/K近似
重构分数，并只重构首块作校准，不需要模型专用训练。论文报告Qwen3-8B与
Llama3.1-8B的LongBench/RULER结果。这是query-agnostic方向的直接近期对照，
不是本文的新idea；当前仅列为后继适配候选，尚无本机资格/性能证据。

## 2026-10-16 16:22 增补：不能通过弱dense或改标签获得收益

[ByteCross（2026-09-27）](https://arxiv.org/abs/2609.33889) 专门研究激活稀疏和
KV稀疏的成本交叉，并以同一split-K注意力实现配对dense/sparse。作者报告仅更换
较弱dense注意力路径就会显著放大相同KV策略的表面收益。因此我们的eager包装
只能作为开发筛查，实际推进要求原生graph及同支持集强执行。字节模型、稀疏分支
组合和简单长度分流也已有直接先例，不能单独换名作为新贡献。

[RelaxKV（2026-09-27）](https://arxiv.org/abs/2609.33503) 在位置无关chunk缓存中
同时选择修复目标与修复上下文；其cache修复问题不同于这里完整前缀的只读共享源。
若转向RAG chunk重用，必须把它与ProphetKV等修复基线纳入，不能宣称查询引导的
部分重算本身新颖。这里仅完成原文摘要及方法范围核查，没有本机复现。

[PageWeaver（2026-10-08）](https://arxiv.org/abs/2610.11201) 已明确区分执行组复用
和在线重组的额外收益，并报告更强原生实现或准备成本可消除优势。当前新增8K/
16K/32K自然支持集网格是补全预算覆盖，不是改变固定支持语义或证明新的贡献；
四个预算均需保留，无论哪个点有利。
