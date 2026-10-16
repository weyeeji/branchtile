# 精确支持下的协同索引排序：先例与可行性审查

日期与检索截止：**2026-10-16**。本次仅联网与 CPU 读源码，不运行 GPU、不修改执行器，不读取事实审阅文件。本节独立于改变支持集的 joint exchange；原始逐请求 mask 在这里完全固定。

**结论：no-new，指当前提出的 owner-bitmap/common-prefix 索引排序不足以成为独立的新方法贡献。** 未核实到完全相同的“少量请求、token 索引、FlashInfer split 对齐”实现，但已有行内稀疏重排、跨行缓存感知调度，以及保持 sparse-attention 支持和后端不变的 L2 优化。把这些技巧用于本任务可以是合理工程对照；不能因为置换的是索引而非 query 行，就宣称新的共享注意力原理。是否有工程收益仍未知，本轮没有实验结果。

## 1. 最直接的一手先例

| 来源与明确日期 | 已有内容；与本提议的边界 |
| --- | --- |
| [ASpT，PPoPP 2019，2019-02-16–20](https://www.sigplan-www.sigplan.hosting.acm.org/OpenTOC/ppopp19.html)，[论文 DOI](https://doi.org/10.1145/3293883.3295712) | 官方论文摘要明确在标准 CSR 内做 **intra-row reordering**，以支持 SpMM/SDDMM 的自适应 tiling 和数据复用。因此“支持不变、只改每行非零项顺序”早已有之。这里只确认官方摘要；出版商全文及原始 artifact 主机本次访问失败，不能据此声称它就是同一个 native attention 排序器。 |
| [Mehrabi et al., Learning Sparse Matrix Row Permutations，ISPASS，2021-03-29](https://research.nvidia.com/publication/2021-03_learning-sparse-matrix-row-permutations-efficient-spmm-gpu-architectures)，[作者全文 §III-B](https://d1qx31qr3h6wln.cloudfront.net/publications/ISPASS_2021_Sparse_Row_Permutations.pdf) | 用 cache-line bitmap、Hamming 相似度及 warp/CTA 映射做缓存感知行排列，保留 CSR，并考虑预处理摊销。这是行排序，不是行内排序；但 bitmap、访问重叠、执行时间接近与复用的组合并不新。 |
| [DTC-SpMM，ASPLOS 2024，2024-04-27–05-01，§4.3](https://home.cse.ust.hk/~weiwa/papers/dtc-spmm-asplos24.pdf) | 两级重排分别优化 Tensor Core 行簇及簇间 L2 locality；依据原非零列集合的相似性。需要配套格式与 kernel，不是本提议的零 kernel 改动，但排除了把跨执行单元的共享地址排序笼统当作创新。 |
| **[WaveAlign，2609.34814v1，2026-09-28，§IV、§V](https://arxiv.org/html/2609.34814v1#S4)** | 最直接的注意力先例：固定选择结果，只排列 query/mask 行并恢复输出；SVD 聚合相似访问，再按 NNZ 做 wave 内排列，缩短跨 wave 的 KV 复用距离，含不值得重排时跳过的机制。后端使用 H100 的 FA4、A40 的 FlashInfer。它没有采用本提议的行内列表置换，但已覆盖“保持精确支持、不增加 union 计算、不改 backend、借已有 L2”的主要主张。 |
| [PageWeaver，2610.11201v1，2026-10-08，§3–4](https://arxiv.org/html/2610.11201v1#S3)；[QUILT，2610.11134v1，2026-10-08，§3.1–3.3](https://arxiv.org/html/2610.11134v1#S3) | PageWeaver 按支持相似性重组 query，使用 union kernel；QUILT 联合排序 Top-K 列表、分解共享/私有段并分层执行。二者不等于本提议，但 owner 分类、共同段提前处理、按支持重叠组织执行已不是空白。 |
| [FlashInfer Cascade Inference，2024-02-02](https://flashinfer.ai/2024/02/02/cascade-inference.html) | 已明确普通独立 decode 的共享 KV 通过 L2/全局存储访问，并以联合处理共同前缀转向 SMEM/register 复用。本提议仍在普通独立 decode 的访存层次内；不能把“独立 decode 可共享 L2”作为新发现。 |

检索覆盖 coordinated/index/intra-row ordering、row/column reordering、SpMM/cache scheduling 及上述最新注意力论文。没有把第三方搜索摘要或未资格化代码当作复现。**这些先例不证明所有行内排序策略都没有研究空间；它们意味着当前简单排序需要额外的、可验证的技术贡献，而不是仅换一个置换维度。**

## 2. 对齐意图与真实执行之间仍有缺口

把原支持看作二部图：请求为行、真实物理 KV 为列，owner bitmap 是列的邻接签名。按签名排各行，相当于对已有边做局部次序调整；不减少边数、union、数学 FLOPs 或源缓存容量。全组交集可被放到同一列表前缀；其余 owner 段在不同请求中前置段长度不同，**全局相同段序不等于相同列表偏移，更不等于同时访问**。

一个无 padding 的反例：三请求支持分别为 `{ab,ac}`、`{ab,bc}`、`{ac,bc}`，每个符号是两个 owner 共用的一个物理条目。每行仅两槽。若每个符号在两个 owner 中必须占同一槽，就要求三角形的三条边用两种槽颜色着色、每个顶点两边异色，这是不可能的。它也适用于“两 split、每 split 恰装一个不可分单元”的限制情形。此反例只否定普遍的完全对齐保证，不证明局部对齐无收益；不能悄悄靠补重复 token 或扩充支持修复它。

本地只读检查 `.venv-latest` 的 **flashinfer-python 0.7.2**：FA2 planner 在 `attention/scheduler.cuh:647` 后按 request、Q tile、KV tile 产生 CTA 元数据；`prefill.cuh:3023` 读取这些映射，按列表偏移切 KV chunk。TC decode 具有调用 prefill 模块的路径，不能仅检查 SIMT 的 `decode.cuh` 就推断当前 E0。索引置换可以改变某个 chunk 的地址集合，**不会自动改变 CTA 元数据排序或提供跨 CTA 同步**；需记录实际 backend 与实际 plan，不能从 `backend='auto'` 推定唯一执行路径。

CUDA 也[不保证 thread block 的分配或执行顺序](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/01-introduction/programming-model.html)。因此同 split 编号不保证共同驻留；动态私有尾增长、有效 batch 改变或不同 backend 还可能改变分块。固定 split 可用于隔离诊断，但不能把候选特调后的结果与未同等调优的 native 基线相比。

本次源码身份，供复核而非性能证明：`decode.py` SHA256 `de1f242bf2f598bc973a9d2f9625f8db1e4ccfc24085f686b2d0304c9076b20d`；`attention/prefill.cuh` `ad8be30492fb04131edc25b539ae77e911177af8647c7e2007cf172a4a69d029`；`attention/scheduler.cuh` `b1fb27078084262158ed070e220ad317467306d1f0b32e0d9624bf85e6ef97ce`。路径均位于 `.venv-latest/lib/python3.12/site-packages/flashinfer/`，headers 在其 `data/include/flashinfer/` 下。

## 3. 若作为工程对照，资格与成本不能省略

- **支持与位置。** 逐请求逐头验证物理 ID 多重集合、KV 值、私有尾及保护项完全不变，ID 无重复/遗漏。logical ID 相同不表示跨请求 private KV 相同。post-RoPE Q/K、`pos_encoding_mode=NONE` 且单 token decode 的原支持全因果时，排列在实数算术下保持 O/LSE；若使用 ALiBi、窗口、位置偏置或多 token causal，则必须携带原位置及对应 mask，不能用重排后的列表位置替代。
- **数值。** 行内排列改变 online-softmax、split 部分和 merge 的浮点归约，不能承诺逐 bit O/LSE 或自然生成不变。应使用原资格容差对相同支持的 FP64 参考验证全部 O/LSE，并记录 logits/token 差异；不得因候选是“精确重排”放宽门槛。WaveAlign 的行置换保留行内计算次序，与本提议的浮点风险不同。
- **完整成本与强控制。** 基线至少包括原索引顺序、物理地址升序、普通共同交集优先、候选 owner 排序，及适用的行排序控制；这不是声称已复现完整 WaveAlign。冻结策略后同时计 owner 识别、排序、上传、plan/索引更新与注意力调用；固定 mask 的预计算可按实际 decode 长度摊销，不能称免费。先固定 Q/K/V 做配对 A/A 和冷热/真实层序诊断，再谈整模型。无硬件计数器只能报告延迟现象，不能将 logical reuse 计作实测 HBM 降幅。

**研究决策：当前不为“新方法”启动 GPU 实验。** 如果主任务需要补强 E0 工程基线，可独立预注册一个很小的排序对照；它即使获益，也先归为已知缓存感知调度的本地应用。只有进一步发现并解决了已有行内重排与 WaveAlign 等无法处理的具体限制，且完整成本与真实层序收益稳定，才值得重新审查贡献；不能用本次 no-new 判定否定所有应用论文，也不能用尚未测量的收益绕过先例。
