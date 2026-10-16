# Joint 停止后的研究判断与最后一项排重

2026-10-16。**本轮推荐进入下一轮实验的候选数：0。** 这不是“经典算法不能发表”，也不是断言 KV 研究已没有空间；而是现有实测、可用资源及截至本日核查的一手资料，还不能支持一个区别明确、成本可信、可证伪的新研究主张。joint 保持停止，不为消耗剩余预算而扩大实验。原生 FP8 KV 的资格仍是补齐强基线，不是新 idea；本判断不启动该资格。

只使用 CPU、项目虚拟环境及公开一手资料；未运行 GPU、安装依赖、修改引擎或冻结评测集，也未读取正式 eval100、事实审阅或身份映射。自有 [CPU 核算 JSON](../results/platform/post_stop_candidate_cpu_audit_20261016.json) 保存输入 SHA、实际观察窗维数和精确有理反例。

## 现有证据尚不足以提出新的经验论文主张

当前 sparse 模型 replay 的 attention 时间占比约 20%–25%，只适用于该 profile 边界，**不是所有 dense 模型的注意力份额**：旧 [native dense profile](../results/application_profile/main30b_docfinqa_g2_rank0_summary.json) 的 attention 占 kernel-time sum 约 53%，两者不能混用。对固定轨迹、非 attention 时间不变的条件估算，令 f 为可替换份额、r 为 attention 速度比、p 为额外准备相对基线总时间，则延迟降低为 `f(1−1/r)−p`。这只是 Amdahl 核算，不是新的性能定律或实测 TP JCT。

| 条件（f=20%–25%） | 条件结果 |
| --- | --- |
| attention 1.06×、p=0 | 总延迟降低仅 1.13%–1.42% |
| attention 1.15×、p=0 | 总延迟降低仅 2.61%–3.26% |
| 要求总延迟降低 5%、p=0 | attention 至少 1.25–1.333× |
| 要求总延迟降低 5%、p=2.5% | attention 至少 1.429–1.60× |

[本轮真实 48 层结果](joint_real48_stop_review_20261016.md) 只有约 105–121 μs/步的正节省、2.81–2.93 s 的当前准备代理；这个具体实现已按原门槛停止。低预算自然质量/覆盖未合格，故也不能拿更激进预算制造性能空间。上述成本不是其他未来算法的下界，但新算法必须提供自己的可测成本，不能默认为零。

把结论改写为“压缩率不代表速度”“自然输出长度改变盈亏”“平均质量掩盖失败”“需强量化基线”也不够新：

- **MLSys 2025，2025-03-31**，[Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving](https://arxiv.org/html/2503.24000v1#S4) 已研究服务实现、压缩后输出变长和逐样本失败；§5 还给出 throughput/length predictor 与 request router。[作者代码](https://github.com/LLMkvsys/rethink-kv-compression)。这直接覆盖最容易把负日志重新包装成论文的几个论点。
- **2025-12-12**，[Hold Onto That Thought](https://arxiv.org/html/2512.12008v1) 已针对 reasoning KV 压缩研究自然长度、低预算变长/不终止与推理成本。不能只把模型换成 Qwen3 Thinking 就主张首次发现。
- **2026-08-22**，[MoEXBench](https://arxiv.org/html/2608.21693v1) 已比较 MoE 的权重量化/KV 精度/部署与任务差异，也包含 Qwen3-30B；它不是本项目同栈 TP2 复现，但足以排除“研究 MoE 上普通组合压缩的质量—效率”这个宽泛新颖性说法。
- **2026-10-05**，[Behavior-Preserving KV Cache Compression](https://arxiv.org/abs/2610.06479v1) 已提出用压缩后 logits/KL 代替 attention importance 代理，并研究额外压缩成本。这里仅以明确的问题设定作先例，不声称已经审计其全部实现或本机兼容性。

单一 SM120 架构并不禁止有价值的系统经验论文，但当前需要的是新的、经干预确认且可在未见文档预测的现象；不是更换标题或补一张 FP8 表。现有数据没有提供这样的增量，且 FP8 KV 前沿尚未资格化。更多同类负点不会自动补齐它。

## 最后筛查的唯一具体机制：多个问题共用非负加权支持

考虑同一公共源 `(K_j,V_j)`，每个 KV head 在所有已观察问题 suffix 的参考 Q 上拟合一个共享非负权重向量 w，零权重代表删除。对每个参考 i，拟合

`Z_i = Σ_j exp(q_i·k_j/√d)`，`N_i = Σ_j exp(q_i·k_j/√d) V_j`，

使加权版本 `Σ_j w_j exp(…)` 与 `Σ_j w_j exp(…)V_j` 同时匹配。private suffix/生成尾和明确保护项须作为不改变的项独立加入；共同源必须是真正相同的物理 KV，不能混入各请求不相同的私有文档尾部。观察位置都在共同文档之后时，这些公共 token 对所有参考 Q 都满足因果可见性。

**排重结论：没有足够的新贡献，GPU 前停止。**

1. **Attention Matching 直接覆盖核心表示与目标。** [2026-02-18 论文 §3、A.2](https://arxiv.org/html/2602.16284v1#S3) 使用 `β_j=log(w_j)`，同时拟合 mass 和 attention output；当 Z 为正且精确匹配时，匹配 output+Z 与匹配 N+Z 等价。其参考 Q 可来自同一 context 的多个 self-study 问题，并显式研究 on-policy Q 和未来固定 KV 的拼接。将 Q 换成实际已知问题 suffix 并追加未见 decode Q 验证，不是独立的新机制。
2. **保留原 V 也不是全新表示。** 作者 [highest_attention_keys.py](https://github.com/adamzweiger/compaction/blob/2020f6d37940fe013e7e7f1ebf409e5d107c5670/compaction/algorithms/highest_attention_keys.py) 提供 `c2_method='direct'`，[base.py](https://github.com/adamzweiger/compaction/blob/2020f6d37940fe013e7e7f1ebf409e5d107c5670/compaction/algorithms/base.py#L381) 直接返回 `V[indices]`。要准确区分：这个现成分支先拟合 mass，并没有替本提议求解固定原 V 的联合 N/Z 稀疏 NNLS；联合求解是约束/拟合方式的差别，**不是已存在逐字相同算法的声明**。但目前没有证据说明这个普通受限变体能提供新的质量—准备成本优势。
3. **其他先例进一步限制宽泛主张。** [BFF/CFF，2026-01-06](https://arxiv.org/html/2601.03067v1) 已做跨请求近似共同 KV 表示；[QuoKA，2026-02-09](https://arxiv.org/html/2602.08722v1) 已做多 Q 的公共支持选择；[PQ-HSA，2026-09](https://arxiv.org/html/2609.33746v1) 为遗漏项补近似 logits/value 背景并统一归一化。后两者不是本 N/Z NNLS 的逐字实现，不能夸大为完全相同；它们意味着“共同支持 + 保留背景 mass + 跨请求复用”的组合描述仍不足以形成独有机制。

## 已完成的最低 CPU 证据

把每个 token 的列写作 `a_j = concat_i exp(q_i·k_j/√d)[1,V_j]`，目标为 `Σ_j a_j`。这是经典非负锥组合；conic Carathéodory 给出的支持上界为 `min(N, rank(A)) ≤ min(N, m(d_v+1))`。保护项先从目标中扣除。若要每个 query head 输出保真，GQA 的八个 Q 不能先平均后当一个 Q。

从两个真实 suffix-only snapshot 的全部 48 层元数据核算：

| 文档 | 每题 suffix Q 数 | 每 KV head 的参考 Q 数 m | m×129 | 共同文档 tokens N |
| --- | --- | ---: | ---: | ---: |
| publicdev1 | 32、31、27 | 720 | 92,880 | 89,726 |
| original | 32、22、21 | 600 | 77,400 | 86,782 |

这个维数上界在第一份文档连非平凡压缩都不保证，第二份也远非 4K/8K。**这是上界不够有力，不是实测矩阵 rank，更不是压缩不可能的下界。** 若朴素显式构建 FP32 moment matrix，每 head 约 31.05/25.02 GiB；隐式乘法/流式求解可避免这个存储，因此也不能把它称为所有实现的最低内存或准备时间。但“直接 NNLS 很便宜”目前没有成立依据。

精确有理反例无需训练/数值容差：取 `k_j=ln(j), V_j=j², j=1..5`，原权重全 1，共享压缩权重为 `(5/4,0,5/2,0,5/4)`。Q=0 时原/压缩 `(Z,N)=(5,55)`，Q=1 时均为 `(15,225)`；Q=2 时 Z 仍均为 55，N 却为 979/985，输出差为 **6/55**。所以两个不同参考 Q 的分子/分母全部精确，也不保证未见 Q。它只反证有限 witness 自动泛化，不能拿这个人工例子断言自然质量必失败。[2026-09-06、v2 09-13 的 whole-query-ball coreset 研究](https://arxiv.org/abs/2609.06327v2) 也已研究更强的统一保证，不能把一般 coreset 定理作为本项目新理论。

执行也不免费：权重必须进入 **logit bias** 才同时影响分母和分子；仅将 V 乘 w 不等价。当前生产 native E0 与 [FI XQA 接口](https://github.com/flashinfer-ai/flashinfer/blob/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/flashinfer/xqa.py) 没有直接暴露该任意 per-token bias 合同。自定义 score kernel、附加维度或其他适配需要重新资格化和完整计时；这不是说 FlashInfer 一切扩展都不可能，而是不能免费继承当前最快路径的计时。完整源仍保活时，物理容量收益也仍不能兑现。

## 停止边界与资源决定

本轮 CPU 筛查已经足够决定**不进入 GPU 阶段**：核心目标/表示已有直接先例，没有独有低成本机制，没有自然未见 Q 质量证据，也没有可继承的原生执行合同。既不启动大矩阵拟合，也不扩写 allocator/quantization/weighted-attention 系统。建议新增 P1/P2 GPU 预算均为 **0**，保留当前约 12.5 P1 / 2.44 P2（含管理预留）供已批准的强基线资格和必要收尾。

若未来出现真正不同的新机制，最低重新进入条件应是：先说明相对 Attention Matching 的具体算法或可推广经验增量；在冻结训练文档上得到有界准备/内存及独立未见 Q 误差证据；与原生 calibrated/uncalibrated FP8、独立 QC 和适用 AM 变体作公平成本比较，并保留既定自然质量与完整应用收益门槛。现在这些条件没有满足，因此不把它列为“待跑候选”，也不重启 joint。
