# 2026-10-16 研究转向筛查：当前 0 个通过新颖性门槛的候选

本轮完成 primary literature / 官方源码核查，以及现有真实 mask 的 CPU 容量与生命周期审计。**没有找到可以诚实宣称为新贡献、并值得立即启动 GPU 实验的转向。** 以下保留三个筛查方向及可证伪条件；它们是被否决或尚未合格的假设，不是三个已提出的新方法。检索截止 2026-10-16，不能把“本次未找到”理解为证明不存在相关工作。

本轮 GPU 使用为 **0 GPUh**。所有计算使用项目 `.venv`，未使用 root、未安装依赖、未修改引擎、未读取正在盲审的 judge 文件或身份映射。源码身份与本地证据 SHA 见 [primary_sources.json](/home/qcjz/iscas/results/pivot_candidates/primary_sources.json)，可复算审计见 [audit_pivot_capacity.py](/home/qcjz/iscas/scripts/audit_pivot_capacity.py) 和 [capacity_bounds.json](/home/qcjz/iscas/results/pivot_candidates/capacity_bounds.json)。

## 实测约束：不能用逻辑稀疏度代替应用收益

- [主模型扩展结果](/home/qcjz/iscas/results/main_capture_expansion/summary.json) 的自然 B3/B4 点仍未给动态行 v2/v3 提供胜过强执行对照的信号。例如 publicdev1 的 L15/s64/4K 热调用，最佳控制约 7.94 μs，v2 16.02 μs，v3 21.40 μs。不能通过遗漏固定 union、最优固定划分或精确层次实现恢复正结论。
- [selector 实测](/home/qcjz/iscas/results/selector/summary.json) 的官方主模型 L31/q0：原两遍+native top-k 为 2719.74 μs，包含完整 FP32 因果分母的融合 selector 为 1556.26 μs，给定已知 FP32 LSE 后为 110.70 μs。1556 μs 是完整 selector，不能全部写成分母本身的测量；110.70 μs 不含 LSE 生产。准备另为 0.423 ms。这是单层单题快照，不是全模型应用时延。
- 原生 BF16 prefill LSE 在两个旧资格点各改变一个所选 token，故复用 LSE 不能默认逐 mask 等价。LSE 复用已有 [MiniKV §3.2](https://aclanthology.org/2025.findings-acl.952.pdf) 先例，top-k/fusion 也不作为候选贡献。
- 低预算质量尚未通过覆盖与事实性非劣门槛；ROUGE 接近不能授权任选 4K 或 10% 作为容量优势点。主模型现有 256-token capture 也不是自然完成质量证明。
- [MSA 适配资格](/home/qcjz/iscas/results/msa_e0_adapter/summary.json) 已说明不能只计裸 attention：准确 gather 后的完整调用明显慢于 native E0。新 allocator 或 selector 同样必须计索引、复制、规划、同步和暂存。

本地 vLLM 0.31.0 的 `block_pool.py`、`kv_cache_coordinator.py`、`kv_cache_utils.py` 同时匹配安装 wheel 的 RECORD 和官方 `db9527a46873454610df6dbedf79a36d6bf1a7f6`，不是本任务补丁。其 [BlockPool](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/block_pool.py#L712) 已有 `unpin_blocks(..., on_reuse)`：引用释放后可读至实际复用。`free_blocks` 归还的是池内可准入块，不释放 CUDA backing allocation；APC 命中又会重新 pin。把弱引用或延后回收单独命名不能成为新意。

[实际主模型初始化记录](/home/qcjz/iscas/results/platform/capture_main30b_tp2_publicdev1_256.json) 为 TP2、48 层、每 rank 两个 KV head、D128、BF16 KV、16-token 页，19095 个池块。一个逻辑池块在每 rank 占 `16×48×2×128×2×2 = 786432` bytes。当前 full-attention 单池的同一 block ID 关联所有层/头；一个 head 或一层留下孔洞，并不意味着整个块可以归还。底层可能采用跨层交织 stride，复制必须按真实 stride 地址处理。[官方池布局计算](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_utils.py#L1767)

该初始化记录每 rank 仅约 1.37 GB 板上空闲。最长文档的完整共享前缀 KV 在每 rank 约 8.52 GB。不能假定在池外再分配第二份完整源；若使用池内空闲页，必须先合法 reserve，迁移完成并停止所有旧 reader 后才能归还源页。保留完整源供未来问题 prefill，会同时保留其容量成本。

## 方向 1：多个消费者共同使用压紧后的精确 KV

**机制与范围变更。** 在所有当前问题完成追加 prefill 和固定 support 后，按层/头取这些已观察 support 的 union，只存一份所需 KV，各消费者继续用自己的精确 token 映射。保留原 RoPE 位置和私有 suffix；原全文末尾 2/10/2 个非整页文档 token 已在各请求 private KV 内，不能丢弃。只有所有旧 dense reader 结束、dense APC 身份失效或转移后，才释放源页。这将明确改变提案“完整源始终只读保留、不减少其容量”的合同；迟到的新问题必须付重新 prefill/重建或保留 dense 源的成本。

**直接先例与剩余区别。** [Zipage §4.4–4.5](https://arxiv.org/html/2603.08743v1) 已解决 token 压缩与 prefix caching 共存、复制后释放原引用和异步压缩。[固定源码](https://github.com/microsoft/Zipage/blob/3cf00ad547e5fe084fa640f431a31e02b45a2869/zipage/engine/block_manager.py#L129) 给不同请求建立独立目标块，未继续合并压缩后仍相同的 token；这留下“压缩后继续共享交集”的实现差异。[Tangram](https://arxiv.org/html/2606.06302v2) 已有按 head group 的独立页与容量回收，其当前 [ragged fast path](https://github.com/aiha-lab/TANGRAM/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/core/kv_cache_manager.py#L122) 明确关闭 prefix caching。这两个具体版本的差异不证明本方向新颖；共享 union 加地址映射、引用计数和复制只是常规工程组合，**当前没有独立的新算法或协议贡献**。

**CPU 预测与已完成证据。** 同一批 3 个真实文档、4 个 budget、2 ranks、层 0/15/31/47、全部当前问题的冻结 mask；每份输入 SHA 已保存。union KV payload 仅为各请求独立所选 payload 总和的 36.61%–43.63%。但这是相对独立复制的 payload，不是相对已有完整共享源的实测容量；只观察 4/48 层。

| 文档 / 自然 B | 4K 不移动可释放页上界 | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: |
| 原 DocFinQA / B3 | 43.07% | 23.33% | 7.32% | 0.18% |
| publicdev1 / B3 | 41.85% | 22.98% | 7.11% | 0.39% |
| publicdev6 / B4 | 49.40% | 30.46% | 13.98% | 3.13% |

上界要求同一 16-token 原页在**所有**观察层、头、rank 和问题中均不再需要；未捕获的 44 层只会降低这些上界。32K 下不搬 KV 几乎没有可兑现整池页，显示必须付迁移代价。JSON 还保存紧密重排所需块数的观察层下界：full pool 至少需要 `ceil((system_tokens + max_layer,head,rank |union|)/16)` 块，而不能用各 head 平均保留率估算池容量。

**最强对照。** 完整共享源 + native E0/E1/E4；Zipage 式每请求独立物理压缩；Tangram 式层/头独立分配；一个普通静态 union 压紧实现。比较容量时必须允许对照用实际释放空间准入额外文档，不能只比较复制字节。比较算子时保持相同 support；PAT、QUILT、PageWeaver 的适用执行机制仍须保留。

**最低实验与停止条件。** 先补全 48 层的整数地址/引用模拟，报告 dense 源仍被 pin、copy destination 峰值、迟到消费者和取消事件。只有提出超出上述普通 union 对照的机制后，才值得做真实 KV 迁移资格：原始 KV 逐元素等价、每 request/head 支持集合等价、无重复归一化、原位置/因果正确、全部层输出与 full-budget logits 资格、归还页可被另一请求真实覆写后原 reader 仍正确。最小资格预估 0.1–0.3 GPUh；这不包含完整服务实验或作者代码完整移植。若必须永久保留 dense 源、峰值无法 reserve、收益只来自独立复制弱对照，或仍只能描述为 union 工程，则停止。**当前在新颖性条件即停止，未启动该 GPU 资格。**

## 方向 2：按消费者退出关系对齐跨层页

**机制假设。** 方向 1 的紧密 union 中，一个物理页可能混有不同消费者需要的 token，只有最后一个消费者退出才能归还。更强的候选设想是：给每个 token 一个当前 consumer bitmap，把同 bitmap 的 token 跨层/头安排到相同 block ID 范围，使一个消费者集合消失时，整池页同步释放。它试图减少的是驻留块随退出事件的累计量，不是 union 的初始字节数。

**区别为何仍不足。** 这增加了跨层池耦合下的布局目标，但 bitmap 分桶、页对齐和引用释放本身都很常规；动态 query grouping 已有 PAT/QUILT/PageWeaver，页回收已有 Tangram/Zipage，上游已有复用通知。必须在这些基线之外提出并验证有意义的布局/迁移策略，不能把“共享 + 生命周期”当论文机制。[PageWeaver](https://arxiv.org/abs/2610.11201) 本身也强调准备成本与强 native kernel 可以消除表面优势；当前提议没有绕过这些成本的理由。

**已完成的最小反证实验。** 在同一 12 个真实 mask 点上做 CPU 整数分配模拟，每个 lane 是一个观察到的 `(rank, layer, head)`。比较三种静态布局：原位置顺序紧密 union；各 lane 独立按 bitmap 排序、不增加容量；每个 bitmap 跨全部 lane 对齐并按最大桶长度预留整页。为保持模型简单，system prefix 统一另占 3 页。枚举 B3 的 6 种、B4 的 24 种退出顺序；等价地穷举所有活跃子集，假设相邻退出事件等时且顺序均匀。**退出时序是合成条件，不是自然完成时间，也没有未来长度 oracle。**

| 比较对象 | 初始块数变化 | 累计 block-step 相对原位置 union |
| --- | ---: | ---: |
| 每 lane 普通 bitmap 排序 | 0 | 减少 0.18%–1.22% |
| 跨 lane 对齐 bitmap 桶 | 增加 20.84%–49.70% | **增加 9.24%–18.24%** |

跨层桶大小不一致使 `sum_signature max_lane bucket_size` 大于紧密 union 的 `max_lane sum_signature bucket_size`。后期确有整页早释放，但本轮所有 12 点的前期 padding 成本都超过收益。两个人工整数 fixture 验证：原本连续分组不产生虚假优势；交织两消费者时，排序确实能显示正确的 4→3 block-step 改善。因此负结果不是指标无法识别早释放。

**最强对照、后续最低实验和停止条件。** 对照至少包括原位置 union、免费 bitmap 排序、完成时才紧密重排、Zipage/Tangram，以及明确标记的知道未来退出顺序的容量 oracle。若有不同的新布局策略，应先在完整 48 层、真实请求退出/迟到 trace 上扣除迁移字节与 peak capacity，不能只挑最后一个消费者剩余的阶段。只有 CPU 证据显示相对上述普通对照至少 10% block-time 改善，且 peak capacity 不增、无需未来信息时，才考虑 0.1–0.2 GPUh 真实复制/reader 生命周期资格；此 10% 是重新进入筛查的工程阈值，不是已证论文收益。**当前简单对齐机制已被本轮 CPU 结果否证；本轮及建议立即投入的 GPUh 均为 0。**

## 方向 3：低秩 query-conditioned observation selector

**机制假设。** 文档 prefill 一次建立低维 key 表示，问题到来后用末尾观察 Q 的小型表示或预测器估计每 head 的 token 重要性，降低现有 QC-SnapKV 的完整历史评分成本。另一变体是近似每条观察 Q 的 normalizer，再保留原池化、保护窗口与 top-k。这会改变 selector 算法与可能的支持集，必须重新验证质量；不能只继承原执行等价性资格。

**直接 prior 已覆盖主要机制。**

| Primary source | 对本假设的约束 |
| --- | --- |
| [Quest, 2024](https://arxiv.org/abs/2406.10774)，[官方代码](https://github.com/mit-han-lab/Quest/tree/01c1623bf9395009520874e989e29f683203b357) | 以 query 和每页 key min/max 估计重要性；query-conditioned 粗筛不是新意。 |
| [Loki, NeurIPS 2024](https://arxiv.org/abs/2406.02542)，[官方代码](https://github.com/hpcgroup/loki/tree/005913bc0b64c3b54d0d96871f4e51e799d7b17b) | 低维 QK 计算排序，再用完整 key 做所选 attention；直接覆盖低秩 selector。 |
| [Q-Filters, 2025](https://arxiv.org/abs/2503.02812) | 从 QK 几何构造单个 context-agnostic 投影；把 query 统计压成廉价过滤器已有先例。 |
| [RetrievalAttention](https://arxiv.org/abs/2409.10516) / [RetroInfer, VLDB 2026](https://arxiv.org/abs/2505.02922) | attention-aware 向量索引、分区与带误差控制的估计是直接强对照。当前 [microsoft/RetrievalAttention](https://github.com/microsoft/RetrievalAttention/tree/03f912c6e917c380d9d90c5ec85bb0f161ba53ef) README 已指向 RetroInfer，不能混称旧版本完整复现。 |
| [SparDA, 2026-06-03](https://arxiv.org/html/2606.04511v1)，[官方代码](https://github.com/NVlabs/SparDA/tree/c8e235bacc3ead8dd19b4ff7a37dce8f09f05e9d) | 蒸馏原 selector 分布、每 GQA 组一个 Forecast head、跳过 softmax 并提前一层选择。训练轻量 predictor 本身不新。其公开代码未附 Forecast 权重，且原实验为 sparse-pretrained 8B，不能假定直接兼容本任务 30B/4B。 |
| [COBS, 2026-07-10](https://arxiv.org/html/2607.09052v1) | §5.2 对各 GQA query 的近似 normalizer 分别归一化后汇总；§5.3–5.4 用低秩 key 协方差和 query 子空间估计 block mass。二阶统计、低秩、保留归一化的组合也已有明确先例。 |
| [SPLA, 2026-01-29](https://arxiv.org/abs/2601.22379) | 二阶 Taylor selector 与未选尾部的 residual linear attention 已有先例。 |

还检查了更广的 reusable cache 路线：[Still](https://arxiv.org/abs/2606.07878)、[ARC-KV](https://arxiv.org/abs/2609.36835)、以及 **2026-10-14** 的 [Latent Compaction with Test-Time Training](https://arxiv.org/abs/2610.20582)。训练或逐上下文拟合共享合成 cache 也不能仅因应用是多问题就被算作新方向。这里不把其摘要质量数字当成本机兼容性或复现证据。

**不能跳过的数学区别。** QC-SnapKV 的未池化分数形如 `s_j = mean_(observation, GQA head) exp(q_i·k_j / sqrt(d) - LSE_i)`，每个 `LSE_i` 覆盖该观察位置的完整因果历史。单条 Q 的 top-k 可去掉公共分母，多条 Q 的归一化分数再平均通常不可以。把 Q 求均值再打一次分，也不能与先指数/归一化再平均交换。COBS 已处理近似 GQA normalizer，因此“加回分母”不足以建立区别。

唯一更严格的目标是：以低成本区间证明池化后 top-k 边界不变，不确定的 token 回退精确计算；池化 halo、保护位置和低原始位置 tie-break 都须进入证明。但一般 top-k/tail 证书已有 [2025 年理论工作](https://arxiv.org/abs/2512.07647)，本轮未得到针对此混合评分的更强、且便宜到可用的界。把普通区间传播套到该公式、或拟合 observation query 的小型 coreset，当前都不足以证明新颖性。

**最强对照与最低实验。** 若后续真的提出区别，应先比较给定真实 prefill LSE 的已知一遍 fused scorer+native top-k，单列 LSE 取得/精度成本；以及 Loki/Quest/RetroInfer 的适用机制、SparDA/COBS 的能力边界和原生 FastKVzip 全局 gate。不要以 2.72 ms 的旧两遍作为唯一基线。最小数值筛查是冻结已有 capture、FP64 小 fixture、score/所选集合/tie 完整比较，并记录 low-rank/index 建立、更新、normalizer、fallback 触及的全部 K、峰值显存与完整调用。此阶段预估 0.05–0.15 GPUh，不含 predictor 训练或自然质量评价。近似 mask 一旦变化，必须另走相同 HF sampling 的开发质量与覆盖检查，不能靠高 mask recall 宣称无损。

**停止条件。** 若机制只是上述 prior 的移植/重命名，则在 GPU 前停止；若声称精确但任何位置/边界 mask 失败，停止精确主张；若完整调用没有超过已知 LSE 强对照的稳定收益，或收益依赖不计文档建索引成本/无效的分母等价假设，则停止性能主张。**当前新颖性条件未通过，不建议为这个空泛方向启动训练或大规模 GPU 搜索。**

## 可以继续的范围核查与不应重启的路线

root 已发现公开训练集存在自然 8/9/10/11/14/19 问的文档，而当前协议人为最多 4 问。扩大到真实 ready 的 B8/B12/B16 可以检验原机制在更多 M16 tile 时是否有机会；它属于原动态行方法的**适用范围扩展**，不是第四个新贡献。需要单独记录从 B≤4 协议扩展的理由，保持自然问题、不复制问题、不等待兄弟请求造并行，并仍与 PAT/QUILT/PageWeaver 和最佳固定组比较。本筛查未修改冻结评价集，也未替 root 运行该 capture。成文期间两份自然 B8 capture 和四个有界 replay 已完成；[B8 汇总](/home/qcjz/iscas/results/b8_pilot/summary.json) 的 dynamic 热调用为 292.99–490.23 μs，四点各自最快的 native/union 对照为 10.84–23.76 μs，全部失败。更多 owner 的尾部碎片，以及 region/split 索引与状态维护，已使这次扩展失去继续布局搜索的依据。不能把只胜过同样缓慢的 region 固定行实现写成成功。

对“把碎 owner 区域合并到 2/4/8 请求 superset，保留 token mask，再切换 superset”的追加筛查同样没有新转向结论。[QUILT §3.4](https://arxiv.org/html/2610.11134v1#S3.SS4) 已有精确 tail push-down：把利用率低的共享尾送到下一层，与其他 workload 合并；不能因它另有可选近似 clipping 就忽略这部分 exact prior。[PageWeaver §3.2/4.2](https://arxiv.org/html/2610.11201v1#S4.SS2) 已权衡 group size 与 masked union，保留跨页 FP32 state，并跳过无有效成员的子组。[PAT §5](https://arxiv.org/html/2511.22333v3#S5) 已按收益合并区域、选择 M/N tile。剩余字面差异是同一 CTA 内跨非层次区域改变行映射并维护 canonical state，这正是原 BranchTile 命题；superset coarsening 是该命题的布局调优，目前不是独立新贡献。不建议因新增此变体无限重启搜索。

普通 union、动态分组、LSE 复用、top-k/gather 融合、跨层 plan cache 均不升级为创新。投机多 token 的简单场景迁移也不重新开题：已有 [SSV](https://arxiv.org/abs/2605.19893)、[SpecAttn](https://arxiv.org/abs/2602.07223)、[S3R](https://arxiv.org/abs/2610.13202) 必须先作为直接 prior。更低预算的优质共享 cache 若能由成熟基线给出，应保留其应用优势，而不是为了原算子强行保留较差 selector。

复算命令：`CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_pivot_capacity.py`。本轮结论是保留可审计的容量机会与负结果，B8 有界扩展也按停止规则收束；不把暂未解决的工程集成缺口包装成新论文机制。

## 追加判断：改成测量/架构论文是否合理

**有可研究的经验命题，但当前还没有足够的新颖性与验证，不能把失败日志改标题就视为完成 ISCAS 论文。** 可以研究的具体命题是：在固定支持集的 sparse decode 中，逻辑 KV 复用率相同的输入，能否由工作集相对缓存容量、N 维尾部碎片、region×split 元数据与状态维护、实际并发 CTA 资源约束，预测 native 独立执行、固定 union 和动态行执行的胜负边界？输出应是对未见文档/shape 的预先预测及失败边界，不是训练集延迟拟合。

现有证据还不足以归因于架构。[B8 producer](/home/qcjz/iscas/src/branchtile/tilelang_kernel_b8.py:108) 的每个 split 串行遍历全部 region，region 内再按 split 划分 token；因此 region 数增加可以引入大量控制、空区间和状态操作。300–500 μs 对 11–24 μs 的差距不能仅用 1–2 倍 tile 数解释，也不能直接证明 SM120 不适合共享。必须首先排除可被普通 worklist/local-index 消除的软件病态；这一控制的价值是因果诊断，不能重新命名为算子创新。

更一般的“共享不必然快”已经不是新结论：PAT 建模 packing/资源与 split-merge 成本，QUILT 处理 tail tile，PageWeaver 明确观察 preparation 和更强 native backend 消除收益。测量论文若要超过这些，至少需要一个**不依赖候选名称、在未见输入和第二架构上仍有预测力的联合模型**，并用匹配输入的干预证明哪些因素引起 crossover。简单把这些已知项相加不是新理论。

可先冻结一个待证、非已建立的模型边界：完整调用包含 `prepare / 实际复用次数 + launch + max(各内存层服务时间, MMA 工作时间, scalar 工作时间) + 未被隐藏的同步/状态时间`。这里的字节必须区分物理 DRAM 与 cache 服务，不能把逻辑加载次数都当 DRAM 流量；重叠部分不能重复累加。`region×split` 只能作为特定 worklist 实现的代价项，不能作为任意共享算法的下界。参数应来自独立微测量/编译资源和固定校准集，而非对每个自然点重新拟合。

**最小剩余验证分阶段进行；下表是建议，不是已分配/已消费 GPU 预算。**

| 阶段 | 必需证据与明确停止点 | 预估新增 GPUh |
| --- | --- | ---: |
| 0：CPU 可识别性 | 冻结 shape/文档分组、有限 backend 集与假设；核查现有完整调用组成。用现有数据比较 bytes-only、tile-only、常数最佳 backend 与联合模型的留文档/留 shape 预测。所有点都选 native 的分类器没有“预测边界”的证据。若缺少可测 crossover 或不能区别实现因素与硬件因素，停止架构主张。 | 0 |
| 1：排除软件病态 | 少量已有 B3/B8 点，matched support 下比较当前 region 扫描与普通非空 worklist 控制，分别记录准备与执行；另作固定物理工作集/逻辑复用、改变 owner 切段及 split 的受控干预。合成输入只用于因果机制，不能充作自然应用样本。若差距主要由索引控制解释，结论限于该实现，不能拟合成普遍架构相变。 | 0.15–0.30 |
| 2：预注册预测验证 | 至少两个模型/两类任务、B1/B2/B4/B8、绝对 budget 与小于/跨越/大于 L2 的有效工作集，覆盖 native 与 union 各自占优的区域。每个自然文档按簇留出，多个 layer/step 不当独立样本。建议进入门槛：留出延迟中位相对误差 ≤20%、p90 ≤35%，选择 regret 中位 ≤5%，且明显优于简单模型；阈值需在新增验证前冻结。这些是建议筛查标准，不是领域公认标准。 | 0.3–0.8（不含新模型下载/捕获） |
| 3：跨架构复核 | 至少另一 GPU 架构，用相同冻结支持、精度合同、完整调用与兼容强 baseline；只重测硬件校准项，不按测试数据换公式。若仅能在 5090 验证，题目和结论收窄到 SM120 个案，不能作跨架构性能定律。八张同型号卡是重复测量能力，不是八种架构。 | 约 0.5–1.0，且目前无第二架构资源保证 |

当前只有同型 5090；在这里模拟更小 SMEM、调 tile 或更改 clock 不等价于第二架构。旧 [ncu 资格记录](/home/qcjz/iscas/results/validation/ncu_permissions.txt) 显示性能计数器权限失败；本轮未尝试提权。若仍不可访问，编译寄存器/SMEM、CUDA event、工作集容量扫描与时序干预可以支持部分分析，但不能把 hot/cold 标签写成已测 L2 hit rate 或已测 DRAM 流量。需要明确保留这项可辨识性限制。

算子测量若只研究执行等价性，可沿用已冻结 support 的数值资格；一旦外推应用选择/压缩价值，仍需质量、真实 EOS、准备摊销和服务容量证据，不能由本轮负算子数据替代。目前最合理的产出是**可复现的 SM120 负结果与研究范围报告**。只有阶段 1 排除实现缺陷、阶段 2 显示新的可预测规律，并补上与主张相称的架构证据后，测量/架构论文才值得独立立项；当前不作录用或满足 ISCAS 要求的承诺。
