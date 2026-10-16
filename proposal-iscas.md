# BranchTile：面向共享稀疏注意力的 GPU 片上数据流

日期：2026-10-16。目标会议：**ISCAS**。拟题：**BranchTile: Active-Query Compaction with On-Chip Accumulation for Shared Sparse Attention**。

本方案研究一个集中的执行机制：**多个问题访问同一文档的不同稀疏 KV 子集时，在同一 GPU 线程块内随 KV 区域切换紧凑 query 行映射，同时把每个原始 query 的 softmax 累积状态保留在片上。** 希望同时减少矩阵 tile 空转和由区域拆分引入的全局中间结果，而不改变保留哪些 token。

实验平台按 **8 张 RTX 5090、每卡 32GB** 规划。主模型采用官方 FP8 的 Qwen3-30B-A3B-Thinking-2507、TP=2；4B BF16 单卡用于边界验证。设备连接、拓扑、引擎兼容性和实际容量尚待核实。这是一份待验证的研究方案，尚无新机制的 GPU 性能、能耗或完整应用结果；本次文档工作不启动模型或恢复已暂停的 GPU 实验。

## 一、问题

### 1.1 真实场景与研究对象

目标场景是长文档的并行分析：给定同一份年度报告，分别回答 2–4 个自然问题，允许逐题返回、验证和重试。输入的共享文档在前，各问题在后。每个问题根据自己的内容选择稀疏历史，生成时只读取所选文档 KV，问题与生成后缀完整保留。

[DocETL parallel_map](https://ucbepic.github.io/docetl/operators/parallel-map/)提供同输入多 prompt 的并行处理；[vLLM APC 文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)将同一长文档的重复提问列为前缀复用场景。这支持应用需求存在，但**不证明**请求一定同 GPU、同时 decode，或自然稀疏集合一定具有有利的交叉结构。[Unstract SinglePass](https://docs.unstract.com/unstract/unstract_platform/features/single_pass_extraction/single_pass_extraction_intro/)也表明把多问合并为一次调用是现实替代，必须纳入应用对照。

令问题 A、B、C 的保留集合分别为 S_A、S_B、S_C。一部分文档 token 可能同时服务 A/B，另一部分服务 A/C，还有一部分服务 B/C。对这种非层次交叉：

| 执行方式 | 已有优势 | 可能付出的费用 |
|---|---|---|
| 每问题独立 attention | 支持集紧凑，接口成熟 | 同一 KV 可能被多个 CTA 读取；实际重复是否进入 GDDR，要看 L2 |
| 固定 query 分组或全 union | KV 加载可复用，输出状态可跨区域驻留 | 每个区域仍占完整 query tile，部分矩阵行没有有效工作 |
| 按精确 owner 集合分别执行 | 活跃 query 少，矩阵计算紧凑 | 短任务、全局 partial、归并及 launch；并非所有实现都有这些费用 |
| 本方案 | 区域间改变活跃行映射，跨区域保留原 query 状态 | 片上重映射、同步、寄存器/共享内存压力和潜在 occupancy 损失 |

这里的 owner 指哪些请求保留一个 token。共享物理 KV 页本身不是新贡献；减少理论读取量也不等于实际加速。需要回答的是：**在 5090 的有限片上资源下，动态压紧行所节省的有效矩阵 tile，能否覆盖行映射和状态维护费用？**

### 1.2 为什么这是一个适合集中研究的问题

以 D=128、BF16 attention、最小 M16 矩阵行 tile 为例，GQA8 模型的 4 个请求共有 32 个 Q 行。如果某一区域只属于 1–2 个请求，紧凑执行可由两个 M16 降为一个；属于 3–4 个请求时通常仍需两个。相比之下，GQA4 的四请求只有 16 行，压紧通常不能减少 M tile。这是可验证的硬件差异，不是“稀疏率越高就越快”的泛化判断。

真正值得寻找的是**长度足够、无法被一个固定分组同时利用的自然区域**。如果支持关系主要是 AB｜CD，最优固定划分可能已经足够；如果回答很短、分支很快退出，准备和重映射也可能无法摊销。B1/B2、完全相同支持、树状支持和 GQA4 都保留为边界，不删除不利样本。

普通顺序聊天和同 prompt 多采样不自动满足本问题。首版不以 Agent 工具循环或跨卡共享 KV 为应用前提，只处理同一 replica 当前步骤已经 ready 的请求，不等待兄弟分支来制造并行窗口。

### 1.3 与既有探索的关系

项目此前在工具边界预算调度中观察到：部分配置没有足够单步加速空间，选择器、准备与额外工作可能抵消局部节省。因此本方案先冻结稀疏语义，测强基线上的完整 attention 费用，再做有界算子设计；不预设稀疏化必然造成重复输出，也不把减少 token 数或降低任务质量当作收益。

既有 [CPU 费用反例](results/analysis/joint_sparse_execution_tradeoff_20261015.json)曾出现“少读 30720B KV，却多出 33280B 逻辑 partial”的形状。它只说明中间结果值得计费，不含真实 split、缓存和 direct store，不能作为新 GPU 方法已经有效的证据。

## 二、最新进展

### 2.1 直接相关方法与具体差距

下表截至 2026-10-16 核查；论文描述、代码能力和本方案的待验证增量分别处理。

| 工作 | 已解决的部分 | 对本方案的约束 |
|---|---|---|
| [SnapKV](https://github.com/FasterDecoding/SnapKV)、[ActKV](https://arxiv.org/html/2609.31395v1) | 根据输入/任务选择或压缩 KV；ActKV 面向 Agent 工作负载 | 不能将 token 选择作为本方案贡献；ActKV 没有“做完”所有执行映射问题，但不能据此推断本方案一定更快 |
| [KVzip](https://github.com/snu-mllab/KVzip)、[FastKVzip](https://arxiv.org/html/2601.17668v2) | query-agnostic 压缩及高效部署，可供多个问题共享 | 如果一个共同压缩历史已经达到更好质量—成本，本方案的适用范围会收窄 |
| [FlashInfer / Cascade](https://arxiv.org/html/2501.01005v1) | paged/ragged、共享前缀执行、split 和 LSE 合并 | 共享源独立执行、精确 cascade 都是强对照；不以一个低效自研 executor 代表现有技术 |
| [PAT](https://arxiv.org/html/2511.22333v3) | 共享结构分组、多 tile、费用评估、pack–forward–merge | 小 tile、费用表、计划缓存不是独立新意；必须比较其适用机制 |
| [QUILT](https://arxiv.org/html/2610.11134v1) | 精确层次交集分解、tail 处理和联合执行 | 保留不开裁剪的 QUILT-exact；全部固定 query 划分不支配其层次分解 |
| [PageWeaver](https://arxiv.org/html/2610.11201v1) | query regrouping、masked union；固定子组可跨页维持输出 | “跨页不写 partial”已经存在，不能把其普通融合实现当作本方案创新 |
| [Fused3S](https://arxiv.org/html/2505.08098v1) | 稀疏 SDDMM–softmax–SpMM 的 Tensor Core 融合、在线状态及布局 remapping | 片上累积和寄存器重排本身已有；拟议区别仅是跨区域改变活跃 query 身份映射，并维持原 query 的累积状态 |
| [cuDNN Block Sparse Attention](https://docs.nvidia.com/cudnn/latest/fe-oss-apis/bsa/) | 当前支持 SM120 的部分 BF16/FP16、D128、GQA 稀疏形状及 split/merge | 若接口能精确表示输入 mask，应比较优化原生实现；block-level mask 不自动等价于逐请求任意 token 支持 |

本轮阅读没有发现上述工作明确完成本方案的全部组合，但这不是首创证明。若后续源码发现同样的数据流，仍可评估特定 GQA/片上布局的具体增量，不靠换应用名称建立创新性。

Tangram 还存在一个必须解决的集成前提：[固定版本源码](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/config/compression.py#L288)在压缩/ragged 路径关闭 APC。记录见[本地源码核查](results/analysis/branch_tile_tangram_source_20261015.json)。不能只移除 flag 就假定共享安全；要实现只读完整源及每请求稀疏视图，并把该工程收益与新算子收益分开。

### 2.2 ISCAS 定位与实际实验规模

ISCAS 官方范围包含 **High-Performance Computing for AI**。本方案定位为 GPU 注意力的数据流与片上资源设计：用真实硬件测量解释矩阵利用率、存储流量、片上状态和延迟/能耗之间的取舍。会议并未要求必须实现 ASIC 或 FPGA。[官方征稿](https://2027.ieee-iscas.org/call-for-papers)

近期一手论文提供以下参照，不能将三篇样本当作统一录用标准：

| 已发表/接收工作 | 真实实验形态 | 本方案可借鉴的尺度 |
|---|---|---|
| [AMUSD，ISCAS 2025](https://bradmcdanel.com/research/speculative-decoding/)，[全文](https://arxiv.org/html/2410.17375v1) | A100 上的推理执行机制；8B verifier＋1B draft；HumanEval 164 题、MT-Bench 80 场景、自建 100 条多轮任务；测延迟、利用率、功耗和能量 | 现成 GPU 软件机制有先例。其异步方案与基线活跃 GPU 数不同，本项目仍采用同资源对照 |
| [PRIMAL，ISCAS 2026](https://arxiv.org/abs/2601.13628) | PIM 设计，Verilog/综合布线加 cycle-level 模拟；1B/8B/13B 三模型，多输入输出长度，功耗/面积/吞吐消融 | 结构清楚、少数模型与关键消融；其结果不是实际流片，也不能替代本项目任务质量评估 |
| [UbiMoE，ISCAS 2025](https://epapers2.org/iscas2025/ESR/paper_details.php?paper_id=1809)，[全文](https://arxiv.org/html/2502.05602v3) | ZCU102/U280 实测；主要为 M³ViT，数据流、资源分配及延迟/功率 | 可围绕一个数据流讲清资源取舍；跨设备/精度结果不能直接视为严格同配置加速 |

因此首版规模定为：**一个 GQA8 主模型、一个 GQA4 边界模型、一个主 QA 数据集、一个摘要边界数据集、同架构重复及强算子对照**。八卡主要缩短实验周转，不扩展成八卡调度论文。

当前 ISCAS 2027 投稿格式为最多 5 页，**技术内容最多 4 页，第 5 页仅参考文献**。投稿系统的延期截止为 **2026-10-22 23:59 UTC−05:00，即北京时间 2026-10-23 12:59**。这是一周内的期限，当前未有实现和资格结果，不据此承诺赶上本届；研究范围按四页技术内容收敛。[官方作者说明](https://epapers2.org/iscas2027/ESR/authorinfo1.php)

## 三、研究 gap

**具体 gap：对于不同 KV 区域需要不同 query 子集的 decode，在一个 CTA 内实现可变的紧凑行映射，并让原始 query 的归一化/输出状态跨映射驻留；证明这种状态组织在有限片上容量下，优于同样融合的固定行映射。** CTA 即 GPU 线程块。

需要同时成立的三个证据是：

1. **机会真实。** 自然支持集里确有相对最优固定划分、精确层次共享仍剩余的 tile/读取空间，并有足够共同 ready 步数。
2. **数据流有效。** 行映射减少的计算及区域中间结果，超过映射、同步、spill、occupancy 和额外 split 的费用。
3. **应用值得。** 同质量下，包含文档准备、私有后缀、模型其他算子和 TP 通信后，完整请求仍受益；容量劣势如实保留。

如果只胜过每区域强制写 partial 的实现，尚未证明新行映射有价值；如果只胜过各请求私有复制的实现，证明的是共享工程；如果仅有合成算子收益，不能宣称真实文档应用收益。

本论文集中于第二项，以第一项建立动机、第三项给出应用验证。预期贡献是一种具体可复现的 GPU 执行增量，不是通用稀疏选择器、最优调度器或全能系统。

## 四、Idea

### 4.1 输入合同：固定稀疏集合、只读共享源

同一文档的完整 KV 只保存一份；只有模型、精度、tokenizer、chat template、RoPE 和实际 token 前缀一致时才能共享。尾部非整页沿用引擎正确的 COW 语义。各问题的追加 prefill 读取完整历史，随后固定该响应的文档支持集；问题与生成后缀始终完整私有。

首个后端采用明确的 **QC-SnapKV 变体**：取问题 suffix 末尾 min(32, suffix 长度) 个真实 post-RoPE Q；对各 Q 的完整可见历史做 causal softmax，再平均同 KV head 对应的 Q heads 与观察位置，文档分数作宽 7 平均池化。文档首 4 和末 32 token 固定保护，计入历史预算；其余按 score 降序、原位置升序稳定取 top-k，重叠去重。观察窗不回填文档 Q，问题/生成 KV 另行完整计费。评分分块扫描，不导出 N×N 矩阵。

历史预算开发网格为每头 4K/8K/16K/32K，加 full-budget 数值对照。主预算按开发质量冻结，不以人为过小预算制造问题。这不是原生 SnapKV 复现，也不是本论文的算法创新；原生 Tangram 的评分/归约语义作为跨算法对照，不能混入同 mask 消融。

只读完整源意味着本方案**不减少完整文档 KV 容量**。共享稀疏索引、观察 Q、workspace、CUDA graph 和私有后缀均计显存；可释放完整源的原生压缩方案保留其容量优势。

### 4.2 按区域切换计算行，按原 query 保持状态

对每个 layer/KV head，构造 `(token 原位置, owner 位图)`。B≤4 最多 15 种非空 owner 集合。将同 owner 的索引组织为区域，区域内保持原位置顺序；不在 GDDR 中重排或复制 KV payload。索引构建、排序、传输及 SMEM staging 都计费。

一个 producer CTA 处理一个 `(文档组, KV head, KV shard)`：

1. 为原始 `request × Q-head` 分配固定状态槽，保存 Q 及 FP32 的 `(m, l, u[D])`。这是 **canonical 状态**，其身份不随区域变化。
2. 顺序读取 shard 内的区域。由 owner 描述符选出本区域活跃 query，将其映射到紧凑 M16/M32 工作行；加载该区域 KV 并执行 QK、softmax、PV。
3. 将结果更新到对应原 query 的 canonical 状态；本区域不活跃的 query 状态保持不变。映射可以用 warp shuffle 和共享内存 staging，不能假定任意跨 warp 重排免费。
4. 继续处理下一种 owner 区域。直到 shard 结束才输出每个相关 query 的 partial；如无需 split，可直接写最终 O。跨 shard 仍采用标准稳定归并。

简化数据流为：

```text
共享文档 KV + 区域索引
        ↓ 按当前区域 owner 选择活跃行
紧凑 QK → softmax → PV
        ↓ 映回原 request / Q-head
片上 canonical (m, l, u)，跨区域保留
        ↓ 仅在必要 KV split 边界写回
标准 split merge → 最终 attention 输出
```

对某 query 的一块新结果 `(m_t, l_t, u_t)`，使用已有 online-softmax 递推：

```text
m_new = max(m, m_t)
l_new = exp(m - m_new) * l + exp(m_t - m_new) * l_t
u_new = exp(m - m_new) * u + exp(m_t - m_new) * u_t
O = u / l;  LSE = m + log(l)
```

空块跳过；全空行 O=0、LSE=-inf，不能直接计算 -inf−(-inf)。统一自然对数/log2 约定。每个 request–token 对恰好参与一次，原位置、RoPE、causal 约束保持。数学上等价于给定稀疏集合的 attention，浮点顺序及自由生成轨迹不保证逐 bit 相同。

**online softmax 与融合不是新贡献。** 待验证的机制是：改变工作行映射时，仍能经济地保持原 query 的状态，并真正减少被发出的矩阵 tile。

### 4.3 用片上容量约束融合，不追求消除所有 split

以 B4/GQA8/D128 为例，32 行 FP32 `(u,m,l)` 共 `32×(128+2)×4 = 16640B`，约 **16.25KiB**；BF16 Q 再需 **8KiB**。N32 的 K/V 双缓冲为 **32KiB**，N64 为 **64KiB**。逻辑总量分别约 56.25/88.25KiB，尚未计临时累加器、mask、barrier 和地址。

这些数据分布在寄存器与共享内存，不能把逻辑总量当作实际 SMEM 用量或承诺全部寄存器驻留。高寄存器压力可能 spill，SMEM 增加可能使每 SM 只能驻留一个 CTA。5090 为 SM120；CUDA 能力表给出每 block 最多 99KB 共享内存，并区分统一缓存和可配置 SMEM。实际资源以设备属性、编译结果和 profiling 为准，不沿用数据中心 Blackwell 的资源假设。[CUDA 能力表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

首版只开发 M16/M32 两种工作行形状，明确在**同一 producer CTA 内**通过分支或一至两轮 M16 MMA 执行，不能切换独立 kernel 后仍宣称状态片上驻留。初始实现把 canonical 状态放共享内存，当前活跃行的工作累加器放寄存器，在区域边界装入/映回；长区域内部继续在线累积，减少状态搬移。编译寄存器与 SMEM 按该 CTA 最大路径计，活跃行从 32 降到 16 不自动降低资源占用或提高 occupancy。N32/N64、4/8 warps、合法的 1/2 stage 在开发 capture 上作有限调优，先剔除超资源配置；状态搬移、同步和 bank conflict 必须实测。

KV shard 的划分仍要提供足够 CTA 并行度。所有强基线同样允许调优 split、直接输出和融合私有 suffix；不能通过强迫基线每区域写 partial，或把候选整个历史塞进一个低并行 CTA 来制造比较。候选只消除**由区域边界额外引起**的写回，不承诺消除所有 split merge。

### 4.4 有界选择与引擎集成

按真实 GQA 行数、区域长度、预估 M/N tile 数、资源配置和 split 数选择紧凑路径或固定行回退。首版使用开发 profiling 冻结的少量配置/阈值，不学习未来输出长度，不把小费用表作为额外贡献。B1/B2、GQA4 或映射频繁而区域很短时允许回退。

在一个成熟推理引擎分支中集成只读源、稀疏视图和联合 attention 接口，优先评估 Tangram/vLLM＋FlashInfer/CUDA 实现；最终版本按 SM120/FP8 MoE 的真实支持冻结。单卡 capture 原型只能证明算子，不代替引擎应用结果。

组内请求自然退出、暂未被调度或迟到时，只用当前 ready 子集。缓存描述符失配立即走正确回退，不阻塞可执行请求。共享源引用计数、异步 reader、取消、抢占、COW、slot generation 和图缓存失效都要通过测试。连续补入沿用引擎原调度，不新增通用调度器。

首版范围固定：一个选择器、一种数据流、两个行 tile 族、一条标准 split merge 路径。动态预算、物理 owner 压实、工具预测、跨卡 KV 迁移、ASIC/RTL 均不进入核心贡献。

## 五、实验设计

### 5.1 研究问题与最小证据

| 研究问题 | 必要实验 | 不能替代它的证据 |
|---|---|---|
| RQ1 自然输入是否有适合压紧的区域 | 自然支持集、M/N tile 差额、共同 ready 时间 | 合成 AB/AC/BC、仅 Jaccard 或 prefix cache hit |
| RQ2 可变行映射是否有独立价值 | 同输入同 mask、与同样片上累积的固定行强对照 | 只胜旧的 per-region partial 版本 |
| RQ3 片上组织如何影响性能和能耗 | 完整 call、资源/流量 profiling、实测 joules | 理论读取量、额定功耗乘时间 |
| RQ4 应用是否得到净收益 | 自然答案质量、冷/暖完整时间、显存和 TP 通信 | 固定输出 replay、只报 attention kernel |
| RQ5 边界在哪里 | GQA4、短回答、树状 mask、自然退出、容量压力 | 删除负向配置或按测试收益挑题 |

### 5.2 数据集和冻结划分

**主数据为 [DocFinQA](https://aclanthology.org/2024.acl-short.42/)**，使用完整年度报告及自然问题。延续项目原 **8 开发公司年度（7 公司）、100 评价公司年度（48 公司）、167 训练公司年度**。8 开发报告不为凑 B4 而替换。

旧评价 manifest 每报告只保留 1 问，需要按输入身份补齐同全文的自然问题：保留旧代表题，其余按问题 SHA 排序补到最多 4 个，去重后不足 4 个就保留实际 B。只有相同 token 全文可共享。100 报告全部保留，并分别报告 B1/B2/B3/B4；此前训练池 115 个 B4 公司年度不等于正式测试都满足 B4。长度分层用目标 tokenizer 重算，不能沿用旧分词长度当实测。

公司年度不是独立公司；同公司跨年样本相关，正式按公司聚类。训练/评价存在同公司不同年份的情况，不宣称公司外泛化。不因模型答错、mask 不交叉或方法无收益删除报告。输入超限、资源失败和不可服务项单列，并保留总体分母。

**次级数据为 [QMSum](https://github.com/Yale-LILY/QMSum)** 的同会议 specific questions：官方 train 开发，官方 test 的 **35 份会议**评价，按同样输入规则冻结自然 B。该场景主要使用 4B 验证摘要及短上下文/GQA4 边界；不拼接会议造长输入，不用人工证据跨度代替模型支持集。

独立问题初始输出 cap=8192；一次多问为实际 B×8192 且满足模型总上下文上限。自然 EOS、无 min_tokens。若开发阶段频繁碰 cap，所有相关臂统一调整后冻结，测试不逐题放宽。分别记录 reasoning/最终答案 token、截断和解析失败。所有方法使用相同文档前置格式，并先测模板修改的质量影响。

### 5.3 8×5090：模型、容量与并行方式

[NVIDIA 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)为桌面 RTX 5090 **32GB GDDR7、PCIe Gen5、无 NVLink**；[公布带宽](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)为 1792GB/s。32GB 按每卡独立容量规划，实际可用字节由设备查询。八张卡不构成一块统一显存，不将 GDDR7 写成 HBM。

| 用途 | 冻结前的首选配置 | 部署条件 |
|---|---|---|
| 主模型与自然 GQA8 capture | [Qwen3-30B-A3B-Thinking-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8)，48 层、32Q/4KV、D128；TP2，BF16 KV | 官方 FP8 权重/scale，成熟引擎 SM120 MoE 支持；每 rank 理想 16Q/2KV，GQA 仍为 8 |
| GQA4 和摘要边界 | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)，36 层、32Q/8KV、D128；BF16 TP1 | 主要自然输入≤64K，较长工作点需实际容量通过 |
| 部署方式控制 | 对应 Instruct-2507，在开发集比较同质量成本 | 若较短输出已更优，不能为获得长 decode 而只选 Thinking；主 checkpoint 在测试前冻结 |

主模型 FP8 指权重部署选择，不是本方法新增量化。候选与所有主基线使用同 checkpoint、权重/KV 精度和模板；数值验证仍覆盖 BF16 attention。保留 2507 官方 RoPE，不用旧库默认配置覆盖。FastKVzip 等需匹配 gate；没有对应 gate 就明确不兼容，新校准单列费用，不用失配结果贬低基线。

容量预估按 `KV字节=2×层数×每rank KV头数×D×token数×2` 计算。以下 K=1024，均为 **每卡 GiB**，只含理想权重、完整源及 B4 每题 8192 token 的生成 KV：

| 模型配置 | 文档长度 | 权重 | 文档 KV | B4 生成 KV | 理想小计 |
|---|---:|---:|---:|---:|---:|
| 30B 官方 FP8，TP2 | 64K | 14.52 | 3.00 | 1.50 | 19.02 |
| 30B 官方 FP8，TP2 | 96K | 14.52 | 4.50 | 1.50 | 20.52 |
| 30B 官方 FP8，TP2 | 128K | 14.52 | 6.00 | 1.50 | 22.02 |
| 4B BF16，TP1 | 64K | 7.49 | 9.00 | 4.50 | 20.99 |
| 4B BF16，TP1 | 96K | 7.49 | 13.50 | 4.50 | 25.49 |
| 4B BF16，TP1 | 128K | 7.49 | 18.00 | 4.50 | 29.99 |

30B FP8 [官方权重索引](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8/blob/main/model.safetensors.index.json)总大小约 29.03GiB，单卡不适合再放长 KV；TP2 均分只是估算。问题 suffix、参数复制、对齐、量化元数据、prefill 激活、索引、图、NCCL 和 workspace 另算，不能把小计当实测峰值。主模型先每 replica 只驻留一个文档组；第二组仅在实际峰值允许时增加。4B 的 128K/B4 不作为承诺可运行的主配置，长报告不静默截断以适配单卡。

八卡按以下**逻辑用途**使用，物理 UUID/卡对由实际拓扑和分配决定：

| 资源 | 首版用途 |
|---|---|
| 一对 5090 | 主模型 TP2，开发和一臂正式评价 |
| 第二对 5090 | 配对对照/重复，随后交换方法与卡对 |
| 两张独立卡 | 单卡真实 capture 重放、数值检查和 kernel 消融 |
| 一张独立卡 | 4B 摘要与 GQA4 边界 |
| 一张独立卡 | 能耗/复核；正式能耗窗口避免邻近重负载干扰 |

先验证一对 TP2，再扩大并行。单卡 capture 重放保存 TP2 rank 的真实两 KV 头形状，不伪装成单卡 30B 完整推理。四对 TP2 replica 可用于后期独立重复，但不是必须完成的扩展性实验，也不能把一对吞吐直接乘四。

运行前核对 `nvidia-smi topo -m`、PCIe 链路、NUMA、peer-access 与 NCCL 实际 transport/all-reduce。无 NVLink 不等于必然无 P2P，也不等于两卡一定直连。主应用完整计入 TP 通信；若平台的 host 路径或兼容性使 TP2 不可用，先记录阻断并调整部署，不能以单卡算子结果冒充完整应用。

沿用 Slurm 规则：独立单卡任务一张卡一个作业；TP2 因权重与长 KV 容量确需两卡，单作业申请两卡并记录原因；监控仅覆盖分配 UUID。新方案按用户给定八卡资源规划，实际并发以可用分配为准，不占用或取消他人任务。本文件不代表已经恢复 GPU 运行。

### 5.4 同支持集的强执行 baseline

所有臂共用同一 Q/K/V、原位置、支持集、物理共享源和完整私有 suffix。分别调优合法 kernel、split 和 tile；完整调用包括准备摊销、producer、suffix、merge、launch 与必要同步。

| 编号 | 方法 | 比较重点 |
|---|---|---|
| E0 | 优化 FlashInfer 独立 paged/ragged，物理源共享 | 不能用重复私有复制或低效自研 attention 代替原生强基线 |
| E1 | 最佳固定 query partition | B4 的全部 15 种划分，组内精确 masked union；部署选择器与每快照事后 oracle 分开 |
| E2 | 固定全 union＋CTA 内在线累积 | **关键控制**：跨区域同样不写额外 partial，只保持完整固定 Q 行映射 |
| E3 | owner 区域紧凑执行＋全局 partial/merge | 活跃行压紧而无跨区域片上状态复用；允许合理 split/直接输出，计真实费用 |
| E4 | QUILT-exact / FlashInfer Cascade | 不裁剪支持集，保留层次交集与 tail 处理，计树构建/merge |
| E5 | PAT / PageWeaver 的适用执行机制 | 多 tile 固定分组、union、直接输出/融合；移植必须清楚标注 |
| E6 | BranchTile 完整候选 | 活跃行映射随区域变化，canonical 状态跨区域驻留，含回退与全部额外费用 |

另做严格 **2×2 机制消融**：固定行/紧凑行 × 区域写回/跨区域片上累积。固定行＋片上累积必须是优化实现；再加入“固定子组内局部融合多个区域”的控制，防止把普通融合误归因为新映射。

优先作者原生代码。SM120/模型不兼容时可做同引擎机制移植，但不能写成完整复现，也不引用不同硬件的原论文倍数作直接比值。Fused3S 与 cuDNN BSA 先检查输入/精度/mask 接口；可合法表达的形状补直接对照，不通过扩大支持集或改变 causal 语义强行适配。不能因 E1 已穷举固定划分就省略 E4。

ActKV 作为相关选择/压缩工作；只有声称兼容其支持集和更新语义时才加入实测扩展。首版固定 mask 不宣称适用于所有动态选择器。

### 5.5 应用 baseline：先确认独立 query-aware 调用值得优化

| 编号 | 方法 | 公平配置 |
|---|---|---|
| A0 | 一次全文、多问题共同回答 | 充分调优 prompt，按 question ID 输出；缺项计失败，组完成时间可比，不能将不完整流式字段提前当成功 |
| A1 | 独立 Dense＋APC＋适用共享 decode | 文档准备一次，允许 Cascade/PAT 等正确强路径；主质量参考 |
| A2 | 共享 KVzip / 可用 FastKVzip | query-agnostic 历史准备一次，多问复用；静态文档适配与原生持续淘汰分别标注 |
| A3 | 调优 RAG | 保留表格的 512/1024-token 分块、4K/8K 检索预算；BM25＋Qwen3-Embedding-0.6B，开发比较 Qwen3-Reranker-0.6B；索引/检索/重排全计费 |
| A4 | query-aware 稀疏强实现 | 原生 Tangram/SnapKV/FastKVzip 按真实语义调优；另有 QC-SnapKV 共享源 E0/E1/E4，分离容量、选择和执行效果 |
| A5 | BranchTile | 与 QC-SnapKV 执行臂保持同一选择规则，仅改变执行；与其他算法比较质量—完整成本前沿 |

RAG 的[embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)与[reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)使用相同资源预算，不免费占额外 GPU。检索 top20 候选、RRF=60、去重/表格规则在开发固定；开发中若明显受检索能力限制，再用更强同系列模型诊断，不能因弱检索器而宣称全文不可替代。gold evidence 仅作质量上界，不进正式速度排名。

A0/A2/A3 先在原 8 开发报告全部筛查；最强应用替代进入正式主表。其余替代也在 100 报告完成至少一次冻结配置的质量/成本筛查，若测试中出现支配候选的证据必须报告，不能因不是主臂而忽略。主要优越性与质量结论使用预定三 seed 配对；单 seed 筛查不承担非劣结论。

不要求未优化的 A4 必须已经快于 Dense。先在固定执行边界下测可替换 attention 的占比，并判断合理算子加速能否弥补差距；若连去掉整个可替换部分也无法接近质量合格应用替代，停止该配置。不能把含排队/重叠的 JCT 直接减去各 kernel 时长之和。

### 5.6 执行阶段与最小矩阵

| 阶段 | 范围 | 交付与继续条件 |
|---|---|---|
| P0 静态准备与部署资格 | 先用 CPU 冻结自然多问输入、评分 fixture、共享生命周期；恢复运行后做最小 GPU 部署小试 | 不消费测试答案选模型；核实 TP2、FP8 MoE、完整 KV 容量和数值。部署小试计入下述 P1 预算 |
| P1 应用资格 | 原 8 开发报告，A0–A4；主模型 TP2，少量 4B；Dense A/A | 实测自然支持、共同 ready、选择开销、完整费用；冻结 checkpoint/预算/应用基线 |
| P2 算子资格 | 主模型 6 开发文档×4 个预定均匀层×自然到达的第 1/16/64/256 步，至多 96 个 layer 快照；每 rank 全部 KV heads | E0–E6、2×2、数值与资源剖析；不存在的自然步不造，capture 数不是任务数 |
| P3 主评价 | 100 公司年度、自然 B；同题同 seed 三轮；核心 Dense、最佳固定组、QUILT-exact、候选 | 完整质量/JCT/显存；主锚点补原生压缩及最强应用替代，最多六主臂 |
| P4 边界与能耗 | QMSum 官方 test35，4B TP1；代表性自然 capture 与主应用能耗；有限连续补入 | GQA4/短输入/自然退出边界；能量可分辨才作正式结论 |

首轮 P1 设 **24 GPU 小时**、P2 设 **8 GPU 小时**的探索上限，按“卡数×分配墙钟”计，TP2 一小时算两 GPU 小时；P0 恢复后的 GPU 部署小试、加载、预填充及失败重试全部计入 P1 上限，不设免费前置阶段。它们是投入预算，不是完成保证。到限记录完成范围和失败开销，再据证据决定下一步。每个初始 GPU 作业合理限时，单卡小试优先≤1小时，较长模型作业按实际加载与预填充费用拆分。

正式预算用试点实测 `每报告耗时×方法×seed×卡数` 加初始化、编译与失败支出估算。八卡并行缩短日历时间，不把四对并行或三个 seed 计为更多独立文档。资源不足先减少次要扩展，不删最强主对照或质量分母。

### 5.7 数值、机会和硬件归因

**数值测试。** 小形状用 FP64 reference，覆盖全共享、不共享、AB/AC/BC、树状、空集合、非整 tile、私有后缀、所有 alive 子集。测最大绝对/相对输出误差、LSE 差及层输出/logit 差；容差按原生 BF16 对 reference 的误差在开发冻结，不在测试放宽。full-budget 与 Dense 相容。mask/位置/重复归一化错误必须修复后才进入性能比较。

**机会统计。** 记录每层头的 owner 长度分布、共享比例、原地址局部性、不同固定划分和精确层次计划、自然共同 ready 步数及其 attention 时间权重。重点是长度加权的 owner=1/2 区域，以及相对最强固定分组仍能减少多少实际 M/N tile；三向交叉的“存在”不足以证明机会。

**性能计时。** 同一物理卡上候选/基线随机交错至少 30 个计时 block，每 block 多次完整调用，另做反序；先 A/A 测噪声。第二物理卡/卡对交换方法，不能长期一张卡只跑候选。记录中位数、分位数、配对区间及最慢边界，正式时间不带 profiler。

**缓存条件。** 同一小 capture 反复重放可能都命中 L2。主结果使用轮转实际 capture、工作集超过实测 L2 或相同模型执行前导；另报 hot-cache 控制。缓存条件一致，不能候选热、基线冷。

**代表性 profiling。** 用 [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)记录实际 DRAM/GDDR 读写、L2、Tensor Core 指令、填充比例、SMEM bank conflict、barrier/stall、寄存器/thread、SMEM/block、spill/local memory、CTA/wave、split 和全局 partial。理论与 achieved occupancy 分开。低 partial、高 occupancy 都不是目的，完整时间才是。

合成数据只补单因素切片：B=1/2/3/4，GQA=4/8，D128，文档16/32/64/128K，支持结构相同/不交/树状/交叉。使用真实可行预算，少数核心点交叉，不跑全笛卡尔积；合成结果不替代自然 capture。

### 5.8 能耗：作为实测指标，不以理论节省代替

主能耗指标为 **GPU 板卡 joules/完整 attention call、joules/文档组、joules/正确回答和 EDP（能量×延迟）**，同时报告正确数、失败数和全部费用；不能只测成功样本的执行而丢掉失败能量。QMSum 没有可信二元成功定义时不构造“正确摘要”分母。

先查询 [NVML 能量计数器](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)支持情况；支持则取窗口差，不支持则积分功率采样。记录设备真实更新粒度，而不是仅记录轮询间隔。微秒级 kernel 使用同工作量的自然 capture 序列连续运行约 20–30 秒，再取总能量/调用数；充分预热、反序配对，窗口不足以分辨时不宣称能耗改善。

报告 gross 板卡能量；扣 idle 的增量能量仅作补充并说明假设。TP2 对两卡求和，功率采样覆盖准备、通信和完成窗口。NVML 板卡能量不等于整机能量，CPU/内存/电源不在该口径内；可用外部仪表时再补整机测量。**575W 是标称 TGP，不能乘运行时间冒充实测。**

候选与基线采用相同支持的功率限制、时钟策略和精度，记录温度/降频/其他负载。正式能耗测量避免八卡高热负载引入不对称干扰。不能由减少 GDDR 字节自动推导节能，也不承诺 5090 存在可用的细粒度累计能量接口。

### 5.9 完整应用、容量与质量统计

主应用先测一个文档组、自然 B≤4：冷文档从提交开始，计全文 prefill、问题 prefill、评分/索引、decode、TP 通信、解析及失败终局；暖文档单列复用延迟，并给准备一次、复用 1/4/16 组的摊销曲线。组 JCT 为提交至最后一题终局；每题延迟、TTFT、TPOT、生成长度和截断同时报告。所有同支持集臂共享同样准备条件。

有限服务验证沿用成熟引擎 continuous batching：新请求实际补入，不等待整组结束；报告共同 ready 比例、自然 EOS 后回退开销及终局排空。固定显存预算下先一组，二组仅在各方法实际可准入时比较；原生压缩可容纳更多组是其真实优势。保持相同绝对到达率，至少一个开发阶段冻结的稳定负载；不把四对 TP2 的并行吞吐当单 replica 算子收益。本论文不铺开大量负载/调度曲线。

DocFinQA 用官方答案评价最终标量，冻结数字、单位、百分比、舍入及 FINAL/JSON 解析规则。已有 GSM8K scorer 不能冒称官方 DocFinQA scorer，先做独立 fixture；不得从输出中挑最接近 gold 的数字或偷偷执行程序补救。首版不加 Python 工具。QMSum 使用官方 ROUGE 并补盲审事实一致性/问题覆盖，judge/rubric/抽查和成本固定，ROUGE 不单独证明事实正确。

三个 seed 按同题配对，不算新任务。DocFinQA 按公司簇重采样至少 10000 次，保留各年份/问题/seed 的相关性；QMSum 按会议及更高源分组。主质量拟设相对 Dense **绝对 2pp 非劣界**，执行前冻结，配对差的 95% 下界高于 −2pp 才支持该口径。48 公司簇可能不足以证明，区间跨界就写证据不足，不以“不显著”声称相同。完全相同配对时补有限样本保守界，不能使用零宽 bootstrap 证明等价。

同 mask 数值正确不等于相对 Dense 无质量损失。固定轨迹 replay 只作算子归因；真实 JCT 主结论保留自然输出变长、提前失败和截断。不同方法分别报告完整质量—成本前沿，而非挑一个有利 seed 或删掉失败报告。

### 5.10 四页技术内容如何组织

主文只保留一个完整论证链：

| 内容 | 建议篇幅/图表 |
|---|---|
| 问题、直接相关工作和自然机会 | 约 0.7 页；一张自然 owner/tile 机会图 |
| 活跃行映射＋canonical 状态数据流 | 约 1.2 页；一张数据流与片上布局图、必要资源计算 |
| 配置与强基线 | 约 0.4 页；紧凑配置表 |
| 完整算子、2×2 归因、能耗与应用 | 约 1.5 页；两张组合图、一张质量/JCT/边界表 |
| 结论与适用范围 | 约 0.2 页 |

完整基线矩阵、调优网格、分层质量、编译资源和原始 capture 放可复现 artifact；正文必须保留最强竞争者、质量和负向边界，不能靠补充材料隐藏关键限制。能耗若测量不可靠，就收缩能效主张，优先呈现可复核的数据流/延迟证据。

## 六、预期收益与推进判断

以下均为**研究目标或估算，不是已有测量，也不是会议录用门槛**：

| 层次 | 值得投入的目标 | 必须满足的前提 |
|---|---|---|
| 完整 attention call | 对最强适用同 mask 执行约 1.15–1.4×，有利自然形状可更高 | 包含索引摊销、suffix、split/merge，超 A/A 噪声；不能只胜 E3 |
| 板卡能量/EDP | 争取 10%–20% 的可分辨改善 | 同工作量、同精度、真实测量；可以与延迟收益不同方向 |
| 完整文档组 | 质量约束下 JCT 约降低 5%–10% | 含全文/问题准备、生成、TP 通信、容量与实际结束 |
| 边界 | GQA4/短回答/树状支持回退或收益近零 | 不额外扩大支持集，不隐藏回退及最坏开销 |

固定执行近似下，若可替换 attention 占总时间 f、加速 s、新增费用占比 p，则 `T_new/T_old≈1−f+f/s+p`。例如 f=0.35、s=1.30、p=0.01，完整时间约降低 7.1%；f=0.10 时仅约降低 1.3%。TP 通信或 prefill 占比高时，算子收益会被稀释；不能由该式直接预测排队吞吐或自然任务质量。

首轮推进标准：自然 capture 上相对固定行片上累积、最佳固定划分和精确层次共享，至少出现约 15% 且超过 A/A 两倍噪声的稳定完整 call 信号，再投入完整集成；较小真实收益保留，不改写原始结果。主应用争取至少约 5% 的可重复净改善。若需要大量额外机制才能补救，则重新收窄或调整算子，而不是扩成大系统。

明确的失败判断包括：自然支持几乎没有可减少的 tile；最佳固定/层次方案已覆盖机会；行切换造成的片上流量、spill 和 split 抵消节省；TP/准备成本使合理上界也无法取得完整收益；或应用替代在同质量下持续支配候选。遇到这些情况记录范围和证据，不将“问题真实”当作“方法必然有效”。

从恢复实验起，预计先用 3–4 天完成输入/部署与机会资格，约 1 周实现和验证核心数据流，约 1–2 周完成主对照/能耗/边界，再集中写作。实际周期由资格结果决定；八卡能并行数据收集，不能替代实现、数值验证和因果归因。

最终要形成的论文主张是：**在自然交叉稀疏支持和合适 GQA 形状下，通过随区域压紧计算行、跨映射保留原 query 状态，取得经片上资源与全局流量测量解释的完整 attention 改善，并在一个真实文档应用中保留净收益。**
