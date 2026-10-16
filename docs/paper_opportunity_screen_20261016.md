# 真实收益空间与下一场景：先测 dense B8，再决定是否换机制

2026-10-16；本次新增 **0 GPUh**。只做非 root、项目虚拟环境下的输入/容量审计、一手资料核查和待调度配置。尚未找到一个已同时满足新颖性、数值资格和端到端收益的替代 idea。这里保留两项可证伪的机会，第一项已给出可运行配置，第二项只是有条件的后继假设。

## 1. 可立即验证的场景：自然 B8、完整历史、共享物理 KV

当前 B3/B4 sparse owner 的微小收益不能代表所有单用户复用场景。真实训练文档确有较多问题：[冻结训练集合](../data/manifests/docfinqa_public_training167_v1.json)的 `natural_question_count` 中，167 组有 **55 组不少于 8 问、11 组不少于 16 问**；原 `questions` 列表人为最多保留四问。已有[独立 B8 manifest](../data/manifests/docfinqa_training_large_group_pilot_v1.json)按冻结文档顺序取前两个自然不少于八问的文档，再按题目 SHA 取八问，不按性能或输出长度挑选。

这支持“一次提交同一报告的八个真实问题”这个范围明确的批量任务。它**不证明**一般 agent 会自然同时产生八个分支，也不授权复制同一个问题、等待未来问题来制造复用。TAPER 的真实任务分析报告的平均分支数约为 2.7–5.2，且已研究每步并发宽度调节；因此 B8/B16 必须作为指定任务设置，不能冒充普遍 agent 分布。[TAPER §2、§3](https://arxiv.org/html/2605.06914v1#S2)

旧 B8 已测的是 **sparse 原动态核**，292–490 μs 对约 11–24 μs 控制，结果仍失败；不建议重跑那个实现。此次不同的问题是：完整 dense history 下，已经启用 APC 和共享物理地址的实际 vLLM graph 模型，有多大 decode attention 占比和可替换时间？

### 输入与容量已核实

审计脚本 [audit_dense_b8_opportunity.py](../scripts/audit_dense_b8_opportunity.py)重新使用本地模型 tokenizer 与 `make_inputs` 构造全部输入，校验文档 SHA；原始数据及来源 SHA 在 [audit.json](../results/dense_b8_opportunity_v1/audit.json)。

| 文档 | 30B 实际八个 prompt 长度范围 | 共享文档 token | 旧 TP2 BF16 dense B8×256 是否完成 |
|---|---:|---:|---|
| train1 / `680456070a0a0714` | 120560–120576 | 120506 | 是，八题均零 preemption |
| train7 / `f1d5c3003725d29c` | 144604–144624 | 144549 | 是，八题均零 preemption |

旧记录是 eager＋capture，所有输出触及 256 cap；23.08/23.54 秒的 warm group wall 只证明历史部署可运行，不能当作新 graph 速度、完整回答或新算子结果。

30B TP2 每 rank 的 BF16 KV 每 token 全 48 层为 49152 B；4B TP1 全 36 层、8 KV heads 为 147456 B，反而是三倍。以下只算保守完整共享页＋八份私有问题/生成尾，每份独立向 16-token 页取整；不包含权重、图、workspace、metadata、FP8 scales 和瞬态 prefill：

| 每 rank KV payload，GiB | train1：128 输出 / 8192 输出 | train7：128 输出 / 8192 输出 |
|---|---:|---:|
| 30B TP2 BF16 KV | 5.58 / 8.53 | 6.68 / 9.64 |
| 30B TP2 FP8 KV | 2.79 / 4.27 | 3.34 / 4.82 |
| 4B TP1 BF16 KV | 16.74 / 25.60 | 20.05 / 28.91 |
| 4B TP1 FP8 KV | 8.37 / 12.80 | 10.02 / 14.45 |

两模型均重新 tokenized；表中输出长度仅为容量条件，不能当作真实 EOS。旧 30B BF16 池为每 rank 19095×16=305520 个全层 token 槽，上表两组即使八份 8192 尾仍在这个旧池容量内。新的 graph 引擎会重新估算池，必须以实际初始化收据为准。4B 单卡 BF16 长尾很可能因另加权重/图而不足，不能以“小模型”推断容量更宽裕；因此目前先用已验证可部署的 30B TP2，4B 只保留 FP8 后继路径。

### 待调度配置与可证伪判断

- [native_opportunity_b8_training_v1.json](../configs/native_opportunity_b8_training_v1.json)：两自然 B8 文档、30B TP2、BF16/FP8 KV，FlashInfer 实际后端，APC、FULL_DECODE_ONLY 图；max model len 163840，max seqs 8。
- [四对卡清单](../configs/native_opportunity_b8_training_jobs_v1.json)：每任务两次未 profile 的固定 128-token 有效模型运行，加一轮独立 profile；不强制输出等于另一精度，不比较跨精度相同工作加速。`natural_repeats=0` 是第一阶段有界诊断，**没有宣称本清单完成自然 EOS 评价**。已有主场景自然 EOS 由 root 另行安排；B8 诊断有空间后再补完整生命周期。
- 每任务超时 720 秒，八卡同时占用的上限为 1.6 GPUh，若按 15 秒终止宽限另计则 1.6333 GPUh；这里只提出清单，未启动，也未改资源账。root 在实际发起前冻结 runner 身份并分配预算。

主报告应区分未 profile 的完整请求墙钟、warm APC 问题组墙钟、实际 decode graph 时间和 attention 组件占比。两个 TP rank 同时执行，不能把二者时间相加当 wall；串行 kernel 累计也不一定等于 critical path。令可被替换的实际时间占比为 `f`，若 attention 加速 `s`、新增生命周期开销占原 wall 的 `c`，条件节省是 `f*(1-1/s)-c`。先报告 `s=1.1/1.3/1.5/2` 的条件曲线，再用实际候选插入测量，不把公式算出的数当端到端实测。

如果 dense B8 的实际 attention 占比仍低、或共享控制在同一部署布局上已接近实际原生，就放弃这条共享执行路线。若只在 warm decode 有空间，cold anchor 占比高，则可以如实收窄到复用已建立后的 decode；不能把忽略 anchor 的结果称整个任务加速。单次“零耗时 attention”替换会破坏后续值，不是有效模型端到端结果；只可将其明确标注为结构性诊断，不能用于论文主加速表。

### 新颖性边界与对照

PAT 已按共享 prefix 打包 queries、分片并归并；扩大 B 或从 sparse 改 dense 本身没有创新。[PAT §5](https://arxiv.org/html/2511.22333v3#S5) 是直接执行机制对照。QUILT 的集合交差和尾部处理、MSA 的 KV-outer worklist、PageWeaver 的页/查询两种调度，同样限制“重新分组就是新算子”的主张；它们的具体近邻审计见 [E3 审查](owner_tasks_e3_prior_review_20261016.md)。TAPER 约束运行时分支宽度调节的创新主张。此次目的只是找真实较大空间，**尚未宣称新的调度方法**。

公平主比较是固定且可部署的 vLLM/FlashInfer 实际路径，以及适用的共享论文实现/机制控制。当前简单 FixedUnion 留作消融；逐输入测 47 个候选的最好结果只作标明 oracle 性质的诊断，不作为投稿必须击败的门槛。

## 2. 条件候选：量化共享 KV 的解码端转换复用

值得区分的物理机制是：共同物理地址可以让缓存复用已压缩的字节，却不自动消除不同 CTA 对同一数据的重复解包、缩放或 codebook 转换。相同量化 KV、相同 support 和原位置下，把一份已经转换的 tile 同时提供给多个请求的 Q，可能在 byte reuse 之后仍省算术。但 FP8 转换可能很便宜或已被 MMA 管线覆盖，额外共享内存、行 padding 和同步也可能更贵；**当前没有此成本的实测，不能称已经找到大空间**。

最小入口直接复用上面的两个自然 B8 输入与 FP8 本机路径，不先创造新量化方案。先用官方 kernel 源码/生成代码确认其实际转换和重复范围；同数据同精度做完整调用的共享组与独立组对照，保留完整 paged layout 和 merge。对所有 arm 使用相同量化字节和 scale，单列转换正确性与量化质量。不能用“提前展开成 BF16”的时间差直接归因反量化，因为这同时改变存储字节数、缓存驻留和加载路径。

**先例很强。** HyQuant 已有低精度 prefix、BF16 重要位置/窗口、生成尾缓冲和融合反量化 attention；v3 为 2026-09-16，论文标注 EMNLP 2026 Main。[论文](https://arxiv.org/abs/2608.27875v3)、[作者实现](https://github.com/jerrysfls/HyQuant)。TurboQuant 的融合 LUT 转换/MMA 也不是新机制；PAT/MSA 已提供跨 query tile 重用结构。因此“加 FP8”“融合反量化”“把转换后的 K/V 放 shared memory”本身均不能成为论文新意。只有能提出并验证超出这些普通组合、同时处理共享组变化和格式约束的具体布局/调度原则，才考虑立项；截至本次没有这样的已确立机制。

如果 FP8 原生 graph 的实际 attention 时间已经很小，或普通 PAT 式分组就获取全部收益，则停止把这条适配叫新算法。若需要更低位宽才能显示优势，应另立量化质量和容量合同，不能借用旧 BF16 的精确性或旧 FP8 两文档质量记录。**本次不为该假设分配 GPU，也不承诺收益。**

## 3. ISCAS 范围与投稿要求的事实边界

[ISCAS 2027 正式 CFP](https://ieee-cas.org/files/ieeecass/2026-07/iscas2027-cfp.pdf)列出 High-Performance Computing for AI，并要求原创研究贡献；可以定位算子的数据流、存储层次和资源约束，但文件没有规定必须多少倍算子或多少百分比端到端加速。项目内部 15%/5% 仅是筛查偏好，不是录用标准。新颖性、可信的实际系统收益与完整实验都需要成立，无法承诺录用。

日期核查存在新旧官方页面差别：上述旧 PDF 写 2026-10-13；本次访问的 [IEEE CASS Active Calls](https://ieee-cas.org/active-calls)把 ISCAS 2027 CFP 截止列为 **2026-10-22（Extended）**。正式提交前仍应以投稿系统最新公告为准；不能把 ISCA 或 ICASSP 的截止和范围混入 ISCAS。

另一个近期边界是微软 [Capillary 官方项目](https://www.microsoft.com/en-us/research/project/capillary-kernels/)已描述通过 Hopper DSM 在共享 prefix 的多个 SM 间交换 partial、以成本模型选 cluster/query grouping。本次只核实项目描述，未核实完整论文和实测结果；它不能作为本机 SM120 可运行能力，也使泛泛的“跨 SM 共享状态/cluster cost model”主张不能直接称首次。

本次产出是可复算的 **真实 B8 原生整模型收益空间测试入口**，不是给大量历史负结果改名字。若第一项无空间、第二项仅是已知组合，继续寻找有不同瓶颈依据的机制，保留失败证据，不将零或微小收益包装成已满足论文目标。

复算：`CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_dense_b8_opportunity.py`。该命令只做 CPU tokenization/JSON，重建上述 audit 和 proposed configs；不运行任何 GPU 工作。
