# 多问题联合 support 选择：先例核查与可证伪筛查

核查日期：2026-10-16。范围：一手论文、作者官方仓库和官方框架文档；仅 CPU/web，没有启动 GPU，也没有更改已锁定的事实审计 v1。

**建议：暂不把论文主张改成“共享 query-aware backbone + private residual”。这个表述不足以构成新意。只保留一个小规模、可终止的 CPU 筛查：在逐请求质量限制下，测量独立 support 还能缩小多少物理 union。** 检索未发现完全相同的“真实同源、同时 ready 的多请求 + 每请求独立约束 + 物理 union/执行成本目标”组合，但这是有限检索中的空缺，不是已经证明的新颖性。若只有加权 attention/affinity 分数和普通贪心选择，建议否决作为新方法。

任务上下文给出的主模型 4K 独立 QC mask 质量退化，是本方向的前置障碍；本轮没有重读或重新统计质量结果。待 8/16/32K 开发网格确定哪些预算可保留质量后，才能从合格点筛查共享空间。不能用“与已退化的 4K 同样差”证明质量保持。

2026-10-16 后续数据修正：旧 TP exporter 会在短 suffix 时用 document Q 回填观察窗，旧导出 mask 不能作为 QC 的证据；生产 `FrozenQCSelector` 不受该 exporter 问题影响。后续探针仅使用 `*_suffix_v2/rank0/selector_q*_l*.pt` 中标记为真实 suffix-only 的 `pooled_scores/supports`。原文第 3 节的零松弛结论意味着有意义的加性评分交换研究通常必须**显式允许正 score loss epsilon**，并与 epsilon=0 控制并列。预冻结的 `[0,0.001,0.005,0.01]` 是相对原选择 pooled-score 总量的探索损失阈值；pooled attention score/mass 不是任务质量，也不是自然生成无损证明。独立的 CPU 前置证伪可在质量网格完成前运行，但不得据此宣称候选质量合格或启动部署。

## 1. 已有工作覆盖到哪里

这里的 request 指同一文档上的不同用户问题；query 指 attention 内的一行 Q。它们不能互换。跨 GQA heads 的共享也不等于跨 request 联合选择。

| 工作与核查版本 | 实际机制 | 对候选方向的约束 |
| --- | --- | --- |
| [KVzip，2505.23416](https://arxiv.org/abs/2505.23416)；[Fast KVzip，2601.17668v2](https://arxiv.org/html/2601.17668v2) | 前者用上下文重建估计可复用 cache 的重要性；后者将重建重要性蒸馏为轻量 gate，prefill 评估采用 query-agnostic 设置。 | “共享压缩文档供多问题复用”已有明确先例。区别只能是使用**已经到达的不同问题**，保留不同 mask，并约束每题，而不是声称首次多问题复用。 |
| [Zipage，2603.08743v1，§4.4](https://arxiv.org/html/2603.08743v1#S4.SS4) | 请求压缩会破坏共同前缀的布局；它将结果写入目标块并管理共享块引用，避免改坏其他请求使用的原块。 | “支持 prefix caching 的压缩”不是空白；但这不是联合选择同一批 token 以最小化压缩后 union。要区分保留原共享前缀的安全性与压缩结果继续共享。 |
| [Tangram，2606.06302v2，2026-06-15](https://arxiv.org/html/2606.06302v2) | 校准 head 预算，按相近预算组织 ragged pages，并提前平衡执行；评分器可替换。 | head 分组、碎片回收、静态预算不能作为新贡献。其当前官方实现仍逐请求评分和提交保留集合，见下方源码核查。 |
| [PageWeaver，2610.11201v1，2026-10-08，§3.1](https://arxiv.org/html/2610.11201v1#S3.SS1) | 在固定的 selected-page support 上重分组 query；维持原 query ID、causal 位置和完整输出归属。 | 用 affinity 减小 union 的**执行分组**已经存在。候选若只重排请求，或固定 mask 后换 grouping，就没有越过这个边界。 |
| [QUILT，2610.11134v1，2026-10-08，§3.3–3.4](https://arxiv.org/html/2610.11134v1#S3.SS3) | 分解共享与私有集合、层次共享；共享尾部下推可精确，最后私有尾部可按重要性裁剪。 | shared/private 分解和为 tile 利用率轻度修改 support 都已有先例。不能把它误写为全程固定 mask 或全程精确，也不能将“剪掉独有低分 token”当作新机制。 |
| [QuoKA，2602.08722v1，2026-02-09，§3](https://arxiv.org/html/2602.08722v1#S3) | 为 chunked prefill 选代表 query，用跨 query 的最大相关性构造公共 KV 子集；另处理 GQA 轴聚合。 | “联合多个 Q 选公共 KV”“用 max 避免平均掩盖稀有需求”不是新意。其轴是同一 prefill chunk 的 query，而不是不同请求的固定预算、独立质量约束与物理 union 目标。 |
| [GVote，2509.03136v1](https://arxiv.org/html/2509.03136v1) | 采样可能的未来 query，汇合所需 keys，估计自适应缓存预算。 | 多 query 投票/union 和动态预算已有先例；“多问题分数求和或投票”不足以构成差异。 |
| [Attention Matching，2602.16284v1，§2–3、附录 A.2](https://arxiv.org/html/2602.16284v1#S2) | 对参考 queries 同时拟合局部 attention 输出和 mass；可选原 keys，再拟合 bias 与 values；还考虑逐层 on-policy queries。 | “以输出误差取代 attention mass”“同时匹配 O 与 Z”“注意压缩引起的后续 Q 漂移”均已有先例。候选只能限制为原 KV 不变、每题不同 support 的联合物理布局问题。 |
| [JSSA，2610.19026v1，2026-10-13，§3](https://arxiv.org/html/2610.19026v1#S3) | 联合 Q/K 低维子空间保持排序；按 GQA 组选 token，并直接索引原全维 KV 执行。 | 改良打分或减少 gather 也已有很新的强先例。它不构成已核实的跨请求 union selector，但后续若做 selector 质量/性能主张，应检查其适配性。此处没有声称已在本机复现。 |

另核实了 TensorRT-LLM 的 `eviction_mode="union"`：其含义是各 **KV head** top-B 的 union 再按 token 最大分数排序；prompt 保留，压缩 generation suffix。不能因名称含 union 就视作跨请求联合支持选择。[官方固定版本说明，配置参数节](https://github.com/NVIDIA/TensorRT-LLM/blob/c76f4a8/examples/kv_cache_compression/triattention.md#configuration-arguments)。

本轮还检索了 joint queries、multi-query compression、cross-request compression、shared/private cache、union-aware、group-aware、batch-aware selection 等组合。大量命中指向 GQA、跨层压缩或缓存路由，不能混同本问题。没有从这些命中推导“世界上不存在同类方法”。PageWeaver、QUILT、QuoKA 的本轮结论以已读论文为依据；没有核实到可用于本问题复现的作者源码，因此不把代码可用性写成已确认。

## 2. 官方源码交叉核查与身份

以下 SHA 由官方 GitHub commits API 查询截至 2026-10-16 的最新可见提交，随后读取指定 commit 下文件。未安装这些实现，也未修改本地运行环境。

| 官方仓库快照 | 读到的证据 |
| --- | --- |
| [FastKVzip `e04afaa6007c58d18500d0698d2d8ae592ff22c2`](https://github.com/Janghyun1230/FastKVzip/blob/e04afaa6007c58d18500d0698d2d8ae592ff22c2/prefill/attention/gate.py#L77)（2026-04-05） | `Weight.forward(hidden_states)` 对单序列产生 per-head、per-token gate 分数；接口没有接收一组其他 request 的需求或 union 代价。文件 SHA-256：`f27e25c54c0ddacdf5b6595296753c215ecae6797e6549d0c0fd6ffcddb6f59d`。不能从该文件单独推断整个仓库的所有扩展。 |
| [Tangram `6fa551fc8f6edcc118a2a39b3554ee1520e91edd`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/attention/compression/compressor.py#L786)（2026-09-09） | Q/K scorer 的注释与路径明确按 request chunk 独立评分，不能拼接请求观察窗；保留决策存在 per-request state。head cluster 共享的是长度约束，位置仍为成员各自排名。文件 SHA-256：`59b1c3cc5a6ef8c81c536e3456abd0670ed85f838b8dab4ed2f988c63901246d`。 |
| [Zipage `3cf00ad547e5fe084fa640f431a31e02b45a2869`](https://github.com/microsoft/Zipage/blob/3cf00ad547e5fe084fa640f431a31e02b45a2869/zipage/kernel/compress_kv.py#L103)（2026-03-13） | 压缩 flags 与源/目标 block tables 带 batch/request 轴；kernel 逐 batch/head 拷贝保留项。与论文的目标块策略一致，但该拷贝 kernel 本身不是 selector 新颖性的证明。文件 SHA-256：`58266f2e6b4cc0faeb0fb0059b22c48d0dcf3edb58bb79e00c500e2895f08ea0`。 |

供后续复现定位、仅核实仓库身份的额外快照：Attention Matching 官方 [adamzweiger/compaction `2020f6d37940fe013e7e7f1ebf409e5d107c5670`](https://github.com/adamzweiger/compaction/tree/2020f6d37940fe013e7e7f1ebf409e5d107c5670)（2026-03-31）；JSSA 官方 [jiwonsong-dev/JSSA `34ce5abd82453b16ca6d5f16a1a8ae50fac5b920`](https://github.com/jiwonsong-dev/JSSA/tree/34ce5abd82453b16ca6d5f16a1a8ae50fac5b920)（2026-10-15）。本轮没有逐函数资格审计这两个实现。

## 3. 先做一个不可能性检查

以下是本次筛查的直接数学推导，不主张它是新定理。固定某请求、层、KV head 的加性分数 `s_t`、必留项及剩余预算 `k`。若独立选择 `S0` 是唯一 top-k，则

```
|S| = k  且  sum(s_t, t in S) >= sum(s_t, t in S0)
    => S = S0.
```

证明只需把被移出的 top-k 项与加入的非 top-k 项一一配对：每项交换都使总分下降。因此，若“每题质量不降低”仅被定义成**同一个评分器的 retained mass 不低于独立 top-k**，零松弛下根本不存在联合优化空间，除精确同分项之外。固定公共 backbone 后补 residual 也不能绕过这个事实。

若边界间隔 `g = s_(k) - s_(k+1) > 0`，允许总分损失 `epsilon`，交换 `m` 个位置至少损失 `m*g`；因此 `m <= floor(epsilon/g)`。这给出极便宜的筛查：当 `epsilon < g` 时该请求不能变。多个请求一起移除一个 union token 还必须让它的全部现有 owners 都放弃它，因此把各请求交换上限相加只是 union 缩减的宽松上界，不是可实现收益。

该推导只约束冻结的加性分数。平滑、pooling、有限观察窗、BF16 同分都可能使该分数偏离真实 attention 或任务质量。不能把分数相等称作语义质量无损；也不能看到无自由度后临时放大 epsilon，制造共享。

## 4. 唯一保留的机制：逐请求误差约束下的联合 set exchange

这是一项**待证伪的可行性机制**，不是已确立的新算法。先让每题独立得到一个已经通过质量门槛的 `S_r^0`，再在原 KV 数值、每题 document-token 预算、必留项和私有尾部不变时，交换边界附近的支持位置。目标按顺序减少物理 union，再减少已测执行布局的 tile 成本；不通过固定加权 affinity 把质量和共享折成一个任意标量。

令 `P_r` 为每请求始终保留的 system/question/generated tail，`W_r` 为仅从已经可见的输入/问题观察窗取得的 Q。对每个 witness、每个 query head 分别计算：

```
O_r(S), LSE_r(S) = Attention_FP64(Q_r, KV[S union P_r])
E_O(r,S) = ||O_r(S) - O_r(dense)||_2
E_Z(r,S) = |LSE_r(S) - LSE_r(dense)|

minimize C_phys(S_1,...,S_B)
subject to |S_r| = |S_r^0|, mandatory_r subset S_r,
           E_O(r,S_r) <= E_O(r,S_r^0) + eta_O,
           E_Z(r,S_r) <= E_Z(r,S_r^0) + eta_Z
           for every r, witness, query head.
```

第一轮设 `eta_O=eta_Z=0`，仅允许固定的 FP64 计算误差容差；有任何违反则回退 `S_r^0`。后续若要使用正质量余量，必须由此前独立质量开发协议决定并预冻结，不能按 union 改善选阈值。比较完整 `S_r union P_r` 的输出及 LSE，避免只比较文档局部输出、忽略其与私有 tail 的 softmax 权重关系。输出/mass 检查沿用已有 Attention Matching 的问题认识，不能主张为原创。

最小实现先做 bounded swap：每题从冻结分数边界取固定数量候选，检查 1-swap / 2-swap，按实际物理 union 变化接受可行交换，固定 token-ID tie-break。这是普通局部搜索；必须报告它的搜索空间与未证最优，不能写成全局 oracle。另用小型 `N<=8, B<=4` 穷举验证约束、物理 ID 去重和搜索结果，不能拿玩具最优性替代自然长文最优性。

真正可能的差异只剩：**对同一真实源缓存上的不同请求，保留 request-specific masks，用逐请求而非平均误差限制，直接优化物理共享代价，并在后续自然生成上验证。** 这不同于 query-agnostic 压缩、QuoKA 的公共集合、PageWeaver 固定 support，以及 QUILT 仅删除私有末尾。但约束优化/局部交换本身是通用方法；没有足够自然收益和完整系统证据，这个差异不够支撑论文。

`C_phys` 必须按已验证的模型/rank/layer/head/原始位置与实际 backing store 计数，不能仅按相同逻辑 token ID 认定相同 KV。共同前缀内只共享真正相同的 KV，私有问题和生成尾部不得跨请求合并。若仍保留完整源 cache，union 变小最多是执行工作量候选收益，不能声称已降低常驻 KV 内存；实际 page 回收、复制与 metadata 成本需另测。

## 5. 最小试验与明确停止条件

**本轮只提出协议，没有运行下列数值或生成实验。** 建议最多先做一个 CPU 阶段，GPU 仍由主任务统一分配。

1. **固定样本与质量合格预算。** 使用现有公开开发协议和自然 TP2 captures，不按重叠挑文档。主模型 8/16/32K 中只将独立 QC 已通过原有质量门槛的格子作为候选；4K 失败格保留为负面背景。B1 是应无跨请求收益的控制，B3/B4 包含不同问题和真实 ready 证据。若任何预算都不合格，停止此方向，先解决独立 selector 质量。
2. **CPU 上算可交换性，而非先写新 kernel。** 保存分数边界间隔、精确同分数量、零松弛可行交换数、每请求 O/LSE 偏移、union 大小、owner 数量分布及 tile 数。验证数值使用原始同源 KV、真实 post-RoPE Q 和 FP64。既存 `s64/s256` Q 可以作未参与选择的回放检查，不能反过来用于在线选 mask，也不能把不同 forward 的 Q 拼成自然 ready batch。
3. **比较简单强控制。** 独立 QC、把多题分数 mean/max/RMS 后选公共集合、同预算 common-core + independent residual、原独立 mask 的固定 union 执行都要保留。QuoKA 风格聚合只是适配基线，不声称完整复现 QuoKA。若简单控制在同一逐请求约束下达到同样结果，联合搜索不具备方法价值。FastKVzip/Attention Matching 属不同准备成本的参照，不能隐去其 gate/重建/拟合成本后比较。
4. **先冻结进入下一阶段的门槛。** 建议 CPU 筛查至少在三个不同文档、两个已质量合格预算上出现方向一致的物理 union 与 padded-tile 降低，且没有单请求误差约束违反。具体最小收益应按已有完整调用的 break-even 成本预定；若必须给探索门槛，可预注册 15% union 减少，但它只是晋级筛查，不是预计速度。用搜索失败只能否决当前机制，不能断言所有联合算法不可能；若仅在同分或已退化预算有效，停止。
5. **只有 CPU 通过后才考虑同源自然 generation。** 沿冻结 prompt/model/template/seed、相同私有 tail、自然 EOS 和原 token cap，生成独立 QC 与 joint-exchange 配对结果，同时保留 dense。用原 strict metric 与盲事实/覆盖审查判断，报告每题失败、长度膨胀和截断，不能只看平均保持。已有 256-token capture 是仪器数据，不是质量证据。选择时不可读取未来输出；全层稀疏后的 Q 分布变化需要自然生成检验，单层离线 FP64 通过不保证答案正确。
6. **最终检验完整成本。** 计入 query-conditioned scoring、交换搜索、mask/metadata、union 准备、gather/复制、attention、LSE 转换及请求等待。比较已资格化的最强 FlashInfer 0.7.2 同 mask 路径和现有 own-union，而非只比旧独立 decode。减小 union 若不能抵消选择和准备，或收益靠等待未来请求聚齐获得，就不成立为部署收益。

现在可排除的是宽泛的新颖性主张；可以保留的是一个窄的、质量优先的可行性问题。若该 CPU 筛查无信号，不建议继续调 affinity 权重、扩大质量容差或挑有利组来挽救这个方向。
