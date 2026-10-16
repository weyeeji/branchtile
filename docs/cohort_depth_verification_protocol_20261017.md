# 共同验证分组与草稿深度：先测耦合，再决定是否实现

状态：2026-10-17 冻结的开发诊断协议；尚未启动 GPU，也没有新颖性或加速成立的结论。优先级高于继续调旧 owner kernel、bitwise 匹配和仅切换现成 cascade。这里的门槛是内部继续投入标准，不是 ISCAS 官方录用标准。

## 要证伪的机制

真实 agent 分支在不同时间产生不同长度的草稿。决定一次验证哪些分支、每个分支验证多少 token，会同时改变：共享前缀实际读几次、同一 prefix 的 query tile 填充率、私有尾部因果 attention 工作量、全模型 MoE/GEMM 形状和等待时间。因此，同样 B、总 query-token 数 K 和上下文长度之和，验证成本未必相同。

候选机制是让**共享物理前缀组、每组 query 数及草稿深度一起决定验证批次**，把一部分低收益草稿深度换成能填满同组 query tile 的另一分支深度。在线只使用已经生成的草稿、历史接受率、已到达请求和测得的边际成本；不使用未来接受长度或 EOS。先不引入新 drafter，不剪掉真实任务分支，也不改目标模型接受/拒绝规则。

这不是“共享 prefix 更快”的重测。所有有竞争力的策略都可使用同一个 full-history prefix-aware attention 后端。研究问题是：**在共同后端上，简单成本表及 greedy 已经足够，还是存在有端到端意义的组合耦合？** 若 greedy 足够，就否定此选题，不能把新增 scheduler 名称当贡献。

## 与最近工作的边界

| 已有工作 | 已经覆盖 | 此处必须额外证明 |
|---|---|---|
| [D-cut，2026-07](https://arxiv.org/html/2607.14647v1)，§3.3 | 跨请求置信度选深度，运行时成本表 C(B,ρ)，全局验证预算 | 在总预算相同的情况下，共享组分配是否改变成本；仅增加表项若已解决，则不声称新调度机制 |
| [ASPIRE，2026-09](https://arxiv.org/html/2609.17943v1)，§4.2 | 异步逐请求 draft/verify；按总 forward token 和各请求上下文长度和估计运行时 | 其线性总和之外的共享组代价，及逐请求决策不包含的组内边际成本；必须对增强后的策略比较 |
| [DSpark，SGLang PR 30261](https://github.com/sgl-project/sglang/pull/30261)，2026-07 合入 | 置信度调度、可变长度验证、token bucket CUDA graph | ragged graph 本身不是创新；图桶选择、切换、首次捕图均须计费 |
| [D²SD，2026-06](https://arxiv.org/abs/2606.04446) | 同一请求的草稿树、cascade 联合验证 | 多个独立 agent 分支的真实共享前缀和 readiness；不能把 cascade verifier 重新命名为创新 |
| [TAPER，2026](https://mast.stanford.edu/pubs/taper/) | 分支外部性、动态宽度及等待/SLO slack | 同等等待预算下的验证深度和执行 tile 联合代价；动态限宽已是已有工作 |
| [TOPAS，2026-08](https://arxiv.org/html/2608.25523v1) | 工作流剩余路径、前缀状态迁移和请求调度 | 单个已驻留 GPU replica 内的验证执行成本；JCT 目标本身不新 |

以上只确定一个尚值得测量的交叉点，不证明文献不存在。若完整论文/代码已包括相同共享组联合决策，则把它作为现成对照并重新判断贡献。

## 不重做哪些分支

[Hive](https://arxiv.org/abs/2604.17353) 已有跨重复采样路径的 logits cache；[FlashAgents](https://openreview.net/pdf?id=m14PPUfgEc) 已有 producer 输出流与 consumer prefill 重叠。因此“相同生成去重”“提前给下游做 prefill”不作为新选题。PCIe MoE 通信也不能从零算创新：[ThunderEP](https://arxiv.org/abs/2609.40093)、[CoMoE](https://arxiv.org/abs/2610.09424) 已直接研究消费卡/host-mediated 路径；通用 PP/JCT 优化已有 [PipeSwift](https://arxiv.org/abs/2609.16491)。可以以后做硬件补充，但本轮只集中验证一个假设。

## 输入与工作量合同

沿用当前本地 Qwen3-30B-A3B-Thinking-2507-FP8 权重、TP2、page16 和实际 layer-stride。KV 首轮 FP8，沿原生 receipt 保存每层尺度；BF16 只在初筛通过后扩展。原 doc1/doc7 的 120528/144576 token 共同前缀作为**已有真实状态的成本诊断**，不是重新挑出的正式任务集。

新的多 query 状态必须从原生完整模型对真实 token 序列的 teacher-forced continuation 捕获。每请求保留至多 8 个连续位置及其每层 Q、K/V、原生 O 和 logits。旧 B8 的单 token Q 不能复制 8 次冒充真实验证状态；也不能用随机 Q 证明整模型收益。可用实际原生已生成 continuation 获取成本输入，但这些已知 token 不是在线可用的 oracle drafter，不计接受收益。

成本矩阵 B=8、K∈{16,32,64}，n_i 是含当前已知 bonus token 的验证 query 数，1≤n_i≤8，Σn_i=K。K32 至少测等深度 `[4,4,4,4,4,4,4,4]` 和不等深度 `[8,8,8,4,1,1,1,1]`。K64 全8是饱和负对照，不能用于证明分配自由度。两种实际组构成：一个共享组 G=[8]，两个不同文档的真实组 G=[4,4]。两组中固定各4条请求，并在组内交换深度，使每组总 query 数分别约为(28,4)、(18,14)、(16,16)。每个具体输入保留 token/位置/分组 hash。

比较深度形状的单步成本不等于比较相同输出工作量。这些是成本标定输入；最终策略必须用真实 draft 接受后的相同任务完成情况比较。共享/不共享的人工物理别名可作独立缓存实验，但不得混入“真实多组”结果。不得复制文本、padding 文档或伪造多请求来制造共享机会。

同一策略输入须固定真实共享页、私有尾长、同一 query token 集、全部 K/V 内容、精度和位置。为了隔离 layout，可重排 query 行和逆排 output；改变深度则另作不同 cell，不能声称与原 cell 是 same-work 加速。

## 执行路径：独立新文件，冻结源不动

1. `src/dense_capture_fixture.py` 可以复用 provenance/真实 page stride 重建逻辑。新增 `src/ragged_shared_verify_plan.py`，不改现有 `src/vllm_dense_cascade_plan.py`。现有类假定每请求1Q、tail `causal=False`，直接调用会让验证 token 看到未来 token，**必须禁止**。
2. 新 plan 的 prefix `qo_indptr` 按共享组的 query 总数划分，所有 query 可见本组完整公共 prefix；private tail 的 `qo_indptr=cumsum(n_i)`，逐请求使用 right-aligned causal mask，使第 j 个 query 只见旧 private history + 本请求草稿前 j 个 token。不同分支草稿绝不互见。prefix 和 tail 的 O/LSE 稳定合并；query 映射后恢复原 token 顺序。
3. 第一版可用 FlashInfer 两个 paged prefill + merge，作为成熟共同后端；后续如有 tile packing kernel，必须与同后端调度提升分开消融。输出合同是完整 O，whole-model 含原生 KV 写入、MoE、LM head 和必要通信；不省略 verifier logits。
4. 新 `scripts/capture_ragged_verify_state.py` / worker 从原生多 token forward 捕获真实 query。本地存在 `vllm/v1/spec_decode/ngram_proposer_gpu.py`、`suffix_decoding.py`，可在第二阶段作无需训练的 draft 来源；模块存在不是已验证 engine 配置。新 worker 必须基于实际 active model runner 的 `prepare_attn` 和 query_start_loc/num_scheduled_tokens 数据，不写死旧 runner 类路径。
5. 当前 `scripts/vllm_dense_cascade_worker.py` 的 prepare 有 `batch.num_tokens == b`，graph 有 `num_tokens == num_reqs`，capture 有 `varlen_decode=False`；所以它**不具备 ragged verification 能力**。借鉴其 native saved forward、shadow 和事件记录结构即可，不删 assert 硬接。新增 graph key 至少包含 K桶/B/必要attention拓扑，检查引擎能否 replay；不可把 eager 与 graph 的差异误归因分组。
6. 先仅捕获并重复真实 complete forward 作为成本实验，保持缓存不在循环中增长或回滚污染。不同 cell 使用新请求快照/独立 cache，或者验证可逆的 snapshot restore。准备、row regroup、KV 更新、候选产生、验收、采样/回滚分别记录，再在在线结果中全部加回。

## 最小实验与预先停止条件

### G0：attention 成本是否存在有用耦合（最多 0.4 GPUh）

root 分配8张独立卡，各≤180秒运行，初始/编译超时也计预算。两doc×两层(15,47)×两rank的真实 capture 分布到8卡；含 B1/2/4/8、n_i=1/2/4/8、16边界尾长、请求退出/行重排、小随机 case 作数值资格，再跑上面有来源的矩阵。未捕获到真实多 query 前，只有随机小资格，不报告真实性能。

数值资格：每行完整因果支持的 FP64参考，冻结 atol=.006、rtol=.015；candidate 和 native 各自对参考判定，不把二者误差相加当真值。输出/LSE及mask检查全部 assert；未来 token K/V 强扰动不能影响较早 O。完整真实层先原阈值验证，失败保留而不改阈值。top1/logit单列；不要求位级等同，也不绕过错误 acceptance。

每cell至少30个随机顺序成对block，AA重复，GPU图事件+外层wall，hot及每block冷缓存诊断分开。比较 raw native verifier、现成共享后端、分组后端；不能拿单token XQA 当多token验证的唯一原生基线。实际原生 vLLM verifier 后端/算法由运行receipt确定。

初筛通过需要：至少两个真实输入上，在相同(B,K)且控制真实长度差异后，分配相关成本差异≥15%，paired95%CI下界>5%，同时大于AA抖动。这个条件只说明继续测全模型，不能证明 scheduler 有收益。若组内query数的普通表已完全解释，仍可进入 G1 但不得声称组合新颖性。

### G1：全模型代价和简单策略是否已足够（最多 0.8 GPUh）

root 分配四组TP2，每组≤360秒，独立完整模型成本矩阵，cold setup单列；只有 G0 成功且真实多query捕获就绪才启动。8卡不同时加载超过四份模型。固定有效查询/草稿输入的 replay 不做EOS质量宣称。

训练成本表与测量集按文档或捕获step划分，不能同一cell拟合再汇报零误差。必须比较：

- C0：原生 prefix-aware verifier + 固定 n_i；实际生产强基线。
- C1：共同共享后端 + D-cut式置信度/接受率分配，表 C(B,K)；这是可复现算法变体，不冒称完整原论文复现。
- C2：同 C1，但普通增强表加入每组query数、prefix长、private tail和graph桶；逐请求/逐token greedy 按“预期接受增量 / 实测组边际成本”选择。它是关键强对照，不能省掉。
- C3：相同共同后端 + 小集合联合搜索分组/深度，使用同样历史接受估计与同样成本表。B≤8可枚举少量候选或有界beam，不允许遍历未来轨迹。
- Oracle：已知未来接受长度/完成长度的离线最优，只用于收益上界，不作为可部署policy，不报告它为加速结果。

初始全部请求ready、等待预算0，排除“多等一会儿自然batch变好”的混淆。之后仅用固定真实ready trace，给所有策略相同最大等待0/1/2个本机实测AR步，并计入每请求JCT。不能只给候选宽松等待预算。未来草稿未ready不能计入cohort。模型估计器若包含全部group状态却成本过高，其维护/选择成本仍计费。

G1继续门槛：真实接受trace可得后，oracle对**最好的C0/C1/C2**的完整decode-time改善至少15%；并且不使用未来信息的 C3 对 C2 至少10%（CI下界>0），含调度、draft、prepare/graph成本。若只有attention改善而whole-model收益<5%，或者C2已到oracle的5%以内，停止新scheduler方向。没有真实acceptance trace时只能报告成本差异，不能计算假想接受率的“端到端收益”。

### G2：真实 agent 任务确认（另立预算，不预先消耗）

按 `docs/real_agent_branch_minimal_protocol_20261017.md` 的两个真实issue、B8、两轮patch+test普通用户venv流程；draft先用本地已存在suffix/ngram实现，测低成本可行性，再决定是否需要已发布兼容drafter。ngram接受率低只能否定该drafter/场景组合，不能拿完美草稿顶替。先冻结任务与seed，所有策略同drafter、相同最大等待和图预算。

报告完整cold/warm session wall、time-to-first-tested-success、all-branches JCT、成功率、每成功任务GPU秒；EOS、cap、patch/test失败全保留。至少4组预定seed和多个issue后才讨论任务级优势。质量差异与长度变化如实报告，固定轨迹执行诊断和自然任务收益分开。成功条件是真实任务成本下降且质量未劣化；不按单seed或“每token归一化”筛选好结果。

## 为什么值得一次小测，以及可能迅速失败的原因

现有 dense B8 microbench 在长prefix上确有约2.2× attention空间，但 B1/B3甚至不加速，整模型自然长度和准备成本能完全吃掉收益。这说明 shape 和分支存活有真实重要性，却不能直接外推 verifier。验证多token可能补满tile，也可能使原生backend已经高效、共享组边际成本近似平滑；若后者成立，C2普通表greedy应当足够，本方向直接退出。

此外，MoE算子可能随K增加而主导，KV共享只剩很小比例；ragged图频繁重捕、FP8数值和因果mask、target/draft缓存回滚都是实际风险。第一轮总预算上限1.2 GPUh，只回答“耦合存在且普通增强策略尚未解决吗”，不先投入新的大规模engine开发。
