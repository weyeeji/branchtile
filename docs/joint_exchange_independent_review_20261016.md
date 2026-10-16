# Joint support exchange：独立新颖性与反例审查

审查日期：**2026-10-16**。结论：**现有 v1 是有逻辑可行性信号的普通启发式，尚没有通过新颖性门槛的算法贡献。** 在当前“每请求、每 KV head 一个冻结的非负加性分数；固定基数；固定必留项”条件下，最小 union 可直接化为经典 submodular cover。没有检索到完全相同的 LLM 场景，不足以推翻这个数学上的已有问题类别。后续值得比较强基线、检验自然质量与完整成本；不应直接将 singleton exchange 命名为新方法投稿。

本轮独立阅读了 `docs/joint_selection_prior_audit_20261016.md`、`scripts/probe_joint_support_exchange.py` 和 `results/joint_support_exchange_v1/`。仅使用 CPU、项目 `.venv` 和公开一手资料；没有运行 GPU、访问事实审阅文件、改动冻结结果或推送 git。

## 1. 一手文献究竟覆盖了什么

以下日期是明确版本的日期，不是搜索引擎的抓取时间。表中“未见”只涉及已核查的算法/章节，不能证明整个领域不存在相同工作。

| 一手来源与日期 | 已有机制及与本方向的关系 |
| --- | --- |
| [Wolsey, *An Analysis of the Greedy Algorithm for the Submodular Set Covering Problem*，1982](https://doi.org/10.1007/BF02579435)；[作者机构记录](https://research.dial.uclouvain.be/entities/publication/a81e631b-bc8e-46b6-8630-dce8b3469c16) | 最小代价满足次模覆盖要求的贪心分析是经典问题。此处未取得该论文受限全文；下项公开原文给出本次需要的覆盖与截断构造。不能直接给浮点 attention 分数套一个与精度无关的整数覆盖近似比。 |
| [Krause et al., *Robust Submodular Observation Selection*，JMLR，2008-12，§4、式(6)](https://www.jmlr.org/papers/volume9/krause08b/krause08b.pdf) | 对多个目标分别截断后相加，将“所有目标都达标”化为 submodular cover，再使用 greedy partial cover；SATURATE 另对目标值做搜索。逐请求而非平均达标、截断覆盖、非单位成本都不能被当作新概念。 |
| [BumbleBee，COLM 2024，§5、Algorithm 1](https://users.ece.cmu.edu/~agr/resources/publications/BumbleBee_colm_24.pdf)；[作者页面](https://lillykumari8.github.io/publication/BumbleBee) | 用 facility-location 与 attention importance 的次模混合选择 KV；原文明确提到共享 prompt 的预计算缓存适合离线总结。其目标是重要性与多样性的固定预算最大化；已读部分未见本项目的逐请求固定 top-k 分数底线与最小物理 union。但“用覆盖/次模优化选择共享 KV”这一宽泛表述已有先例。 |
| [Multi-Query Diversification in Microblogging Posts，EDBT，2014-03-24–28](https://www.cs.ucr.edu/~vagelis/publications/MicroblogDiversificationEDBT14.pdf) | 在多个 query 需求下寻找覆盖它们的最小代表集合，有近似算法和小实例精确算法。其时间/情感覆盖模型不同于 attention，但“多个查询都必须满足、最小共享集合”并非新问题模板。 |
| [Jina 官方 weighted/saturated coverage 演示，2025；2026-10-16 访问](https://jina.ai/news/submodular-optimization-for-text-selection-passage-reranking-context-engineering/) | 用 query relevance 与 passage coverage 做多 query 的饱和覆盖和 lazy greedy。它是作者技术说明，不是本项目的同约束 KV 复现或新理论依据；说明检索领域的覆盖目标不能仅因换成 KV 就获得算法新颖性。 |
| [GVote，2509.03136v1，2025-09-03，§3.1–3.2](https://arxiv.org/html/2509.03136v1#S3) | 从合成未来 Q 得到 top-k 集合，取 union 决定预算；实现定义按 request/head 独立。它已经用多 Q 的需求来决定保留集合，但不是在已到达的多个 request 间重新交换支持来最小化 union；也没有保证采样外 query 的真实误差。 |
| [QuoKA，2602.08722v1，2026-02-09，Algorithm 1、§3.3](https://arxiv.org/html/2602.08722v1#S3.SS3) | chunked prefill 中先选代表 Q，GQA 方向平均、Q 方向最大聚合，再取公共 KV 子集。不是每个 request 保持不同 support 的 constrained union，但“跨 Q 共同选择、避免平均掩盖需求”已存在。当前 common-max pooled-score 控制没有复现它的 cosine scorer 或 Q 子选择。 |
| [Attention Matching，2602.16284v1，2026-02-18，§3.1–3.3](https://arxiv.org/html/2602.16284v1#S3) | 对多个参考 Q 匹配 attention 输出及 mass；包含 RMS 选 keys、OMP keys、NNLS bias、value 拟合以及 on-policy queries。仅改成“输出误差约束”“观察窗 witness”或“共享参考 Q”不足以构成新颖性；其修改 bias/value 的方法与保持原 KV 的子集选择不是同一约束。 |
| [Joint Encoding of KV-Cache Blocks for Scalable LLM Serving，2601.03067v1，2026-01-06，§4](https://arxiv.org/html/2601.03067v1#S4) | BFF/CFF 跨 request/chunk 融合相似 KV blocks，以共同表示和 block table 共享减少存储；有相似性阈值与 rate-distortion 分析。它会改变表示，未见逐请求原 KV 子集的固定分数约束优化。它足以排除“首次用近似使不同请求共享更小缓存”的宽泛主张。 |
| [Zipage，2603.08743v1，2026-03，§4.4](https://arxiv.org/html/2603.08743v1#S4.SS4) | 压缩到目标 blocks，以免破坏仍被其他请求引用的 prefix；不是本次已核实的联合支持选择器。但原共享缓存释放、独立压缩副本、引用计数与异步成本，都是容量比较必须纳入的强系统基线。 |
| [CORE，2610.02235v1，页面日期 2026-09-28，§3、§4](https://arxiv.org/html/2610.02235v1) | 页面题名为 *One Distribution, Two Channels*；结合 query utility、log-det set coverage 与 evicted-mass compensation。原文明确区分质量、方向与 mass。未见跨 request 的 min-union 约束；但“set-aware coverage 比独立排名好”及用输出误差指导保留已有很新的先例。 |
| [JSSA，2610.19026v1，2026-10-13，§3.2–3.3](https://arxiv.org/html/2610.19026v1#S3) | joint 是 Q/K scoring subspace；GQA 内 mean-query 排序后 indexed attention，保留不同 KV-head 支持。不是多个 request 的联合 mask 优化，不能因标题 joint 就视作等价；也不能忽略其作为高效 selector/indexed 执行参照的地位。 |
| [QUILT，2610.11134v1，2026-10-08，§3.3–3.4](https://arxiv.org/html/2610.11134v1#S3.SS4)；[PageWeaver，2610.11201v1，2026-10-08](https://arxiv.org/html/2610.11201v1) | 前者已有 shared/private 层次分解以及近似裁剪低重要性私有尾部；后者基于既定 support 重组 query。不能把 QUILT 说成完全不修改 support，也不能把 common-core/residual 或物理 union 执行重新命名为本次创新。 |

官方代码补充核查：Attention Matching 的 [highest_attention_keys.py，commit `2020f6d37940fe013e7e7f1ebf409e5d107c5670`](https://github.com/adamzweiger/compaction/blob/2020f6d37940fe013e7e7f1ebf409e5d107c5670/compaction/algorithms/highest_attention_keys.py)（2026-03-31）明确接收一个参考 Q 矩阵并实现 mean/max/RMS、可选 pooling、top-k 及后续拟合。BFF 的 [官方 README，commit `233833584c6ad45beb159f18f2b990bdef2c18cb`](https://github.com/sef1/kv_fast_fusion/blob/233833584c6ad45beb159f18f2b990bdef2c18cb/README.md)（2025-09-18）指向 vLLM `e4ca6e3...` 与外部 patch；本轮没有下载、安装或资格化该 patch，不能写成已复现的性能基线。GitHub commit 身份由官方 API 截至 2026-10-16 的查询确认。

## 2. 当前加性目标可严格化为已有覆盖问题

以下是针对本项目约束的推导，不主张为新定理。先固定一个 layer/rank/KV-head 和已观测请求组。物理源必须相同：同模型版本、相同因果前缀与位置、相同原 KV backing 数据；逻辑 token ID 相同不够。

定义：

- `V`：实际共同源里允许交换的文档 KV 条目，不含任何必留项。
- `M`：共同源内固定保留的文档条目；本实现 first4/last32 中落在共同源的部分属于这里。system prefix 固定保留，可作为另一个固定物理成本。
- `D_r`：文档末尾不完整共享页内、实际存在于请求 r 的 private KV 的必留文档条目。它们必须逐请求保留，不能把相同逻辑位置去重为共享 KV。
- `P_r`：题目及生成尾部等其余 private KV，固定保留，不参与 document budget 和当前 pooled-score 选择。其存储/执行是固定项；本项目分数约束不是对它们重新评分。
- `k_r = K_r - |M| - |D_r|`：该请求可交换文档条目的精确个数；当前点之间 `K_r` 相同，但推导不要求所有 r 的 `k_r` 相同。
- `s_rt >= 0`：冻结的 pooled score；`A_r^0`：原文档 mask 的总分；`c_r`：`M ∪ D_r` 的固定分数贡献。
- `tau_r = (1-epsilon_r) A_r^0 - c_r`：可交换部分的固定分数底线。`tau_r <= 0` 时该分数约束为空，但精确基数约束仍在。

考虑可用共同集合 `U ⊆ V`，以及

```text
g_r(U) = max { sum(t in T) s_rt : T ⊆ U, |T| ≤ k_r }
       = U 中最多 k_r 个最高分数的和。
```

存在满足原约束的 `T_r ⊆ U`，当且仅当：

```text
|U| ≥ max_r k_r，且对所有 r，g_r(U) ≥ tau_r。
```

因为可以直接取 U 内每请求自己的 top-k；各请求可以取不同支持。原问题的最优 union 大小等于满足这些条件的最小 `|U|`：任意可行 masks 的 union 满足条件；反之，从可行 U 取各请求 top-k 得到可行 masks，其 union 不大于 U。固定的 `M`、system prefix 共享成本和 `D_r/P_r` 私有成本不改变这个最优化。

非负分数的 top-k 和是 uniform-matroid weighted-rank function，单调且 submodular。直接看加入一个 token 的增益：不足 k 个时是其分数，已有 k 个时是它超过当前第 k 大分数的正部分；集合越大，第 k 大分数越高，增益不会变大。`k_r=0` 的函数恒为零。

对所有 `tau_r>0` 定义：

```text
F(U) = sum_r min(g_r(U)/tau_r, 1)
       + min(|U|/k_max, 1)       # k_max>0 时加入此项
```

每项截断仍单调次模，和也如此。F 达到全部项数，恰好表示所有请求达标且可填满基数。这正是经典 submodular cover；截断各函数以避免某请求的过量收益掩盖另一请求的不足，已经是 Krause 等 §4 的构造。**这是 constrained joint selection 的一个标准化简，并不是 v1 新发现了一个新的优化类别。** 对实数分数不能不加条件直接沿用整数版本的对数近似比；任何阈值松弛、精度离散化与末尾修复都要单独说明。

固定物理页映射时也没有自动产生新算法类别：以 page 为 ground element，令 `T(p)` 为该页实际拥有的共同条目，使用 `g_r(union_{p in A} T(p))`；它同样具有递减边际收益，固定页成本可作为权重。mandatory 已占用的页应计作固定成本，页内可用槽位不能重复收费。**这只适用于已知不变的 backing/page 映射。** 重排压紧后才决定页归属、跨层共享 allocation、请求到达退出和源缓存保活，不是上述固定 page cover 的现成解；本轮也没有证明这些系统约束带来新贡献。

特别限制：该等价仅对应 v1 每 request/head 的**一个**聚合分数底线。若同一 support 必须同时通过多个 witness 的 O/LSE 或 attention-mass 条件，分别计算每个 witness 的 top-k 是不充分的：`k=1`，witness1 只偏好 a、witness2 只偏好 b，U={a,b} 的两项独立 top-1 都能达标，但没有一个单元素支持同时达标。输出误差还含归一化和 value 方向，不能直接宣称保持单调次模。全层稀疏生成改变后续 Q/private KV，也超出冻结实例的推导。

## 3. v1 的交换实际做了什么

v1 只允许删除 owner 数为 1 的非必留 token，并加入当前已被其他请求选择的 token。每个请求轮到时，删除候选按本题分数升序、加入候选按分数降序，取满足剩余损失额度的最长前缀；两轮、两种固定顺序分别记录。

在这一冻结的局部邻域里，前 m 个最低删除项与前 m 个最高加入项，确实使 m 次替换的加性损失最小；无需搜索所有配对。每个成功交换恰好减少一个 union token。此性质解释了代码结果，但不是全局近似保证：它禁止临时不降低 union 的交换、禁止协调删除多 owner 条目、禁止引入原 union 外的新共同条目；也不能撤销先前共享化的选择。

任何 mask 集合都可写成公共交集加各自 residual；residual 之间仍可能部分共享，并不天然是物理私有集合。v1 是把 singleton 条目改成已共享条目的单向规则，明显不同于完全固定 mask 的 prefix caching，但与“保留公共部分、减少低重要性私有部分”的启发式只有约束与细节差异。它尚未展示超越普通 common-core/residual、覆盖贪心或协调删除的机制优势。

## 4. 不依赖 tie 的可复现反例

以下在 `.venv` 上直接调用冻结的 `exchange`，两个请求顺序均测试；每行分数都互不相同。无保护项只是去掉了固定常数；可给所有请求附加相同必留项而不改变停滞机制。

### 4.1 多 owner 的联合删除被禁止

| 请求 | a | b | c | 唯一初始 top-2 |
| --- | ---: | ---: | ---: | --- |
| r0 | 102 | 101 | 100 | {a,b} |
| r1 | 102 | 100 | 101 | {a,c} |
| r2 | 100 | 102 | 101 | {b,c} |

`epsilon=.005`，每题初始总分 203。每个 token 有两个 owners，故 v1 完全不交换，union=3。三题全用 `{a,b}` 时，r0 不损失，r1/r2 各损失 `1/203≈0.4926% < 0.5%`，union=2；由基数下界它就是最优。一次“从可用 union 删除 c，然后所有请求各自 top-k refill”即可完成，根本不需要新算法。

### 4.2 最优公共位置可能在原 union 之外

| 请求 | a | b | c | 唯一初始 top-1 |
| --- | ---: | ---: | ---: | --- |
| r0 | 100 | 1 | 99 | {a} |
| r1 | 1 | 100 | 99 | {b} |

`epsilon=.01`。v1 只能尝试 a↔b，损失过大，停在 union=2；两题都用 c 的损失各 1%，union=1。restricted-U0 reverse-delete 也解不开此例，必须与允许完整 V 候选的 cover 或 common-core 控制区分。common-mean 在这个例子就能找到 c；因此“用复杂联合交换一定胜过简单聚合”也不成立。

### 4.3 score、输出、页数不能互相替代

两个 keys 得到相同 attention/pooled score、values 分别为 `+M/-M`，保留其中一个的输出分别为 `+M/-M`。零 score loss 仍可伴随 `2M` 的输出变化。该例不否定当前实测可行性，只否定“低 pooled-score 损失保证输出或事实质量”的推断。

对于 16-token 页，union 从 `{0,1,16,17}` 缩为 `{0,16}`，token 数减半而仍需要两页。若完整源 cache 仍保活，物理回收更是零。每请求精确 k 不变时，独立 E0 的 token 处理数也不变；共享地址可能改善缓存复用，但不能把 union 降幅直接作为 E0 FLOPs/延时降幅。

### 4.4 本轮 CPU 验证记录

冻结脚本 SHA-256：`97625c9207a2541ab76cef5eedf5200740ab6bf066abae5d9b89cd49b56aa651`。严格 triangle 与 outside-union 两例的 `exchange` 输出分别为 `[3,3]`、`[2,2]`；枚举全部可行 mask 组合所得最优分别为 2、1。

另对 120 个 `N=5,B=3,k=2` 随机小实例（NumPy seed 8821），枚举所有满足阈值的 masks 与所有候选 U，最优值全部相同；再对 120 个不等 `k_r∈{1,2,3}`、包含非零 mandatory/private 分数常数的实例（seed 9917）做同样枚举，全部相同。这是推导与实现解释的 CPU 资格检查，不是自然数据收益、质量或算法新颖性证据。CUDA 未初始化。

反例可独立重现，不写入冻结结果：

```python
import importlib.util, itertools
import numpy as np
spec = importlib.util.spec_from_file_location("probe", "scripts/probe_joint_support_exchange.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for scores, bits, eps in [
    ([[102,101,100],[102,100,101],[100,102,101]], [[1,1,0],[1,0,1],[0,1,1]], .005),
    ([[100,1,99],[1,100,99]], [[1,0,0],[0,1,0]], .01),
]:
    s, baseline = np.array(scores, float), np.array(bits, bool)
    k = int(baseline[0].sum())
    feasible = [[set(c) for c in itertools.combinations(range(s.shape[1]), k)
                 if s[q,baseline[q]].sum()-sum(s[q,t] for t in c)
                    <= eps*s[q,baseline[q]].sum()] for q in range(len(s))]
    oracle = min(len(set.union(*sets)) for sets in itertools.product(*feasible))
    unions = []
    for order in [list(range(len(s))), list(reversed(range(len(s))))]:
        masks, _ = m.exchange(s, baseline, np.zeros(s.shape[1], bool), eps, order)
        unions.append(int(masks.any(axis=0).sum()))
    print(unions, oracle)  # [3,3] 2 ; [2,2] 1
```

## 5. 可以直接实施的强基线

所有新基线应使用独立输出目录和预冻结协议，不能回写 v1 或按文档挑顺序。以同一输入 SHA、同一保护范围、同一逐请求阈值、同一 tie-break、同一候选域比较；最后用原始 float64 分数和 `math.fsum` 检查每请求真实总分，不能仅相信增量或浮点 F 已接近上限。

**A. Union reverse-delete + top-k refill：最先运行。** 初始 `U0=union(S_r^0 \ mandatory)`；固定删除顺序为 `max_r(s_rt/A_r^0)` 升序，token ID 打破同分（该顺序只是预注册启发式，不称最优）。对每个 t，令 `U'=U\{t}`，重取所有请求在 U' 的 top-k；只有每题基数、必留项和底线全部通过才删除。最终用这些请求自己的 top-k 作为 masks，删去未使用条目。它能协调移除多 owner token，直接解决 4.1；但初始域受限，不能解决 4.2。

可执行伪代码：

```text
U ← union(original free masks)
for t in fixed ascending priority:
    if |U|-1 < max(k_r): continue
    T_r ← stable_topk(scores_r restricted to U\{t}, k_r) for every r
    if every exact_score(T_r) ≥ tau_r: U ← U\{t}
return mandatory ∪ stable_topk(U,k_r) for each r
```

实现可为每请求保留按 score/token-ID 排好的 U0 列表、当前 top-k membership 与下一个 alive replacement 的游标。删除 t 仅影响当前选中 t 的请求；试算 `g_r-s_rt+s_r,replacement` 后再精确验收。随着 U 缩小，一个因阈值不可删的 t 不会重新变为可删，因为 `g_r(U\{t})` 只会降低。因此固定顺序一遍足以产生该域内 inclusion-minimal 的可行 U，但不保证最小基数，不能写成全局 oracle。

**B. Saturated greedy cover + refill + reverse-delete：数学上的直接强对照。** 分开记录候选域 U0 与完整 V 两个 arm。从空集合开始，每次取 `ΔF/cost` 最大的条目；每题维护 top-k heap，计算真实边际增益；直到全部 score 底线和 cardinality 检查通过。取各请求自己的 top-k，删除 unused U，再运行 A 的可行删除过程。可用 lazy heap：次模性保证旧边际增益是上界，每次取出候选后重新计算并决定是否接受。精度处理不能暗中放松约束；记录所有重新评估次数、wall/CPU 时间和峰值内存。最坏情况下 lazy greedy 仍可能昂贵，超时也要保留，不能拿离线 CPU oracle 成本当在线生产成本。这是经典覆盖算法的适配，**不是本项目的新方法**。

**C. 小实例 exact/MILP/LP bound。** 变量 `x_rt∈{0,1}` 表示请求是否选 token，`y_t∈{0,1}` 表示共同 backing 是否启用；最小化 `sum c_t*y_t`，约束 `x_rt≤y_t`、`sum_t x_rt=k_r`、`sum_t s_rt*x_rt≥tau_r`，mandatory 已预处理。固定页则用 `x_rt≤y_page(t)`。小实例枚举或整数求解器提供最优；LP 松弛提供下界。自然长文只在明确的候选域和限时内求解，必须报告 domain、gap、未完成状态，不能称整篇原问题最优。

**D. 受同约束约束的 common-core + independent residual。** 对 mean/max/RMS 等已知 aggregate 排序，预定 core 长度网格；每题选 core 外自己的剩余 top-k。只比较满足同一底线的格子，保存所有失败和额外搜索成本。core=0 必须退化为独立 QC；不同 core 大小不能按最终质量事后择优。v1 的全公共 mean/max 只是一种极端格子，且大多违反 1% 约束；它们不是足以建立算法优势的强对照。

**E. 必须单列的部署控制：独立达到相同 score floor 的最短 top-prefix。** 每请求独立保留 mandatory，加自身排序前缀，直到达到 `(1-epsilon)A_r^0`；其 `k'_r≤k_r`。它不满足 v1 的“恰好 k”，所以不并入同基数优化表，但实际部署通常关心预算上限与质量。它减少每请求条目数，无需等待其他请求；若在同质量/成本前沿已优于联合交换，强制补满 k 可能只是人为制造研究问题。GVote 的 top-p 思路也使这一控制不能被省略。

执行侧继续使用已经资格化的强原生路径：独立 FlashInfer E0、原 mask 的 native union、联合后 mask 的相同强执行器。Zipage 是容量/生命周期的系统参照；QUILT/PageWeaver 是共享执行参照，但都不能只引用其宣传速度来替代本地适配资格。FastKVzip、Attention Matching、BumbleBee 等准备阶段不同的方法，应在同 query 可见性、自然质量和完整准备成本下作补充，不能把未经资格的适配称原作者完整复现。

## 6. 现有 132 行结果支持的结论

v1 使用 3 个真实文档、rank0/layer15、2 个预算、2 个 KV heads；**12 个 head cells 不是 12 个独立文档，更不是 132 个独立质量样本**。每题相同 scorer 的独立 top-k 是起点，允许 score loss 后有可交换边界，本身符合预期。

1% pooled-score loss 下，两个固定顺序的逻辑 union 减少范围分别为 17.65–22.99%、18.12–22.87%；零松弛全部不变。它说明独立 top-k 在这个代理目标的近优集合中存在共享机会。完整源 cache 未释放，物理容量收益为零；没有 O/LSE 和自然质量验证。不能将结果改写成“1% attention 误差带来 20% cache 节省”。

common-max/mean 的最大逐请求 score 损失分别约 23.18%/18.66%，不满足同约束；它们损失较大不能证明 exchange 比所有共同选择更好。也尚未与 A/B/D/E 相比。选择阶段 CPU 约 2.465 秒为一次包含多配置的数据运行；既不是每层在线 selector 延时，也不是重复测量分布。

## 7. 成为可投稿贡献还缺什么

最强已知算法对照是**经典 truncated submodular cover 的适配及精确/松弛界**，不是仅 common-mean/max。最强直接工程控制是 **reverse-delete/refill**，最强预算上限控制是 **独立阈值 top-prefix**。当前算法没有理论或实测证据超越它们。

若继续，优先做低成本、可停止的验证：

1. 冻结同阈值 CPU A/B/D/E，B1 控制、上述反例和小实例 exact；报告胜负与成本，而非只报 union 有下降。现阶段可继续把结果称 heuristic feasibility，无需制造新名称。
2. 对独立 QC 已质量合格的预算，再做逐请求/逐 witness/full-private-tail 的 attention 检查和全层自然生成；不能用与失败 4K 相同的代理质量来晋级。输出误差约束与未来 Q 漂移已有 Attention Matching 等先例；仅补该检查不会自动成为创新。
3. 选择 union 流量或常驻容量中的一个明确目标。前者需要强执行器的完整调用收益；后者需要真实 allocator 释放、mandatory/private、全层全头页和源 cache 生命周期。必须包含协调等待、打分、交换/cover、索引、复制/压紧、metadata、attention 和回退成本。
4. 只有出现通用覆盖方法解决不了的**具体**约束，并提出可证明或有强消融证据的高效机制，或形成足够广泛且可推广的系统结论，才重审论文贡献。把这些已有组件组合起来本身不够；本轮没有认定其中任何一项已经成立。

停止条件：若相同约束下 A/B/D 已达到相当收益，v1 不再作为新算法推进；若 E 在同自然质量下更便宜，联合协调缺少部署理由；若收益只在质量失败预算、只在单层逻辑 union、或无法抵消准备/等待成本，停止性能/容量路线。可以保留公开可复现的负面结果，但不能据此直接宣称满足 ISCAS 投稿贡献要求。

## 8. 检索边界

本轮检索包含 `KV cache multi-query joint compression union`、`KV cache submodular shared`、`per-request joint selection`、`shared prefix compression joint`、`per-query set cover`、`minimum union attention`、`multi-query weighted coverage retrieval`，并沿论文与作者官方仓库核对。已加入旧审计中缺失的 BumbleBee、经典 robust/submodular cover、跨请求 BFF/CFF、CORE 和多 query 检索覆盖先例。

未找到已核实的“逐请求固定原 KV mask、逐请求真实输出误差预算、在线最小可兑现物理页成本”完整同构实现；也没有完成全球专利、所有未索引论文或所有作者代码分支的穷举。**该有限空缺不是新颖性证明；对于当前 pooled-score v1，经典优化归约已经足以否决仅靠问题措辞和局部交换规则成立的新算法主张。**

## 9. 初稿提交后的分阶段证据补充

初稿独立审查提交为 `2c7fc83`；上文“没有 O/LSE”描述的是被审阅的 CPU exchange v1 本身，不能理解为后续始终没有这一阶段。随后团队另行完成了 [held-out FP64 attention 阶段](../results/joint_attention_fp64_v1/README.md)（`56dd877`）：1% pooled-score loss 对应的最大 retained dense-softmax mass 下降达到 4K 的 6.50 个百分点、16K 的 4.92 个百分点；单 query-head 的 dense-reference 相对 L2 误差增量最大为 +0.133470/+0.044762。平均或中位改善不能抹去个别恶化。该阶段使用 dense-trajectory captures，仍不证明 on-policy 自然质量；它进一步支持本文区分 score floor 与真实误差的结论。本审查只读取其公开结果/README，没有重跑 GPU。

root 新增的 `reverse_delete` 已接受只读约束检查：fixed-k、mandatory、活跃域与 next-best refill 的逻辑成立，最终 `fsum` 检查没有采用 1e-12 放宽验收。初稿指出零松弛分支仅验证 canonical top-k、未证明 strict boundary；团队随后以独立 [boundary_gap_audit.json](../results/joint_support_reverse_delete_v1/boundary_gap_audit.json)（`1af149a`）确认当前 40 个请求/预算/head 边界均为正 gap，关闭这些自然输入的证据缺口。这是后验输入审计，没有静默修改原算法或结果。

本审查员另按预冻结 `e06c531` 完成了第 5 节 E 的 [relaxed-budget 独立最短 top-prefix CPU 控制](../results/joint_support_relaxed_prefix_v1/README.md)：48 cells、160 request checks、0 GPUh，1% 阈值下 union 下降仅 1.95–4.13%，sum selected entries 下降 2.37–4.49%，12 个 head cells 均未得到比两种 exact-k exchange 更小的 union。它保留了“身份协调超出直接剪尾”的代理几何信号，但约束是 ≤k，尚无该 arm 的 O/LSE/自然质量或执行性能证据，也没有改变本文的新颖性否决。
