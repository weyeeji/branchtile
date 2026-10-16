# Packed owner-task E3：近邻、主张边界与必要对照

审查日期：2026-10-16。范围是原支持集上的算子执行，不评估新的 KV
选择器；本次只读本仓库和一手论文/官方代码，没有启动 GPU，也没有修改
正在实现的 `attention_owner_tasks`。该候选尚不能从本审查获得数值、性能或
新颖性资格。

**判断：适合立即作为缺失的强 E3 控制测试，目前更像已知技术的具体组合，
不足以预设为新的论文机制。** 每个 owner-region/chunk 一个 CTA、把共享该
片段的 query 压进 MMA 行、写 partial 再按原 query 归并，分别受到 PAT、
MSA 和 split-KV 的直接先例约束。严格只分配活跃 task 行的 packed workspace
是可核实的实现差异；它不能自动升级为首次稀疏 attention、首次紧凑执行或
首次避免跨区域状态搬运。论文中“没有同字实现”的证明责任远高于这一轮检索。

## 最直接的一手先例

| 工作与可核实日期/状态 | 已覆盖机制 | 对本候选的约束与区别 |
|---|---|---|
| [PAT v3](https://arxiv.org/html/2511.22333v3)，2026-03-16；论文页标注 ASPLOS 2026；[作者仓库](https://github.com/flashserve/PAT) | 将共享 prefix 的 queries 打包给 CTA，选择多种 M/N tile，长 KV 分片，输出 partial 后逐 query 合并；§5 的代价已计 partial 读写。 | pack→forward→merge、多 tile、长 KV chunk 和逐 query 稳定归并均非新范式。其 prefix 树结构不同于本仓库非层次的任意 owner 集合，但这种适配差别本身还不证明算法创新。 |
| [MiniMax Sparse Attention v2 §4.2](https://arxiv.org/html/2606.13392v2#S4.SS2)，v1 2026-06-11、v2 2026-06-12；本轮核实为 arXiv 论文及[官方代码](https://github.com/MiniMax-AI/MSA) | KV-outer 反向稀疏索引聚合 queries；预调度 worklist/chunk；给每个 query 的 partial 预分配槽位；按 query 的有效槽位数 combine，attention forward 不需原子写。 | 这是比普通 split-KV 更近的先例。其主要 chunk 维度是某 KV block 的 query fanout，本候选主要沿 owner-region 的 KV token 长度切块；不能因此重新声称 KV-outer、query gather 或 slot-aware merge。 |
| [PageWeaver v1 §3.2、§4.2](https://arxiv.org/html/2610.11201v1#S3.SS2)，2026-10-08；本轮只核实到 arXiv 版本，未确认作者执行代码发布 | 明确研究 page-oriented worklist、公共 query 集、partial 输出与最终归并，并讨论该路径的准备/归并费用；最终 query-owned 路径跨页保留状态并直接写完整输出。 | “共享就值得按 KV 分任务”和“避免 cross-page partial 才是首次”都不成立。其 page-pair 替代路径还给出相反结果的测量理由，不能省略准备和 partial 成本。 |
| [QUILT v1 §3.2–3.4](https://arxiv.org/html/2610.11134v1#S3.SS2)，2026-10-08；本轮只核实到 arXiv 版本，未确认作者执行代码发布 | 稀疏集合交/差分解、不同 query 粒度共享、query-specific residual、tile tail 处理。 | shared/private 分解、压紧活跃集合、多粒度执行和尾块优化已有。精确 tail push-down 与最终 residual clipping 要分开：同支持集控制必须关闭会删支持边的 clipping。 |
| [Fused3S §3](https://arxiv.org/html/2505.08098v1#S3)，2025-05-12，ICS 2025；[官方仓库](https://github.com/HPCForge/Fused3S) | CSR 稀疏布局压紧、Tensor Core 的 SDDMM–softmax–SpMM 融合、row-window 调度及寄存器布局 remapping。 | 不可把稀疏 MMA 融合、片上 softmax 状态或 layout remapping 作为新原理。它不是本候选逐 owner task＋跨 CTA softmax merge 的同字实现，且原工作是图/稀疏 forward 输入，适配范围须实证。 |
| [FlashInfer §3.3、附录 D](https://arxiv.org/html/2501.01005v2#S3.SS3)，v2 2025-04-21，MLSys 2025；2026 官方实现仍继续更新 | KV 按长度拆成 work item、调度、partial workspace 和组合；短请求可直接写输出，只有需要 split 的请求产生中间输出。 | 非均匀 split、跳过不需要的 partial/merge、workspace 压缩和直接输出均有先例。图稳定地址需要最大容量预留也已有，不应把它漏出显存报告。 |
| [NVIDIA cuDNN BSA 官方接口](https://docs.nvidia.com/cudnn/latest/fe-oss-apis/bsa/)，访问于 2026-10-16；产品代码/文档，不是本轮新论文 | block sparse 索引、可变有效块数、empty rows；适用架构的 split-KV 写 FP32 partial 并 combine。 | “稀疏索引＋split＋FP32 merge”是现成工程能力。SM120 BF16 GQA/D128 已有可调用路径，但文档没有把所有架构的 split-KV 能力赋给 SM120，不能混写支持矩阵。 |

本表是正面检索到的机制重合，不是“没有其他相关工作”的断言。没有核实到
QUILT/PageWeaver 作者代码不意味着代码不存在，也不豁免适用机制控制。

## 官方代码的精确边界

PAT 检查点为
[`8cb067f0a31dabe800958f657093f17ac5d7366d`](https://github.com/flashserve/PAT/commit/8cb067f0a31dabe800958f657093f17ac5d7366d)，
GitHub committer 日期 2026-05-26；kernel 的最早公开文件提交为
[`fdd6176f875ddbe29a34617b8aa800b056ae4e39`](https://github.com/flashserve/PAT/commit/fdd6176f875ddbe29a34617b8aa800b056ae4e39)，
2025-12-15。[api.h L219–221](https://github.com/flashserve/PAT/blob/8cb067f0a31dabe800958f657093f17ac5d7366d/csrc/api.h#L219)
分配 `[batch, heads, max_split_per_seq, D]`；
[merge L330](https://github.com/flashserve/PAT/blob/8cb067f0a31dabe800958f657093f17ac5d7366d/csrc/pat_fwd_kernel.h#L330)
读取每个序列自己的 `num_split_per_seq`，并对单个 partial 单独处理。
因此它已有有效 partial 计数和按 query 归并，但本次所读版本不是严格的
`sum(active task rows)` 一维 partial 分配。不可把“有有效计数”与“分配完全
ragged”混为一谈。

MSA 检查点为
[`80434d7f67877c6570ca19cac444b84bc9855dac`](https://github.com/MiniMax-AI/MSA/commit/80434d7f67877c6570ca19cac444b84bc9855dac)，
2026-07-30；所读 scheduler 文件的公开提交为
[`9175bebf2b623ca0ea8ad5247bb615cd424598a0`](https://github.com/MiniMax-AI/MSA/commit/9175bebf2b623ca0ea8ad5247bb615cd424598a0)，
2026-06-11。[prepare_scheduler.py L125](https://github.com/MiniMax-AI/MSA/blob/80434d7f67877c6570ca19cac444b84bc9855dac/python/fmha_sm100/cute/src/sm100/prepare_scheduler.py#L125)
构造 compact flat worklist；L349 起打包 `q_idx/split_slot`；
[combine.py L1214](https://github.com/MiniMax-AI/MSA/blob/80434d7f67877c6570ca19cac444b84bc9855dac/python/fmha_sm100/cute/src/sm100/fwd/combine.py#L1214)
接受逐行 split counts，但 partial 的逻辑形状仍为
`[num_splits, total_q, heads, D]`。所以其 worklist/slot 方法与本候选高度接近，
而本候选更紧的全局分配不能反过来归给 MSA 原实现。原论文描述的是 sparse
prefill 的 KV-outer 核心；本仓库 GQA8/B3–B4 decode 的任意 token owner 集合
适配，仍须单独证明输入语义及实现收益。
Scheduler 的 L476 仍用 `atomic_add` 分配 slot，因此“attention forward
不需 atomic”不能扩写为整个准备/执行流程完全没有 atomic。CSR 反向索引、
prefix offsets 和 ragged 列表是常规数据结构；更紧的分配也需要测出有意义
的增量，不能仅凭换一种数组形状宣称新的算法。

Fused3S 官方检查点
[`65d43a9cf212d6f18d8931de8e1c66a728756d7d`](https://github.com/HPCForge/Fused3S/commit/65d43a9cf212d6f18d8931de8e1c66a728756d7d)
日期为 2025-10-13。BSA 所属仓库检查点
[`ea97aab73c1dc38d464a50212d37a4808fa47311`](https://github.com/NVIDIA/cudnn-frontend/commit/ea97aab73c1dc38d464a50212d37a4808fa47311)
为 2026-10-16 11:34:25 UTC；该目录截至审查时最近提交
[`5225ce3f15fae6358ce32fc8ddb505582f220b2d`](https://github.com/NVIDIA/cudnn-frontend/commit/5225ce3f15fae6358ce32fc8ddb505582f220b2d)
为同日 09:48:25 UTC。这里记录的是代码观察点，不把仓库最新提交日期当作
所有功能的发布日期或正式二进制发行日期。BSA 的 token/page、causal、layout
及 GQA 约束必须按合法接口适配；不能扩大 support 来换一个可跑的对照。

## 本仓库 E3 为什么还缺一个强版本

[提案 §5.4](/home/qcjz/iscas/proposal-iscas.md:235) 已明确 E3 应为“owner
区域紧凑执行＋全局 partial/merge”，允许合理 split/直接输出，并要求固定行/
紧凑行 × 区域写回/跨区域片上累积的 2×2 消融。
现有 [attention.py](/home/qcjz/iscas/src/branchtile/attention.py:107)
及 v2/emptyguard/FullCol wrapper 的 `region_partials=True` 将输出 split 数
改为 `R*S`，分配 `U[H,R*S,rows,D]` 与同形 M/L；通用 merge 对每个 query
扫描全部 `R*S` 槽，包括不属于该 owner 集合的行。
[FullCol producer](/home/qcjz/iscas/src/branchtile/tilelang_kernel_fullcol_guard.py:96)
即便只处理一个 region，仍建立完整 canonical shared CU/CM/CL，再写完整行。
这是可复现的旧消融 hook，不能等同于最强紧凑 E3。

`bench_kernel.py` 接受 `compact_partial/fixed_partial` 以走该 hook；其默认
方法却是 `compact,fixed`。因此不能只凭脚本存在就宣称已经完成最强 E3
评价，旧实验也不应被新实现覆盖或改名。
本仓库已有 [msa_e0_adapter](/home/qcjz/iscas/src/msa_e0_adapter.py:1) 是
FlashInfer SM120 sparse-decode 的逐 request 精确 gather 适配，明确不是
完整作者算法复现；它也不能代替上述 MiniMax 论文 KV-outer 跨 query
worklist 机制的控制。

设 task 已包含 head/region/chunk 身份，活跃 GQA 行数为 `a_t`，
`P = sum_t a_t`。FP32 的 U/M/L payload 为 `4*P*(D+2)` 字节，而旧 E3 是
`4*H*R*S*rows*(D+2)`；还必须另加 task descriptors、反向索引/CSR、对齐与
容量余量、O/LSE 以及编译/allocator 所需内存。只在 payload 上填一个更小
数字，不等于总显存更小。

若沿用相同 uniform S、全部非空 owner 标签，B3 的有效行总和为每 head/split
96（旧全行 168），B4 为 256（旧 480）；这只是 1.75×/1.875× 的行槽缩减。
改成 256/512/1024-token chunks 后，task 数变为实际长度的 ceil-div 总和，
不再等价于 uniform S。相对 fused 方案仅 `H*S*rows` 的 partial，packed E3
甚至仍可能更大。必须用实际 task/row 计数报告，不能把稀疏分配当成必然少写。

## 必要的强对照与最小消融

1. **先补齐 E3 自身。** 测全部预声明 chunk 长度 256/512/1024 与 N32/N64，
   输出全部资格/失败记录；计 producer＋流式 sparse merge 的完整调用、冷热
   cache、workspace、规划、上传及初始化。空 region 不建无用 task；只剩一份
   partial 的 query 应有合理的直接输出/简化 merge 控制，或明确说明尚未实现。
   不能故意保留明显无用槽位让 E3 给 fused 方法让路。

2. **分开 task 调度与 partial 压紧。** 在同一 Q/K/V/mask、相同 M16/M32、
   FullCol policy、精度及 KV 加载管线下，至少保留“原 uniform S＋矩形行”
   与“相同 task 边界＋packed 活跃行/稀疏 merge”的配对；再比较 uniform S
   与有界 chunk。若要单独声称 chunk 或 workspace 的因果收益，补齐
   uniform/chunk × rectangular/packed 四格。避免将旧单 region 的 shared CU
   往返、GEMM policy、线程数和新局部初始化一起改变后，统称为压紧收益。

3. **保留最强适用执行对照。** E0 原生 ragged/paged 独立请求、全部 E1 固定
   query partitions、固定 union/E2、精确 hierarchy/E4、适用 PAT/PageWeaver
   机制、现有 v2/FullCol/wide/persistent/v4/lane 均不能因新 E3 出现而消失。
   作者原生实现能合法表达的输入优先测试；否则称“同引擎机制适配”，说明
   适配和缺失能力。E3 只胜旧 partial hook 或未优化自研 fused，不构成论文
   性能结论；E1 已穷举也不等于已经覆盖 E4/E5。

4. **用匹配工作量确认机制。** 每格报告真实 KV visits、Q reloads、MMA padded
   rows/tiles、非空 task 数、有效 partial rows、producer 写字节和 merge 读
   字节、每 query partial 数分布、最大 task 时长或可用负载代理。布局/资源
   用生成源、cubin REG/SMEM/LOCAL/STACK 和 SASS spill 审计。没有硬件
   计数器时将字节写为逻辑 proxy，不能称已测 HBM 流量。task 化与 v4 的
   比较同时改变 CTA 并行度和全局归并，不能当成纯 shared-U transport 消融。

5. **正确性资格先于筛选快格。** 验证所有 owner labels、cross/tree/full/
   disjoint/empty、非均长和极端 skew、单 token、chunk/tile 边界、真实完整
   suffix；每条原 query–token 边恰好一次，不重复、不遗漏、不跨 query
   混入 support。partial 未归一化 U/M/L 与已归一化 O/LSE 两种表示不能
   混用；自然 log/log2 必须一致；空输入返回 O=0、LSE=-inf，不能产生
   `-inf - -inf` NaN。保持原 FP64 容差，不为某个 arm 改接受门槛。

6. **图与生命周期单列。** fresh Q/KV/indices/offsets replay、active B 退出
   和 plan/task 改变分别资格化。固定图地址并不意味着 host plan 和 task
   列表免费，也不能将超出捕获拓扑容量的变更视为已测。可复用 plan 的真实
   次数须实测/标注；一步、自然 EOS 生命周期及首次使用成本分列，不能拿
   热 graph 时间冒充 JCT。统计按同实际轨迹配对，按独立 capture/request
   聚类 bootstrap，不能把每次 CUDA event 或六个 mutation 阶段当独立样本。

这些要求是区分收益来源所需的证据，不是要求先补做完整作者系统才能开展
首批有界资格实验。新控制可以先证明“原区域执行的调度/输出布局太弱”，
再决定是否值得扩大；本文件没有要求停止正在进行的原候选测量。

## 当前能写与不能写的主张

可以写：本仓库补充了一个保持原 mask 的 owner-region task 基线，采用
实际活跃行 packed partial 和逐 request 稀疏归并，用来检验跨区域片上
状态维护的必要性。其增量性能、资源及准备代价待独立实验决定。

不能写：首次 KV-outer/shared sparse decode、首次 packed query/多 tile/
分片 softmax merge、首次用 worklist 避免无效行、首次无 atomic partial、
首次避免跨区域 canonical state transport，或仅凭 local 源数组就声称
通信被消除。任务化本身是“不跨区域保留累加状态”的另一种常规分工，
它用 Q 重读和全局 partial/merge 交换了区域状态维护。

如果 E3 最终胜过 v4/lane，只能首先说明原状态维护方案不划算，且过去的
E3 控制不够强；若它胜过全部强 native/union/hierarchy，才有条件讨论
有价值的工程贡献。若要上升为新论文机制，还需超出上述组合的、可辨认的
调度/布局原则及跨输入证据。未检索到同字实现不提供“首次”的依据。
