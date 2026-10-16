# 下一步算子机制：先测剩余空间，再投入实现

2026-10-16。范围仅为单用户、多 agent 的固定支持集精确 decode；本次只读本地源码/结果、CPU 解析和一手资料，新增 GPU 用量为 **0 GPUh**，没有修改执行器。这里列出三个可证伪的工程假设，**没有一个已获得新颖性或性能资格**。它们的目标是同格最快的合格 FlashInfer/E1/E4，而不是旧 v2。简单重述 PAT/MSA 的 pack–forward–merge、worklist、稀疏归并，不构成新机制。

优先顺序：先测不可隐藏的 producer/merge 开销；再做同 kernel 的物理布局干预；只有它们存在真实净收益，才实现随 ready 集变化的长期调用。当前没有依据再启动无边界的寄存器、tile 或 owner 分组搜索。

## 1. 当前输入与强对照已经排除什么

[wave16 独立审计](../results/operator_lane_wave16_audit_v1/README.md)的同波热态目标如下，单位为完整 GPU 调用微秒；参数是已执行有限网格中的选择，**不是所有可能实现的全局最优**。原生完整调用包含必要的 Q packing、O restore/merge 和统一 LSE；不能以删除某一项的候选去比较它。

| 输入 | 被选强控制 | 完整热态目标 | 再降低 15% 的目标 |
| --- | --- | ---: | ---: |
| D0 / B3 / L15 / 4K | E1 own，全部请求固定 union，S64/N64 | 8.375 | 7.119 |
| D0 / B3 / L31 / 32K | E1 native，`{0,1}\|{2}`，native automatic split | 20.220 | 17.187 |
| D2 / B4 / L31 / 32K | E4 native，`{0,2}\|{1,3}` hierarchy | 21.650 | 18.403 |
| D2 / B4 / L31 / 128K | E4 native，`{{0,2},3}\|{1}` hierarchy | 98.752 | 83.939 |

来源分别是 `results/operator_overhead_wave16/overhead_w16_{docfinqa256,publicdev6_256}_l{15,31}_b{4096,32768,131072}.json` 的 `selected.hot.best_lane_control.config` 与 `summary.hot.median_us.best_lane_control`。15% 只是本项目的原推进门槛，不是已观察的收益。更强资格结果出现时必须更新目标，不能冻结一个较慢旧对照。

**成文时的新结果。** wave20扩大owner chunk网格后，D2/B4/L15/4K的C128/N64热态为9.6024us，同格最强控制10.0440us，`control/owner`配对中位1.04795、95%区间[1.04575,1.05148]；人工冷态则22.5280对20.4800us。其余七格owner更慢，固定C128/N64在新rank/layer/step上的确认仍在安排。这一处约4.6%的时间下降值得确认，但没有达到15%门槛，也不是应用结论。[该格原始结果](../results/operator_overhead_wave20/overhead_w20_publicdev6_256_l15_b4096.json)。该JSON的`planning.hot`记录owner warm prepare约13.551ms、被选E0约1.055ms；只按本格0.4416us热态差额抵消适配器准备差，需要约2.83万次同收益调用。这只是当前单层静态费用的条件算术，不能外推自然ready长度或全部层。它加强了先做producer/merge组件诊断和真实生命周期账的优先级，不授权立即实现另一个已知融合控制。

三个已知约束决定下一轮的设计：

1. **共享地址本身已经有效。** [native physical-alias 实验](../results/native_physical_alias_v1/README.md)只改变物理地址，不改变同一 FlashInfer plan/算法。4K–16K 热态 private/shared 比值在地址轮换后接近 1；32K 热态约 1.72–1.80，人工冷态也有共享收益。因此不能把独立请求的逻辑重复读取全部算成新增 GDDR7 流量，也不能拿 private-copy 作为唯一控制。这里没有可用的实测 L2/DRAM counter。
2. **换成预打包 FA4 已有负向诊断。** [once-packed 下界](../results/fa4/prepacked_attention_lower_bound.json)中 main30b 的 FA4 E0/union 热态 attention-only 分别约 114.48/172.01 微秒，已去掉每次 payload gather。它不是完整部署比较，也没有与 wave16 做配对；但它足以否定“只要不重复 gather，现有 FA4 就会快”的推断。下一轮布局实验必须保持算子族/调度不变。
3. **当前 owner-task 不是在线可变拓扑算子。** [wrapper](../src/branchtile/attention_owner_tasks.py)在构造时 D2H offsets/counts/row_map，CPU 建 task，上传并分配 packed U/M/L；[kernel](../src/branchtile/triton_kernel_owner_tasks.py)每 task 读原 KV 的间接 token 地址、写 FP32 partial，随后始终启动独立 merge。Q/K/V 和原 indices 值可原址更新；offsets/labels/counts/row_map、tensor 身份或布局改变需要 rebuild 与 graph recapture。`graph.replay()`绕过 host guard，不能据此宣称可变 ready 集已被验证。

波次16的 native/固定/树控制 warm prepare 在小预算格已约 0.85–6.84ms，明显大于约 8–10us 的一次热 GPU 调用；这些构造时间包含当前 Python 适配器，不是 FlashInfer 必然的硬件下界。不能把整个毫秒差额当作新算子可兑现的节省，也不能每 decode 步都重新构造一个本来可缓存的 native plan。

## 2. 候选 A：保留 task 并行，在最后一个贡献 CTA 内完成最终输出

**实际改动。** 保持已资格的 task 边界、M16/M32、KV 加载与 packed partial 顺序。每个 `(KV head, request)`另设独立 completion counter，task 写完其完整 GQA8 的 U/M/L 后，用正确的 device-scope 发布/获取协议报告完成。最后完成该 request 贡献的 CTA 按**固定 task 顺序**读取全部 partial，稳定归并并直接写最终 O/LSE，再复位 counter。没有忙等，未完成的 CTA 不占着 SM 等其他 CTA；一份贡献直接输出，不经过全局 partial。一个 task 可属于多个 request，分别报告完成，最后者可各不相同。空 request 的 O=0/LSE=-inf 必须有唯一写者。

它只尝试消除独立 merge 的 launch 与部分终端依赖开销，**没有消除全局 partial 流量**。若生产者原来只有很少几个 CTA，最后者又要独自归并大量行，长尾可能更差。改变后固定归并顺序可保持归约次序，但整个实现仍须重新做原 FP64 O/LSE 资格；counter 的存在不是数值正确性证明。

**直接先例。** NVIDIA 的 [threadFenceReduction 官方样例](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh)已经由最后完成的 block 归约并复位。更直接地，本次核查的 [FlashInfer PrimTS decode 源码](https://github.com/flashinfer-ai/flashinfer/blob/38e6b8944fe8b6497d793ef21fbaadd797ea39d8/flashinfer/attention/prims_ts/decode.py#L3551)明确采用 fused split-KV completion counter、最后 CTA 复位、仅准备时清零；[官方 counter API](https://docs.flashinfer.ai/generated/flashinfer.get_dcp_spec_counter_bytes.html)还公开 Cake DCP 的对应空间。因此本候选只能称 **SM120 任意 owner-task 的已知单次启动归约适配**，不能称新 reduction、首次无单独 merge，或新 softmax 算法。

**能赢的条件。** 实测 `独立 merge + 不可隐藏的启动/依赖` 明显大于 task counter、fence、较差尾部并行和额外寄存器费用，且完整调用能超过同格最快 E0/E1/E4。最可能的区域是低中预算、已有足够 producer CTA 且每 query 的贡献数适中；绝不是认为所有 128K 长输入都受 launch 限制。

**最低开销门槛。** 先测原 complete call、相同 producer-only、已填真实 partial 的 merge-only、相同节点数的轻量 graph 对照，以及 FlashInfer 合法路径的 PDL 开/关。组件相加未必等于完整调用，不能据此假定所有时间串行。`T_full - T_producer_only`可作为“只改终端归并”的乐观剩余空间诊断，不是所有 attention 的普遍下界。若最乐观的终端节省仍到不了该输入的强目标，停止这条纯终端改动。

FlashInfer 当前 `run(enable_pdl=None)`会按设备支持默认选择 PDL，本地 decode.py L2815–2816、prefill.py L3419–3420 已核查；不能假定 native 完全没重叠。PDL 允许依赖 kernel 的部分执行重叠，CUDA Graph 也可使用，实际重叠并无保证。[NVIDIA PDL 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

**最小八卡实验。** 四个上表输入 × 两张卡重复，卡内全部 arm 随机/反序配对并轮换卡号。先仅做组件诊断；有 headroom 才比较两 kernel、last-producer、合法 native PDL、native disable-split/direct-output。partial 数和 task 顺序冻结；每卡不同时运行其他工作。资格必须覆盖连续 graph replay、不同 Q/K/V、ready mask 切换、空行、异常中止后的 workspace 重建、不同 stream 各自独立 counter。跨 stream 共享同 counter 的并发调用不在合同内，不能用其结果。没有 deadlock 压测和 memory-ordering 资格，不筛选快点。

**预计失败。** 小预算的完整强基线已只有约 8us；一次 launch 的节省可能不足以追上 producer 差距。长输入 counter 数量、cache-line 争用和最后 CTA 的大 merge 尾部更可能抵消节省。它即使成功，也首先是一项工程控制，不能恢复原动态行的新颖性主张。

## 3. 候选 B：一次准备的只读 owner slabs，改变物理加载管线

**实际改动。** 在当前问题组的文档支持集冻结后，每层/头将所选文档 union 按 owner 桶复制为一份 N32/N64 对齐的只读 slabs，保存真实 token 位置、有效长度及 owner。Qwen 的输入已 post-RoPE，复制时保持 K/V 原值；padding 不构成支持边。producer 用 `slab_base + tile_offset` 直接计算地址，采用合法的连续向量加载/异步 staging；私有问题与生成 suffix 使用独立的 append-only 段，不能每步把整个文档或整个已生成后缀重新打包。原完整文档源仍保留，遵守当前提案合同。

**先用单变量干预。** 同一 kernel、同一 task 顺序、M/N、split、有效 token 与归并，比较：原 source + 间接索引；等容量 contiguous slab + identity index；contiguous slab + 去掉索引直接地址。先不同时换软件流水、task coarsening、精度或 reducer。若只换加载路径就没有强收益，停止布局实现。若有收益，再给同一 slab 数据和相同一次准备机会给 tuned FlashInfer E0/E1/E4，允许合法的大页与 ragged/paged 适配；不能把 native 锁在原始随机 page1 后称 candidate 胜出。原位置掩码无法被较大 page 精确表达时，保留 page1 或精确 custom-mask 控制，禁止扩充支持。

**能赢的条件。** 原 indexed load 的地址依赖、TLB/sector 行为或不能形成有效 staging pipeline，而连续源足以改善**完整**调用；与此同时 shared-source native 的 L2 复用仍不足以抵消候选减少访问/增加 tile 利用率的收益。这里需要同 kernel 的实测干预；logical visits、索引排序和理论 coalescing 本身不提供正证据。

**准备与容量门槛。** 对整个实际共同 ready epoch，若新增准备为 `C_pack + C_plan`、每 decode 步相对 strongest 同布局控制净节省为 `delta`，只有 `delta > 0` 且 `L > (C_pack + C_plan)/delta`才可摊销；自然 EOS 的剩余共同 ready 步数不能预知。应同时报告仅一步、首用、稳态和自然生命周期，不以 8192 cap 代替真实 L。所有层的 slab、原源、对齐、索引、workspace 和 graph 必须同时计显存。只把一个层的额外 buffer 计作整个模型代价不成立；每层复用一个 scratch 但每步重打包，也不属于本候选。

现有 alias 数据 4K/32K 的单层 shared payload 约 6.3/40MiB，仅作容量尺度参考；若 48 层都相近，额外驻留分别约 0.30/1.88GiB，**这不是已测全模型容量**。现有池初始化的约1.37GB板上空闲见[旧容量审查](pivot_candidates_20261016.md)，所以不能假定较大预算可以同时驻留全层第二份布局。合法 reserve 失败即保留失败，不能悄悄释放仍需的完整源。保留源时这是加速副本，绝不声称减少 KV 容量。

**先例与边界。** [JSSA v1，2026-10-13](https://arxiv.org/html/2610.19026v1#S3.SS3)已经直接从标准 GQA KV 索引所选项，排除“无 gather 原布局稀疏 GQA 首次”；本候选则明确支付一次物理复制。[Zipage v1，2026-03-01](https://arxiv.org/abs/2603.08743)已研究压缩页与 prefix caching 并存；本候选不释放源，不继承其容量结论。[WaveAlign v1，2026-09-28](https://arxiv.org/html/2609.34814v1#S4)已按访问相似性与 wave 调度改善缓存，排除笼统的首次 cache-aware sharing。owner 分桶、静态压实、按真实 L 摊销都是既有工程原则。目前没有可据此主张的新布局算法。

**最小八卡实验。** 仍用同四输入 × 两卡，完整配对比较原/contiguous-identity/direct-slab 三格及同布局最强 E0/E1/E4。先 pack 一次作为**乐观单层诊断**；只有 complete GPU call 已胜强控制，才计真实一次复制+所有层持久容量和 natural-ready 生命周期。冷热都保留，人工 cold 不当作48层真实模型；至少追加真实层序工作集重放。没有 counter 时只报告延迟与工作集干预，不能写“L2 hit提升”或“DRAM bytes下降”。

**预计失败。** 每个 D128 token 原本就是连续256B，token 间稀疏不等于 token 内未合并；native L2 已可重用共同地址。复制后仍有相同的 softmax/MMA 和 partial 工作，收益可能只有很小索引开销；一次准备和全层第二份 payload 则可能较大。更可能的正结果是 native 也因同布局变快，届时应把它计入强控制，结果仅支持存储工程。

## 4. 候选 C：固定文档 owner，late-bound ready rows 与 append-only 私有尾

**实际改动。** 固定文档 tokens 的原始 owner `o`及 task 身份，维护运行时 ready bitmap `R`，执行行由 `o & R`得到。B≤4只需最多15个非空子集的 row-map/count/有效 task 目录，均从已知当前支持预建；EOS/暂未调度只改小型 live descriptor，无 D2H owner 重算、重新分桶或 payload 搬迁。不同原 owner 桶投影成同一当前集合时，可以继续为两个不同 token 段，不必立刻合桶以追求最小 descriptor。

私有 suffix task 单独给出实际 base、device length 和 epoch/slot 身份；最大容量在构造时预留，完整新增 KV 沿合法 append 路径写入。最后一个 task 的 end 每步由 live length 计算，跨 chunk 边界也不重做文档计划。新 agent 带来新支持/新 owner 关系时是**新的 topology epoch**，明确付构造与必要 recapture，不能假定预知未来问题。相同地址的新 slot 不可误用旧 generation。

**能赢的条件与主张类型。** 当前反复 rebuild/replan/recapture 是真实在线成本，descriptor 更新明显较小，且固定容量/死 task 的热态罚时没有吃掉节省。它可能改善**有限生命周期的总调用成本**；本身不证明一次热 GPU attention 更快。必须与已有 native plan cache 比较：固定 Bmax 的空 inactive 行、合法 graph wrapper buffer 更新、以及最多15个实际子集的缓存计划/graph。native 可按开发集冻结每子集策略；不能强迫它每步搜索全部 partitions/trees。缓存元数据、workspace、graph及首用全部计费，避免缓存15份512MiB workspace却只报一份。

**直接先例。** [FlashInfer 原生 wrapper 文档](https://docs.flashinfer.ai/api/attention.html)已有 graph 用固定 buffers、固定 batch，`plan()`不可捕获；fixed split 在长度变化时也不保证图兼容。其新的 task-scheduled sparse 路径更明确区分图外容量计划与热路径 live route metadata。当前官方 signoff 为 SM100a/B200，SM103a/B300仅 architecture-gated，不能把这写成 SM120现成支持。相关[固定版本文档](https://github.com/flashinfer-ai/flashinfer/blob/38e6b8944fe8b6497d793ef21fbaadd797ea39d8/docs/api/attention.rst#L12)与[metadata源码](https://github.com/flashinfer-ai/flashinfer/blob/38e6b8944fe8b6497d793ef21fbaadd797ea39d8/flashinfer/attention/prims_ts/q_token_kv_block_sparse_metadata.py#L168)已核查。故 plan cache、graph稳定地址、live route 都不是新贡献；剩余只是当前 SM120 owner adapter 的生命周期实现。

**最小八卡实验。** 四条控制 trace × 两卡：固定 B、逐个退出、退出后slot复用、私有 suffix跨 chunk/page/预留容量边界。先用固定真实 Q/K/V构造这些**明确标注为控制事件**的支持等价资格，记录每步 plan/graph身份、descriptor传输、host wall与GPU call；不拿合成退出间隔当自然收益。随后使用真实已测 ready trace，重放同一工作轨迹，比较 owner rebuild、owner live descriptors、native cached subsets、native合法固定容量方案。首用与每epoch重建均进入总和；最终另测自然EOS和完整模型时延。

**预计失败。** 自然 ready 集长时间不变时，native也能充分摊销；只发生三次EOS的B4不会产生每步毫秒级开销。owner全容量网格中的死CTA、固定M32行、device分支或为最大私有尾保留的split可能慢于重新选择的小图。如果原生同样缓存后收益消失，就只是修好了当前适配器，不能称胜过现有attention算法。

## 5. 决策与最小可交付证据

这三条各自针对一个可测成本，不应同时改完后把总差额归为“共享数据流”。最小有用产出依次是：A的组件时序与终端空间；B的同kernel三格布局干预；C的实际epoch完整账。每条都保留当前强E0/E1/E4和适用PAT/QUILT/PageWeaver机制控制。调优、JIT、失败资格、图、warm prepare、payload副本与自然ready时间分别如实记录。

[PAT v3，2026-03-16](https://arxiv.org/html/2511.22333v3)、[MSA v2，2026-06-12](https://arxiv.org/html/2606.13392v2#S4.SS2)、[PageWeaver v1，2026-10-08](https://arxiv.org/html/2610.11201v1#S3.SS2)及[QUILT v1，同日](https://arxiv.org/html/2610.11134v1#S3)已覆盖共享query打包、KV-outer worklist、partial/merge、query-owned累积、精确集合层次和tile-tail处理；本次不会把其中某种重组再列为第四个“新机制”。PageWeaver的强native/在线准备反例尤其说明，现有控制越来越强时，剩余空间必须实测。

若A没有足够终端空间、B在乐观single-layer完整调用下仍不胜同布局native、C只胜无缓存Python构造，就停止相应加速主张。**目前建议的是有界反证实验，而不是已找到三项可发表新算子。** 只有相对同等准备/缓存/布局的最强控制仍有可重复净收益，才值得重新审查更窄的算子贡献；新颖性仍需独立成立。

## 一手源码观察点

官方FlashInfer main读取时为 `38e6b8944fe8b6497d793ef21fbaadd797ea39d8`，committer时间 `2026-10-16T10:08:39Z`；不把此时间当功能首发。只下载文本到内存检查，未安装/运行外部代码。

| 文件 | 固定官方版本SHA256 |
| --- | --- |
| `flashinfer/attention/prims_ts/decode.py` | `102a81f4d8b9b9353fabbed71725f0a541b85c045cedb6873b92676636f3fe58` |
| `flashinfer/attention/prims_ts/q_token_kv_block_sparse_metadata.py` | `61b0d418aa7f8d68841cea1c83281b4e33f97773aadf9491e715ce7e1c609a7c` |
| `docs/api/attention.rst` | `1a270068b20377eddee72c0f10aa195818009cd6aa0bd733429824f66f387a3f` |

安装环境 `.venv-latest` 的 decode PrimTS 文件另为 `7c4dc3e44cd5e920f3b927d006f4289ecfa0850076a1ac00f8b8d931b37c7ec0`；其completion说明在L3549，官方版本在L3551。不能把两个版本的行号或身份混写。安装环境普通decode.py SHA为 `de1f242bf2f598bc973a9d2f9625f8db1e4ccfc24085f686b2d0304c9076b20d`，PDL默认选择在L2815–2816。本文没有宣称这些SM100/SM103路径能合法服务本项目SM120的任意token-owner输入。
