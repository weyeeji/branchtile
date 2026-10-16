# 八卡算子探索：2026-10-16 22:03 快照

这轮确实改了算子并做了广泛实验；目前还没有超过强原生/固定划分/精确层次共享对照的候选，不能据此报告论文成功或应用加速。原来的停止结论不因局部优化而自动撤销。

## 执行与覆盖

七批性能实验共 **50 个正常结束作业、2,852 个配置，全部通过冻结 FP64 O/LSE 检查**。这些是三份文档的 **17 个不同真实输入设置**的重复方法比较，不是50个独立文档。覆盖B3/B4、TP rank0/1、层15/31、4K至128K及full预算、热缓存和人工冷缓存。掩码保留系统前缀、完整私有后缀及受保护文档尾部。

每一波用不重叠GPU并行多个真实输入，八卡没有用于同一个attention的跨卡切分。这里优化和测量的是单卡内部的数据流、状态搬运和完整算子开销，不能把结果说成跨卡通信加速。所有Python都在项目虚拟环境、uid1000，无root或驱动权限变更。

50项性能作业合计0.9130001002 P2 GPUh；资格、JIT、超时、解释器失败另外完整记P1账。汇总见 [JSON](../results/operator_8gpu_exploration_summary_v1/summary.json)，原始配置和逐项结果均保留。

## 改了什么，说明什么

| 实验 | 解决的问题 | 结果与边界 |
|---|---|---|
| 空分片guard | 空的region/split也走初始化和状态交换 | 有真实17.19%空分片的输入仍未净赢；不能只凭空分片比例预测收益 |
| FullCol-only | M16/M32 Tensor Core线程分工改变 | 数值与资源通过；局部调参快点存在，最终仍慢于强对照 |
| 固定M32持久化 | 整个循环保留输出和softmax状态，取消canonical状态交换 | 比FullCol局部快约4–22%，仍未赢强对照；MMA浪费和Q staging也改变，非纯搬运单变量消融 |
| 静态owner寄存器bank | 行集合变化时让同一请求的输出保持同lane，不再显式shared-U往返 | CPU路由和GPU数值通过；B3调分片后比原动态实现更快；B4 N64寄存器255并出现stack/LDL/STL；所有已测格仍慢于强对照 |
| 两形状runtime bank选择 | 用少量M16/M32代码替代7/15份GEMM展开 | 数值通过、无shared-U回退、无spill；选择和调度成本仍使完整call变慢，未赢强对照 |
| 256-thread FullCol | 用更多warp改善并行、降低每线程寄存器 | 合格目标中REG114–124、无spill，最终仍未赢；一次B3/S3解释器SIGSEGV独立保留，不冒充成功 |

消除某一条显式搬运路径确实实现了，但没有自动转化成整体attention获胜。降低寄存器数、无spill或更短JIT都不能单独代替完整调用的性能证据。

## 可直接核对的同波结果

下表均来自wave16，同一格方法使用相同Q/K/V、mask、实际位置；数字为热态完整producer+merge微秒。不同方法独立选择冻结网格中的配置，所有对照保留。

| 真实输入 | 强对照 | 原动态v2 | 静态bank | runtime lane bank |
|---|---:|---:|---:|---:|
| D0 B3, 4K, L15 | 8.38 | 15.63 | 14.00 | 18.68 |
| D0 B3, 32K, L31 | 20.22 | 27.42 | 24.83 | 30.37 |
| D2 B4, 128K, L31 | 98.75 | 118.15 | 142.73 | 129.52 |

[独立审计](../results/operator_lane_wave16_audit_v1/README.md)复算了568个配置、52个capture文件SHA、352项原始配对统计和80项作图区间。图只用这一波，未跨波拼最优值：[PNG](../results/operator_lane_wave16_audit_v1/paired_time_ratios.png) / [PDF](../results/operator_lane_wave16_audit_v1/paired_time_ratios.pdf)。静态bank的格数是每batch已资格的4/6网格点，不声称穷尽所有实现。

## 比较与研究范围

论文范围按用户要求采用单用户、多agent复用；不扩展多租户安全研究。数值正确性、私有后缀与共享源生命周期仍是执行语义要求。

输出长度问题使用两层比较：[固定真实工作重放](operator_comparison_protocol_20261016.md)比较同一条实际Q/KV/mask/位置轨迹的执行开销；自然EOS另报任务质量、总完成时延、GPU费用及真实长度。不截断到等长，不把更短答案带来的时间下降当作算子加速。这里的cold flush是人工冷缓存，不是实际48层工作集或应用JCT。

ISCAS方向契合需看可证明的机制和完整实验，不保证算子改动本身足够；原15%算子/5%应用是本研究推进门槛，不是会议硬性规则。[范围审查](iscas_operator_contribution_review_20261016.md)。

## 还在推进的工作

原矩形region_partials未覆盖最强E3：新owner-task控制将非空owner区域分成独立任务，只保存活跃query partial，再做稀疏流式merge，避免CTA内部owner切换。源与CPU数学已提交，专用GPU资格脚本在准备；尚无GPU或速度结论。它与PAT、PageWeaver、MSA已有直接先例，按强控制评估，不预先声称新范式。CPU拓扑准备、额外workspace和重建graph成本将计入。

[最新一手文献增量](operator_register_bank_literature_update_20261016.md)已核查2026-10-13 JSSA、2026 AMD实现与TyphoonMLA，也排除了已撤稿GraniKV的性能数字。NCU计数器无普通用户权限时不提权，不把逻辑字节/源码/SASS位置数量当作动态DRAM/L2流量。
