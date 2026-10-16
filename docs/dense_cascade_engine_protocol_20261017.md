# 完整模型中共享前缀对照的测量协议

目的：将缓存O诊断发现的空间转换成真实执行验证。候选采用已有两段cascade机制，完整共享prefix只计算一次多query attention，每个query独立计算全部private tail，按两段log2 LSE合并。它是成熟算法基线，不构成论文创新声明。

实现为 `src/vllm_dense_cascade_plan.py` 与 `scripts/vllm_dense_cascade_worker.py`。缓存直接使用真实4D `[physical_page,2_heads,16_tokens,256_KV]` 存储的strided K/V alias；FP8 byte存储重解释，scale传入两段，不复制KV payload。每层当前token的KV写入仍由原engine完成。token公共前缀向下取整到16tokens，并在每个实际步检查所有活跃请求的物理共享页地址相同。

每个B有独立prefix/tail plan，48层共享其元数据；预先构建B2至B8图，B1、prefill与混合调度沿用native。private tail包含公共区间以外的所有prompt、问题与新生成tokens，允许长度不等和退出/重排。每步prefix地址检查、tail plan、host同步和已有native plan全部计入generate walltime；图构建和plan初始化另外记录，首次使用必须加回。

资格模式使用独立图保存每层实际Q/O。用同一个Q和当前完整KV运行原生XQA O-only参考，逐元素检查 `abs(candidate-native) <= 0.006 + 0.015*abs(native)`。这是一项事先冻结的工程门槛，不能代替数学参考精度或自然任务质量。资格图的保存与参考重算不进入性能图；任何资格失败立即停止，不能放宽门槛后重命名为通过。独立失败捕获runner在原断言前保存完整Q、两种O、物理页和重映射table供FP64裁决。

性能阶段按native A0、cascade0、cascade1、native A1执行各128tokens的真实整模型生成。每阶段清APC、先生成一个cold anchor、再同时就绪全部请求。记录anchor、完整group wall、两者之和、所有token、缓存命中数和实际逐步batch/positions/question轨迹。两次重复仅够初步筛查；CPU分析不为其捏造总体置信区间。自然EOS阶段另跑native/cascade，保留全部输出和退出原因；输出改变可能改变总工作量，因此单次任务耗时比不能冒充同轨迹算子加速，需任务答案评分及更多独立场景验证。

## 首轮资格状态

train1 FP8（120528-token公共prefix）完成15个B8 decode步、每步48层、两rank，共1440项逐层Q/O资格检查，全部通过。两rank调度轨迹一致，private tails为33至63tokens，未发生plan topology变化。额外prepare host中位每步0.4105/0.4009ms；候选图与plan构建2.408秒。资格generate耗时含检查，不是候选速度证据。原始与审计见 `results/dense_cascade_model_qual_v1/train1_fp8/`。

train1 BF16在rank1、layer47的原定same-Q比较失败，记录max_abs 0.03125。绝对最大误差并不直接表示失败坐标的误差；门槛含相对项，需保存失败元素再判断。worker错误后原engine未干净返回，root终止进程，账本exit -15；summary残留initializing，不能当成仍运行或已完成。原始日志 `logs/dense_cascade_model_qual_train1_bf16_v1.log` 保留。后续独立失败快照将检验candidate/native各自对FP64的误差，而非用二者互相不一致直接判定谁错。

CPU审计工具 `scripts/audit_dense_cascade_model.py` 核对源哈希、48层覆盖、实际共享页资格、完整tail与图topology、两rank轨迹、固定token数、anchor账目、额外准备成本及ABBA/APC一致性。自然结果同时输出每种B实际存活步数和caps，避免把初始B8收益外推全部请求。

## FP8 两文档完整生成与自然结束结果

后续两个TP2作业均完成。固定128输出tokens、native/cascade各两次的完整生成，train1 ready-group比值1.17036，train7为1.16521；包含cold anchor后仅1.01401和1.01158。native AA耗时比max/min分别1.00076和1.00104。同方法两次输出tokens完全相同，两方法间tokens不同；四个phase的APC命中数、实际调度batch、question和position轨迹一致。因模型含MoE、token变化可改变路由，这只能称**固定输出长度的完整生成**，不称严格相同工作量或相同模型轨迹。

| 文档 | 自然native/cascade ready wall | 自然native/cascade cold文档wall | 总输出tokens native→cascade | ready比值 |
| --- | --- | --- | --- | ---: |
| train1 | 33.702 / 31.986 s | 54.805 / 53.074 s | 12999→10750 | 1.05364 |
| train7 | 20.841 / 59.570 s | 49.764 / 88.476 s | 7317→13915 | 0.34986 |

四次自然组共32个请求均正常EOS，无8192tokens截断。train1候选虽然总tokens减少，最长请求从3022增到3473，单请求尾部B1占2063步；train7最长请求从1854增到5643，候选B2占2803步、B1占1840步。B1按预定策略回退native；自然输出长度和batch存活分布变动可以抵消算子收益，甚至使完整任务明显更慢。不得隐藏train7负结果，也不得把train1单次正结果视为稳健自然任务加速。自然答案仍需独立打分。

额外图/plan初始化分别2.469和2.389秒，以上cold文档wall未含这项；首次使用须另加。生产阶段prepare记录elapsed约9–10ms包含 `.item()` 等待前一GPU步的时间，不能直接把它称为9–10ms新增CPU计算开销或从wall中再次扣除；资格阶段同步后的约0.4ms也不构成严格因果overhead分解。下一步应在同一真实decode快照内直接比较native与真实cascade完整模型图，分离prefill和逐步CPU准备，保留实际算法计算而非预知O。

CPU审计结果：`results/dense_cascade_natural_v1/train1_fp8/analysis.json`、`train7_fp8/analysis.json`。逐rank记录覆盖B8自然收缩至B1、全部完整tail/实际共享页和图topology检查，未发生遗漏private suffix或越界plan变化。

## BF16：独立FP64裁决与完整测量

原BF16失败快照的CPU FP64审计表明，唯一pairwise失败元素两侧误差方向相反：candidate=-0.1689453125、native=-0.16015625、FP64=-0.166339348589，candidate/native各自对FP64仍满足原0.006/0.015门槛。新独立reference版本保留原pairwise结果，仅在pairwise失败层对两臂计算完整FP64 reference并使用相同不变门槛；pairwise已通过层不冒称独立FP64认证。数值失败只记录，各rank完成资格phase后driver共同判定，避免单rank先raise造成TP挂起。

train1资格1440次逐层检查中有1项 `pair_failed_but_reference_passed`，train7没有。两作业完成，SHA/两rank轨迹/真实共享页/完整private tail/topology审计通过。固定128输出tokens完整生成ready比为train1的1.20771和train7的1.24765；含cold anchor仅1.02075和1.01860。native AA波动max/min为1.00973/1.01339，同方法两次token相同、跨方法token不同，APC和调度positions一致，依然不能称严格相同内部工作量。

| 文档 | BF16自然native/cascade ready wall | BF16自然native/cascade cold wall | ready比值 |
| --- | --- | --- | ---: |
| train1 | 33.698 / 24.875 s | 53.523 / 44.685 s | 1.35471 |
| train7 | 44.782 / 46.605 s | 73.021 / 74.844 s | 0.96087 |

BF16共32个自然输出均正常stop，无cap。train1输出长度明显缩短；train7最长输出从3726增到4148。结合FP8，四个单seed条件中两个自然ready wall正向、两个负向；这支持开展预先冻结多seed评估，不能从单个收益好的条件挑选结论。已另行冻结三个新增seed20261017/18/19，方法顺序BA/AB/BA，与原20261016的AB构成两AB两BA；保留全部负结果和质量差异。

原始与分析目录为 `results/dense_cascade_reference_natural_v1/train1_bf16/` 和 `train7_bf16/`；高精度失败裁决见 `results/dense_cascade_failure_fp64_audit_v1/audit.json`。原失败未删除、未重新标成pairwise通过。
