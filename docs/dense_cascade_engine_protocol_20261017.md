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
