**BranchTile 可行性实验记录（2026-10-16，本轮可行性结论）**

当前结论是：**动态活跃行映射及跨映射 FP32 状态保持在数值上可实现，但本轮 SM120 原型没有达到约 15% 的算子集成推进门槛；主模型完整应用收益仍未验证。** 三轮布局实现和独立调参后，候选仍输给同样跨区域累积的固定行实现。加入可跨 owner 边界切 tile 的固定 union 后，部分自然快照的差距更大。因此停止当前动态行方案的扩展，不进入完整应用集成。[原始方案及推进标准](proposal-iscas.md)、[审稿式审计](results/kernel_audit.json)

这个结论限定于本轮实现、RTX 5090/SM120 和已测输入。它不证明所有动态压紧算法都不可行，也不等于完成了方案中的 P1 应用资格或全部 P2 算子资格。P1 的 A0–A4、正式主模型、质量和完整费用比较尚未完成；本轮触发的是方案第六节“先出现稳定算子信号，再投入完整集成”的停止判断。

实验使用项目 `.venv`、隔离的 `.venv-latest`/`.venv-engine` 和普通用户权限，GPU 按作业分配并行；编译、部署失败和重试保留记录。资源预算以分配卡数乘墙钟计，不把八卡并行当作单个算子的加速倍数。[作业台账](results/job_ledger.jsonl)、[v2 消融作业配置](configs/branchtile_v2_2x2.json)、[v3 有界作业配置](configs/branchtile_v3_qualification.json)

**实现与正确性。** 候选读取同一物理 Q/K/V 和逐请求支持集，只整理索引，不按请求复制 KV payload。一个 CTA 内根据运行时活跃行数实际执行 M16 或 M32，并为原 query 保留 FP32 softmax 状态 `(m,l,u)`。固定行对照同样跨区域融合，Q 常驻 shared memory，`m/l/u` 常驻寄存器；它没有被人为加上动态行方案才需要的状态搬移。[v1 与固定行源码](src/branchtile/tilelang_kernel.py)、[v2 源码](src/branchtile/tilelang_kernel_v2.py)、[v3 源码](src/branchtile/tilelang_kernel_v3.py)

| 版本 | 实际改动 | 结果与依据 |
|---|---|---|
| v1 | 同 CTA 动态 M16/M32；canonical FP32 状态放 shared memory | 初筛 480 条配置记录，420 条成功、60 条因 N64 shared-memory 容量超限失败；失败记录不作性能样本。[原始八卡结果](results/kernel_screen/)、[初筛汇总](results/kernel_screen_summary.json) |
| v2 | M16/M32 分支显式复用 Q/K/V/P shared buffers，并拆开 Q gather 与状态 gather | 消除 v1 分支缓冲叠加的容量问题；独立调 split、N tile 和合法线程数后，稳健配对结果仍未出现目标收益。[配对原始结果](results/paired/)、[编译资源](results/kernel_resources.json) |
| v3 | canonical `u` 改为 FP32 register fragment，在区域边界借用 K shared scratch 按位保存和恢复；`m/l` 仍为 FP32 | 降低 SMEM，但提高寄存器数；预定的 cross16K/128K 两个形状均未产生稳定 15% 信号。保守迭代占用约 0.153 GPU 小时，低于 0.5 GPU 小时上限，随后停止。[资格记录](results/kernel_ablation_v3/qualification.json) |

FP64 reference 直接根据原始 owner 位图计算 QK、softmax 和 PV，避免 TF32 干扰；候选与强固定行对照均检查输出和自然对数 LSE。已有日志包括 v1 的 17 项 FP64 测试、v2 的 17 项 memcheck 测试、v3 的 5 项 memcheck 测试，以及复用同一编译 kernel、改变 Q/运行时映射并切换 M16→M32 的测试。v2/v3 相应 memcheck 与 runtime racecheck 均未报告错误或 hazard。[FP64 日志](results/validation/fp64_tests.txt)、[v2 memcheck](results/validation/v2_memcheck.txt)、[v2 racecheck](results/validation/v2_racecheck.txt)、[v3 memcheck](results/validation/branchtile_v3_memcheck.txt)、[v3 racecheck 及执行日志](results/validation/branchtile_v3_qualify_bench.txt)

v3 scratch 中的两个 BF16 类型存储单元承载一个 FP32 值的原始 32 位，**没有把 canonical `u` 数值量化到 BF16**。早期输出布局排列错误已在进入保留的性能实验前修复。测试覆盖全共享、不共享、交叉、树状、混合、空集合、空行、非整 tile、split 和部分小 B/GQA4 边界；它们不等于穷尽全部 alive 子集或完成层输出/logit、自由生成质量验证。[测试源码](tests/test_attention.py)、[v2 测试](tests/test_attention_v2.py)、[v3 测试](tests/test_attention_v3.py)

**计时口径。** 下文 μs 均为 steady-state attention 执行时间。候选和固定行实现包含 producer、split partial 写回和 merge；自然 replay 包含系统前缀、被选文档 token 和完整私有 suffix。CPU metadata 构造、选择、编译、分配和模型准备不在区间内，因此不能把这些数值称为包含准备摊销的完整应用成本。[计时与 capture loader](scripts/compare_attention.py)

主要配对实验在同一物理 GPU 上分别调优各方法的 hot/cold 配置，再执行 30 个随机交错 block 及其反序，并测固定行 A/A。Hot 为每图 20 次调用；cold 为每次调用前清理超过两倍 L2 的缓冲，清理不计时。区间按配对 block 重采样。冷缓存的单次计时存在约 2.048 μs 量化，零宽区间不代表总体零不确定性；人工清缓存也不等于真实 capture 轮转或模型执行前导。[配对原始结果](results/paired/)、[审计及逐文件复算](results/kernel_audit.json)

**历史 native 列有接口限制：** 本报告引用的 FlashInfer 原始计时返回 log2 LSE，而候选/固定行返回自然对数 LSE；乘 `ln(2)` 只在正确性检查中执行，未进入这些历史 native 时间。若要求统一输出自然对数 LSE，该列只能视为原生接口时间，不能直接宣称已经计入格式转换。本轮停止结论由候选与固定行/union 的同一 LSE 接口比较独立支持。[FlashInfer wrapper](src/flashinfer_baseline.py)、[历史计时审计](results/kernel_audit.json)

**强固定 union 的早期五份结果。** 初版固定行控制仍逐 owner 区域补齐 N tile。更强控制把已整理的 union 索引连续切块，允许 tile 跨 owner 边界，每个 token 用 owner 位图精确屏蔽行，从而减少短区域的尾块和 split 碎片。它是已有 masked-union 执行机制的实现，**不是本项目的新颖贡献**。[固定 union 源码](src/branchtile/union.py)、[数值与内存验证](results/validation/fixed_union_memcheck.txt)

以下各行均为各自调参后的中位时间；“固定区域”指原先保留 owner 边界的强固定行控制，“固定 union”指允许跨边界切块的控制。合成实验为 B4/GQA8、D128、两个 KV heads。两份自然实验来自旧版 Qwen3-30B-A3B BF16，仅作诊断，不能替代官方 2507 FP8 主模型。

| 输入与原始结果 | 缓存 | 动态 v2 μs | 固定区域 μs | 固定 union μs | Native 历史口径 μs |
|---|---|---:|---:|---:|---:|
| [合成交叉 16K](results/union_paired/union_v2_cross_n16384_g8.json) | hot | 16.6664 | 14.9128 | 11.1512 | 10.5448 |
| 同上 | cold | 30.6880 | 26.6240 | 24.5760 | 24.5760 |
| [合成交叉 128K](results/union_paired/union_v2_cross_n131072_g8.json) | hot | 94.5328 | 92.3112 | 91.7056 | 83.0664 |
| 同上 | cold | 106.4960 | 100.3520 | 104.4480 | 106.4960 |
| [合成 mixed 64K](results/union_paired/union_v2_mixed_n65536_g8.json) | hot | 52.1592 | 38.9424 | 27.9008 | 27.5256 |
| 同上 | cold | 77.7920 | 63.4560 | 59.3600 | 57.3440 |
| [旧 30B，QMSum3，L31，预算4096，step256](results/union_paired/union_old30b_qmsum3_l31_b4096_s256.json) | hot | 27.4752 | 21.9632 | 7.9496 | 8.3160 |
| 同上 | cold | 40.9280 | 34.4000 | 16.3840 | 18.4320 |
| [旧 30B，QMSum7，L47，预算4096，step256](results/union_paired/union_old30b_qmsum7_l47_b4096_s256.json) | hot | 27.2552 | 21.4240 | 7.8176 | 8.1104 |
| 同上 | cold | 38.9120 | 32.7680 | 16.3840 | 18.4320 |

相对动态 v2，五个 hot 结果中固定 union/候选的配对时间比分别为 0.6691、0.9732、0.5349、0.2895、0.2871；比值小于 1 表示候选较慢。两份自然快照的该比值 95% 配对区间分别为 `[0.2889, 0.2920]`、`[0.2868, 0.2875]`。128K 的接近结果不应泛化为大优势，但也没有达到候选约 15% 改善的目标。更强 union 并非处处最快，例如 128K cold 的固定区域控制更快，必须保留各自调优后的最佳适用对照。原始文件中的额外 `fixed_AA` 方法标签/比值不作为一个方法或独立样本使用。

旧模型另有 QMSum0 的四层 step1 replay：动态 v2 hot 约 26.22–27.41 μs，固定区域约 21.22–21.61 μs；cold 同样候选较慢。四层属于同一个文档的快照，不是四份独立任务。[L0](results/natural_paired/natural_old30b_qmsum0_l0_b4096_s1.json)、[L15](results/natural_paired/natural_old30b_qmsum0_l15_b4096_s1.json)、[L31](results/natural_paired/natural_old30b_qmsum0_l31_b4096_s1.json)、[L47](results/natural_paired/natural_old30b_qmsum0_l47_b4096_s1.json)

这些自然输入的误差不能统一概括为“小于 1e-3”：L47 候选/固定区域输出相对 FP64 的最大绝对误差约 0.0037284，native 约 0.0090291，在现有绝对加相对容差下通过；候选/固定区域 LSE 误差不超过约 3.68e-6。测试通过支持本轮算子比较，不证明相对 Dense 的任务质量无损。[L47 原始数值记录](results/natural_paired/natural_old30b_qmsum0_l47_b4096_s1.json)

**保留局部正向结果及其边界。** v2 的 2×2 消融共保留 48 条原始记录。下表采用各方法各自调优的 split；partial 版本给每个区域独立 CTA，并写回完整 canonical 行槽，包含 inactive 零行，因此它不是优化的 packed-active E3。[fused 原始记录](results/kernel_ablation_v2/fused.jsonl)、[partial 原始记录](results/kernel_ablation_v2/partial.jsonl)、[汇总及限制](results/kernel_ablation_v2/summary.json)

| 合成交叉输入 | 动态融合 μs | 固定融合 μs | 动态区域 partial μs | 固定区域 partial μs |
|---|---:|---:|---:|---:|
| 4K | 12.8173 | 11.2045 | 7.4253 | 7.1850 |
| 64K | 30.2778 | 32.4416 | 30.0054 | 25.2410 |

64K 的动态融合比固定融合有约 **1.071×** 局部加速，即时间降低约 **6.7%**。这个结果保留，不改写为“所有点都慢”。但该消融是固定顺序的 hot-cache 诊断，尚无同条件的随机配对确认；收益小于约 15% 的推进目标，同表固定 partial 仍更快，也未覆盖强固定 union。因此它不支持投入完整集成。

v3 的最后两点也完整保留。16K hot 候选 19.7568 μs、固定区域 14.6016 μs，固定/候选配对比为 0.7383；128K hot 候选 92.9480 μs、固定 92.2704 μs，比为 0.9929。Cold 对应为 34.7840/26.6240 μs 与 104.4480/100.3520 μs。两点均无稳定 15% 信号，未继续扩大搜索。[16K 原始记录](results/kernel_ablation_v3/paired_cross_n16384_g8.json)、[128K 原始记录](results/kernel_ablation_v3/paired_cross_n131072_g8.json)

**资源与机会解释。** 下表是实际 cubin/launch 资源，不是手算估计。共同形状为 B4/GQA8、D128、128 threads；v1/v2/固定为两个 heads、4096 tokens、6 regions、split16，v3 为两个 heads、16384 tokens、6 regions、split64。SMEM 总量包含 1024B static memory；这些 cubin 的 stack/local allocation 均为零。[完整资源及源码/cubin 哈希](results/kernel_resources.json)、[v3 cubin](results/kernel_ablation_v3/compiler/register_u_n64.cubin)、[v3 资源工具输出](results/kernel_ablation_v3/compiler/register_u_n64.resources.txt)

| 实现 | N tile | registers/thread | 总 SMEM/CTA（B） | 说明 |
|---|---:|---:|---:|---|
| v1 | 32 | 164 | 67,840 | 分支缓冲未充分复用 |
| v1 | 64 | 190 | 103,680 | launch 超出允许容量，非有效性能样本 |
| v2 | 32 | 164 | 46,336 | 显式复用缓冲 |
| v2 | 64 | 190 | 64,768 | 允许执行，但状态搬移仍存在 |
| 固定区域 E2 | 32 | 124 | 28,672 | 固定行在线累积 |
| 固定区域 E2 | 64 | 148 | 47,104 | 固定行在线累积 |
| v3 | 64 | 240 | 48,384 | 降低 SMEM，增加 register 压力 |

可以确认的是：减少理论 M tile 没有自动变成更短的执行时间，v3 也没有通过降低 SMEM 恢复目标收益。代码和资源记录支持“映射、片上状态组织、区域尾块与并行划分存在代价”的解释，但尚不能把延迟差归因于某个具体 bank conflict、stall、DRAM 流量或 achieved occupancy。普通用户运行 Nsight Compute 时返回 `ERR_NVGPUCTRPERM`，本轮未使用 root 解除限制，也没有以理论 occupancy 或逻辑读取量冒充硬件计数器。[权限记录](results/validation/ncu_permissions.txt)

自然支持的逻辑机会也偏小。旧 30B 的 8 组诊断输入在预算4096、N32 下，按文档 token 统计的动态/最佳固定划分 tile 比为约 0.9889–1.0218；即使去掉区域尾块取整，放松模型的比值约为 0.9586–0.9871。这只是枚举固定划分后的逻辑 M/N tile 估计，未包含 prefix/suffix、真实缓存、执行资源或调度，不能直接当作可实现加速上界。它表明大范围交叉关系的“存在”不足以提供大的剩余计算机会。[8 组原始机会统计及计算规则](results/data_natural_tail_opportunity.json)

**官方 4B 边界实验。** 已核验官方 Qwen3-4B-Thinking-2507 权重身份；8 组 QMSum exploratory capture 已完成，模型为 GQA4、8 KV heads。自然生成允许 EOS，但这批每组四题均到达 256-token 仪器上限，不能计作完成回答、自然 EOS 或质量合格。[权重校验](results/data_4b_download_identity.json)、[8 组 capture 索引与机会统计](results/data_official4b_opportunity.json)、[QMSum0 示例元数据](results/captures/official4b_boundary/qmsum0/metadata.json)

GQA4/B4 总共只有 M16 行，动态压紧不能减少最小 M tile；已有文档部分的逻辑分析中，预算4096、N32 的动态/最佳固定 tile 比约为 1.0425–1.0530，去除区域尾块取整后为 1.0。这是边界机会分析，不是 GPU 性能结果。[官方 4B 机会原始记录](results/data_official4b_opportunity.json)

官方 4B 的 QMSum0 四个预定层已完成配对重放，使用全部 8 KV heads、B4/GQA4、预算4096、自然到达 step1；Q 形状为 `[8,16,128]`。本组原生 FlashInfer 0.6.12 **已将 log2→自然对数 LSE 转换计入计时**，与前述历史 native 列不同。每层独立调参，30个随机/反序配对 block，固定 A/A；四层来自同一文档，不能当作四份独立任务。

| 层与原始结果 | 缓存 | 动态 v2 μs | 固定区域 μs | 固定 union μs | Native（含转换）μs |
|---|---|---:|---:|---:|---:|
| [L0](results/official4b_paired/official4b_qmsum0_l0_b4096_s1_allheads.json) | hot | 27.2664 | 22.5896 | 10.3552 | 17.3152 |
| 同上 | cold | 40.9600 | 36.8640 | 24.5760 | 28.6720 |
| [L11](results/official4b_paired/official4b_qmsum0_l11_b4096_s1_allheads.json) | hot | 29.2560 | 24.8528 | 10.3120 | 17.2736 |
| 同上 | cold | 47.0720 | 38.8800 | 26.5920 | 28.6720 |
| [L23](results/official4b_paired/official4b_qmsum0_l23_b4096_s1_allheads.json) | hot | 29.5480 | 24.8920 | 10.4696 | 17.2352 |
| 同上 | cold | 47.1040 | 40.9440 | 26.5920 | 28.6720 |
| [L35](results/official4b_paired/official4b_qmsum0_l35_b4096_s1_allheads.json) | hot | 28.6488 | 23.9328 | 10.3344 | 17.3760 |
| 同上 | cold | 42.9760 | 38.8800 | 24.5760 | 28.6720 |

动态版本仍慢于固定区域和固定 union，未通过推进门槛。固定 union 热缓存约10.3–10.5 μs，明显快于本组 native 约17.2–17.4 μs；这是既有机制在本形状上的正向工程结果，不能证明动态压紧有效或建立新颖性。最新 FlashInfer 0.7.2 的独立环境复查和更强的原生联合 prefill 对照已经完成，见下节；联合原生对照消除了这个幅度的热缓存优势。

数值检查沿用预先使用的 `atol=0.006, rtol=0.015`，没有为4B放宽。L35动态/固定/union对FP64最大绝对误差约0.03055，native约0.05174；相对容差下通过。这些大幅值BF16输出的绝对误差不能隐去，也不能由逐算子通过推导整层/logit或任务质量非劣。另有[原生vLLM 0.31.0单卡部署记录](results/platform/engine_official4b_tp1.json)：官方权重可加载、四请求及APC可运行；首轮含额外JIT且两轮缓存状态不同，不作为A/A或加速比较。

**截至 2026-10-16 的最新基线复查。** 在独立 `.venv-latest` 安装前一天发布的 [FlashInfer 0.7.2](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2)，保持 Torch 2.11.0、Triton 3.6.0、TileLang 0.1.9 不变；实际 Tensor Core decode 后端为 FA2。原生 probe 通过数值与 CUDA Graph 检查，动态 v2/固定 union 的 23 项测试在此环境通过。[环境身份](results/platform/latest_flashinfer_environment.json)、[probe](results/platform/latest_flashinfer_probe.json)、[测试日志](results/validation/latest_tilelang_tests.txt)

八张卡并行复测四个官方 4B 快照、一个旧 30B 诊断和三个合成输入。动态方案仍未通过门槛；自己的固定 union 相对逐请求原生 decode，在官方 4B 的四个热缓存点有 1.33–1.62× 收益，但冷缓存有两个点几乎持平。为排除“只是原生调用方式没有表达联合查询”的影响，又实现并测量 `BatchPrefillWithPagedKVCacheWrapper`：将 KV head 作为 batch、请求作为 query 位置，保留 GQA 结构，以精确 custom mask 联合执行。它直接读取同一物理 KV payload，无按请求复制。[原生联合 wrapper](src/flashinfer_union.py)、[backend 元数据勘误](results/platform/native_union_metadata_erratum.json)、[八卡配置](configs/latest_eight_gpu_paired.json)、[联合对照配置](configs/native_union_paired.json)

最后一轮每个输入在同卡重新独立调参、交错测量全部七个方法。下表为 hot 中位数，单位 μs；raw 列返回原生 log2 LSE，ln 列把转换计入执行。自己的 union 和动态版本返回 ln LSE。

| 官方 4B 快照 | 动态 v2 | 自己的固定 union | 原生逐请求 raw | 原生联合 raw | 原生联合 ln |
|---|---:|---:|---:|---:|---:|
| [QMSum0 L0 step1](results/native_union_paired/native_union4b_qmsum0_l0_b4096_s1.json) | 26.9584 | 10.1456 | 16.3152 | 9.5496 | 10.4888 |
| [QMSum0 L35 step1](results/native_union_paired/native_union4b_qmsum0_l35_b4096_s1.json) | 28.8008 | 10.2504 | 16.5496 | 10.1720 | 11.2944 |
| [QMSum3 L23 step256](results/native_union_paired/native_union4b_qmsum3_l23_b4096_s256.json) | 31.9696 | 13.0680 | 17.3376 | 12.4320 | 13.4496 |
| [QMSum7 L35 step256](results/native_union_paired/native_union4b_qmsum7_l35_b4096_s256.json) | 30.8160 | 12.1864 | 17.0176 | 11.7984 | 12.7944 |

原生联合 raw 在四点均更快。统一为 ln LSE 后，原生联合/自己的 union 的配对 block 中位时间比约为 1.037、1.102、1.028、1.049，仍低于 1.15。Cold 自己的 union 约 24.576/26.592/34.816/30.720 μs，原生联合 ln 约 26.624/26.624/34.816/32.768 μs；单次事件量化明显，也没有稳定 15% 信号。所有这四点 hot/cold 的 ln 比值 95% 区间上界均低于 1.104。逐 block 比值和置信区间单独保留，不能把上表中位数相除当作配对统计。[完整配对汇总与图](results/summary/union_signal_latest.json)

**最新原生实现并非所有数值配置都合格。** 沿用输出 `atol=0.006, rtol=0.015`、ln-LSE `atol=0.0003, rtol=0.0001` 的原门槛，逐配置通过后才计时；部分 native split 因数值失败而剔除，原记录保留。新增的七项原生联合小形状测试为 **4 通过、3 失败**，失败仅在 LSE。三例直接调用 single-prefill FA2 也违反相同阈值，其中两例输出与联合 wrapper 逐值完全一致；源码中 BF16 概率舍入后累积分母可解释这一限制，未发现 mask/reshape 语义错误。未放宽阈值、改输入或把失败改为 xfail；含全空 KV head 的 union 明确不支持。这限制了它作为通用可替换基线的资格，但不改变四个自然点在原门槛下的逐配置合格结果。[原始失败日志](results/validation/native_union_qualification.txt)、[诊断及资格限制](results/platform/native_union_qualification_status.json)、[直接 FA2 诊断](results/platform/native_union_lse_diagnostic.json)

因此，固定 union 值得作为后续研究的必要强基线，却不能作为本轮新贡献。最新文献中的 query regrouping、union membership mask、完整输出 ownership 和层次共享也进一步提高新颖性要求。[2026 文献核查](results/literature_review_20261016.md)

**本轮完成范围与资源。** 汇总保留 31 组配对配置、62 个 hot/cold 结果；这些包含同一文档的不同层与重复基线版本，不是 31 个独立自然任务。所有记录中动态/固定比较均未达 1.15 推进比值。[可重建表格](results/summary/paired_comparisons.csv)、[图](results/summary/dynamic_vs_fixed.png)

两模型各 8 个探索文档、4 题、4 层、4 个 decode step 共保存 1024 个主要张量快照；它们来自 dense 轨迹，不能用于宣称 sparse 质量。官方 4B 的原生 vLLM 单卡部署已实际执行，正式 30B FP8 主应用在早停后不继续投入。台账核对 69 个已结束作业，无未配对记录或活跃实验；分配时间含编译、初始化、失败和重试，P1 约 0.937 GPUh，P2 约 3.093 GPUh。另显式保留初期手动 probe 的不完整计时和管理预留，不能把台账当作数学上完备的总耗时上界。计入预留后 P1/P2 分别约 1.937/4.143 GPUh，低于方案 24/8 GPUh 的预算。全部实验已结束并释放 GPU，未用 root；初筛和最新复测均采用八卡作业分配。台账存在同时分配全部八卡的时段；一秒 NVML 采样观察到的最高同时非零利用率为七卡，没有观测到八卡同时非零，因此不声称八卡持续满载。[预算审计](results/budget_audit.json)、[并行与采样证据](results/platform/eight_gpu_parallelism.json)

**未完成范围。** 官方 Qwen3-30B-A3B-Thinking-2507-FP8 的下载在动态算子未过门槛后停止，保留可续传文件；部署资格和 TP2 主应用未完成，旧 BF16 30B 结果始终保留诊断标签；官方 4B 不能替代 GQA8 主模型。[主模型下载记录](results/Qwen3-30B-A3B-Thinking-2507-FP8_download.json)、[旧模型身份](results/data_local_old_30b_identity.json)

DocFinQA 原始 dev 与转换后的 validation 已逐行核验，且保留冻结文档 ID；这解决的是输入身份问题。历史正式划分仍未恢复，当前 exploratory manifest 不能冒充原 8 开发报告、100 公司年度主评价或 QMSum 官方 test35。上下文长度检查也不等于容量与运行资格。[DocFinQA 校验](results/data_docfinqa_parquet_validation.json)、[当前 exploratory manifest](data/manifests/docfinqa_exploratory_dev8.json)、[上下文审计](results/data_context_audit.json)、[QMSum exploratory manifest](data/manifests/qmsum_train_exploratory8.json)

尚缺的核心证据包括：真实共同 ready 与生命周期、自然完成及全部失败分母、完整 JCT/TTFT/TPOT、A0–A4 应用替代、任务质量与三 seed 非劣检验、TP 通信/选择/准备摊销、正式能量/EDP；算子侧还缺全部固定划分的执行级 E1、QUILT-exact/Cascade 的 E4，以及适用 PAT/PageWeaver、Fused3S、cuDNN BSA 原生或明确标注移植的比较。当前枚举逻辑 tile 的机会分析不等于完成这些强执行基线，2×2 partial 也不等于优化 E3。[方案中完整评价约定](proposal-iscas.md)、[审计的未支持主张](results/kernel_audit.json)

**下一研究方向及重新投入条件。** 当前动态行 v3 搜索已经结束。官方 4B 边界、最新原生版本和联合 prefill 强对照均已完成；固定 union 的正向信号没有通过最后的强对照门槛，故本轮不更换为一个已被原生接口覆盖的“新 idea”。无需继续无界扩大 layout、pipeline 或 split 搜索。

若开启新一轮，值得研究的问题是：**在精确固定 union、最佳固定划分及层次共享之后，自然输入是否还存在足够大的、可付清计划费用的执行空间。** 本轮暴露出短 owner 区域与 N 轴尾块的代价，可以作为候选假设；直接使用跨区域固定 union 只是采用既有机制，不能据此改名宣称新 idea。任何新的布局或执行选择方法先给出区别于这些基线的可检验增量，再决定是否实现。

重新投入分三道证据门槛：首先在身份和划分冻结的目标模型自然 capture 上，包含真实 prefix/suffix，量化相对固定 union、全部固定划分与 QUILT-exact 的剩余机会，并确认请求实际共同 ready；其次在同 mask、同精度、完整执行接口和准备摊销口径下，独立调参、同卡随机配对及第二卡复查，取得约 15% 且超过 A/A 两倍噪声的稳定收益，所有失败与边界保留；最后才进入主应用，验证质量约束下约 5% 的可重复净改善。任一道门槛没有足够证据就停止或更换问题，不由逻辑字节减少推导节能，也不由算子 replay 推导应用收益。[原定推进标准](proposal-iscas.md)
