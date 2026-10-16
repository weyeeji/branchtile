# GQA8 固定寄存器状态的剩余实现空间：CPU 审查与停止判断

2026-10-16。本轮推荐新增 GPU 候选数为 **0**。动态 v1/v2/v3 和 real48 joint 均保持停止；不扩展 GQA16/32 人为制造优势，不修改原有收益门槛。这是基于现有证据的技术与成本判断，不是证明所有片上数据流都不可能。

本轮只使用普通 uid、项目虚拟环境和 Python 标准库读取既有结果；无 GPU、安装、编译、掩码修改或新增试跑。可复算工具为 [analyze_register_bank_residual.py](../scripts/analyze_register_bank_residual.py)，输出 [CPU 审查 JSON](../results/platform/register_bank_residual_review_20261016.json)。JSON 保存实际输入字节 SHA256、脚本及被审源码 SHA256、原实验冻结源码收据及其当前匹配情况，不以当前文件冒充原运行版本。

## 唯一具体但尚未实现的布局缺口

GQA8 的每个请求有 8 个 Q 行。NVIDIA [PTX `mma.m16n8k16` FP32 累加器定义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type) 中，同一 lane 的 `c0,c1` 属于 `lane_id >> 2` 行，`c2,c3` 属于该行加 8。因此可以设想：原请求 a、b 的 U 永久保存在固定 lane-local register banks；针对当前 owner 对 `(a,b)`，将两组原寄存器直接作为 MMA C/D 操作数的上下两半。PV 更新回原 bank，owner 变化只选择静态代码分支，不将全部 canonical U 暂存再恢复。singleton 使用一半加 dummy，B3 全活跃使用两组 M16；m/l 按原请求保持，并进行各自 softmax rescale。

这不同于当前 [v3](../src/branchtile/tilelang_kernel_v3.py)：v3 虽然将 canonical U 保留为 FP32 fragment，但每个 region 仍把 U 的原始 32 位写入两格 BF16 类型的 K scratch，再 gather active U、更新、写回并恢复 canonical U；它没有将 U 数值转成 BF16。[v2](../src/branchtile/tilelang_kernel_v2.py) 则使用 canonical shared U gather/scatter。固定 union 已将 U 保留在寄存器，并在固定行布局中跨 owner 边界执行。[源码与既有数值审计](../results/kernel_audit.json)

若四个 warp 各负责 32 个 D 列，B3 的 U 存储下限是 `3×8×32/32=24` 个 FP32 寄存器/线程。这**不是完整寄存器需求或编译结果**：QK/score、m/l、地址、P、MMA 输入及控制流仍要资源，静态分支也可能增加代码与寄存器压力。QK 分布在不同 warp 时，归约和 P 暂存仍需跨 warp 协作与同步；不能宣传为“零同步”。实现需要专门的寄存器布局和 MMA 代码，并非直接删除 v3 的几条 copy 就成立。本轮没有编写或编译该内核。

## 真实全 48 层几何的复算定义

使用已成功完成的 real48 replay **replica0** 的两个文档，各含 48 层、2 个 TP rank、每 rank 2 个 KV head，共 192 个几何 cohort。它们仍是两个开发文档，不能把层、头、设备副本当成更多独立文档。读取各 `joint_variants.json` 的 `epsilon01.layer_records[].records[].independent_geometry`，只取**原 independent mask**；脚本检查其几何及 mask SHA 与 epsilon0、epsilon001 容器中的原 mask 完全一致，不使用交换后的 new_geometry。

- publicdev1： [rank0](../results/joint_real48_replay_v1/joint_real48_doc0_replica0/rank0/joint_variants.json)、[rank1](../results/joint_real48_replay_v1/joint_real48_doc0_replica0/rank1/joint_variants.json)。
- original： [rank0](../results/joint_real48_replay_v1/joint_real48_doc1_replica0/rank0/joint_variants.json)、[rank1](../results/joint_real48_replay_v1/joint_real48_doc1_replica0/rank1/joint_variants.json)。

固定 B=3、GQA=8、D=128、document budget=32768。令 `n_o` 为原 owner bitmask `o∈{1,…,7}` 的共享文档 token 数，`r(o)=ceil(8×popcount(o)/16)=ceil(popcount(o)/2)`。

1. 理想动态计算量 `C=Σ_o r(o)n_o`，单位为 M16 行组×token；所有方案相同的 D/K 维系数省略。
2. 枚举全部五种 B3 固定划分，bitmask 为 `(1,2,4)、(1,6)、(3,4)、(5,2)、(7)`。组 g 的 union 长度 `L_g=Σ_{o:o&g≠0} n_o`，固定计算量 `F(P)=Σ_{g∈P} r(g)L_g`。固定组允许跨 owner 边界连续执行，与强 fixed union 对照一致。
3. N64 取整分别为 `C64=Σ_o r(o)ceil(n_o/64)` 与 `F64(P)=Σ_{g∈P} r(g)ceil(L_g/64)`。动态每个非空 owner region 一个尾 tile，固定每组一个尾 tile；两者均未加入 split 边界额外尾 tile。
4. 每个 cohort 独立选择最小 F、最小 F64，再各自跨 192 个 cohort 求和并取比。它是**事后逻辑工作量最小值**，不是实测延迟胜者、可部署选择器或平均各格速度比。

system prefix、非整页文档 private tail、question suffix、generated tail 全部省略。脚本验证各题共同源所选数为 `32768−private_document_tail`；所有这些 private token 在真实执行中仍必须保留。未计 prepare、planning、最终 merge、output copy、selector、owner 转换与同步成本。省略这些成本及 private 工作对动态计算量核算偏乐观。

| 文档 | C：M16×token | 最佳 F | 最佳 F/C | C64 | 最佳 F64 | 最佳 F64/C64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| publicdev1 | 12,649,031 | 13,107,783 | 1.036268× | 198,386 | 204,939 | 1.033032× |
| original | 12,626,559 | 12,940,687 | 1.024878× | 198,039 | 202,301 | 1.021521× |

单一全 union 与理想动态的 C 比为 1.151190× / 1.129425×；只比较这个较弱固定选项会夸大相对最佳固定划分的余量。全部三题共享的 owner7 占 union token 比例的 cohort 中位数约为 72.1% / 76.6%；这些 token 在两种布局都需要两组 M16。

**上述逻辑 MMA 比不是完整 runtime 上界。** 它既不是原生 SIMT/XQA 的实际指令数，也不能排除新访存组织带来的额外收益；占用率、缓存、流水、同步、指令和调度都可能改变时间。仅在“attention 全部时间与这些算术量成正比、相同单位吞吐且新增成本为零”的条件下，最佳固定划分到理想动态的 attention 延迟降幅才为约 3.50% / 2.43%；再条件性取 attention 占整步 20%–25%，整步降幅约 0.70%–0.88% / 0.49%–0.61%。这只是算术路径的乐观 Amdahl 例子，不是实测模型收益或普遍硬件上界。

独立逻辑 KV 读取数 `Σ_o popcount(o)n_o` 与 union `Σ_o n_o` 的比为 2.591587× / 2.646875×。**这不是 GDDR 流量比**：native E0 已能通过 L2 重用相同地址，cache 热状态和真实模型交错会改变读取。现有权限不能读取所需硬件 counter；没有真实 GDDR counter 或严谨 cache 证据，不给这一项任何已证实收益信用。

## 实测差距、资源与新颖性边界

既有修正 suffix mask 的 original 文档 L15、step64、rank0、32K 点已测完 85 个配置，包含全部五种固定划分、三棵二叉层次树和动态 v2/v3。hot 中位数为 native E0 23.53 μs、固定 union 21.51 μs、E1 native 19.89 μs、E4 native 19.19 μs、v2 27.07 μs、v3 32.27 μs。[原始计时、FP64 数值合同和调参记录](../results/suffix_v2_replay/main30b_budget_grid_rank0_l15_s64_b32768.json)。它是单个真实 operator 点，不能冒充新寄存器方案的测量，也不能与 FI0.7.0 的 real48 生产链直接拼接速度比。

[保存的 shape-specific cubin 资源](../results/kernel_resources.json) 中，v2 N64 为 190 registers/thread、64,768 B shared，v3 N64 为 240 registers/thread、48,384 B shared，固定 E2 N64 为 148 registers/thread、47,104 B shared；这些收据没有 local/stack spill。它们支持当前资源压力及状态转移成本需要关注，但不能把未测过的 spill、bank conflict 或 DRAM stall 当作已定位根因，也不能把该 B4 微形状资源当作所有生产形状。

[Cascade](https://flashinfer.ai/2024/02/02/cascade-inference.html) 已做多 Q 片上 KV 复用和 state merge；[PAT §5](https://arxiv.org/html/2511.22333v3#S5) 已讨论 region 分组与资源约束；[PageWeaver §4.2](https://arxiv.org/html/2610.11201v1#S4.SS2) 已使用稳定 query 身份、持续 FP32 状态及共享 KV 协作。不能声称这些工作逐指令实现了上面的 GQA8 register-bank routing，但也不能仅凭“不 canonical 回写”就推出一个新论文机制。这个区别目前仍是原动态 owner 映射目标内的实现优化缺口，不是通过新颖性与收益门槛的新 pivot。

结合 [既有 pivot 排重](pivot_candidates_20261016.md)、[real48 joint 停止证据](joint_real48_stop_review_20261016.md) 和 [停止后的研究判断](post_stop_research_decision_20261016.md)，本轮只保留“技术上可能进一步减少状态往返”的未实现事实；不为此启动 GPU、下调 1.15× 完整调用门槛，或扩大主模型 GQA。若未来有真实访存证据，它应作为新的证据重新审查，而不是本报告承诺的待跑工作。

复算与字节一致性检查：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/analyze_register_bank_residual.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/analyze_register_bank_residual.py --check
```

脚本只写自己的结果 JSON；`--check` 不写文件。原有 raw、源码、计时、中心报告和预算账本均不修改。
