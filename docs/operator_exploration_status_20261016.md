# 算子探索进度与比较边界

2026-10-16；本报告冻结 wave1 / wave2 完成后的结果。只读现有收据，不运行 GPU、不修改实验源码。后续资格实验若有新终态，应由其独立收据更新判断。

## 当前范围与实验原则

用户已明确选择继续寻找有新意的加速机制，并要求重新挖掘原 idea 的算子实现，尽量八卡并行测试。本轮范围是**单用户、多 agent、共享文档**。安全不再作为研究方向或新增实验项，论文不提出跨租户部署主张。

输出长度变化按两个层面比较，完整规则见[比较协议](operator_comparison_protocol_20261016.md)：

1. **固定工作量、配对测算子。** 相同真实 Q/K/V、mask、私有尾部、位置和物理共享关系，比较完整 attention 调用；索引、merge、必需输出 copy 分清是否计时。分别保留热态、人工清缓存冷态、A/A、随机/反向配对 blocks。它回答“同样的计算是否更快”。
2. **自然 EOS、联合测质量与任务成本。** 各方法正常生成至 EOS，记录真实长度、退出后的 batch、完成时间及质量；失败保留在完整分母，多 seed、按文档来源 cluster 配对统计。它回答“同质量完成任务是否更省时”。不得把不同自然长度的总耗时直接解释为纯算子收益，也不强行裁成等长后声称自然质量相同。

GPU 软件数据流、片上状态组织与流水线设计符合 ISCAS 的研究范围，已有 AMUSD 等接收先例；但不能据此断言会议更偏好该形式。具体证据与先例边界见[ISCAS 范围核对](iscas_operator_contribution_review_20261016.md)。

## wave1 / wave2：扩大真实工作集后，空分片 guard 仍未产生净优势

原始结果位于 [wave1](../results/operator_overhead_wave1/) 和 [wave2](../results/operator_overhead_wave2/)。全部使用普通 uid 1000、项目 `.venv-latest`、实际 BF16 Q/K/V、GQA8、每 rank 两个 KV head、同事件 step64。两波每次七卡并行筛查，另一卡承载寄存器候选资格任务；单卡的 rank capture 重放不是实际 TP2 模型运行。

| 已完成覆盖 | 数值资格与搜索 | Emptyguard 配对速度比：热态 | 冷态 |
| --- | --- | ---: | ---: |
| wave1：长 B4，rank0/1，L15/31，64K/128K，共 7 个状态格 | 434 / 434 配置通过冻结 FP64 O/LSE 检查；全部完成 | 0.549–0.851× | 0.830–0.913× |
| wave2：三份文档 B3/B4，rank0，L15/31，4K 六格；长 B4 全文 173232 一格 | 306 / 306 配置通过；全部完成 | 0.341–0.931× | 0.476–0.928× |

速度比定义为 `已选强 control 时间 / emptyguard 时间`，超过 1 才表示候选更快。共 **28 个状态×冷热条件均未超过 1**；各格配对 block bootstrap 95% 区间上端也小于 1。只在已测 split / tile 范围内选参，属于固定状态上的筛查，不能称全局最优实现。不同层、rank、冷热与 tuning 配置均不是更多独立文档样本。

强 control 在原生独立、全部固定请求划分的原生/自写实现，以及层次共享树之间选择；B4 枚举 15 个划分和 15 棵树。选择来自单独 tuning，测量使用后续随机/反向 blocks，但仍是每个 capture 的事后配置选择，尚无可部署 selector。它使比较覆盖已有共享能力，而不只对比旧 v3。

`emptyguard` 的改动是遇到没有 token 的 region shard 时，以 CTA 一致分支跳过该次区域处理。它相对旧 v3 快了约 1.03–1.46 倍，但这不构成相对强基线的胜出：wave1 每格均比 v2 慢约 1.45%–7.11%；wave2 也大多更慢，冷态微小相当/反转受约 2.048 µs 的计时量化影响。wave2 的独立等配置和几何核对见 [CPU 审查](../results/operator_emptyguard_wave2_audit_v1/audit.json)。

例如长 B4、rank0、L15、64K 的热态，强 control 为 **46.05 µs**，v2 为 **70.40 µs**，emptyguard 为 **74.30 µs**；长 B4 全文的热态为 **117.94 / 125.94 / 126.75 µs**。扩大到 64K、128K 和真实全文后，候选仍没有超过已测强 control。它说明“简单跳过空 shard”不足以抵消当前动态组织成本，**不证明所有寄存器或流水线实现都不可能更快**。

两波表格是已构建算子的完整 GPU 调用时间，准备和规划另列在各原始结果 `planning` 中。即使这个较有利的稳态边界都未胜出，也没有理由据此宣称计入准备后会有应用净收益。静态计划复用 1/16/64/256 次只是摊销情景，不是已测请求生命周期；并行运行的初始化/JIT 时间也不是孤立编译速度。

## v4：仍处于编译与数值资格阶段

固定寄存器 bank 的 CPU 路由审查已见[寄存器映射报告](operator_register_routes_review_20261016.md)，但 CPU 整数证明不能代替 GPU 编译、数值和 graph 资格。

截至前三次资格尝试：[attempt1](../results/static_bank_v4_qualification/attempt1.json) 遇到静态嵌套过深；[attempt2](../results/static_bank_v4_qualification/attempt2.json) 遇到片段布局推导冲突；第三次编译超时，由 [job ledger](../results/job_ledger.jsonl) 的 `static_bank_v4_qual_attempt3` 终态记录 `timed_out=true`、exit -9。其 [attempt3 JSON](../results/static_bank_v4_qualification/attempt3.json) 因进程终止残留 `running`，不能据此认定仍在运行或资格通过。三次均没有有效性能结论。

后续 wave3 的 B3/B4、N32/N64 及 full-column guard 是新的资格尝试，状态由独立收据决定。**v4 尚未通过资格，不能报告其加速比，也不能把 wave1 / wave2 的 emptyguard 成绩归给 v4。** 下一项有意义的进展是通过编译与独立数值检查，再在相同真实状态上与上述强 control 配对测量；当前未满足新的论文加速结论。
