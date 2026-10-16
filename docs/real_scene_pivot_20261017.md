# 真实场景验证与下一步选择（2026-10-17）

目前没有达到论文结论所需的已验证新机制。新的整模型干预说明长上下文 attention 仍有显著空间；新完成的自然生成则否定了当前 4K 稀疏 owner 桥接版的实际价值。这两件事不矛盾。

## 1. 当前 owner 的完整自然生成

原开发文档 `docfinqa-c0f11a67aae850fc`，自然 B3，官方 FP8 权重、BF16 KV、TP2、实际 FlashInfer 后端。三种方法使用相同输入、sampling seed、temperature 0.6/top-p 0.95，最大 8192 token；未强迫相同输出长度，没有 shadow 或 logits CPU copy。各方法独立重置 APC 并运行冷 anchor。

| 方法 | 完整 generate 秒 | anchor + generate 秒 | 输出 token 总数 | EOS / 3 | 严格 FINAL / 舍入敏感性 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原生 full-history | 13.123 | 24.065 | 3066 | 3 | 1/3、2/3 |
| 相同 4K 支持集的 E0 | 53.535 | 64.518 | 16299 | 2 | 0/3、0/3 |
| 相同 4K 支持集的 owner | 104.225 | 115.220 | 17768 | 2 | 0/3、0/3 |

这是单文档单 seed 探索结果，不是正式质量比较。两种稀疏方法各有一个请求触及长度上限；所有失败保留。生成轨迹不同，不能把这些比值称为相同工作量的算子减速倍数。owner 的自然长尾还暴露了当前单 CTA 私有尾/桥接准备的成本，但 host prepare 时间包含等待 GPU，不能全部归为 CPU 元数据开销。

第二份已预选开发文档 `docfinqa-593ec0c33f45ea02` 也已完成：dense/E0/owner 的 generate 分别 30.839/42.376/44.686 秒，anchor 加和分别 42.152/53.716/56.020 秒；输出分别 6732/11492/11128 token，全部 EOS。严格指标均为 0/3，舍入指标分别 2/3、0/3、0/3。第二文档也没有支持当前稀疏 owner 的净收益；不因第一文档结果差而删除它。[第二文档记录](../results/paged_owner_natural_newdoc_exploratory_v1/natural_execution_summary.json)

模型初始化 43.945 秒、选择器 dense warmup 12.915 秒和图构建另行记录，表中没有把它们假装免费；anchor + generate 也是两个实际阶段的加和，而非包含客户端所有开销的一体请求测量。不同方法的图同时常驻，原生独立部署可释放额外候选 workspace。

证据：[原始结果](../results/paged_owner_natural_exploratory_v1/summary.json)、[冻结指标汇总](../results/paged_owner_natural_exploratory_v1/natural_execution_summary.json)、[预先标识的解释合同](../configs/paged_owner_natural_exploratory_v1_interpretation.json)。先前模型 logit cosine 门槛失败保留；局部 attention 检查通过不能自动授予任务质量资格。

## 2. 真实整模型中仍有空间

在同一真实 decode 状态，用已保存的正确 attention 输出替换该算子，并逐 bit 核查全部层 Q/O、最终 hidden/logits 和本步写 KV，B3 BF16/FP8 模型图分别减少约 41.5%/29.6%；更长的 B8 文档分别约 67.3%/59.3%。这只说明当前实际模型图中可以被 attention 替代影响的范围。预知 O 不能用于下一 token，替代改变图依赖且保留复制成本，不能称自然服务收益或严格数学上界。

B8 的自然请求会陆续退出。第一个文档的全八路共同活跃仅占 BF16/FP8 模型步的 11.1%/25.6%，其后必须处理 B7…B1；因此固定 B8 的比例不可乘整个自然任务。两种精度的生成长度不同，也不能仅用每 token 指标判断任务更快。

第一个 B8 文档的原生严格 FINAL 得分本身较低：BF16 1/8、FP8 0/8，舍入敏感性都是 2/8。全部完成 EOS。这一训练组仅作真实负载探索，不能凭少数样本宣称任务质量保持，也不能在看到结果后挑选高分题替换。

第二个 B8 文档原生 BF16/FP8 的 anchor + generate 分别 71.868/49.826 秒，输出 10003/7317 token，16 个答案均 EOS；严格得分 2/8 与 1/8，舍入得分 3/8 与 2/8。自然长度和质量变化都影响耗时，FP8 更短不能单独作为精度无损的证据。[四个原生自然单元的冻结指标](../results/native_opportunity_b8_natural_v1/quality_all4.json)

证据：[整模型干预报告](cached_o_opportunity_first_result_20261017.md)、[自然活跃期分析](native_opportunity_b8_analysis_20261016.md)、[原生质量记录](../results/native_opportunity_b8_natural_v1/train1_quality.json)。

## 3. 转向的实验顺序

1. 完成第二文档的自然 owner 结果，保留不利结果；不再围绕 4K 微小算子差值反复调参。
2. 在真实完整 KV 上实现成熟共享前缀对照，直接面对原生 XQA O-only。前缀只处理共同完整物理页，非对齐残页、问题和生成尾全部进入私有路径，允许请求自然退出和重新排序。
3. 核查本地 FlashInfer 新接口对 BF16 Q / FP8 KV、真实交错层 strides 的支持，先小规模 FP64/图重放/页变化资格，再整模型固定轨迹与自然生成。不能从旧 vLLM 注释推断当前库不支持。
4. 普通 cascade、共享解量化、split/merge、增量元数据都有先例。以成熟实现为底座，再提出可区分的片上数据流或执行机制；没有新意与完整应用证据时，不宣称已达到 ISCAS 投稿结果。

所有任务以普通用户和项目虚拟环境运行；四组 TP2 尽量并行，失败和初始化也计入 GPU 分配时间。进展及原始记录持续提交 Git。

## 4. 四个随机种子的完整结果与实计算模型图

追加种子在运行前固定为 20261017/18/19，与原 20261016 合计四个；顺序 AB/BA 平衡。两个自然 B8 训练文档分别用 BF16/FP8 KV，共 16 对方法比较、256 个完整答案，全部 EOS，无截断。每个阶段重置 APC、执行冷 anchor，完整保存输出长度、原冻结 FINAL 指标与所有不利结果。四组不是独立的 16 个文档，不能以种子数替代任务覆盖。

| 文档与 KV | 四 seed 生成总耗时 native/cascade | 含 anchor 总耗时 native/cascade | cascade 冷任务耗时变化 |
| --- | ---: | ---: | ---: |
| train1 BF16 | 0.9464 | 0.9657 | 增加 3.6% |
| train1 FP8 | 0.9598 | 0.9740 | 增加 2.7% |
| train7 BF16 | 0.9715 | 0.9818 | 增加 1.9% |
| train7 FP8 | 0.6402 | 0.7707 | 增加 29.7% |

比值小于 1 表示候选慢。计时不包括模型加载与候选图构建，不能通过另加准备成本恢复正结果。答案质量也未建立非劣：例如 train7 BF16 舍入敏感性评分 native 为 13/32，cascade 为 10/32；这只是小样本描述，不作统计显著性断言。[可复算四 seed 审计](../results/dense_cascade_four_seed_audit_v1/audit.json)

另在相同真实输入 token、位置、先前 KV 下，实际计算 full-history cascade 的完整 48 层模型图，BF16 为 14.3256→10.0965 ms，FP8 为 11.4039→8.3069 ms（较慢 TP rank 的配对事件测量）。每层同 Q 数值资格通过，最终原生 hidden/当前写入 KV 逐 bit 恢复，checked/production 图一致。此实验不使用已知 O 替代算子，但仍只是一份真实状态，未计每步 host 准备、prefill，且候选中间数值/MoE 路由可变化。因此能证明计算路径存在约 27%–30% 的模型步机会，不能称为自然完整任务收益。[BF16](../results/dense_cascade_snapshot_v1/train1_bf16/analysis.json) / [FP8](../results/dense_cascade_snapshot_v1/train1_fp8/analysis.json)

Q 双分量在真实分页布局失去原连续布局的微小优势；Q+PV 双分量四点均通过数值资格，但只有一个点相对匹配控制节省约 0.6 μs，全部落后成熟 FA2，且共享内存和 spill 增加。关闭这两条具体算子假设，保留源码与完整失败记录。

下一步的具体可证伪工作是：保持完整注意力与原 FA2 算术，使用一次规划和 GPU 元数据更新去掉每步 host 规划/同步，先测试跨页、重排和自然退出正确性，再测完整任务。这是执行开销控制，尚不宣称新颖性。另外独立筛查共同前缀组与 speculative verification 深度的成本耦合；必须与支持同样前缀共享的已有调度策略比较，并先证明确有可兑现的模型调用收益。

[实测模型步与全部自然任务结果对照图（PDF）](../results/real_scene_opportunity_figure_v1/opportunity_and_task_outcomes.pdf) 同时展示四个种子的单次结果与汇总比值；两侧测量边界不同，不用模型图加速替代完整任务结论。
