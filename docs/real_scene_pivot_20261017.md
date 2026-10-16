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

模型初始化 43.945 秒、选择器 dense warmup 12.915 秒和图构建另行记录，表中没有把它们假装免费；anchor + generate 也是两个实际阶段的加和，而非包含客户端所有开销的一体请求测量。不同方法的图同时常驻，原生独立部署可释放额外候选 workspace。

证据：[原始结果](../results/paged_owner_natural_exploratory_v1/summary.json)、[冻结指标汇总](../results/paged_owner_natural_exploratory_v1/natural_execution_summary.json)、[预先标识的解释合同](../configs/paged_owner_natural_exploratory_v1_interpretation.json)。先前模型 logit cosine 门槛失败保留；局部 attention 检查通过不能自动授予任务质量资格。

## 2. 真实整模型中仍有空间

在同一真实 decode 状态，用已保存的正确 attention 输出替换该算子，并逐 bit 核查全部层 Q/O、最终 hidden/logits 和本步写 KV，B3 BF16/FP8 模型图分别减少约 41.5%/29.6%；更长的 B8 文档分别约 67.3%/59.3%。这只说明当前实际模型图中可以被 attention 替代影响的范围。预知 O 不能用于下一 token，替代改变图依赖且保留复制成本，不能称自然服务收益或严格数学上界。

B8 的自然请求会陆续退出。第一个文档的全八路共同活跃仅占 BF16/FP8 模型步的 11.1%/25.6%，其后必须处理 B7…B1；因此固定 B8 的比例不可乘整个自然任务。两种精度的生成长度不同，也不能仅用每 token 指标判断任务更快。

第一个 B8 文档的原生严格 FINAL 得分本身较低：BF16 1/8、FP8 0/8，舍入敏感性都是 2/8。全部完成 EOS。这一训练组仅作真实负载探索，不能凭少数样本宣称任务质量保持，也不能在看到结果后挑选高分题替换。

证据：[整模型干预报告](cached_o_opportunity_first_result_20261017.md)、[自然活跃期分析](native_opportunity_b8_analysis_20261016.md)、[原生质量记录](../results/native_opportunity_b8_natural_v1/train1_quality.json)。

## 3. 转向的实验顺序

1. 完成第二文档的自然 owner 结果，保留不利结果；不再围绕 4K 微小算子差值反复调参。
2. 在真实完整 KV 上实现成熟共享前缀对照，直接面对原生 XQA O-only。前缀只处理共同完整物理页，非对齐残页、问题和生成尾全部进入私有路径，允许请求自然退出和重新排序。
3. 核查本地 FlashInfer 新接口对 BF16 Q / FP8 KV、真实交错层 strides 的支持，先小规模 FP64/图重放/页变化资格，再整模型固定轨迹与自然生成。不能从旧 vLLM 注释推断当前库不支持。
4. 普通 cascade、共享解量化、split/merge、增量元数据都有先例。以成熟实现为底座，再提出可区分的片上数据流或执行机制；没有新意与完整应用证据时，不宣称已达到 ISCAS 投稿结果。

所有任务以普通用户和项目虚拟环境运行；四组 TP2 尽量并行，失败和初始化也计入 GPU 分配时间。进展及原始记录持续提交 Git。
