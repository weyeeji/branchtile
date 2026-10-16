# 原生 Q/O 地址输入：整模型开发试验

`results/raw_direct_production_v1/train26` 完成全自然四阶段 ABBA。对同一次进程的原生实现，ready比值 **1.141677×**、cold比值 **1.109863×**；计入每文档完整额外setup和receipt导出后，主净比值 **1.087977×**。原生A0/A3 generate最大/最小为1.003095，只有一次种子/进程，不能据此给总体置信区间。

原生cold均值123.817114秒；候选cold均值111.560682秒，另完整收费2.244264秒，包括离线核对0.030207、配置0.153477、production建图2.046711及setup receipt导出0.013869秒。共同模型初始化和输入准备未包含在常驻服务的文档cold指标内。若将这些共同组件相加，首用组件估计比值1.061879×；它不是两个独立冷启动服务的直接测量。

该文档原始自然B21，名义共享前缀135376。候选执行2082/5551个完整模型图步骤，B6..21符合冻结策略的部分使用候选，其余原生回退。全部问题自然结束，无8192 cap、无记录抢占。四阶段的完整token、APC、finish/stop、问题和位置图轨迹一致；严格4/21、rounded10/21，1个预测解析失败，两臂相同。它是训练文档上的开发试验，不加入新next8首次或重复主表。

## 实现合同与资格

新机制身份为 `raw_pair_native_state_direct_input_v3`，direct build `c28391a037f29141`，继承逐原生组hint修复的base `5e5940e0894a8c8a`。与先前拷贝版相比，该bank在图捕获时将每层实际原生Q/O地址直接绑定给prefix与native tail/merge调用，避免候选自己的Q规范化、打包/metadata刷新和O搬运。

实际记录验证了每rank 16个B形状×48层：Q/O shape `[B,16,128]`，stride均 `[2048,128,1]`，Q/O alias标记为真、Q规范化copy标记为假；非连续布局会被拒绝。单套共享workspace/state仍按串行执行合同使用，KV更新、checkpoint导出和tail/merge完整计费。源代码和捕获receipt支持这个地址合同；它们不能替代独立GPU trace，也不能说整个模型没有数据拷贝。

生产使用新的离线资格：每rank125个真实原生轨迹状态，B6..21全覆盖（B13只有5个实际状态，其余每B8个），每rank6000层Q/O比较。全部12项包括hidden、logits、当前KV、输入/位置/页表、native AA、GPU metadata alias均通过。新source和fingerprint与生产一致，旧拷贝版资格不能替代。生产只捕获production图，没有偷偷在此session重跑在线严格资格。

`scripts/audit_native_direct_model_e2e.py`、`scripts/audit_native_direct_production_setup.py`及 `scripts/audit_direct_model_layout_receipt_cpu.py`完成独立CPU审计，结果在各运行目录的`independent_analysis.json`中。全部clean exit、运行源和生成代码哈希通过。

## 与历史运行的描述性关系

同一训练文档旧运行 `results/native_raw_pair_production_natural_v1/train26_bf16` 的ready/cold/完整setup净比值为1.129030/1.100257/1.078239；新运行分别为1.141677/1.109863/1.087977。CPU对照确认两次的prompt token哈希、全部四阶段自然token/text/APC/finish/stop、评分及图输入轨迹相同。

**这不是同session、同source的“只移除copy”配对消融。**两次运行不仅时间不同，还跨过grouped-hint数值修复版本；旧实现曾在其他文档出现数值分叉。不能把这两个开发净比值之差孤立归因于移除搬运。当前能够直接支持的是：新direct实现相对本次原生ABBA的整模型收益，以及实测资格覆盖中的逐位一致性。更广输入/长度与运行重复仍需独立验证。
