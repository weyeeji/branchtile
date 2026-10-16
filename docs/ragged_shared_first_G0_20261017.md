# 真实连续 query 的第一轮 G0 证据与近邻工作

当前新增代码已经可运行，不再停留在计划。capture 使用原生完整模型自然产生的连续8步 Q，保留0/15/31/47层和两rank的最后完整paged KV；后续只是这些真实 AR 状态的多query因果attention replay，**不是实际整模型speculative verifier**，没有接受率或端到端加速结论。

第一轮128MiB工作区在长prefix多query plan时不足，保留 `results/ragged_shared_verify_v1` 的失败。独立v2使用512MiB，原生完整causal prefill、prefix prefill、private causal prefill共用这块工作区，并在receipt记录。train7原capture在最后precision receipt错误地假设每条eager decode metadata都有kernel字段，run仍failed；独立CPU恢复审计验证8份文件完整性，显式recovery入口要求随后原AR O及full/shared O分别过FP64。没有将未知eager子kernel默认为XQA。

## 已完成数值和成本结果

`results/ragged_shared_verify_v2` 中四项qualified：train1 layer15、train7 layer15、两个真实文档各4请求合并的layer15和47。每项6种depth，共24 cell；原AR输出、full causal与shared输出各自FP64检查通过，LSE和未来KV扰动不泄漏检查通过。CPU另有210个精确支持集检查，含B8→B3重排→B2→B1；这些CPU检查不能冒充小batch GPU资格。

以下hot中位数来自每cell内部30 paired random blocks，单位微秒。不同depth cell尚未交错，因此**只算初步成本信号**，不能据此直接判G0严格门槛已通过。

| 实际组 | K32深度分配 | 每组query数 | shared L15 | shared L47 | full L15 |
|---|---|---|---:|---:|---:|
| train1单组8请求 | 4×8 | 32 | 168.92 | 未测此cell | 336.54 |
| train1单组8请求 | 8,8,8,4,1,1,1,1 | 32 | 168.91 | 未测此cell | 337.09 |
| 两文档各4请求 | 4×8 | 16,16 | 188.26 | 187.96 | 374.79 |
| 两文档各4请求 | 8,8,8,4,1,1,1,1 | 28,4 | 262.86 | 263.33 | 375.36 |
| 两文档各4请求 | 8,1,8,1,8,1,4,1 | 18,14 | 263.65 | 262.66 | 377.05 |

单组同K32的深度差别几乎不影响shared成本；两个真实组中，分配不均约慢40%。这首先否定只看(B,K)的简单成本假设，但**普通group-aware成本表仍很可能充分解释它**。不能把约2×的known-cascade/full attention比值当成新scheduler的收益。短prefix synthetic的shared反而约0.5× full，是两次调用和merge开销的负对照。

## 极近先例：DScale

[DScale（2026-09-29）](https://arxiv.org/html/2609.37532v1) 已把验证和prefill采用不同query tile，做逐请求长度预测、固定总预算分配及GPU打包接入CUDA graph。其§II-A/III-B的attention tile对应单个请求；不同上下文不能直接混用该请求的causal tile。因此，“tile-aware验证＋动态depth＋GPU packing”不是我们的新颖点。当前仅剩的待验证区别是：具有**同一物理公共prefix的独立请求**能在公共部分合并query，分配depth使同组query数跨过tile边界，改变公共KV扫描次数。因果private尾部仍分开。

这个区别是机制范围判断，不是首次性证明。必须保留增强成本表＋逐token边际greedy（C2）对照，若它消除差距，应当停止复杂scheduler或将研究改成小而明确的执行改进。

## 代码层解释与当前最大风险

本地FlashInfer `data/include/flashinfer/attention/scheduler.cuh` 的 `PrefillSplitQOKVIndptr` 逻辑：当wrapper内部CUDA graph模式关闭时，用平均packed query长度选CTA tile，再对每组累加 `ceil(group_Q * GQA / tile_Q)`。本机GQA为8；若实际tile_Q为128，则(16,16)只需2个query tile，而(28,4)/(18,14)需要3个。该推断与阶跃符合，但最终以独立交错runner导出的实际plan_info和后续kernel profile为准。

独立 `scripts/measure_ragged_group_coupling.py` 已冻结：三种K32分配以及各自拆成单组执行共9个plan，全部输出重新FP64；随机交错全部9种和AA，hot/cold分别30 blocks；记录plan_info、捕图和4.5GiB工作区驻留。拆组两次调用之和只用于成本模型，不是公平在线策略对手。

**G0使用的wrapper内部graph模式为False，只在外部手动捕获固定输入图。** 真正动态graph-safe计划会按最大可能query分配选择tile和padding数量，本地源码存在单独分支。它可能让当前观察到的2→3 tile差异消失，或者需要多个预捕图桶才保留收益。因此，离线最优plan收益不等于在线可实现收益；G1之前必须测真实graph-safe计划、图桶/切换/准备成本，不能假设每步重新plan免费，也不能无限复用错误拓扑。
