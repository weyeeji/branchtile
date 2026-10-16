# 双 FP8 QK 第一轮：长前缀有小增量，尚未构成论文结果

root 调度的 wave33 四个独立单卡实验已完成，本代理只做 CPU 审计，没有另起 GPU。全部 QK/完整共同前缀 O 数值门通过。结果表中时间是完整 producer＋O-only merge 的 CUDA graph 调用中位数，含每个 split CTA 的 Q 分解；每个 case 有20个随机顺序配对 block，每 block 重放10次。

| 输入 | tile M | 原生 XQA O-only | 同形 BF16-QK 控制 | 双 FP8-QK | 相对同形控制降时 |
|---|---:|---:|---:|---:|---:|
| 合成8K，B8 | 32 | 16.84 µs | 33.11 µs | 33.12 µs | 无收益 |
| 合成8K，B8 | 64 | 17.28 µs | 47.50 µs | 49.03 µs | 慢3.2% |
| train1，真实FP8，rank0/layer15，120528-token共同前缀 | 64 | 128.05 µs | 74.81 µs | 71.51 µs | 4.4% |
| train7，真实FP8，rank0/layer15，144576-token共同前缀 | 64 | 151.03 µs | 74.75 µs | 72.45 µs | 3.1% |

逐 block 配对差值 `control-candidate`，两真实 case 均20/20为正，中位数3.194/2.056 µs。按20个计时 block bootstrap 的中位数差95%区间为 `[2.858,3.702]` / `[1.574,2.987]` µs。合成 M32 区间跨0，M64明确为负。区间只描述这次计时重复，不能作为跨文档、跨层或自然任务的统计证据。[可复算脚本](../scripts/analyze_query_components_gpu.py)、[完整结果及源SHA](../results/platform/query_components_gpu_analysis_20261017.json)。

## 数值与资源

同形控制与候选的完整前缀 O 对 FP64 relative-L2，在两真实 case 分别约0.00180、0.00166；原生 XQA 约0.00274、0.00260。两个新臂都保留 BF16 P/V，不能把主误差下降归给双FP8 QK。候选 QK 的 span-wide 1024个位置抽样，对 FP64 relative-L2 为 `4.97e-8` / `4.46e-8`；不是全部 token QK 的穷举资格，也不能代表模型 logits。

M64候选255寄存器，同形控制251–252；两者 `n_spills=0`，shared memory均24576 B。不能将候选的短输入负结果归因于已测 spill。M32均168寄存器、0spill、16384 B shared。短8K只有16/32个 producer CTA，长输入达到236/284个M64 CTA，可能影响占用与额外Q工作摊销；这是从grid计算的解释假设，不是profile证明。两个FP8 MMA、额外score与Q分解仍会消耗资源。

## 严格的归因边界

**相对 XQA 的约1.8–2.1倍算子差距，不等于新机制的贡献。** 自写两个臂都使用已有共享 packing、连续紧凑K/V；它们的 Q 也在计时前重排。XQA 用B8相同物理页的页表别名和独立HND K/V，尚非原引擎fused/interleaved存储。Q重排、CPU紧凑导出、页布局转换都没有进入本轮计时。只有同形自写两臂之间的2–3 µs差值隔离了本轮QK修改。未包含private tail、在线页管理、48层模型图、自然输出或答题质量，不能宣称端到端收益。

因此继续优先完成成熟 cascade 的原位执行对照。新机制若要继续，应固定这组形状和参数，在已有其余真实层/另一rank验证增量，再将两臂同时接入同一成熟数据布局；没有依据扩大tile盲扫。即使新机制微基准稳定，端到端增量也可能很小。

## 新原生FP8 Q的独立CPU检查

[脚本](../scripts/audit_query_fp8_native_cpu.py) 验证 `dense_capture_b8_train{1,7}_fp8_v2` 全部completed收据及Q文件SHA，覆盖2文档×2rank×48层、24576行：24078行数值精确重构，506个元素非精确，最大绝对表示误差 `1.52587890625e-5`，最大单层relative-L2 `3.56376e-8`。每文档只是一份真实第三步B8状态，不是全生成序列。[完整CPU结果](../results/platform/query_fp8_native_components_20261017.json)。v1失败export没有混入分析。
