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

## 有资源依据的后续变体：仅保留一次 Q 分解的候选

不据255寄存器直接宣称已跨占用率台阶：251–255在线程束分配粒度下可能同属一个资源档位，且两臂都没有spill。M32是已经预声明的对照，但它会把长前缀CTA数量和K读取副本翻倍；仅有8K结果不支持直接宣布M32更快。

若原paged同合同比较后仍有空间，最有明确结构依据的变体是**在既有Q生产操作的尾部只计算一次高低分量**，让所有prefix split读取同一份Qh/Ql与scale。当前120528/144576-token前缀分别有118/142个chunk，单个查询行的分解就重复118/142次。Qwen3MoeAttention源码中顺序是Q/K RMSNorm→RoPE→attention；可研究RoPE输出处融合分解，而不是新增一个独立quantization kernel。B8/TP2单层有128行：两个FP8分量共32768 B，加两份FP32 scale共1024 B；原BF16 Q要保留给private-tail和数值对照，额外payload约33 KiB/层，若静态缓存全48层还须计48倍。

必须先保留与原生相同的BF16 Q舍入边界，再作分解；直接使用RoPE内部未舍入FP32值会改变输入合同。RoPE融合还需两次每行max归约、转换和额外写出，其新增成本不能当作免费。该变体可能减少重复指令，但循环中的双score/PV accumulator仍可能决定寄存器峰值，因此不保证降到更低寄存器档位。独立Q拆分kernel的启动、依赖和读写很容易吞掉当前每层2–3 µs增量，不作为优先性能方案。

静态PTX审计支持的是“少做K转换”：M64控制128条BF16 MMA，候选64条FP8 MMA加64条BF16 PV，`cvt.bf16.f16`静态指令从128降到64。两臂shared仍相同，不能把本轮增量解释为已经减少共享内存。静态指令计数也不是动态耗时profile。

## 新原生FP8 Q的独立CPU检查

[脚本](../scripts/audit_query_fp8_native_cpu.py) 验证 `dense_capture_b8_train{1,7}_fp8_v2` 全部completed收据及Q文件SHA，覆盖2文档×2rank×48层、24576行：24078行数值精确重构，506个元素非精确，最大绝对表示误差 `1.52587890625e-5`，最大单层relative-L2 `3.56376e-8`。每文档只是一份真实第三步B8状态，不是全生成序列。[完整CPU结果](../results/platform/query_fp8_native_components_20261017.json)。v1失败export没有混入分析。
