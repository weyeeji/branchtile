# 短 Q 分解、长 K 保持 FP8：待否证的共享前缀数据流

**状态：只有 CPU 表示检查和 SM120 离线编译，尚无 GPU 正确性、加速或论文新颖性资格。**
独立原型是 [`triton_kernel_query_components.py`](../src/branchtile/triton_kernel_query_components.py)，不改原 owner/cascade 实现。

## 针对的瓶颈与区别

本机 XQA 的 `mha.cu` 在 QK 和 PV 的 warp fragment 上把 FP8 KV 转成 InputElem。`utils.cuh` 的 FP16 路径有 packed E4M3→FP16 特化，BF16 使用逐元素转换。共享 prefix packing 消除重复全局读取后，长 K 的转换、16bit staging 和 fragment 布局仍可能重要；是否真的主导尚未测得。

[PAT](https://arxiv.org/html/2511.22333v3) 已覆盖 shared-query packing、多 tile 和 split/merge；[HyQuant](https://arxiv.org/html/2608.27875v3) 已覆盖低 bit KV 与融合解量化。[FlashInfer #3485](https://github.com/flashinfer-ai/flashinfer/pull/3485) 更直接：把 FP8 KV tile 一次解量化到16bit共享内存后复用。本机 FA2 `use_kv_repack` 对 D128、query tile>16 已具备该策略。因此简单“共享解量化”不是本假设的新意。

这里只检验另一个数据流：把短 Q 表成两个 FP8 分量，长 K 保持现有 unit-scale E4M3 字节，执行两个原生 FP8 QK；V 的解量化、BF16 P/V 乘法、online softmax、split/merge 全部先与控制相同。它把一部分转换工作从长 KV 维移到短 query 维，代价是第二次 QK MMA、行 scale、额外 score 临时量和寄存器。现有 SM120 FP8 attention 并不缺失，见 [FlashInfer RFC #3628](https://github.com/flashinfer-ai/flashinfer/issues/3628)；这里不是 MXFP8 QKV 或“首次 SM120 FP8”。

多分量低精度矩阵乘法也早有[Henry 等，2019](https://arxiv.org/abs/1904.06376)。只有特定共享 prefix 数据流相对上述成熟实现的新增收益、数值合同和机制消融，才可能形成后续贡献；目前没有完成这项证明。

## 公式及实现成本

对每个 D128 的 BF16 查询行，在 FP32 中计算：

```
sh = 2^ceil(log2(max(abs(q))/448))
qh = E4M3(q/sh)
r  = q - sh*FP32(qh)
sl = 2^ceil(log2(max(abs(r))/448))
ql = E4M3(r/sl)
Sh = FP32_MMA(qh, K8^T)
Sl = FP32_MMA(ql, K8^T)
S  = (sh*Sh + sl*Sl)/sqrt(D)
```

高低 scale 相互独立、均按行取二的幂。零行用一个极小正尺度避免 log(0)。softmax 在 S 合成后才执行；不能分别 softmax 后相加。两个 FP8 MMA 都指定 `max_num_imprecise_acc=0`。相对单个 QK score tile，增加最多一个 M×N FP32 score 临时量：M64/N64 为16 KiB逻辑数据，另有两份 FP8 Q fragment 与行尺度。实际寄存器和 spill 由最终 SASS/GPU 收据决定，不能从逻辑容量推断。

prototype 每个 split CTA 只在 KV loop 前计算 Q 分解，**仍会随 split 重复**；这项成本包含在 producer 计时内。两个 FP8 MMA 可能已经耗尽 FP8 的理论吞吐优势，故不能从算力峰值预测加速。V 仍转换 BF16，未消除全部 KV 转换。

## 已完成的 CPU 检查

[`audit_query_fp8_components.py`](../scripts/audit_query_fp8_components.py) 读取两个真实 B8 旧 BF16-KV 轨迹的完整导出：train1 使用完成的 `export_retry1`，train7 使用完成的 `export`。4层×4步×8题×2rank×2文档，共512文件、8192个查询行。

- 8142/8192 行数值精确重构；50个元素非精确。
- 最大绝对表示误差 `3.814697265625e-6`；最坏单文件 relative-L2 `4.25814e-8`。
- 未初始化 CUDA；逐文件 SHA 保存在[结果](../results/platform/query_fp8_components_20261017.json)。

这不是任意 BF16 的精确分解证明，小值/subnormal 确实会丢失。即使 Q 数值重构精确，两个 MMA 的累加次序也不同，不能推出 QK、attention、logits 或生成相同。这些旧轨迹不是当前 FP8-KV 原生模型状态。新 FP8 捕获须独立检查。

[`compile_query_components_cpu.py`](../scripts/compile_query_components_cpu.py) 离线编译 SM120：M32/M64×BF16-QK控制/双FP8-QK候选，以及 O-only merge，共5个 kernel，通过且未初始化 CUDA。PTX/cubin hash、MMA 指令和共享内存见[编译收据](../results/platform/query_components_compile_20261017.json)。编译中最初 FP8 masked load 的整数字面量和 JSON metadata 序列化失败已在独立文件中修正；这不是 GPU 资格。

## 最小 GPU 否证方案，未运行

先用新原生 FP8 B8 的同一真实 Q/K/V，限制4层、2文档、2rank。共同前缀是原物理页共享部分；CPU 紧凑导出仅供离线算子实验，不能免费计入引擎。

1. 同一 Q/K8/V8、相同 chunk1024、N64，比较 M32/M64 的两种 QK；双方 O-only、相同 BF16 PV。另保留原生 FlashInfer 共享 prefix FA2/cascade，防止只赢自写控制。
2. 先测 QK 相对 FP64 的误差，再测 attention 相对相同量化字节的 FP64 完整 prefix 输出，并对照原生/自写控制误差。记录每层/行失败，不以平均 cosine 掩盖异常。暂不替换整模型。
3. 计完整 producer+merge，包含所有 Q 分解；冷热缓存、图重放、初始化/JIT和workspace分别报。所有 variants 预声明，不按输入逐点选优冒充一套实现。
4. 若候选未优于同形 BF16-QK控制，停止这条数据流；若只赢同形控制但不赢成熟共享实现，不进入整模型。若跨输入有稳定空间，才加入页寻址、private tail、真实逐层尺度和模型级数值/自然质量试验。

目前保留的只有这一个可证伪机制假设，成熟 cascade 的结果优先于本原型。正负结论均不能由 CPU 表示误差或离线编译直接代替。
