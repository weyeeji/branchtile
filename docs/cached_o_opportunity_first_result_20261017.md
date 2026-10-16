# 保持真实输出一致的整模型 attention 替代诊断

首次原文档B3、BF16 KV、TP2实验成功。它回答“如果当前真实decode状态的attention结果已知，完整模型图能省多少”，不提供可部署方法。所有图在新真实请求之前构建，随后于第三个真实B3 decode步内部冻结同一tokens、positions、tables和KV地址。

两个rank各48层Q/O逐bit一致，最终hidden、local vocab logits逐bit一致，rank0完整gathered logits逐bit一致；实际被写入的KV槽逐bit相等，最终inputs/positions/block tables不变。保留KV写入、投影、MLP、TP通信，只有attention核心返回预存O。witness和逐层检查图不参与性能对照。

| 条件 | 原生完整模型图，max-rank中位数 | cachedO完整模型图 | 配对速度比，95%bootstrap CI | 中位数耗时下降 |
| --- | ---: | ---: | --- | ---: |
| 模型热重放 | 8.335 ms | 4.873 ms | 1.7111 [1.7083, 1.7141] | 41.54% |
| 图前256MiB人工flush | 8.353 ms | 4.893 ms | 1.7085 [1.7069, 1.7096] | 41.43% |

每种条件12个block，每block随机与反向顺序，各自运行native/cachedO/nativeAA。两rank各48行原始测量；AA在两种条件的CI均包含1。CPU分析保存各rank独立统计和每对max-rank事件值，没有将两rank时间相加。

这给出比kernel累计时间占比更有信息的干预证据：前述同类B3 BF16 profile的attention/kernel累计约26%，存在PDL依赖重叠；本次在下游输出一致时，实际完整模型图的可替换部分约41.5%。不过两者时间窗不同，不能把其差值精确解释为PDL成本。

**限制**：只测单个真实状态。缓存O通常不能用于下一token；不能据此声称自然输出加速41.5%。替代图仍有O copy、图依赖结构改变，是乐观实验参照，不是严格数学上界。没有测自然请求或调度walltime，未摊销图构建/witness成本；完整KV只核对当前写入槽，不伪称对整个pool做checksum。

对研究方向的含义：真实整模型解码仍有明显可优化空间；当前owner方案失败不能推出attention方向没有空间。需要设计可以实际计算正确attention的新机制，并在上述可替换空间里兑现一部分，而不是把预知O当成方法。

复现CPU统计：

```sh
.venv/bin/python scripts/summarize_cached_o_diagnostic.py \
  --input results/cached_o_diagnostic_v1/original_bf16/summary.json \
  --output results/cached_o_diagnostic_v1/original_bf16/analysis.json
```

源版本 `869e6c3`；B8独立扩展为 `c56eb85`，扩展尚待自身bitwise资格与测量，不能从B3结果推断B8。

## 四个已完成条件：FP8与B8扩展

后续原文档B3 FP8和train1 B8两种精度均clean exit0，并通过同样全48层Q/O、hidden/local logits/rank0 gathered logits及当前写KV槽逐bit资格。四个case共有8个rank、384组逐层Q/O检查及384行配对计时，所有native AA区间包含1。

| 场景/精度 | 热图原生→cachedO | 热配对速度比及95% CI | 热耗时下降 | 人工冷耗时下降 |
| --- | --- | --- | ---: | ---: |
| B3 / BF16，86.8K | 8.335→4.873 ms | 1.7111 [1.7083,1.7141] | 41.54% | 41.43% |
| B3 / FP8，86.8K | 7.135→5.026 ms | 1.4196 [1.4178,1.4213] | 29.56% | 29.59% |
| B8 / BF16，120.5K | 14.224→4.646 ms | 3.0632 [3.0602,3.0647] | 67.34% | 67.21% |
| B8 / FP8，120.5K | 11.436→4.659 ms | 2.4558 [2.4500,2.4579] | 59.26% | 59.08% |

后两项分析分别在 `results/cached_o_b8_diagnostic_v1/train1_bf16/analysis.json` 和 `train1_fp8/analysis.json`；原文档FP8在 `results/cached_o_diagnostic_v1/original_fp8/analysis.json`。B8与B3文档长度和题目不同，不构成纯batch scaling消融。

**B8场景的固定整模型图有很大机会，但仍不能推出自然任务3倍加速。** 同一train1的自然EOS测量中，8题共同活跃仅BF16的291步/2612总步、FP8的773步/3022总步；冷文档还需约20秒anchor。候选需要支持逐步变化的Q、private tail、B7…B1，并支付元数据/切换成本。实际共享机制只能兑现缓存O乐观参照的一部分。
