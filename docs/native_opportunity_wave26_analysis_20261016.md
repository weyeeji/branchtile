# 实际原生 XQA 机会测量：wave26

四个 TP2 作业完成，两个开发文档 × BF16/FP8 KV；实际 vLLM full-history、FlashInfer/XQA、APC、48层，未执行新候选。本轮给出真实基线与瓶颈证据，不能称为候选端到端加速。

## 完整引擎测量

| 文档 | KV | 固定128 tokens×3请求，两次完整generate秒 | 自然generate秒 | 共享文档anchor秒 | 冷文档总秒 | 自然每题输出tokens |
| --- | --- | --- | ---: | ---: | ---: | --- |
| c0f11a | BF16 | 1.408 / 1.394 | 12.859 | 10.788 | 23.647 | 1461 / 696 / 909 |
| c0f11a | FP8 | 1.265 / 1.258 | 9.235 | 11.462 | 20.697 | 952 / 819 / 1131 |
| 593ec0 | BF16 | 1.431 / 1.438 | 31.112 | 11.373 | 42.485 | 1511 / 1681 / 3540 |
| 593ec0 | FP8 | 1.247 / 1.251 | 17.271 | 11.886 | 29.157 | 1252 / 1388 / 2292 |

12/12自然输出均 stop，没有达到8192 cap。固定AA的 tokens、APC命中、两rank调度(batch/questions/positions)全部相等，墙钟max/min为1.0031–1.0101。只有两个重复，不能构造稳健统计结论。跨精度输出轨迹不同，即使固定长度也可能影响MoE专家路线，因此上述跨dtype差异不应命名为相同工作量加速。自然文本没有在本轮完成人工质量评分。

所有普通墙钟均不启用torch profiler，也不插入CUDA events；已有CPU原生metadata/identity hooks仍计入。共享文档冷总时包括一次anchor及整组生成，初始化分开报告。这是同步ready-group，不代表在线到达分布。

## 原生 attention 占比与一个关键计时现象

实际元数据确认 `FlashInferTrtllmAPIDecode / FlashInferDecodeKernel.XQA`，`KVCacheLayout.LBNHC`、`use_cascade=False`；所有48层均为FlashInfer。XQA导出的符号是 `kernel_mha(...KVCacheList...)`，名称不含xqa；已在当前engine安装的 `flashinfer/data/csrc/xqa/mha.cu` 核实，分析脚本保存源码SHA。每个rank恰有 **1536=32×48** 个匹配attention kernel，8个trace合计12288次。

| KV | Attention / kernel累计时间 | Attention累计 / trace首末kernel跨度 |
| --- | ---: | ---: |
| BF16 | 25.31%–26.02% | 33.72%–34.20% |
| FP8 | 19.55%–20.10% | 23.24%–24.00% |

**两个分母均不是可直接删除的端到端成本。** Trace显示明显重叠：例如原文档BF16 rank0，第一个MHA持续69.76µs；紧随的量化kernel仅晚0.83µs启动，报告持续71.01µs，与MHA基本重叠。它们具有同stream、同graph、连续node id。当前XQA源码包含 `ENABLE_PDL` 下的 `preExit/acqBulk`，工具函数包含 `griddepcontrol.launch_dependents/wait`；这与依赖启动重叠相符，但仅凭trace尚不能证明每个量化kernel的等待分解。

因此，不能把量化kernel的71µs再算成独立计算成本，也不能依据kernel累计占比约26%认定attention真实机会只有26%。分析保留kernel sum、union busy、span和所有名称，避免相加重叠导致错误判断。进一步可靠的可替换边界需要保持下游输入完全一致的干预。

单独profile固定128段的墙钟5.7–6.1秒，远高于无profile约1.25–1.44秒，包含profile控制/导出等开销；它不是运行性能。模型图events记录127个B3步，但trace仅截取32步，分析不混用两个窗口构造占比。

## 固定真实状态、缓存 attention O 的后续诊断设计

这项尚未实施。目标是固定一个实际decode状态，在两rank同步测量相同完整模型输入、位置、KV映射和MoE输入时，替换attention核心可省多少。

1. 预先创建三个独立图路径：原生完整模型；原生attention并把每层O复制到持久缓存的witness图；每层返回该O缓存的替代图。所有图保留KV写入、投影、MLP、通信与输出拷贝。仅witness负责采集，不能拿有额外48次copy的witness时间当原生对照。
2. 在真实B3 decode调用内部、scheduler尚未推进时冻结实际tokens、positions、slot mappings、block tables、metadata和必要输入缓冲。让witness运行一次采集48层O；缓存O不随诊断计时刷新。TP两rank必须走同一有序RPC/同步序列，避免collective失配。
3. 先在同一冻结状态重放原生与替代图，逐层检查attention输入/输出、最终hidden及logits；KV写入仍执行且应幂等。MoE路由要求完全相同，不能仅凭最终argmax相同就视为同轨迹。
4. 对冻结状态做原生/替代/原生AA的随机与反向成对计时，报告各rank CUDAevents及同步TP墙钟。保存参数、输出哈希、KV变化检查、图构建开销及原始配对数据。
5. 得到的是**固定真实状态的乐观可替换边界诊断**。缓存O在下一token通常无效，不能用于自然生成、质量或宣称可部署加速。copy成本和图结构变化意味着它也不一定是严格数学上界，应称乐观实验参照。

实现风险是新版 `ModelCudaGraphManager.capture` 会构造dummy输入和metadata，必须在实际状态捕获前完成图构建，或完整恢复输入/metadata；不能在原生模型状态已被dummy capture覆盖后直接声称冻结真实状态。现有共享计划图可提供接口经验，但不能免除上述逐层和logits验证。

可重复分析：`.venv/bin/python scripts/summarize_native_opportunity.py --root results/native_opportunity_wave26 --output results/native_opportunity_wave26/analysis.json`。旧sparse20–25%历史证据仍独立保存，不能与本轮dense XQA混用。
