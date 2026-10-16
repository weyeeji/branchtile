# 判断：保持原生 XQA 轨迹的共享数据流

2026-10-17。仅 CPU 源码分析和一手文献检索，没有启动 GPU，没有实现或验证新的加速核。结论：**作为兼容性约束及有上限的小实验值得保留，暂不升格为主论文方向。** 目前“只共享 KV 读取，保持每行算术”本身既缺少新颖性，也有实际并行度代价。不能因为某个种子自然输出变短就选实现，更不能通过选择种子制造收益。

## 必须区分的三个契约

| 契约 | 要求 | 与当前设想的关系 |
|---|---|---|
| 同配置可重复 | 同输入、同批次、同硬件/软件再次运行结果相同 | 是必要检查；不保证不同实现之间相同。 |
| Batch-invariant | 某请求独立于其他同批请求和批大小 | 既有方向；可能需要替换 GEMM、归一化、attention及通信算术。 |
| 原生兼容的共享读取 | 在**同一当前批次和调度条件**下，候选 O 逐比特等于冻结版本原生 XQA | 比两个实现各自确定更严格；又比所有批大小恒等更窄。原生自身随 B 改算术时，候选必须跟随。 |

Thinking Machines 已在2025年明确固定每行归约和 fixed split-size 来实现 batch invariance；FlashInfer 已公开 `fixed_split_size`/`disable_split_kv`，vLLM 也已有 batch-invariant 模式。这些都不能当成我们的新机制，也不自动承诺与当前默认 XQA 的比特一致。[Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)、[FlashInfer API](https://docs.flashinfer.ai/api/attention.html)、[vLLM](https://docs.vllm.ai/en/latest/features/batch_invariance/)。

## 2026 年的直接相关工作

- **Taming Bitwise Behavior，2026-09-10，arXiv:2609.11356**：已有归约描述、重建供应商 GEMM 算术、编译器约束和比特等价类内调优。其 §3.3/§8 明确 attention 主要给理论描述、实现留待后续。因此“固定归约还能优化”不是新点；面向真实分页共享历史、增长后缀和退出请求的完整 attention 接入仍可能有工程切口，但不能据一次检索声称首创。[论文](https://arxiv.org/html/2609.11356v1)。
- **DASH，2026-01-29，arXiv:2601.21824**：确定性 attention 的训练反向调度，研究固定累加依赖与硬件调度的冲突。不是本项目的解码共享前缀，但排除了泛称“调度与数值次序解耦”为独有贡献。[论文](https://arxiv.org/abs/2601.21824)。
- **VeXact，2026-05-14，arXiv:2605.14220**：训练与推理概率对齐，说明数值一致性存在真实应用需求；它不是“任意当前原生 XQA 加速版”的直接保证。RL 也不能只靠改 attention 就声称零 mismatch。[论文](https://arxiv.org/abs/2605.14220)、[实现](https://github.com/verl-project/vexact)。
- **PageWeaver，2026-10-08，arXiv:2610.11201**：对固定稀疏支持做页面亲和查询分组、保持查询输出归属，报告算子收益与较小的在线增量收益。这与旧 owner/query-union 路线高度相关；它不提供本设想的原生逐比特兼容保证。[论文](https://arxiv.org/abs/2610.11201)。

已有 Hydragen 和 FlashInfer Cascade 都使用共享前缀分解、查询合并和状态合并，数学等价不等于匹配另一内核的浮点轨迹。当前成熟对照已经验证有算子空间，而不是证明这种分解具有新颖性。[Hydragen](https://arxiv.org/abs/2402.05099)、[Cascade](https://flashinfer.ai/2024/02/02/cascade-inference.html)。

## 本地 XQA 的关键约束

精确本地文件 SHA 见 `source_audit.json`。以下是已安装版本的源码事实；外部 main 链接用于定位项目，不代替本地冻结版本。

1. `mha.cu:3123` 的实际 `launchMHAFlashInfer` 选 `S=min(max(1,SM/(B*Hkv)),ceil(maxSeqLen/256))`。CTA在KV维度按 `s,s+S,s+2S…` 交错遍历，随后按固定子序列索引合并。默认 XQA 因而不随 B 保持同一算术。不能把成组后的 batch 传给原 launcher 再声称保持原生结果。
2. 必须保持 QK/PV 的 MMA 指令、每行 tile 边界、在线 softmax 更新位置、量化尺度、转换和舍入点、split 内顺序及最终 merge 顺序。重新拆 prefix/tail 后额外合并 LSE，即使数学完整，也破坏这个原生兼容契约。
3. **已有 beam 复用**：`mha.cu:1809` 的 `isConvergedTile` 与后续 `BEAM_WIDTH>1` 分支在公共 context tile 复用 KV，分叉后按 beam 间接索引。`jit/xqa.py` 当前固定 `BEAM_WIDTH=1`，公开接口也只支持1。直接开放旧 beam 路径属于已有功能移植，不能包装为新加速原理。普通独立请求还需处理不同长度、真实分页表和退出，而非假设 beam 同步。[XQA kernel](https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/xqa/mha.cu)、[JIT](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/xqa.py)、[API](https://docs.flashinfer.ai/generated/flashinfer.xqa.xqa.html)。
4. Hgrp=8 时，原生 row tile 补齐至16；两请求填入16行可能复用同一KV读取和原来空闲的计算行，但第二请求映射到高8行是否仍逐比特等价，必须实测，不能凭数学独立性认定。旧 beam 路径也已涉及此类布局。
5. 原代码使用有屏障的 `CTA_ROW_MAX_BACKWARD_METHOD=1`。旁边注释的无保护 method4 是另一方案，当前未启用；不可把其非确定性当作我们的实际基线行为。

### 一个会抵消收益的结构性问题

RTX5090既有收据为170 SM，Hkv=2。长上下文原生CTA总数接近170：B8的S=10、CTA=160；B4的S=21、CTA=168；B2的S=42、CTA=168。若每两个请求融合且必须保持原S，CTA分别变80、84、84，只有约半数SM有线程块。这不是性能测量，却说明“省一半读取”不能直接推导“快一倍”。为恢复并行度而提高S，会改变原生浮点遍历/合并。

保留各CTA而跨CTA共享暂存需要额外通信、同步和硬件条件；不能凭Hopper/SM100设计推定SM120相同。另一低成本变体只重排CTA访问顺序以改善L2局部性，保留每个CTA的算术；但这属于常见线程块调度技巧，无法保证实际执行次序或省掉多少内存流量，单独新颖性弱。

## 是否继续及如何判定

**不建议现在大规模投入新的 bitwise XQA 内核。** 如果其他路线暂时没有更强结果，可将它限制为一个明确的小原型，贡献目标须是“在兼容原生算术的约束下，仍能实现真实共享读取及端到端增益”，而非“避免浮点误差”的笼统说法。

首个原型只接受两种有限探索：保持CTA算术的地址/线程块调度，或填充Hgrp8原生16行中的空行并保留原B决定的S。每个原型必须先经过：原生AA逐比特自检；同Q/真实KV候选逐比特O；B1/2/3/4/8、不同尾长、物理重映射和退出；全48层O及同teacher logits；自然生成逐token检查。若任何环节有差异，就只能归类为数值近似兼容，不能继续称trajectory-stable。

即使 O逐比特相同，任意在线请求流下的完整轨迹也未必相同：速度改变可能改变同时到达请求的批次，其他算子及采样状态也需要一致。因此首先仅承诺固定已就绪请求组、相同引擎调度和按请求固定RNG下的兼容性。若要跨负载承诺，则需与batch-invariant整套系统比较，不能只做attention。

停止条件：只有修改split/归约才能获得收益；少量共享读取收益被并行度/通信/准备成本抵消；或最后只是旧beam支持的开放且没有更普适机制。现阶段这些风险都很实在，故它应保持次要实验而非靠“稳定”一词转成新论文故事。

下一轮统计必须同时对原生与候选使用预先固定的多个seed和请求集合，随机/交错实验次序，报告文档级成组JCT、anchor与一次性准备、EOS/长度分布、冻结质量指标和逐题得失。固定工作量测量隔离实现速度，自然生成测量用户实际等待，两者均保留；不挑选更短的候选轨迹作为收益证据。
