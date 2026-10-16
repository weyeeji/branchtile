# XQA 两请求填充 M16：不等尾长与原生顺序的必要合同

这是给最小算子诊断的源码审查，不是恢复已经关闭的泛化 bitwise 调度方向，也不构成新颖性结论。已有 `results/trajectory_stable_shared_dataflow_review_v1/review.md` 中 beam 和占用率风险仍适用。

## 实际原生 split 的含义

本地 `.venv-engine/lib/python3.12/site-packages/flashinfer/data/csrc/xqa/mha.cu` 的 `launchMHAFlashInfer`（约3123行）按整个batch的 B、Hkv、maxSeqLen 一次确定 S：`min(max(1, SM/(B*Hkv)), ceil(maxSeqLen/256))`。不是按每请求实际长度分别选S。没有sliding window时，subseq s读取的CTA tile是 s、s+S、s+2S……；每请求自身的实际长度决定循环何时结束，以及最后一块中哪些列有效。

同一原生batch内的不等长请求，因此可共享prefix部分的tile遍历顺序。把B8先变成B4再让原生函数自动计算S会改变它；pair接口必须仍按原始请求数和原始maxSeqLen保留S，才能讨论同顺序。B8/Hkv2/170SM时S10，pair后仍S10意味着CTA数约减半，可能损失占用率；不能为追求occupancy增加S后还宣称同一归约。

## 原有 beam 不自动支持 agent 的不等长度

`mhaUtils.cuh` 的 `getSeqLen`（约264–275行）只读取第一个beam的长度，并对其余beam逐一assert相同。`mha.cu` 中 `cacheSeqLen` 是CTA标量；它决定页数、`nbSeqIters`、K/V异步加载有效长度和`applyMask`的末列。已有BEAM_WIDTH>1共享prefix和beam尾indirection路径，所以“打开beam2”本身不是新机制，而且不能直接覆盖不等agent尾长。

两请求路径若支持不等长度，至少需要：

- 独立的每请求长度和尾部页表；逐行mask只解决数学支持，加载边界也必须各自正确。
- 每行保持原生同一个s,s+S遍历顺序。较短行完成后冻结其max/sum/PV状态，不应额外执行任意空块重标定。
- 只把两个请求完全一致的物理KV区间当共同数据。原生CTA tile宽256，page16对齐不代表prefix结束恰好处于CTA边界。首个混合tile应按两行的真实地址处理，不能把prefix余块和private尾各算一遍再merge。
- 末tile各行的有效列严格匹配原生。对一个请求有效、另一个无效的列不能共享同一个标量validTokenEnd。
- 最终scratch仍按原始subseq编号读入归约；同一算子调用的原始scale、FP8解码、概率截断、max/sum归约路径均保留。

M16 tensor-core计算本来就有16行，填原padding行不一定增加矩阵尺寸，但 `nbValidRows` 从8变16会改变Q拷贝是否full-tile、有效scratch布局、输出拷贝循环等编译路径。因此“同M16”不足以证明原逐行指令顺序和资源不变；需检查编译后寄存器/SMEM和实际逐行bit结果。新增第2请求的地址/尾mask也可能增加寄存器。

## 有界小资格建议

只用GPU1张、两层实际capture或独立随机qualification，不先接整模型。先原生A/A重复10次确认该固定输入自身可重复，然后对每行检查native与pair O；数学正确性另外用FP64，不能仅因bitwise失败就称数学错误。

固定原始B8、Hkv2、S10，至少覆盖下列长度配对。P先取256对齐，再取仅16对齐的P+16，以覆盖共享/私有混合tile：`(P+1,P+1)`、`(P+1,P+15)`、`(P+1,P+257)`、`(P+256,P+257)`。改变页物理地址但保持内容、调换pair内请求、去掉单个请求后的奇数B处理，也要单列。测试中不通过新B重算S来掩盖顺序变化。

资格后仅做同S的完整O算子成本对照，并记录有效CTA、寄存器、SMEM。若内存读减半却因CTA不足无加速，直接记录负结果。若只有改S才快，另作数学近似/浮点次序不同的普通算法结果，不将其包装成trajectory-stable优势。跨层LM token一致性不是这个小实验的前提，但不能由局部bitwise通过外推整模型性能。
