# BranchTile 文献核验与实验解释（2026-10-16）

核验范围是一手论文与公开作者仓库入口。提案六个 arXiv 编号均真实存在；论文未明确描述某机制，不等于源码证明其不存在。下述代码入口已核实，但没有完成核心源码逐行审计。本文区分直接核实、由相关论文转述的线索，以及本项目实验。

| 工作 | 已核实机制 | 对本项目的含义与限度 |
|---|---|---|
| [QUILT，2610.11134v1](https://arxiv.org/html/2610.11134v1)，2026-10-08 | 稀疏 prefill；SCSD 构造交集/差集并与 attention 流水重叠；递归提取不同 query 子组共享区域。共享尾块下推保持精确语义；最终私有尾裁剪改变支持集。实验在 Ascend 910C。 | 同 mask 对照必须关闭私有尾裁剪，同时允许精确尾下推。最佳固定 partition 不能替代层次共享。正文没有足够细节判定其是否实现同一 CTA 内动态紧凑 query 身份与 canonical 状态；不能武断假设它每区域写全局 partial。本轮未定位作者公开源码。 |
| [PageWeaver，2610.11201v1](https://arxiv.org/html/2610.11201v1)，2026-10-08 | GPU 有界重分组生成 query IDs。H200 Union8 由两个 CTA 各处理四个 query；固定 CTA 子组跨页保存 FP32 状态、直接写最终输出。整个子组不访问某页时可跳过算术。另测 KV-page pair 的 partial/merge 路径。 | 固定组跨页驻留、ID 间接访问、masked union 和普通融合已经存在。论文中的 CTA query 身份在一次 traversal 内固定，没有明确描述逐 owner 区域更换紧凑身份。正文提及 artifact，但未定位作者公开 artifact URL；不能给出源码排除性结论。 |
| [PAT，2511.22333v3](https://arxiv.org/html/2511.22333v3)，v1 2025-11-27、v3 2026-03-16，ASPLOS 2026 | 面向 decode 的共享前缀 pack–forward–merge、多 tile、KV split 和多 stream，最后稳定归并。 | [作者仓库 flashserve/PAT](https://github.com/flashserve/PAT) 包含 CUDA 实现、Python 包、vLLM plugin 和测试。artifact 以 A100/CUDA 12.4 为环境，SM120 兼容性需实测。共享分组、tile 选择和小费用表不能单独作为新意。没有完成其 CUDA 源码审计。 |
| [Fused3S，2505.08098v1](https://arxiv.org/html/2505.08098v1)，2025-05-12 | 图稀疏 attention 的 SDDMM–softmax–SpMM 融合、在线归一化、片上状态、row-window 调度和寄存器布局 remapping。 | [作者源码 HPCForge/Fused3S](https://github.com/HPCForge/Fused3S) 存在。论文 §3.4 remapping 是 Q/K/V 特征维与寄存器布局置换，不能直接等同于跨 owner 切换 query 身份；融合和片上累积本身已有先例。decode/GQA/任意 token 索引兼容性未实测。 |
| [Fast KVzip，2601.17668v2](https://arxiv.org/html/2601.17668v2)，首稿 2026-01-25 | 轻量 sink-attention gate 保留重要 KV；冻结 LLM 权重，使用任务无关重构目标训练 gate；覆盖 prefill/decode。 | 属于应用质量—成本对照，不是同 mask 执行对照。[作者项目页](https://janghyun1230.github.io/fastkvzip/) 与 [NVIDIA kvpress 实现](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/fastkvzip_press.py) 可访问。代码按模型名加载 gate；未核实目标 Qwen3-30B-A3B-Thinking-2507-FP8 是否有匹配 gate，不能套用其他 checkpoint 的权重。 |
| [ActKV，2609.31395v1](https://arxiv.org/html/2609.31395v1)，2026-09 | 标题与正文结构已核实，研究 action-guided Agent KV 管理，包括 action-preferred entries、预算充分性和 page-aware compression。 | 提案的固定文档支持集不构成其动态生命周期复现。本轮未完成算法与源码细审，未定位可确认的作者仓库。同名第三方实现不能当作者代码。 |

## 应补充的直接相关工作

[LiteTopK / LiteDSA 作者仓库](https://github.com/Heisenberg-Yin/LiteTopK) 已确认存在。PageWeaver 明确将相邻 query 索引 union 与逐 query membership mask 归于 LiteDSA，参考文献给出 [LiteTopK，2607.11976](https://arxiv.org/abs/2607.11976)。本轮没有完成该仓库内核审计，不能断言它实现或没有实现 canonical 动态行压紧。固定 masked union 的收益应先表述为既有机制在本硬件/任务的效果。

PageWeaver 还引用 [MiniMax Sparse Attention，2606.13392](https://arxiv.org/abs/2606.13392)，描述其 KV-outer 路径将访问同页的 query 组织为有界工作单元并合并 partial。这是 owner/page 驱动执行的重要相关线索。此机制归属来自已读 PageWeaver，MSA 原文和作者代码尚未独立细审。[SGLang PR 37236](https://github.com/sgl-project/sglang/pull/37236) 是 PageWeaver 指明的 native SM90 基线来源，不表示本轮已确认 SM120 兼容。

原生软件也按查询日期更新：本轮实际测试 [FlashInfer 0.7.2，2026-10-15 发布](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2)。它在隔离虚拟环境中使用 FA2，未因引入新版本而更换 Torch/TileLang，见 [环境身份](platform/latest_flashinfer_environment.json)。论文新近发布、软件新近发布与我们完成硬件复现是三种不同证据，不能相互替代。

## 强对照与实验解释

应保留原生独立 attention、原生 joint masked attention、优化固定 union、B4 的全部 15 个固定 partition，以及精确层次共享机制对照。所有方法保持支持集、物理共享源、精度和完整调用口径一致。固定 union 必须允许跨 owner 边界填充 N tile；否则每区域补齐会人为削弱对照。

逻辑 M/N tile 与逻辑 KV 访问量应共同报告。最少 tile 的 2+2 固定组可能重复读取 KV，不能仅凭 tile 数认定速度最优。CPU 层次枚举只能称机会分析，不能称 QUILT 复现、运行时间或实际 DRAM 流量。本轮未完成全部 partition 与精确层次共享的 GPU 执行对照，见 [完整范围](../FEASIBILITY.md)。

本项目最终实测：动态行 v1/v2/v3 均未通过预定性能门槛；固定 union 的初步正向信号在加入最新 FlashInfer 原生 joint masked prefill 后明显收窄。官方 4B 四个热缓存 capture 的原生 raw 时间为 9.55/10.17/12.43/11.80 μs，自研 union 为 10.15/10.25/13.07/12.19 μs；统一自然对数输出口径后，自研 union 的配对加速比仅 1.028–1.102×，仍不足 1.15 门槛。数值来自 [实际配对结果及置信区间](summary/union_signal_latest.json)，不是引用论文的作者结果，也不是应用层加速。

这个结果与文献共同支持收缩结论：共享和融合可以改善较弱的执行路径，但尚未证明跨区域动态压紧 query 行有独立价值。PageWeaver 的 B300 实验也出现在线 regrouping 无法超过每输入更强既有后端的情况；不能把该现象直接外推为本机原因，但它说明强原生对照不可省略。

## 可行转向与新颖性风险

当前应停止围绕失败动态布局继续无界调参，完整交付已经测得的自然 owner 分布、尾块浪费、资源占用和调用结果。状态搬移的独立耗时归因、实际缓存/DRAM 计数和完整应用成本仍缺证据，不能由逻辑统计补造。现有证据可形成可复现的负向研究记录，不能包装成原 idea 已成功。

若继续研究，最具体的候选问题是：在不扩大支持集的情况下，跨 owner 区域填充 N 维短尾，同时约束 membership mask 和 canonical 状态访问成本。但这仍是待验证假设，必须先核查 LiteDSA/MSA，且必须击败原生 joint masked attention；不能把它简单改名为新 masked union。

固定组/独立路径的有限选择器可作为工程优化，但 PAT、PageWeaver 等已覆盖分组和成本权衡。仅增加阈值、小费用表、应用名称或共享物理 KV，不足以建立论文新颖性。在没有超过最强兼容基线的稳定自然形状收益前，不建议扩大到完整引擎、多模型和长周期能耗评价。

## 未完成核验项

QUILT/PageWeaver 的作者 artifact、LiteDSA/MSA 核心 CUDA 数据流、PAT/Fused3S 的逐行源码检查、ActKV 作者实现和目标模型 Fast KVzip gate 均保留为未核实项。本轮只能说“已读论文未明确给出与 BranchTile 完全相同的组合”，不能据此声称首创。
