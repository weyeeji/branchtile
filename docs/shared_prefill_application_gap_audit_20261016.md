# 同文档多问：共享预填准备的剩余空间审计

2026-10-16，CPU only，**0 GPUh；本轮通过新颖性门槛的候选为 0**。这只关闭下述普通共享准入提议，不代表全部应用优化不存在。已读旧 pivot 和 joint 停止记录；本轮不改模型、自然答案合同、指标或 formal eval，不安装、执行第三方代码。

可复算证据：[脚本](../scripts/audit_shared_prefill_application.py)、[逐 arm/重复/rank 核算](../results/shared_prefill_application_audit_v1/accounting.json)、[一手来源版本与 SHA](../results/shared_prefill_application_audit_v1/primary_sources.json)。脚本重现两个公开开发文档的分词，逐题核对冻结 token SHA、长度与文档边界；读取 4 arms × 2 cold repeats × 2 ranks 的既有元数据，CUDA 未初始化。所有数值只适用于这两个文档和当前部署。

## 已经复用的工作不能再记为收益

当前 `benchmark_native_precision.py` 每次清空 APC 后，先用完整 q0 prompt 预填并采样一个丢弃的输出，再原子准入同文档三个自然问题。**完整文档前缀已经只计算一次。** 该 anchor 约 10.7–11.9 秒包含必要的首次预填，不能全部作为新机制可消除的准备。`max_tokens=1` 从 prefill logits 采样，不意味着多执行了一次自回归 decode forward。

下表的“上限”仅是原输入里重复计算的 **prompt token 工作量**，不是延迟上限或已有可用 API 的保证。它等于 q0 再执行的最后页残余，加上两个 follower 未命中的精确公共前缀残余；没有把其真正私有的问题 suffix 算作可去除工作。

| 文档 | 完整 prompt 长度 | 精确共同前缀 | 三题 APC 命中 | anchor 后三题仍需预填 | 可去除重复 token 工作上限 |
| --- | --- | ---: | --- | --- | ---: |
| original / ANSS | 86850 / 86840 / 86839 | 86820 | 86848 / 86816 / 86816 | 2 / 24 / 23 | `2 + 4 + 4 = 10` |
| new / publicdev1 | 89802 / 89801 / 89797 | 89772 | 89792 / 89760 / 89760 | 10 / 41 / 37 | `10 + 12 + 12 = 34` |

其中真正文档内容的重复分别仅 4 / 20 tokens，其余公共尾部包含模板。两种 KV 精度、两次重复的 APC 命中一致。私有分支后的相同词不能仅凭 token ID 相同复用 hidden/KV。若做不完整页复用，还需保留分叉后的写隔离、位置、logits 与正确 copy-on-write；不能默认零复制成本。

本机 vLLM `kv_cache_manager.py:264–300` 明确仅复用完整块；全命中也保留末 token 重算以取得 logits，因此可能重算末块。`scheduler.py:3003–3037` 已有 in-flight prefill 剩余块预留。它们排除“普通块共享/容量预留是新机制”的主张，但不能由静态检查声称 vLLM 已完整实现任意 cold follower 合并。

## 当前可量化的 CPU 准备较小，排队机会尚未测量

| arm | 输入准备秒数 | anchor + 自然生成秒数，中位数 | 假设整个输入准备免费后的乐观占比 |
| --- | ---: | ---: | ---: |
| original BF16 KV | 0.816 | 23.604 | 3.34% |
| original FP8 KV | 0.842 | 20.621 | 3.92% |
| new BF16 KV | 0.838 | 42.486 | 1.93% |
| new FP8 KV | 0.811 | 29.127 | 2.71% |

最后一列仅为 `prepare / (prepare + median(anchor+generate))` 的条件核算；不是综合 client JCT 测量。输入准备包含分词器加载、模板渲染、重复分词、offset 和 hash，不能全部归为可共享的重复成本。精确分词缓存是合理工程控制，但不能称为新贡献。表中不同精度改变了自然输出长度和设备分配，不用于隔离执行器速度。

三个请求的 CPU 身份 hash 每 rank/组合计约 18–19 ms；这还是仪器开销，去掉也不是产品机制。已有 `prepare_attn` 主机调用累计每 rank/生成约 0.157–0.487 秒，详见 JSON；它不包括全部 prepare_inputs，也不是未重叠的关键路径。不能把两个 rank 加起来，也不能直接从 JCT 扣除该和。

自然 EOS 导致 batch 从 3 降为 2/1，原生执行已随实际 ready 集合变化。现有轨迹没有独立文档到达、队列延迟、缓存压力或多副本放置的干预；“用别的文档填空”属于连续批处理/调度，当前没有新的可预测收益依据。八张同型 GPU 可以做四 TP2 重复，不能凭设备空闲假定应用有额外 ready 请求。

## 唯一具体提议及排重结论

被筛查的提议是：同文档请求共用一次 admission/prefill，follower 等待 leader 发布精确公共前缀，缓存引用和 suffix 容量提前保护，再分叉自然生成。**未进入 GPU：这是成熟共享准入机制，且本协议已经由 anchor 消除了其主要重复工作。**

- [WavePP v1，2026-09-28](https://arxiv.org/html/2609.35263v1) 的 §4 已组合异步准入、跨 stage 前缀一致、local lease、suffix capacity escrow 和动态 wave。论文的 PP/独立 stage cache 场景与本机 TP2 有区别；这里没有同类跨 stage 一致性问题，因此把机制简化移植不是新颖性依据，也不能移用它的吞吐数字。
- [Gufo 官方文档，固定 commit ec69aa3](https://github.com/gufo-org/gufo/blob/ec69aa328ea78f00090c42a78034cc012eae151e/docs/KV-CACHE.md#concurrent-requests-sharing-a-prefix) 直接描述 cold follower 等待正在预填的 leader 在共享位置保存 checkpoint，再只预填私有尾；等待不占 session，失败则自行预填。这是公开实现文档先例，非本机 TP2 资格或同模型性能证据。该文件最后提交为 2026-10-15；没有依赖晚于截止日的版本。

因此不申请 GPU 预算。若另为工程基线排除原生 cold B3 的重复，最低对照应是同 token/模型/精度下的 cold B3、当前 anchor+B3、leader 保持为真实 q0 的原生准入；包括首次 prefill、所有排队/准备、两次独立冷重复与自然 EOS。冻结全部题分母，不能只计 follower TTFT，不能通过延后兄弟请求制造原合同不存在的并行，不能把相同随机 seed 当作改变调度后的逐位输出保证。该实验即使正向，也首先是 baseline 完整性检查，**当前不是待跑新论文候选**。

## KV² / EchoPress 的新增约束

[KV² v1，2026-10-02](https://arxiv.org/html/2610.03198v1#S3) 从完整预填 cache 出发，用轻量 proxy 选择文档 token，再把它们追加于 cache 后进行真实因果 reconstruction，提取局部 chunk 的最终淘汰分数。它减少的是额外压缩重构；不是跳过首次长前缀预填，也不只是对本项目已有 suffix Q 重评分。论文默认 4K chunk、选择比例随保留比例设置，实验主要是 HF/FA2 的 8B 模型与 A100/H100。它应加强 query-agnostic 压缩对照，排除“少量公共探针重构是新方法”；没有直接资格化本机 FP8 MoE TP2/SM120。

[EchoPress v1，2026-09-30](https://arxiv.org/html/2610.00412v1#S2) 利用 prefill Q/K，加入 RoPE 位移和虚拟因果副本，首 chunk 做精确重构，以逐层/head quantile map 校准后续分数。§3.3/附录 B 的加速相对 KVzip 的**含压缩预填**，其 plain prefill 仍存在；不能将约 2.9× 套到当前 native anchor。它已覆盖“复用预填状态、少量精确校准、摊薄共享 cache 准备”这一常见转向。实际压缩时间包含校准、选择和 compaction；现有 suffix_v2 保存的是问题 observer/稀疏 decode anchors，不能免费当作它的全篇 prefill Q。文中的低预算质量不能外推本机 89K–175K 金融推理。

两者都改变保留 cache，可能改善旧 QC 的低预算质量，但需要独立、完整成本的成熟 baseline 资格。它们不保持原 dense 数学合同，也不证明本项目质量门槛；本轮只核论文一手描述，未审计或执行其实现。新增文献进一步缩小重构共享的可主张空间，**不改变当前预填准备分支的停止判断**。

复算：`CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_shared_prefill_application.py`。冻结输入 SHA、完整逐 rank 数值与外部文本 SHA 已保存；没有新增 P1/P2 事件。
