# 官方主模型 A4 稀疏执行资格计划

本计划继续验证只读共享源是否存在应用价值；已有动态行 v1/v2/v3 的负证据不变。E0 与固定 union 都是已有机制，不能据本集成重新声称动态行贡献。

## 固定环境与数据

- 使用项目 `.venv-engine` 的已通过真实主模型 TP2 部署的 vLLM 0.31 环境，不修改系统或原安装包；独立 worker extension 实施实例级适配。
- 官方 `Qwen3-30B-A3B-Thinking-2507-FP8`，revision `60d80c83c53c3b611c642dbb8c942b3f90c5948a`，BF16 KV、真实 TP2/GQA8，每 rank 2 KV heads。
- 先使用当前冻结开发报告的自然 B3；完整文档、原 prompt/tokenizer/RoPE、自然 EOS、不补第四题。算子检查不能当作正式应用质量。
- 源完整页结束于 86816、文档结束于 86818，最后两文档 token 用本请求实际私有 KV 并计入保护预算。

## 可执行接口

1. 复用 `scripts/vllm_rank_capture.py` 已验证的 `prepare_inputs` 包装、实际 request ID/idx_mapping、position 和 per-layer metadata 读取方法，新增独立 worker 文件，保留原 capture 实现。
2. vLLM 0.31 FA 后端 `forward_includes_kv_cache_update=False`；`do_kv_cache_update` 已在独立操作写入真实 cache。只包装各 attention 实例的 `impl.forward`。正常纯 decode 批次调用新的稀疏 executor 后写入既有 output buffer，直接返回，**不执行原 dense attention**。
3. dense prefill 保持 proposal 输入合同。在每层每请求 suffix prefill 完成后，保存最后 `min(32,suffix长度)` 个真实 post-RoPE Q，按既有 `selector_baseline` 语义固定历史 mask。不得使用 dense 自由生成后续 Q 代替稀疏轨迹 Q。
4. APC 跳过的观察位置，只可复用已执行过且原 token 前缀与物理页身份均一致的 Q；逐位置验证、记录命中与缺失，缺失必须显式失败或另行计费重算。不能借用文档 Q 补观察窗。观察缓存的存储/生命期属于准备费用。
5. 当前 FA 物理 cache 的逻辑形状为 `[physical_page,Hkv,page_tokens,2D]`，实际主模型 stride 是 `[8192,256,512,1]`，不能使用 contiguous `view` 或假定 HND 索引。已实现以 `as_strided` 暴露原 storage 地址跨度，保留真实 `storage_offset`，K/V 两个半区组成 `[physical_token,1,1,D]` 页1别名。物理索引为 `(page_id*stride0 + head*stride1 + offset*stride2)/(2D)`；跨层空洞不进入有效索引。独立原 cache 坐标 FP64 与实际 FA 对照验证地址、私有末页和无执行 KV payload gather/copy。
6. E0 将 `(request,KVhead)` 展成 batch，GQA8 不变，逐行索引为系统前缀、冻结文档 mask、完整问题/生成 suffix 的去重集合。fixed union 以 KV head 为 batch，对同 ready 组的物理 token ID 求并集，并使用精确 per-request custom mask；Q gather 和 output scatter、native plan 全部计入。
7. 每步从实际 block table/seq_len 构造增量 suffix；任何 alive 子集必须自然支持。两种执行方式使用同一份不可变历史 mask，不能按候选更改预算或选择轨迹。
8. 混合 prefill/decode 批次不能默默让 decode 走 dense。首版可明确记录仅混合批次的 dense-prefill-plus-sparse-overwrite 冗余费用以维持准确语义；纯 decode 必须完全旁路原 dense。若无法做到，停止该资格配置并报告限制。

## 顺序资格与终止条件

- 首先仅对物理 cache 页1视图做小形状 FP64 输出/LSE 检查和原 cache 索引核对，包含非整页、私有末页、空/退出请求。
- full-budget 在相同 prefill 下检查每层输出与原 FA、TP 后 logits 和短 greedy 轨迹；同时保存 dense A/A 作为本环境数值噪声。任何不一致先定位，不能扩大 tolerance 来换取通过。
- 再测冻结稀疏预算 E0 与 fixed union 的每层输出/logits、teacher-forced 同 token 轨迹和自由生成 EOS；自由生成可因 FP 运算序不同分叉，须单独报告质量，不把逐 bit 不同直接视为数学错误。
- 计时分清 selector、观察 Q 保存、mask/plan、索引更新、完整 decode step、prefill、峰值显存与自然结束。单步各 kernel 和不可直接从重叠/JCT 中相减。
- 先做一组有界资格，不预设稀疏 A4 快于 dense。若完整费用没有可辨净价值或即使移除可替换 attention 也不能接近质量合格替代，则停止该配置；不无限优化 Python metadata 或重包装已有 union。

## 已完成阶段与证据边界（2026-10-16）

- [物理 cache 资格](../results/main30b_sparse/simt_paged_executor_qualification.json)：B1/B2/B3、连续和跨层交错 storage、改变 Q/cache，36 条输出检查全部通过。SIMT 的 12 条严格 LSE 检查全过；其他原生 TC 路径有严格 LSE 超差，原始记录保留，没有放宽。FI 原生 LSE 为 log2，检查时转为 ln；真实模型只需要 O，不添加 LSE 转换工作。
- [独立训练报告原生校准](../results/main30b_sparse/native_calibration_summary.json)：以此前未用于候选调参的公开训练文档 CCI_2003/HUM_2005 做 full-history FA、FI SIMT/TC 和独立 FP64 层探针。HUM_2005 最初 retained-cache A/A 不匹配，保留为失败归因证据；每 pass reset + 相同 anchor 后 FA A/A logits exact。TC 的完整 forward trace 与 FA 相同，SIMT 仍有异步入队造成的初始批形状差异，逐项标记。原生后端也会出现完整 logits/top1 差异；不由这些结果拟合新的宽阈值。
- [真实冻结 mask 执行资格](../results/main30b_sparse/frozen_4096_qualification/execution_qualification.json)：主模型 B3、预算4096，两 rank 每个 pass 使用完全相同的 mask hash、APC 命中与 forward trace，E0 SIMT A/A logits exact。四方法/两 rank 共3456次局部同 mask 检查和32次 FP64 探针全过；每 pass/rank 的336次纯 decode 层调用旁路 dense，96次混合批次调用明确保留 dense 冗余。每 rank selector 一次性准备约1.195秒，包含临时完整 K gather；没有把该费用隐藏为执行收益。
- [Graph 算子合同](../results/main30b_sparse/graph_executor_qualification.json)：B1/B3/B4 × SIMT/TC/union ×5次变化，共45次重放通过原 cache 独立 FP64 输出检查。固定 cache/output/workspace 地址、图外 plan、plan_info 保持不变；若 plan 拓扑变化必须重捕获。初始 CPU fixture 边界失败的 GPU 分配也完整计账。该证据仅为算子图合同。
- [真实 vLLM FULL graph 初版资格](../results/main30b_sparse/fullgraph_qualification/graph_execution_qualification.json)已经完成：保留原 dense manager，通过第二 manager 显式捕获 E0/union 的 B1/B2/B3 图；实际运行 B3 的7次纯 decode 重放，三 pass/两 rank 共2016次图内真实 Q/cache 影子检查全过，48层输出范数都随真实重放变化，E0 A/A logits exact。原 dense attention 没有在这些纯 decode 图中执行。当前图外每步48层 Python metadata 重建/plan 中位约80–181ms，是原型的主要阻碍，不能宣称系统加速。该运行关闭 torch.compile、保留 FULL graph，最终计时仍需匹配最强原生生产配置。
- 下一有界阶段为原生跨层 plan 复用与 GPU 持久索引更新。E0 固定每head预算使48层的长度/indptr相同，可共享一次原生 plan；这属于常规缓存/原生 API 复用，不声称新颖。必须新增页/split边界、B3→B1→B3、映射重用和前缀身份变化的资格，再进行无影子、同资源、同工作量完整费用测量。

以上八步 teacher-forced 结果不是完整自然生成质量。自然 EOS、任务配对质量、长期 ready/alive 生命周期、强原生图模式下的完整费用性能仍未通过。所有实际 GPU 尝试（含失败、初始化、编译）通过 `scripts/run_jobs.py` 记账。
