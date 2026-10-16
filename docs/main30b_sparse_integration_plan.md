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
5. 当前 FA 物理 cache 为 `[physical_page,Hkv,page_tokens,2D]`。以 `cache.view(-1,2D)` 的 K/V 两个半区形成 strided `[physical_token,1,1,D]` 页1视图；物理索引是 `page_id*Hkv*page_tokens + head*page_tokens + offset`。先独立验证 FlashInfer 接受该 stride、指针确实别名且无 KV gather/copy。
6. E0 将 `(request,KVhead)` 展成 batch，GQA8 不变，逐行索引为系统前缀、冻结文档 mask、完整问题/生成 suffix 的去重集合。fixed union 以 KV head 为 batch，对同 ready 组的物理 token ID 求并集，并使用精确 per-request custom mask；Q gather 和 output scatter、native plan 全部计入。
7. 每步从实际 block table/seq_len 构造增量 suffix；任何 alive 子集必须自然支持。两种执行方式使用同一份不可变历史 mask，不能按候选更改预算或选择轨迹。
8. 混合 prefill/decode 批次不能默默让 decode 走 dense。首版可明确记录仅混合批次的 dense-prefill-plus-sparse-overwrite 冗余费用以维持准确语义；纯 decode 必须完全旁路原 dense。若无法做到，停止该资格配置并报告限制。

## 顺序资格与终止条件

- 首先仅对物理 cache 页1视图做小形状 FP64 输出/LSE 检查和原 cache 索引核对，包含非整页、私有末页、空/退出请求。
- full-budget 在相同 prefill 下检查每层输出与原 FA、TP 后 logits 和短 greedy 轨迹；同时保存 dense A/A 作为本环境数值噪声。任何不一致先定位，不能扩大 tolerance 来换取通过。
- 再测冻结稀疏预算 E0 与 fixed union 的每层输出/logits、teacher-forced 同 token 轨迹和自由生成 EOS；自由生成可因 FP 运算序不同分叉，须单独报告质量，不把逐 bit 不同直接视为数学错误。
- 计时分清 selector、观察 Q 保存、mask/plan、索引更新、完整 decode step、prefill、峰值显存与自然结束。单步各 kernel 和不可直接从重叠/JCT 中相减。
- 先做一组有界资格，不预设稀疏 A4 快于 dense。若完整费用没有可辨净价值或即使移除可替换 attention 也不能接近质量合格替代，则停止该配置；不无限优化 Python metadata 或重包装已有 union。

本文件是待实现计划，不是实现完成或质量通过的声明。所有实际 GPU 尝试（含失败、初始化、编译）通过 `scripts/run_jobs.py` 记账。
