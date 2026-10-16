# Native FP8 KV 强基线资格审计（2026-10-16）

结论：**值得纳入必须处理的强工程对照缺口；源码适用，但尚未取得本机数值/质量资格。** 若继续声称主模型应用效率或容量优势，应补这个基线，或如实报告其资格失败。它是成熟量化工程，不构成新 idea。本轮只读源码、checkpoint 索引/头部及官方资料，未运行 GPU、加载模型、安装依赖或修改配置。机器可读证据见 [audit JSON](../results/platform/native_fp8_kv_source_audit_20261016.json)。

## 当前栈确实允许，但不能套旧参数

主推理环境 `.venv-engine` 是 **vLLM 0.31.0 / FlashInfer 0.7.0.post1 / torch 2.13.0**，并非 kernel 实验的 FI 0.7.2。vLLM 四个关键源码文件逐字等于官方 `v0.31.0`（commit `db9527a4…`）；FI `xqa.py` 逐字等于官方 `v0.7.0.post1`（`946200de…`）。

官方主模型是 **Qwen3 MoE 的完整上下文 attention**，不能把“dense attention 基线”写成 dense-parameter 模型。当前 FP8 权重、BF16 激活、TP2 下每 rank 有 16 Q heads / 2 KV heads、D128、GQA8、48 层。源码允许 SM120 上 `kv_cache_dtype="fp8_e4m3"`：prefill 保持 BF16 Q 并走 native FI/FA2，decode 保持 BF16 Q、优先 dedicated XQA；vLLM 对此路径允许 page16/32/64。可用 cubin/依赖、禁用 batch-invariant、未强制关闭 TRTLLM/XQA 等仍是实际选择条件，须保存 resolved backend，不能只记录 `FLASHINFER` 标签。[vLLM 路径](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flashinfer.py#L805)、[平台判定](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/utils/flashinfer.py#L592)、[XQA 实现](https://github.com/flashinfer-ai/flashinfer/blob/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/flashinfer/xqa.py)。

**`calculate_kv_scales` 已删除，不能传 True 来解决校准。** 官方 PR #49389 于 2026-08-03 合并；本地整个 vLLM Python 树也没有这个名字。移除前的实现取首次 eligible forward 的 Q/K/V `absmax/range` 后关闭计算，会依赖首批数据、长度和顺序，不是全开发集独立校准；当前 scalar FP8 路径只读 checkpoint scale，缺失即 1.0。[删除 PR 及旧实现差异](https://github.com/vllm-project/vllm/pull/49389/files)、[当前加载器](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/quantization/kv_cache.py#L115)。

模型 config/index 与官方 HF revision `60d80c83…` 逐字相等。四个 safetensors header 合计 37,491 个 tensor，与索引键集合完全一致；18,624 个 scale **全部是 `weight_scale_inv`**，没有 Q/K/V/KV/prob scale。config 的 `activation_scheme="dynamic"` 属于权重线性层的激活量化，不能当成 KV 校准。因此不改变 checkpoint 直接开 FP8 KV 是 **uncalibrated scale=1**。这不证明质量必坏，也不证明安全；E4M3 的范围/低幅值分辨率风险需要长上下文实测。[官方 config](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8/blob/60d80c83c53c3b611c642dbb8c942b3f90c5948a/config.json)。

v0.31 文档推荐离线数据校准；其 Llama 示例不能直接证明“已预量化 Qwen3 MoE FP8 权重 + KV-only 校准”的导出兼容性。若需要校准，应单独资格 **只增加 per-layer K/V scalar、原权重/weight_scale_inv/tokenizer 不变** 的派生 artifact，并验证实际加载的所有层/rank scale。不要重新量化权重后仍叫同 checkpoint；不要给 BF16 Q 暗中加 FP8 Q 量化。FI backend 不接受 `fp8_per_token_head`；文档的 calibrated per-head 路径限定 Flash Attention。E5M2 也不能仅凭 vLLM 通用枚举当作本次 XQA E4M3 的替代资格。[v0.31 校准说明](https://docs.vllm.ai/en/v0.31.0/features/quantization/quantized_kvcache/)。

## 最小后续资格建议（尚未执行）

1. **冻结一个首选 arm**：原 checkpoint、`fp8_e4m3`、所有层、默认 unit scales，明确命名 `native-full-FP8KV-unit`。其他设置继承现有 BF16 KV native-full 对照；启动后保存每 rank 每层的逻辑 KV dtype、scale、page/layout/stride、prefill/decode kernel 和实际 block 数。若 unit arm 失败，不把失败泛化为 calibrated FP8 不可用；先报告，再决定是否值得单独批准校准分支。
2. **先数值，后质量**：用真实长短输入检查 chunked prefill、16 个 teacher-forced decode steps、B3→B2→B1、page 尾、相同前缀 APC reuse，以及更新 Q/KV/block-table 后的 graph replay。FP8 kernel 应与**同一量化后再反量化 KV** 的高精度 attention 参考比；另列 BF16→FP8 的量化误差、logit drift、top-1/margin/目标 token logprob，不要求跨精度逐 token 相同，也不能靠相同 token 掩盖 kernel 错误。沿用既定 kernel 数值门槛；新 logit 资格门槛在看到结果前冻结。
3. **eager 与 FULL_DECODE_ONLY 都要验**，包括 graph capture 后变长、page 替换、请求退出和 prefill 插入。曾有 5090/XQA FULL graph 静默错误报告，但它是 v0.25.1/FI0.6.13、自行接入 XQA、不同模型，不能判定当前 v0.31 有相同 bug；它说明只做短文本 smoke test 不够。[历史 issue #49010](https://github.com/vllm-project/vllm/issues/49010)、[2026-08-11 合并的 SM12x 正式支持](https://github.com/vllm-project/vllm/pull/49718)。
4. 数值通过后，先冻结开发集一短一长文档、再覆盖原有相同开发组/问题；同 sampling/seed、原始位置/RoPE、自然 EOS 和原 8192 cap。记录覆盖/事实、解析失败、截断、自然长度及成对 wall time；16-token 资格不计质量样本，不能用 ROUGE 近似替代已暴露的覆盖问题。不读正式 eval100，不因 FP8 输出较短就直接称 decode 更快。

冻结 protocol 至少包含：模型/config/index/tokenizer/revision/hash；软件和 cubin/JIT 身份；TP2、BF16 compute、KV E4M3、每层 K/V scale 来源和值/hash、skip-layers 空集；page/layout、DCP=1、无 speculative；max_model_len=98304、max_num_batched_tokens=4096、max_num_seqs=3、APC 开关及 warm/cold 状态；attention backend **及实际 kernel**、graph capture sizes/mode；全部采样字段和 prompt token hash；校准若启用则训练文档 IDs、样本顺序/长度/observer 方法及与测量集隔离；资源预算、数值门槛、失败条件和质量规则。保持现有 `disable_custom_all_reduce` 等 engine 字段一致；native arm 不带研究 sparse executor。

容量方面，每 rank 原始 KV payload 是 `2×48×2×128×2=49,152 bytes/token`；E4M3 是 `24,576 bytes/token`，这是**相同实际存储 token 的 payload**，不是进程显存减半承诺。vLLM 固定 memory utilization 下通常把空余转成更多 blocks；应同时报告实际 payload、KV 池/可用 blocks、峰值显存与完整调用延迟，避免把预留池大小不变误写成容量无收益。该基线与 BF16 sparse kernel 的同精度资格不同，应放在质量—成本应用对照里。资格失败就停止该配置并留证据；不在本轮扩展精度/校准/后端搜索。
