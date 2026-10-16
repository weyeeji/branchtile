# 原生 FP8 KV 的长上下文容量边界

2026-10-16。**当前不追加 GPU 资格、不启动正式评估，也不将 FP8 设为默认。** 更大 KV 池补充了成熟工程基线的可执行性边界，但没有解决当前研究贡献与质量证据缺口。单位 scale 的两文开发对照不能支持质量非劣；缩短生成也不能直接解释成算子加速。

本审计使用普通用户、项目虚拟环境和 Python 标准库，GPU 使用为 0。复算脚本为 [audit_native_context_capacity.py](../scripts/audit_native_context_capacity.py)，逐文档输入长度、条件预算及全部来源 SHA 为 [native_context_capacity_20261016.json](../results/platform/native_context_capacity_20261016.json)。未读取正式答案或正式模型输出；正式 manifest 仅作为字节计算 SHA，长度来自已有输入资格表。没有重新分组、截断、挑选正式题目或重新执行 tokenizer。

## 模型上限、配置上限和 KV 池是三件不同的事

固定官方主模型 revision `60d80c83c53c3b611c642dbb8c942b3f90c5948a` 的 [config](../data/models/Qwen3-30B-A3B-Thinking-2507-FP8/config.json) 为 `max_position_embeddings=262144`、`rope_scaling=null`。本轮原生两文对照设置的 `max_model_len=98304` 是实验配置限制，不能当作模型原生上下文上限。

两 rank 的实际缓存收据一致：同为 TP2、`.80` GPU 池比例、16-token 页、BF16 compute、官方 FP8 权重时，BF16 KV 池为 **206,528 token / rank、10,151,264,256 bytes**，FP8 E4M3 unit KV 为 **413,072 token / rank、10,151,657,472 bytes**。两者约 9.45 GiB；不是 FP8 进程显存减半。这里取已有 [BF16](../results/native_precision_natural_development_v1/original_bf16/summary.json) 与 [FP8](../results/native_precision_natural_development_v1/original_fp8/summary.json) 的内存字段，不重新裁定任务退出状态或质量。

本地 vLLM 0.31 的 `kv_cache_utils.py` 在生成 KV 配置时先为永久 null block 留空间，再检查是否能服务一条 `max_model_len` 请求。以当前池推算，BF16 显式设置 262144 会在该检查失败；显式值不等于自动缩到可容长度。FP8 的 413K 池槽则仍不能让单请求超出模型原生 262144。不得为提高覆盖率偷偷引入 RoPE 扩展。

## 冻结输入的可容长度

复用 `results/data_qualification/docfinqa_public_{development8,training167,evaluation100}_v1_context.json`；三份文件的 manifest SHA 均与当前冻结文件一致。两篇已运行开发文档的 prompt 长度和文档边界又与本轮原生实验 protocol 逐项相同。**这是已有准确长度的复用，不是一次新的全量 tokenizer 资格。** 正式集列只说明输入长度分布，不是正式评估结果。

每题保留完整 8192-token 输出余量，要求 `max_i(prompt_i)+8192 ≤ max_model_len`：

| 冻结 split | 文档 / 题数 | 98,304 配置 | 196,608 配置 | 原生 262,144 |
| --- | ---: | ---: | ---: | ---: |
| development8 | 8 / 24 | 2 / 6 | 5 / 17 | 5 / 17 |
| training167 | 167 / 567 | 27 / 93 | 149 / 505 | 157 / 535 |
| evaluation100，仅输入元数据 | 100 / 371 | 7 / 22 | 82 / 302 | 86 / 315 |

开发集另外三篇最大 prompt 为 288,574、292,307、442,485；FP8 不会使它们进入原生上下文。**开发覆盖从 2/8 到 5/8 主要是放开人为 98K 限制，不能归为 FP8 的独有贡献。** BF16 在适合其池容量的约 196K 配置下也覆盖这五篇的单请求长度。

## 有公共 anchor 时的整组驻留预算

令页长 `P=16`，文档末尾位置为 `E`，各完整 prompt 长度为 `p_i`，每题输出上限为 `T=8192`。已核相同的系统与文档前缀至少共享

`C=P floor(E/P)` 个整页 token；保守的整组池需求为

`L = C + Σ_i P ceil((p_i+T−C)/P) + P`。

最后一项是永久 null block。文档非整页尾、问题后缀和各题完整输出余量都算入独立私有部分；不额外奖励可能延伸到问题里的公共前缀。公式假设先完成同文档 anchor、这些公共页仍可命中、无其他活跃文档、全部 B 题共同驻留。没有把可驱逐的闲置 APC 块永久占用，也没有模拟 allocator 临时页、lookahead、COW、抢占或重算。

同时满足原生上下文与 `L≤当前池槽数` 的条件计数为：

| split | BF16 当前池预算可容 | FP8 当前池预算可容 |
| --- | ---: | ---: |
| development8 | 5 / 17 | 5 / 17 |
| training167 | 141 / 474 | 157 / 535 |
| evaluation100，仅输入元数据 | 76 / 275 | 86 / 315 |

这些**不是实测服务率，也不是 BF16 显式 262K 引擎已能启动**。BF16 需要选择不超过实际初始化容量的上下文配置。提高 `max_model_len`、把 `max_num_seqs=3` 扩为 4 会改变 buffer、图和 profile，原池大小不能直接继承。开发 group6 的 B4 预算为 206,240，仅比目前 BF16 池少 **288 tokens / 18 页**，尤其不能据公式保证无 OOM。多数公开文档是 B4，当前 B3 配置也不能直接代表整组服务。反之，未通过满 8192 预留预算不代表自然短输出必然失败。

## 后续资格边界，当前不启动

若以后确需补原生成熟基线，最有区分度的单点是按训练 manifest 顺序选出的首个 B2/B3、原始 prompt 已超当前 BF16 池、但仍符合 FP8 与原生上下文的文档：**training group22，`docfinqa-8fca6f30ec6b8575`**。三题长度为 236978/236978/236977，加单题输出余量为 245170，整组保守驻留为 261680。它没有使用正式输出或质量结果来选点。

最低资格需要重新冻结 `max_model_len=262144` 的资源与软件身份；重新测真实池和峰值；核 48 层 scale、FP8 存储及原生 FA2/XQA 路径；在真实长位置检查共同页命中、位置/因果、短 B3→B2→B1 数值与 FULL/eager 一致、A/A 和干净退出。短 token 资格不算任务质量；若后续需要自然答案，仍须单独冻结自然 EOS/cap、同栈采样、逐题质量和完整准备成本。预期会失败的 BF16 262K 初始化可由源码/池预算解释，不为证明已知拒绝再消耗 GPU。

**本次决定仅归档边界。** 当前没有通过门槛的新候选，FP8 质量也未获非劣支持；扩大基线覆盖不是下一步关键阻断，因此不启动该单点、更不扩大为正式集评估。

复算：

```bash
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/audit_native_context_capacity.py
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/audit_native_context_capacity.py --check
```
