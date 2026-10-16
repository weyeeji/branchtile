# 原生整模型收益空间测量协议

本轮回答：在实际 vLLM、FlashInfer/XQA、完整模型与真实长文档请求中，attention 还占多少运行成本？本轮提供真实服务基线与机会诊断，不运行新候选，因此不把它称为候选端到端加速证明。root 统一分配 GPU 和账本；本协议作者未运行 GPU。

## 当前已经核实的历史依据

`results/native_opportunity_protocol_v1/historical_evidence.json` 核验旧 trace SHA，并从原始汇总的分子分母重新核算八个 rank 记录：独立 sparse32K、B3 的 attention 加必需 output copy 占匹配整模型图 kernel sum 21.57%–25.89%，占单独记录的 instrumented model-graph CUDA event 20.62%–24.62%。这不是完整引擎或自然请求墙钟占比。历史 dense53% 来自没有显式指定 FlashInfer 的 runner，以 `flash_fwd_splitkv` 名称分类；不能直接作为当前原生 XQA 占比。

## 可运行新工具

- `scripts/benchmark_native_opportunity.py`：原生完整模型，FlashInfer backend，APC 开启，TP2，2 KV heads/rank，48 层，BF16 激活与现有 FP8 权重；分别使用 BF16 或官方 unit-scale FP8 KV。复用已冻结两个开发文档的 prompt 哈希与模型配置。
- `scripts/vllm_native_opportunity_worker.py`：继承已用过的原生精度 worker，只在单独 profile 阶段增加每次完整模型 CUDA graph 的事件；无 profile 测量不插 CUDA events，不修改 attention、cache 或图执行。
- `scripts/summarize_native_opportunity.py`：CPU 核验墙钟、token 数、固定 AA token/APC/真实调度轨迹、trace SHA、kernel 名称分类及 instrumented graph events。对全部 kernel 名保留明细，遇到未识别 attention 需先审计分类完整性。

四组 TP2：GPU0/1 文档0 BF16、GPU2/3 文档0 FP8、GPU4/5 文档1 FP8、GPU6/7 文档1 BF16。具体启动由 root 新建 `run_jobs.py` manifest 完成，绝不独立抢卡。示例 argv：

```sh
.venv-engine/bin/python scripts/benchmark_native_opportunity.py \
  --document-id docfinqa-c0f11a67aae850fc --kv-dtype auto \
  --output results/native_opportunity_v1/original_bf16
```

另一文档为 `docfinqa-593ec0c33f45ea02`。默认 `--fixed-tokens 128 --natural-repeats 1`；自然阶段沿用原协议 max8192、ignore_eos=false。已有 summary 或 trace 目录时拒绝覆盖。trace 放 `results/raw/native_opportunity/<output-basename>/`，普通 summary 记录 SHA 和路径。默认原协议文件 `configs/native_precision_natural_development_v1.json` 内容完整嵌入 summary。

## 每个作业的顺序与时间边界

1. 初始化引擎，记录加载时间；真实形状 warmup；记录实际 FlashInfer/XQA、KV dtype/scale/layout receipts。
2. 两次固定128 token 的正常模型生成，保持全部请求活跃。每次先清 APC，再一次真实共享文档 anchor，然后 ready-group 生成。固定段允许忽略 EOS，故不用于自然任务质量。它没有 teacher forcing，也不删 attention。
3. 一次独立自然 EOS 生成，仍先清 APC、计时 anchor，再运行完整请求组；保留每题文本、tokens、长度、finish_reason 和缓存命中。达到8192 cap 的输出必须保留，不能称作自然完成。
4. 最后单独 profile 固定128 token：延迟64个引擎iteration，采样32个iteration。该阶段单独开启模型图 CUDA events，保留其余时间段events作诊断，绝不把不同时间窗的event分母和trace分子相除。

`complete_generate_wall_seconds` 是整个 ready-group 调用，包含同步准入、输入准备、调度、模型、采样与自然完成；`cold_shared_document_total_wall_seconds` 加上每文档一次真实 anchor。初始化不混入稳态生成，但单独报告。CPU identity/metadata hooks 的成本仍计入墙钟，未来候选需要保留同样 hooks 或同时关闭。天然 ready-group 阻塞准入不代表真实在线到达流，应限定为同步单用户分支请求。

fixed AA 同时核验 tokens、APC hits、两 rank 的 `(batch, questions, positions)` 序列。AA不一致不隐藏；两次样本只能显示漂移，不能支撑置信区间。跨 BF16/FP8 自然轨迹可能不同，时间比不可命名为相同工作量的加速。profile 始终独立于非profile墙钟。

## 分析和后续决策

```sh
.venv/bin/python scripts/summarize_native_opportunity.py \
  --root results/native_opportunity_v1 \
  --output results/native_opportunity_v1/analysis.json
```

需先检查真实 XQA kernel 名及所有 kernel 明细，确认 primary/merge 没有漏项。kernel 累计份额、kernel union busy、第一至最后 kernel span 分别报告；不把累计份额视为关键路径上的可删除比例。每 rank 独立报告，不把 TP 两张卡的时间相加当墙钟。

若要测严格的可替换边界，应在固定状态完整模型上进一步做相同 attention 输出的回放干预；若删除 attention 或注入零输出，会改变下游 MoE 路由和计算，此类干预只能单独标无效轨迹诊断，不能计为自然质量或实用端到端收益。本轮不使用此类替代。

机会足够时，下一步测候选真实集成：同时包含 prep、metadata 更新、graph recapture、private tail 与请求退出。原生主基线使用实际 O-only XQA 路径，不强制增加主应用不需要的 LSE 导出。简单共享实现和相关论文用于消融/定位，逐输入 oracle 仅作诊断。没有固定端到端百分比是 ISCAS 硬性投稿要求，也不据本轮基线测量宣布论文成功。
