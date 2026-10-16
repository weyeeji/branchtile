# TinyGEMM 成熟基线补齐：仅准备，尚未启动

目前没有发现足以支持新论文机制的精确 MoE 数据流。唯一保留的检查是把已存在的 FlashInfer `tinygemm_bf16` 作为 GateLinear 强基线：本机 vLLM 的特殊 router GEMM 条件排除 SM120，而安装版 FlashInfer 的 TinyGEMM 形状检查明确接纳 SM120。它是成熟实现的兼容性与成本检查，不是新算法。原动态行、register-bank 和 joint selector 的止损结论不变。

## 已有成本证据与限制

[可复算脚本](../scripts/analyze_tinygemm_opportunity.py) 和 [原始来源 SHA 与结果](../results/platform/tinygemm_opportunity_20261016.json) 只读取已有八份 rank trace：两文档、两设备副本、TP2，每份四个实际 B3 模型图步骤。以原 attention 归因收据的 correlation/graph ID 选择完整模型图，每步 1348 个 GPU kernel。

| 项目 | 每步 kernel 时长之和 |
|---|---:|
| 48 个 BF16 gate GEMM | 600–611 µs |
| 48 个 top-k router | 132–136 µs |
| 96 个 MoE grouped GEMM | 1220–1250 µs |
| MoE scatter/gather | 382–398 µs |
| 97 个 NCCL all-reduce | 747–2082 µs |

每个上述 BF16 GEMM 后紧接实际 `topkGating<8,128,...BF16>` kernel，支持 router 归因；本轮仍要求实际 X/W、输出类型和运行分支收据，不能仅靠 kernel 名称确认全部语义。已有 profile 的 CUDA 模型事件为 6.82–8.20 ms/step，来自 32K sparse E0、BF16 KV；尚无本轮完整原生 FP8 KV 的 B1/B2 gate profile。首个 collective 有明显 rank 到达等待（最慢约 1.39 ms），不能当作可消除的通信成本。kernel 求和不是关键路径；FULL graph 下 1348 个 kernel 也不是 1348 次 Python 启动。

原生已经包含小 M active-expert 限制、SM120 grouped-GEMM BLOCK_M 调整、排列/融合激活量化/归并等机制。TP2 的跨卡 peer-access 实测为 false，打开 custom-all-reduce 开关不能绕过其 P2P 资格。现有量化或需要训练的专家参数共享不是本检查的精确替换。

自然 FP8 对照第一重复的 cold anchor+generation 为原文档 20.588 s、新文档 29.109 s，分别 1131/2292 个实际 graph 步骤。若要 1.05× cold 加速，平均每步须减少 867/605 µs；若定义为延迟下降 5%，须 910/635 µs。即使**假设**所有 B1/B2/B3 步骤均有最大已测 B3 gate 成本、且 gate 成本完全消失，cold 延迟也仅下降约 3.36%/4.81%。这些是条件情景，不是完整 runtime 上界：B1/B2 和完整 FP8 成本尚未测，prefill/init 未加速，也没有模型级性能测量。

## 安装源码与同步检查

冻结协议保存所有本地源码 SHA。安装 FlashInfer 为 `0.7.0.post1`，build metadata revision `946200de1ae94fc93fdd0926f0a13afd1fa7f0f1`。`tinygemm2.cu` 的 warp>=4 外层分支、内部 elected DMA lane、分支外全 CTA `__syncthreads()` 已具备 [FlashInfer #2996](https://github.com/flashinfer-ai/flashinfer/pull/2996) 所述 barrier 修复结构。原 [TensorRT-LLM #10873](https://github.com/NVIDIA/TensorRT-LLM/pull/10873/files) 涉及同步与 PDL 完成信号；本轮固定 `use_pdl=False`，不把不同仓库文件视为逐字相同。

安装版 launcher 根据 shared-memory 容量只选 STAGES=16 或 4，SM120 默认 4。未合并的 [stage autotuning #3459](https://github.com/flashinfer-ai/flashinfer/pull/3459) 不进入实验；不安装、不改库、不进行 stage 搜索。K=2048、N=128、B=1/2/3、BF16 连续 row-major、无 bias 符合该 API 的公开形状条件，但还不是已运行资格。

## 预先冻结的最小 gate

[协议](../configs/tinygemm_gate_qualification_v1.json)、[job 配置](../configs/tinygemm_gate_qualification_jobs_v1.json)、[driver](../scripts/qualify_tinygemm_gate.py)、[worker](../scripts/vllm_tinygemm_gate_worker.py)。一份原文档、一次 TP2，现有官方 FLASHINFER/FP8-KV 引擎；240 s 硬限及 15 s 清理，最多 0.141667 P1 GPUh，失败也计账、无自动重试。当前只准备，未启动。

1. 活模型始终执行原 gate/router/expert。新增 FULL B1/2/3 图仅在真实 gate/router 边界复制 X、BF16 logits、selected IDs 和 FP32 normalized weights，防止编译中间 storage 被复用。每种 B 的首次快照前将收据 buffer poison，再验证实际图完整覆写。
2. 两次独立冷 APC anchor 后原子加入 B3，greedy、ignore_eos、各题 cap=8/12/16，仅用于产生 B3→B2→B1 仪器状态。要求 A/A token、APC hits、位置/问题序列一致。**每 phase/B 仅首次实际状态**保存 all48 层，不声称全生成步逐位资格。
3. 共 2 phase×3 B×48 layer×2 rank=576 个资格点。各自实际 BF16 X/W 的 FP64 完整 logits 参照只报告 finite、绝对/relative-L2/RMSE；另报告 Tiny 相对原生 BF16 误差、编码步差、完整 IDs/weights 及变化数。没有沿用 attention 容差，没有根据候选结果拟合阈值。
4. 保守的直接替换门：原生重算须逐位匹配实际图；Tiny 通过**同一个实际官方 router**所得 expert IDs（含顺序）与归一化 FP32 weights 须逐位相同。源码已断言 modular experts 只消费这两个路由结果，EPLB/额外 routing capture/fused gate 关闭。任一 rank 任一点不满足，双方都跳过候选计时。此时结论只是“未获本轮 bitwise drop-in 资格”；浮点舍入不同不等于 TinyGEMM 错误或永远不适用，后续非逐位替换需要另冻模型级数值/质量合同。
5. 全部通过才测 gate-only 与 gate+实际-router 两种完整48层链；真实冻结 X/原 W、B1/2/3、native A/A 和 Tiny 各自输出保活、20 个 random/reverse block，hot=3 repeats/cold=1。256 MiB 清缓存在计时外。报告 CUDA event 与完整 host call，两 rank 独立，不当作同步 TP2 整模型一步。原 GateLinear 直接返回 GEMM 输出，两臂均不人为增加 `output.copy_`。

初始化、warmup、新图 capture、Tiny JIT 首调用、资格、benchmark graph prepare、全部计时分列。二十个 block 才是计时重采样单位；两个 rank、副本和各层不是新增独立文档。即便该成熟替换通过并更快，也须另行验证真实模型图关键路径及自然 JCT，不能从 gate 链推出应用 ≥5%。

CPU 检查已覆盖 IDs 顺序变更、FP32 权重一 ULP 变更、dtype、NaN、signed-zero 的逐位合同，2 tests passed；无 CUDA 初始化。独立只读源码审查未发现 fatal finding，并促成 poison-overwrite 收据增强。
