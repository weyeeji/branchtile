# E3 强消融：spatial owner tasks 与 packed reduction

本实现是原实验矩阵中尚未完成的 packed-active partial 对照，不主张新颖性。PAT、PageWeaver、Cascade 已有直接相关先例；此处的目的是检查：将 owner cohort 分给独立 CTA、完全去掉串行 region 切换，是否能改善原动态实现。当前仅完成源码、CPU coverage/数学检查及 Python 语法检查，**尚无 Triton 编译、GPU 数值资格或性能结论**。

## 执行与内存合同

[attention_owner_tasks.py](../src/branchtile/attention_owner_tasks.py) 提供 `Attention(plan, splits=64, block_n=64, threads=128, compact=True, region_partials=False, chunk_tokens=512)`。默认 chunk 为 512 个索引位置，可显式设置 256/1024；N32/N64、B3/B4、GQA8、D128、连续 BF16 Q/K/V。`splits` 仅保留调用兼容，**不参与该版本的任务划分**，metadata 明确记录此区别。不能把它与原等分 split 的任务数、工作量或公平性直接画等号。

CPU prepare 读取原 plan 的 offsets、row_map、counts 和 labels，验证每个 owner 的 active rows 是完整、顺序一致的 GQA8 请求组。将每个非空 region 切为固定长度 chunk，生成 `(head, region, begin, end, active_rows, packed_row_base)` 六个 int32 字段；空 region/shard 没有 CTA。每个 CTA 仅执行一个 owner cohort/chunk，读取原 `Indices` 和原物理 K/V，M16/M32 两个分支进行 BF16 QK/PV 与 FP32 online softmax。没有 canonical U，也不在 region 之间搬运它。

只有有效 query rows 的未归一化 U、max、denominator 写入 packed 缓冲区。设任务 t 有 A_t 个 active rows，P=Σ_t A_t，则 FP32 partial 存储为 **`P×(4D+8)` 字节**；没有分配 `heads×regions×splits×rows` 的矩形 partial。任务描述符占 `24×task_count` 字节；另有 `[H,B,maxContrib]` 的 int32 merge 索引表和 `[H,B]` contribution counts。输出 O/LSE 也计入 `workspace_bytes_by_buffer`。原 KV、原 Indices 和原 row_map 被引用，未复制 payload。

merge 索引记录 `packed_row_base + request_slot×GQA`，每个 query head 再加自己的 GQA offset。归并每次只取 **16 个 partial**，以稳定 max/den/U 在线合并，避免把整个长 contribution 列表铺成一个巨大的寄存器 tile。全空返回 BF16 O=0、自然对数 LSE=−inf。

完整 GPU 调用是 **producer + packed merge**。只计 producer 会隐藏 partial 写回和 reduction 成本，不构成收益。prepare_ms 包含本构造函数中的 metadata D2H、CPU validation/task 构建、H2D、buffer allocation 与结束同步，首次 Triton JIT 在首次调用中另计；原 region plan 创建发生在外部，实验必须同时报告其成本，不能因传入已有 plan 而自动免除。

## 与已有对照的区别

- 原 `region_partials=True` 采用 region×split 的矩形 launch/partial 空间，包含空任务及固定 rows 的输出存储。本实现显式去掉空任务，且只保留 active rows；它没有宣称这一定抵消 CPU descriptors、更多 global partial 及 merge table 的成本。
- ExactCascade 根据树进行 intersection/residual 分解并归并原生输出。本实现直接使用互斥 owner bitmask regions，不采用树；每个 `(request, token)` 仅出现一次，chunk 仅细分其任务。
- 串行 v2/v3/v4 在一个 CTA 内跨 region 保持状态。本对照将状态拆为多个空间任务，增加全局 partial/merge，换取没有 canonical state transport 的执行方式。必须实测其完整成本。

## Graph 拓扑边界

相同 topology 的 graph 可原地修改 Q/K/V 和 Indices 值。更改 offsets、counts、row_map 或 labels 会改变 descriptors 或 merge table，必须**重建 Attention 并重新捕获 graph**。`validate_plan()` 使用 host Tensor version 检查，能拒绝经 `__call__` 使用的过期对象；**`graph.replay()` 绕过该 host 检查**，不能声称自动保护任意旧 graph。

专用 GPU 资格需要保留原 mutability 问题，但区分两种能力：同 graph 下修改 Q/K/V/Indices；修改 topology 后旧 host 对象拒绝，然后完整 rebuild/recapture 并核 FP64。两者不能合并报告为与旧动态 kernel 完全相同的 graph 功能。CPU prepare 与 recapture 成本均需保留。

## CPU 资格与后续边界

[qualify_owner_tasks_cpu.py](../scripts/qualify_owner_tasks_cpu.py) 对 B3/B4 的全部 owner masks、unequal、empty、one-token，分别检查 chunk256/512/1024，并额外检查 33 个 contribution 跨三组流式归并，共 **26 个 case、1456 个 query rows**。每个 `(head, query-row, selected source-token)` 的覆盖精确一次，packed rows 无重叠/洞，merge 索引完整，Python FP64 partial+merge 与直接 attention 一致，误差≤1e−12；还检查错误 metadata 拒绝和 host stale-version 拒绝。

这些只验证任务分解、映射及精确数学，不验证 Triton lowering、BF16 rounding、GPU shared/register 资源或速度。[CPU 收据](../results/operator_owner_tasks_cpu_v1/qualification.json) 保存 source SHA。后续可从 `last_producer_compiled` / `last_merge_compiled` 取得 Triton 的 asm/资源；全空情况没有 producer launch，其 compiled 值为 None。不能套用 TileLang host-stub 资源 helper。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/qualify_owner_tasks_cpu.py
```
