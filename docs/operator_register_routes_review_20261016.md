# GQA8 静态 owner 算子：可去掉的状态搬运与布局条件

2026-10-16。本次按用户要求重新评估算子实现，限定为单用户、多 agent、共享文档。此报告提供 **CPU 整数路由证明和实现条件**，没有编译、运行 GPU 或新增性能结果；不覆盖旧实验结论。

## 优先验证的一个具体修改

v3 每个 owner region 都将全部 canonical FP32 U 拆成两个 BF16 类型的位容器，写入 K scratch；再 gather 本 region 的 active U、执行注意力、scatter active U、恢复 canonical U。这是精确保留位模式的状态搬运，并非 BF16 数值量化。保存的 [生成 CUDA 源](../results/kernel_ablation_v3/compiler/register_u_n64.cu) 中这些 shared 写读确实存在。资源收据为 240 registers/thread、48,384 B 总 shared、零 local/stack spill；因此不能把问题归因于已发生的寄存器溢出。

`make_plan.labels` 全 head 统一、已排序，region 的 active requests 可以在编译时从 label 得到。可以据此将 region schedule 静态展开，使用常量 request 编号完成 canonical fragment 与 active output fragment 的寄存器拷贝，并在 `begin < end` 时才进入 region。它同时移除动态 `RowMap` gather、`Count <= 16` 分支和 U 的共享内存中转。实际 owner/token indices 仍是运行时数据，源 K/V 保持只读、不打包复制。

## 必须处理的布局陷阱

**不能保留当前 M16/M32 的默认 FullRow 分配后，就假定静态切片都是同线程拷贝。** 当前 TileLang 的 FullRow policy 优先按 M 分配 warp：M16/D128 使用 `m_warp=1,n_warp=4`，M32/D128 使用 `m_warp=2,n_warp=2`。后者会使部分 U 由其他 warp 持有。旧生成源中 M32 的 row expression 包含 `((threadIdx.x & 63) >> 5) * 16`，与此一致。

建议最小实现先将 QK/PV policy 固定为 **FullCol**，保持 `m_warp=1,n_warp=4`；M32 时由每个 warp 计算两组 M16。设 `i` 为 fragment 行、`d` 为 D128 列，固定 bank 的布局可以写为：

```python
canonical_layout = T.Fragment(
    (rows, 128),
    forward_thread_fn=lambda i, d:
        (d // 32) * 32 + (i % 8) * 4 + (d % 8) // 2,
    forward_index_fn=lambda i, d:
        (i // 8) * 8 + ((d % 32) // 8) * 2 + d % 2,
)
pv_layout = T.Fragment(
    (M, 128),  # M = 16 or 32
    forward_thread_fn=lambda i, d:
        (d // 32) * 32 + (i % 8) * 4 + (d % 8) // 2,
    forward_index_fn=lambda i, d:
        (i // 16) * 16 + ((d % 32) // 8) * 4
        + ((i % 16) // 8) * 2 + d % 2,
)
# T.annotate_layout({canonical_u: canonical_layout, out: pv_layout})
# All owner/request numbers below must be compile-time constants.
# out[slot*8 + q, d] = canonical_u[request*8 + q, d]
# canonical_u[request*8 + q, d] = out[slot*8 + q, d]
```

`T.Fragment`、`T.annotate_layout` 和 FullCol 在当前安装的 TileLang 中存在；上面是**未编译的布局示意**。若编译器不接受该 fragment copy，下一步应使用 `TensorCoreIntrinEmitter`/MMA lowering 明确 local register slots，而不能退回动态索引的 local array。`T.gemm` 的现有 PTX intrinsic 要求 contiguous C buffer；直接将两个不连续 request banks 作为 C/D 还需要自定义 lowering，优先先验证同线程 register copy 的版本。

## CPU 证明与预期移除范围

[audit_operator_register_routes.py](../scripts/audit_operator_register_routes.py) 穷举 B3/B4 的 7/15 个 owner、全部 8 个 query heads 与 D128 坐标，共 **45,056** 条 active U 路由。每条 FullCol route 的源、目的 thread ID 完全相同；每个 buffer 的 `(thread, register)` 映射为双射。原 FullRow M16/M32 混用时，同样路由有 **9,728** 个坐标跨 thread。脚本没有导入 CUDA 库，输出含源码和历史收据 SHA256，可复算：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_register_routes.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_register_routes.py --check
```

[CPU 收据](../results/platform/operator_register_routes_20261016.json) 中，B3/B4 canonical U 分别需要每线程 24/32 个 FP32 register slots；这只是 U 存储数，不是完整编译寄存器数。对于一 CTA 覆盖全部 7/15 个 region 的情况，v3 显式 U 中转逻辑量分别为 **233,472 / 638,976 B**，按 `full canonical store + active gather + active scatter + active canonical restore` 计算；不计 Q/K/V/P、m/l、编译变换和硬件缓存。新版本目标是将该项 shared U 字节降为零，仍允许 active U 的同线程寄存器 copy。

QK/P 暂存、softmax 归约及 K/V 生命周期仍需协调；m/l 可暂时保留共享存储。不能把这个方案称作“零同步”。也不能仅凭 Python 使用 fragment 就宣称达成目标：生成 CUDA/PTX 必须证明确实没有 U 的 shared store/load、没有 local spill、没有额外跨线程重排，并保存资源与代码收据。

进入性能比较前应先通过：所有 owner、空 region/shard、不整除长度与 split 的 FP64 O/LSE 合同；真实 B3/B4 masks 的完整 producer+merge；相同输入、相同源 KV、成对冷热计时。之后再和最佳固定划分、原生 E0/XQA 及 v2/v3 比较。FullCol 可能改变吞吐、归约和寄存器压力，静态展开也可能增大代码，因此目前结论只到“有可落实的结构性 overhead 去除点”，并无速度承诺。

此前全 48 层几何核算仍显示：仅依靠少做矩阵乘，动态布局相对最佳固定划分的差距通常只有几个百分点。新实现如果成功，其贡献必须由实测的搬运、资源或完整调用收益支持；不能把逻辑去重比当作实际显存流量或整模型加速比。
