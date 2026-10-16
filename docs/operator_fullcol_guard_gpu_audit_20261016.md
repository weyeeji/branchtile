# FullCol guard：已保存 GPU 资格产物的独立审计

2026-10-16。审计使用普通用户、项目虚拟环境及 CPU 上的 `cuobjdump`，不初始化 CUDA、不重新编译或运行 kernel、不改动 wave3 源 JSON、CUDA、cubin 或 SASS。

**四个 FullCol control 均已通过数值、可变输入 graph 和本次独立资源审查。它们保留共享内存中的 canonical U gather/scatter，不是固定寄存器 bank v4。** 原收据的 `resource_review_pending` 表示数值资格已经通过、等待资源人工审查，不能解释为数值失败。本次结论写在独立 [audit.json](../results/operator_fullcol_guard_gpu_audit_v1/audit.json)，不改写原始状态。

## 检查范围与结果

四种形状为 B3/B4 × `(N32, splits64)` / `(N64, splits3)`，均 128 threads、GQA8、每 rank 两个 KV head。每种形状有 `all_masks`、`unequal`、`empty`、`one_token` 与 `real_capture` 五个 case。共 20 个 case、120 次冻结 FP64 O/LSE 检查通过，覆盖 eager、graph baseline、改变 Q、改变 K/V、改变索引、改变 offsets 与索引。eager 和首次 graph 的 O hash 在全部 case 一致；数值误差门槛未调整。空输入/offset 本来未变化的情况保留原记录，不把它强称为有效 offset 扰动。

审计重新对 20 个 cubin 执行 CPU `--dump-resource-usage` 和 `--dump-sass`，输出逐字等于已保存文件；CUDA/cubin/SASS SHA 均与原资格收据相同。四个 job 的账本终态均 exit 0、未超时。

下表只列真实 capture case。shared 总量是 **cubin 静态 shared + 原保存 launch 的动态 shared**；只读 cubin 得到的 `SHARED:1024` 不是全部共享内存。

| B / N / splits | registers/thread | 动态 shared（B） | 总 shared（B） | LOCAL / STACK | SASS LDL / STL |
| --- | ---: | ---: | ---: | --- | --- |
| B3 / 32 / 64 | 140 | 41152 | 42176 | 0 / 0 | 0 / 0 |
| B3 / 64 / 3 | 174 | 59584 | 60608 | 0 / 0 | 0 / 0 |
| B4 / 32 / 64 | 140 | 45312 | 46336 | 0 / 0 | 0 / 0 |
| B4 / 64 / 3 | 174 | 63744 | 64768 | 0 / 0 | 0 / 0 |

全部 20 个 case 的 LOCAL、STACK、LDL、STL 均为零；不能据此称寄存器压力很低、occupancy 已高或没有其它瓶颈。原生成 host 的动态 shared 数值来自保存的 launch 收据，本次未重新发起 launch。

## 为什么可以确定它仍在搬 canonical U

1. 源码声明 `cu = T.alloc_shared((rows, dim), "float32")`。每个区域按 `RowMap` 把 `CU[src,d]` 读入 fragment `out`，PV 更新后再把 `out` 写回 `CU[src,d]`。B3 的 U payload 为 12288 B，B4 为 16384 B；m/l 另占空间。
2. 全部 20 份生成 CUDA 均保留 M16/M32 两条分支的实际 shared-load gather 与 shared-store scatter。canonical U 位于 `buf_dyn_shmem` 的 float offset `2×rows`，即 B3 的 192 B / B4 的 256 B 起始处；地址以运行时 `RowMap` 返回的 src 乘 128 索引。最后向 U 全局 partial 输出也从同一区域读取。逐行证据保存在独立 JSON，例见 [B3 N64 real_capture CUDA](../results/static_bank_v4_qualification/wave3/staticbank_w3_fullcol_guard_b3_n64_s3_compiler/real_capture.cu)。
3. 重新反汇编的真实 case 含 LDS/STS，静态站点分别为 B3 N32 `66/92`、B3 N64 `66/110`、B4 N32 `68/94`、B4 N64 `68/112`。这些是静态指令站点，包含 Q/K/V、P、归约等多种流量；不能当成 canonical U 的执行次数、动态字节量或延迟占比。对 CU 的具体归因来自上述生成 CUDA 地址与源语义，而非把所有 LDS/STS 都算成 CU。

对 FullCol 与 emptyguard 做 AST 比较：移除模块说明并仅删除两个 GEMM policy 参数后，producer AST 完全相同；wrapper 仅替换 producer import 后也完全相同。原 emptyguard 两次 GEMM 使用 `FullRow`，此 control 改为 `FullCol`，其余区域 guard、状态格式、launch 和 merge 语义保持一致。因此，它可用于排除 warp policy / 编译器映射差异，不能替代待验证的 register-bank 数据流。

## 可以支持的下一步

FullCol 是已有数值与资源证据的对照候选，可以进入相同真实状态、相同完整调用合同的配对计时。**本审计不产生任何性能、应用质量、能量或 TP2 关键路径结论。** 新 v4 的编译和数值资格由其独立收据判断，不能继承 FullCol 的通过状态，也不能用 FullCol 的 shared 结果证明 v4 已经消除了 state round trip。

复算：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_fullcol_guard_gpu_artifacts.py
```

脚本只写自己的独立 `audit.json`；所有 wave3 输入文件及原收据保持只读。
