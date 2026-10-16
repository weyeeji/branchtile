# Static-bank v4：独立 CPU 编译产物审计

本目录仅核查已存在的 GPU 产物，不启动 GPU、不修改原始 JSON、不改变任何失败或计费状态。可复算入口为 [audit_static_bank_gpu_artifacts.py](../../scripts/audit_static_bank_gpu_artifacts.py)，机器可读结果为 [audit.json](audit.json)。审计读取 frontend_trace、B3 extended、资源解析器 v2 probe，以及 wave8 各 job 的当前快照；未完成 job 保留 pending 状态，不能视为资格通过。

当前快照共 **11 个干净退出 job、14 个 case、84 个数值检查阶段**。所有被审资格收据的 v4 kernel SHA256 必须为 `afa47ed8aca50923e7426096d6c17be0c0830bbfd8687b3d0fb99aee17ec5a85`，对应 `8e92a61` 冻结版本。每个合格 case 均要求 eager、graph baseline、修改 Q、修改 KV、修改 indices、修改 offsets+indices 的六项检查完整通过；这些阶段不等于六个独立自然任务。整体 clean exit 另以预算账本的 end 事件核对。

## 已确认的实现事实

- 生成 CUDA 的 canonical `cu` 只用于初始化、与 active `out` 的直接线程私有数组赋值、最终全局 U 输出；所有使用位置均分类核查。`out` 只额外进行初始化、softmax rescale 和 MMA 累加器调用，没有经共享内存暂存。每个 CUDA 线程访问自己的 local array，这证实当前实现消除了 v3 的 shared canonical-U 往返。
- 保存 cubin 的 REG/LOCAL/STACK/static SHARED 与收据逐项一致；SASS 独立检查 LDL/STL。JSON 逐 case 保留寄存器、shared 和 spill 状态，不能用源代码中的数组声明替代该项证据。
- 最大值及分母 CM/CL 仍写入 shared；初始化后发布 barrier、每个 region 末尾无条件 `__syncthreads()` 均保留。单 region 时循环被编译器展开，但末尾 barrier 仍存在。Q/K/V/P、归约等其他共享内存操作仍然存在。

目前已通过 B3 的常规 BN32 case 为 **182 registers/thread、35,072 B 总 SMEM**，BN64 真实 case 为 **214 registers/thread、53,504 B**；empty BN32 为 120 registers/thread、28,928 B。B4 BN32/split64 已通过，为 **230 registers/thread、43,264 B**。这些形状和分支数不同，不能将其与旧 v3 的另一形状资源直接相减作为性能提升。

**B4 BN64 确实存在 spill。** split64 为 255 registers/thread、STACK 16 B、SASS 4 处 LDL/STL；split128 为 255 registers/thread、STACK 24 B、6 处 LDL/STL。二者 LOCAL 字段虽为 0，仍不能称为无本地访存；总 SMEM 均为 61,696 B。B4 BN32 的 split32/64 则均为 230 registers/thread、STACK/LOCAL/LDL/STL 为 0。这项资源成本与已通过的数值合同分开报告，不把数值合格改判失败，也不将所有 v4 笼统称为无 spill。

BN32 的 B3 源码含 52 处、BN64 含 53 处、B4 BN32 含 110 处、B4 BN64 含 115 处静态 barrier 文本。这是生成代码的静态位置数，**不是运行时 barrier 次数或同步时间**。

## Launch 参数的独立重建

旧 helper 的 `[2,2,splits]` 等 grid 字段保持原样，不用于本审计。工具从匹配 CUDA 源的 TileLang cache executable 中解码 `FunctionInfo` 的二进制记录，并要求其后嵌入的 cubin 与保存 `.cubin` **逐字节相同**，然后按真实 metadata tags 和 host packed-call slots 重建 launch。这样不依赖从 `strings` 输出猜测标签归属，也不加载 executable 或调用 CUDA。

全部已审 producer 的实际 grid 为 `[KV_heads, splits, 1]`，block 为 `[128,1,1]`，dynamic SMEM 与保存收据一致。新增 `resource_parser_probe_v2` 实际 GPU 作业保存的 `.tvm_meta.json` / `.host.c`，与上述独立二进制解码和解析结果一致。完整 tags、slots、参数表达式、host/executable SHA 和 cubin SHA 位于 JSON。

重建所用 host 全源以确定性 gzip 存入 [launch_hosts](launch_hosts)，解码后的 metadata 同目录保存。归档名称按 cubin SHA 去重；JSON 同时保存源 executable、host、归档文件哈希。后续 job 可通过重复 `--input` 参数纳入新版本输出，避免覆盖本快照。

## 结论边界

这些证据说明固定 bank 的 U 搬运修改已经落实到可通过数值与可变输入检查的 GPU 代码。它们不证明运行时显存流量降低、缓存命中率提高、占用率提高、完整调用加速或模型加速；这些结论必须使用独立性能实验。局部 case 通过也不会将总体超时作业改判成功。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_static_bank_gpu_artifacts.py
```
