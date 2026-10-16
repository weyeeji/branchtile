# Static-bank 编译缓存与 launch grid 更正

2026-10-16。本审查使用普通 uid、项目虚拟环境和 CPU 读取既有文件，未启动 GPU，也没有修改旧源码、旧 JSON、计时或预算账本。

## Wave4 与后续快速通过并不矛盾

Wave4 的八个 job 整体均超时，继续按失败计费。但四个 B3 job 的部分证据已经写入：all_masks、unequal、empty、one_token 各自通过 FP64 及可变输入 graph 检查，随后进入 real_capture 编译才超时。B3 all_masks 首次初始化约 84.74–92.65 秒；unequal 与其 `(heads,tokens,rows,regions,splits,block_n)` 相同，复用同进程 kernel，初始化约 0.0002 秒。empty、one_token 的 region 数不同，分别再次花约 84–93 秒编译。这些累计成本足以解释 B3 在到达真实 capture 时耗尽作业时限。四个 B4 job 尚停在第一个 fixture，不能视为已通过。

后续 `frontend_trace_v1` 仅跑同一个 B3/BN32/split3/all_masks，初始化 1.132 秒，总资格阶段 2.513 秒，干净退出。它的生成 CUDA 源 SHA256 为 `c57c44d1561bdf5593acb4e77abda7cc30eb1c0f18bc9810966d282e49a7f2f3`。完全一致的 CUDA 源存在于 TileLang cache key `3017940eb2a019adc26f59f38ff2aa78cdf98982391f93e16c285598ef40b5fb`；device/host 源在 **13:18:00 UTC** 写入，完整 executable/params 于 **13:18:00.862 UTC** 前形成，早于 trace 的 **13:24:13 UTC**。

TileLang 的磁盘命中路径直接调用 `JITKernel.from_database`，绕过 lower/compile。缓存键包括 TIR、target、后端、pass configs、版本和运行库内容哈希；并不按 GPU 序号区分。上述时间、字节一致性和快速执行强烈支持磁盘命中。trace 没有显式记录 cache-hit 事件，故不将缓存命中描述为有直接命中日志的事实，更不能据此声称出现了锁竞争或编译器死锁。后续真实 capture 的独立 faulthandler trace 已定位到 LayoutInference，不能将所有 CPU 耗时笼统归因于 race checker。

## 旧 helper 的 grid 假设不成立

旧 `resources.py` 取 host 源最后七个整数 cast，假定依次是 grid xyz、block xyz、dynamic SMEM。然而 TVM 根据 `launch_param_tags` 只传实际存在的维度。当前二维 grid 的 CUDA metadata 有：

```text
blockIdx.x, blockIdx.y,
threadIdx.x, threadIdx.y, threadIdx.z,
tir.use_dyn_shared_memory
```

没有 `blockIdx.z`。host 的实际 `main_kernel_packed` 调用有 15 个参数：9 个 tensor 指针，以及 slots 9–14 的 **`[2, 3, 128, 1, 1, 34048]`**。TVM runtime 将缺省维度初始化为 1，所以正确值是 **grid `[2,3,1]`、block `[128,1,1]`、dynamic SMEM 34,048 B**。旧 helper 多取了上文无关常量 2，误报 grid `[2,2,3]`。

错误只发生在资源收据解析：真正启动 kernel 的是 TVM runtime，它使用实际 metadata 与 host 参数。它不影响已执行的 FP64 检查、graph replay 或计时，也不改变 cubin 中 REG/LOCAL/STACK/static SHARED 的数值。当前 trace 的 dynamic/total SMEM 34,048/35,072 B 经 host 参数复核仍正确。旧 grid 字段在逐条复核前不得用于工作量、占用率或“实际启动了哪些 block”的论证。旧 JSON 保留原样，不能静默重写。

## 新 helper 与验证边界

[resources_v2.py](../src/branchtile/resources_v2.py) 让 CUDA module 导出 cubin 及其 `.tvm_meta.json`，再按真实 kernel packed call 的 stack slots 和 metadata tags 解码。缺省维度补 1，动态表达式或未知 tag 明确拒绝。它同时保存完整 `.host.c`、metadata、CUDA 源和 cubin，并记录各 SHA256。

[qualify_static_bank_v2.py](../scripts/qualify_static_bank_v2.py) 复制旧资格脚本，仅更换资源 helper，并在源码收据中同时保留新旧 helper 及原资格脚本身份；数值合同和所有可变输入检查保持相同。[trace_operator_compile_v2.py](../scripts/trace_operator_compile_v2.py) 包装该新入口。

四个 CPU 回归检查覆盖二维缺省 z、三维 grid、无关常量、拒绝不支持的参数，以及读取上述真实缓存 host 的纠错。测试没有导入 CUDA 库。新的 CUDA metadata 导出路径本身仍须在后续已分配 GPU 作业中取得完整资源收据；本次 CPU 通过不冒充新 GPU 资格。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_resources_v2_cpu.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/qualify_static_bank_v2.py --cpu-check --batch 3 --output /tmp/unused-static-bank-v2.json
```
