# Owner-task E3 独立 CPU 审查

日期：2026-10-16。审查基线为实现 `1185774`、CPU 证明 `bc475a3`；最终
host guard 复验绑定修复 `aac3e85`。审查者未编辑实现、planner、kernel、资格
或 benchmark 脚本，未启动 GPU、CUDA driver、kernel 或 graph。所有 Python
命令使用普通 uid 1000 的 `.venv/bin/python`；涉及 Torch 或离线编译的进程均
设置 `CUDA_VISIBLE_DEVICES=''`。

**结论：发现的 host 元数据对象替换缺口已经由实现作者修复，并通过独立
CPU 回归。有效原始 RegionPlan 下，未发现 task 覆盖、packed row 或 merge
索引、自然对数 softmax 合并公式的其他错误。可继续 GPU 资格；本记录本身
不授予 GPU 数值、图功能、资源或性能资格，也不构成新颖性结论。**

## 审查内容与证据

| 项目 | 独立结果 | 证据边界 |
|---|---|---|
| 原 CPU coverage / FP64 partial+merge | 修复前后各复跑 26 case、1,456 query rows，通过 | 复用了既有 CPU 脚本；不是 Triton 数值执行 |
| 随机覆盖及 merge 索引 | 200 case，通过；随机种子 1185774 | 单独生成 owner 数据并逐边枚举，不依赖 kernel |
| 原 host guard 失败复现 | 真实 CPU `row_map` 被不同值、相同 `_version=0` 张量替换后仍获接受 | 独立于任何 CUDA 执行 |
| 修复后真实张量 guard | 16 类改变被拒绝，原 Indices 原地值改变获接受 | 从实际源码 AST 提取 snapshot / validation 方法执行；CUDA 未初始化 |
| CPU 调用边界 payload alias | 两次 host 调用保持 Q/K/V/Indices/row_map 对象及数据指针 | 使用真实 CPU 张量和记录参数的 producer/merge stub；不代替 GPU graph alias 资格 |
| Triton 3.6.0 离线编译 | sm120 共 12 格生成 cubin，通过 | 显式 GPUTarget，禁止 runtime driver 访问；无加载、launch 或 GPU 测量 |

原数学检查命令为：

```bash
CUDA_VISIBLE_DEVICES='' timeout 60s .venv/bin/python scripts/qualify_owner_tasks_cpu.py --output /tmp/owner_tasks_independent_cpu_fixed_20261016.json
```

随机检查覆盖 B3/B4、1–4 个 KV heads，token 数包括
1/2/31/32/63/64/255/256/257/511/512/513/1023/1024/1025；各 head 独立生成
owner bitmask，chunk 数包括 1/31/32/63/64/255/256/257/511/512/513/1024。
它核对每条 `(head, query row, source token)` 边恰好一次、packed row 无洞无
重叠，以及每个 merge base 加 GQA offset 后精确返回对应 head/request 的
全部 task rows。既有 fixture 另外覆盖全空、单 token、全部 masks、显著不均长
与每请求 33 个 contribution 的三组流式归并。

## 覆盖与数值推导

[planner](../src/branchtile/owner_tasks_plan.py) 顺序遍历每个 head/region，
将 `[offset[r], offset[r+1])` 切成不交叠非空 chunk。每个 chunk 的 packed
基址按 active rows 递增。对某请求，`packed_base + request_slot*8 + g` 与
该 owner 的有序 GQA8 row map 一致；merge table 没有跨 head/request 混行。
不同 head 缺失某 owner 时，其对应 region 长度为零且没有 task；counts 和
row map 仍描述合法 cohort，这与原 `make_plan` 的行为相符。

[producer / merge](../src/branchtile/triton_kernel_owner_tasks.py) 的每份
partial 是未归一化 `(U, m, l)`。scores 已除以 `sqrt(128)`，U 与 l 相对于
同一自然指数最大值 m；merge 用 `exp(m_i-new_max)` 同时缩放 U 和 l，最后
以 U/l 输出，LSE 为 `new_max + log(l)`，没有混入 log2 单位或重复除分母。
masked rows 的 safe maximum 为零，因此全空路径不依赖 `-inf - -inf`；无
contribution 时 producer 不发射，merge 输出零和负无穷。masked padding
可能形成负地址表达式，但其 load/store mask 为 false；仍须 memcheck 验证
实际 lowering 与硬件执行。

producer 在 PV 前将概率转换为 BF16，分母仍累加 FP32 概率，最终 O 再舍入
为 BF16。不同 chunk 边界会改变概率的局部缩放和舍入，因此 Python FP64
恒等式不能证明实际输出满足原 FP64 资格阈值。必须使用
[原 reference](../src/branchtile/attention.py) 的同一接受标准测试所有 arm；
不能因为该数学证明通过而放宽误差门槛。

## Host 合同缺口与修复

`1185774` 仅保存 `_version`，无法识别相同版本的新张量或新 plan。
尤其替换 row_map 后，producer 可能读新配行，而 merge 仍使用旧 task table，
形成实际错误。该问题已及时报告，修复由作者提交为 `aac3e85`。

独立回归确认修复拒绝：四个使用中 plan 张量的同版本替换；offsets/counts/
row_map 原地修改；plan 替换；batch/GQA/rows 或 label 值变化；Indices 的
shape、stride、storage 或 dtype 变化。原 Indices 的原地值变化，以及 labels
相同值的 list 替换仍可通过。强引用防止对象或地址复用掩盖替换。作者的更完整
持久化回归见 [host guard 记录](owner_tasks_host_guard_fix_20261016.md) 与
[19-case 收据](../results/operator_owner_tasks_guard_cpu_v1/qualification.json)。

该 guard 不是内容 checksum。独立确认 `row_map.data.add_(1)` 可以在不增加
原张量 `_version` 的情况下改值；device kernel 写 metadata 同样不受普通
host 版本检查保证。直接 `graph.replay()` 完全绕过 host validation。
因此所有 metadata 内容都必须由调用方保持固定；改变拓扑必须重建并重新捕获。
Q/K/V 和原 Indices 只可在既定地址、shape/layout 及合法 region/index 对应
关系内原地更新。

## 离线编译与 payload / prepare 计费

离线编译使用 `triton.compile(ASTSource(...), target=GPUTarget('cuda',120,32))`，
producer 固定 `D=128, num_warps=4, num_stages=1`，覆盖 B3/B4 × N32/N64；
merge 覆盖 B3/B4 × `MAX_CONTRIB=0/1/17/33`。`DriverConfig.active/default`
与 `_create_driver` 被替换为直接抛错的函数，前后 `_active/_default` 均为
None；未请求 compiled kernel 的 run/launch handle。命令有 60 秒进程超时。
编译器报告 producer shared memory 为 N32 的 18,432 B、N64 的 28,672 B，
merge 为 0 B。这些仅是编译 metadata，不是已测寄存器、spill、occupancy
或硬件吞吐；GPU 运行时的指针对齐 specialization 也可能改变资源结果。

源码仅把原 Q/K/V、Indices、row_map 引用传给 producer，没有复制或重排 KV
payload。CPU 调用边界也确认指针保持原样；`payload_aliases_source=True`
目前有源码和 CPU 边界证据，捕获地址及其原地 mutation 仍待 GPU 资格。
`logical_kv_token_visits` 是所有 chunk 的长度之和，即有效原始 KV token
位置总数；它不计每请求的重复 MMA 行、Q 重读、cache miss 或实测 HBM 字节。

构造函数的 `prepare_ms` 从 metadata D2H 前计至 CPU 验证、task/table 构建、
H2D、buffer allocation 与结束同步之后，覆盖本构造函数准备工作。原
`make_plan` 在外部，其时间与内存必须另外记账；首次 Triton JIT、warmup、
graph capture/recapture 也在外部。输出 O/LSE 已计入 owned workspace；
原 plan、原 payload、host Python lists、allocator reserve 和编译临时内存
不能被误解为已全部包含在此数字内。结束同步在非静默进程中可能包含其他
待执行工作，因此实际计时仍须检查实验隔离。

## GPU 资格仍必须完成

所有 chunk256/512/1024 × N32/N64 都需在 B3/B4、实际 source shape 和完整
suffix 上核对原 FP64 reference，覆盖不同 head 的 owner 集、空请求/全空、
不均长与 skew、tile/chunk 尾块、跨 16/32 个 contribution 的 merge。
同一 graph 原地更新 Q/K/V/Indices、拓扑改变后的 host 拒绝与完整
rebuild/recapture 必须分开报告。另须验证真实 pointer alias、memcheck /
racecheck、全空路径、编译资源及完整 producer+merge 计时。CPU 通过不应
隐藏某个配置的失败记录，也不能自动转成八卡性能结论。

最终复验源 SHA256：

```text
attention_owner_tasks.py       cc9d59e704cff075eaba2b4e92b5fa199e52475de4b5241ee58d6d2b6aa3987d
owner_tasks_plan.py            e1d41289a183518f716fbd43bf6c98d4412f9da24ebf8ae7070cce3a739f4ec2
triton_kernel_owner_tasks.py   48e8cd01e2aeae1d8f8c06e6370f06c3f186c7d3a51e2b1ba614ad3ede9e698e
```
