# Owner-task 阶段诊断：wave23 独立 CPU 复核

2026-10-16。**八项诊断全部完整结束，原始中位数可精确复算。32K 下独立 merge 图约9.4–9.7us，值得继续调查终端归并；128K 下 producer-only 已约100.3us，不能靠删除一次 merge 启动解释整个长输入瓶颈。** 这些只是不同图的测量，阶段时间不可相加或相减成可移除开销，也没有测到融合后的速度。

本轮复核没有运行 GPU、导入 torch、编译 kernel 或修改 runtime，新增 GPU 用量为0。复算入口是 [audit_operator_owner_stages_wave23.py](../scripts/audit_operator_owner_stages_wave23.py)，完整证据为 [audit.json](../results/operator_owner_stage_diagnostic_v1/audit.json) 和 [全部64项中位数CSV](../results/operator_owner_stage_diagnostic_v1/stage_medians.csv)。

## 核查范围与结果

- [冻结协议](../configs/operator_owner_stages_wave23_protocol.json)的7项 source/jobs SHA全部匹配；8份结果各自记录的6项源码SHA也全部一致。producer和merge launch的AST逐项等于 `Attention.__call__` 的对应调用，grid、buffer、constexpr、warps与stages均相同。
- 8个作业各有唯一start/end，uid1000、项目`.venv-latest`、P2、exit0、无timeout；分配GPU为0–7，每个区间没有账本中的同卡其他作业重叠。费用合计 **0.02044224234375482 P2 GPUh**，包括启动、初始化与资格开销。结果未保存GPU UUID，本复核不把账本索引当成独立的UUID实测。
- 每项80行，均为hot/cold各20个random/reverse配对block；全部随机顺序按冻结seed重建一致。总计 **640行、2560个计时值、64项完整/阶段/A/A中位数**，与原JSON逐项精确相等。hot图每次20次调用、cold图每次1次调用；两种cache状态分别计算。
- 32条原始完整O/LSE检查均记录passed：原wrapper、独立调用同两kernel、hot图后、cold图后。记录的最大FP64绝对偏差分别约0.00120717和9.60e-6，原assert_close容差未改。CPU复核只检查这些已有记录，没有重新执行数值验证；它们不替代wave17/19的完整形状与可变输入graph资格。
- 根据loader实际访问集合重hash **48份capture文件**，与前次冻结receipt一致；metadata、source、mask、各自然active request的decode记录均保留。源码中system prefix和完整private suffix路径未改变。前次审计的总SHA检查数还含其他文件，不能误写为本轮实际加载了52份capture。
- chunk选择与wave20的对应hot赢家一致。旧波结果只用于配置和provenance核对，**没有用旧波strong时间除本波producer-only时间**，也没有产生跨波speedup或移除merge的预测。

这8格来自3份文档、TP rank0、每rank两KV头、GQA8、D128、BF16、step64；不是8份独立文档或每格两卡复测。chunk来自前次有限网格的逐capture选择，本轮没有新的参数搜索或部署策略资格。

## 同波热态中位数

单位us。`complete`含producer+merge，输出O和自然对数LSE；两个stage-only图分别重复一个阶段，使用同一组buffer。

| 真实输入 | chunk | complete | producer_only | merge_only | complete A/A |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 B3，L15，4K | 128 | 9.457 | 5.870 | 3.821 | 9.457 |
| D0 B3，L31，32K | 256 | 22.523 | 13.386 | 9.422 | 22.572 |
| D1 B3，L15，4K | 128 | 9.564 | 5.883 | 3.926 | 9.556 |
| D1 B3，L31，32K | 256 | 23.307 | 13.786 | 9.709 | 23.308 |
| D2 B4，L15，4K | 128 | 9.781 | 6.118 | 3.863 | 9.791 |
| D2 B4，L31，32K | 256 | 26.242 | 17.115 | 9.355 | 26.190 |
| D2 B4，L15，64K | 512 | 63.881 | 53.032 | 9.631 | 63.842 |
| D2 B4，L31，128K | 1024 | 111.869 | 100.305 | 9.693 | 111.891 |

D0=`docfinqa256`、D1=`publicdev1_256`、D2=`publicdev6_256`。32K的逐request贡献数为130–133，64K为131–132，128K为131–134；chunk随预算增大，使这些格的merge循环规模仍接近。4K贡献数为34–38。这个结构与merge热态时间变化一致，但没有把循环次数与其他布局因素单独控制，不能直接宣称它是唯一原因。partial payload与workspace均已按实际packed rows独立复算；这些是逻辑字节，不是动态DRAM/L2流量。

## 时序解释的边界

**stage-only与真实调用有不同的驻留、依赖和调度。** hot producer-only反复覆盖partial；hot merge-only反复读取同一批partial；它们不等于真实producer刚完成后的merge环境。cold merge则先人为flush partial，与真实两kernel链尤其不同。由此，`producer_only + merge_only`、`complete - producer_only`都不是已测可移除merge/launch成本、有效overlap时间或融合速度上界。所谓“producer-only已经比旧strong快”也不是完整attention的性能证据。

热态complete/complete_AA的配对ratio中位约0.9990–1.0021，个别小偏移的bootstrap区间不跨1，全部保留。人工冷态存在明显粗粒度计时与A/A差异：D1/4K的两个完整图中位18.432/20.480us，D2/32K为53.248/55.296us。不能过滤这些点，也不能用[1,1]的窄bootstrap区间宣称没有系统误差；区间只反映这20个配对block内的重采样，不包含计时量化、板卡或跨文档不确定性。各格cold全部四个中位数与A/A区间保存在CSV/JSON，未与hot混合。

原始 [bench_owner_stages.py](../scripts/bench_owner_stages.py)没有修改任何attention算法；32条输出检查也都使用未变化的同一Q/K/V。因此本轮不提供新的ready-set切换、graph拓扑更新、自然EOS、任务质量、完整模型时延、能耗或硬件流量结论。

## 后续优先级

1. **32K先调查终端归并的实现空间。** standalone merge约9.4us足以构成有意义的测量对象，但不是可免费拿回的9.4us。若尝试last-producer或其他终端组织，应保留同一producer工作量，重新测完整O/LSE调用、A/A、准备和workspace，并在同一波重新执行最快合法E0/E1/E4；不能以stage-only代替候选完整调用。counter/fence、尾部串行和额外寄存器都可能抵消节省。
2. **64K/128K优先解释producer。** 长输入的producer-only仍为53/100us，当前资料不足以把它归因于带宽、寄存器或CTA不平衡。下一项有信息量的干预是同一kernel、同一task边界下比较原indexed source、contiguous identity-index、direct slab地址，并同时给native控制相同布局机会；先证实加载路径有完整调用增量，再考虑一次准备/所有层驻留和真实ready生命周期。只看logical visits不足以选择路线。
3. **4K继续先看完整调用与生命周期。** producer-only约5.9–6.1us、merge-only约3.8–3.9us不能相加为新预测。当前图外构造、描述符与native plan缓存也应同等处理；修复反复Python重建只能先归为适配器工程。复用长度按真实ready epoch记账，不按生成cap摊销。

上述优先级不产生新颖性资格。[固定FlashInfer PrimTS源码](https://github.com/flashinfer-ai/flashinfer/blob/38e6b8944fe8b6497d793ef21fbaadd797ea39d8/flashinfer/attention/prims_ts/decode.py#L3551)已有fused split-KV completion counter和最后CTA复位；[官方counter接口](https://docs.flashinfer.ai/generated/flashinfer.get_dcp_spec_counter_bytes.html)公开一次清零、自复位的存储约定。普通last-block归并更有[NVIDIA样例](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh)先例。那些原生路径的架构/shape支持不等于本任务SM120任意owner输入已被支持，但适配不能重新命名为首次融合归并。相关固定版本、SHA与布局/图生命周期边界见[下一步机制审查](next_operator_mechanisms_20261016.md)。

复算命令：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_owner_stages_wave23.py
```
