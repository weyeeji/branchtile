# Paged owner 集成的独立源码性能审查

2026-10-16；CPU 只读审查。未修改运行中的 frozen plan、kernel 或 runner，未运行 GPU。观察文件的 SHA、函数行号和静态 tail 工作量记录在 [source_audit.json](../results/paged_owner_source_review_v1/source_audit.json)。本记录不授予数值资格，不把源代码操作计数当成实测延迟或 occupancy。

## 结论

当前桥接版有可定位的额外成本：每步 CUDA page equality 检查导致主机同步、全容量全层索引重映射、pageable H2D 小拷贝、长私有尾单 CTA、输出 shadow 副本和常驻全层 partial。它们足以要求一次“实际时间花在哪里”的诊断，不能预先解释为共享算法本身的下界。

修复这些主要属于成熟技术的工程适配，尚没有超越 PAT/MSA/PageWeaver 的新机制。新 O-only 同合同结果和真实 XQA 路径应成为比较起点；旧额外计算自然 LSE 的 baseline 优势不再是性能依据。当前 task-parallel / packed / split / merge 结构本身不能重新命名为创新。

## 1. 每 decode 步共享页验证：主要风险是同步

`LayerSharedPagedOwnerPlan.prepare` 对 CUDA `tables[:B, :common_end//16]` 做逐元素相等、`.all()`，最后转成 Python `bool`。这需要归约和 scalar D2H，主机必须等结果。B3 的约 86K 上下文只是约 1.6 万个表项比较；字节数量小不能说明成本小，因为额外 kernel launch 和 stream 同步可能打断本来可重叠的调度路径。

`complete_prepare_host_seconds` 用主机时钟包围该检查，可能包含此前尚未完成的 GPU 工作等待，不能把这个字段全部叫“页检查自身耗时”。应同时看 CPU timeline 中 scalar extraction/copy 与 CUDA stream 依赖；不要把它和 GPU kernel 累加时间再相加。

**可验证的工程控制：**在实际 block allocator / `StagedWriteTensor.stage_write` 接收 CPU block IDs 时维护每 request 的页表版本，针对当前 `request_id + slot generation + document identity` 验证共同完整前缀；只有页表/slot/anchor epoch 变化才重验。热路径只读取已经验证且仍被 pin 的版本。需要覆盖 APC reset、页回收后重用、请求退出重排、原请求恢复、相同地址的新请求。`state['table_cpu']` 在每次 prepare_inputs 被置空，并非可靠的常驻新鲜 CPU 镜像，不能直接拿它替换检查。

另一个安全但未必快的控制是把验证放在 GPU，使用 failure flag 阻止非法共享并回退准确独立路径，异步回收错误；必须真的具备错误前的正确输出保障，不能只是在错误发生后告警。因此不能简单删除当前检查或延后读取 bool 然后继续错误 replay。

上述是版本化映射/普通缓存，不是新算法。若版本控制带来收益，同等机会应给 native sparse / 固定 union 控制。

## 2. 全容量重映射与 H2D 元数据

当前 `_map` 每步按 `ceil((HISTORY + B*max_generated)/256) × (layers*heads)` 发射。它在 48 层、每 rank 两个 KV head 上重新处理全部 frozen history，并为未到达的私有尾容量发射 masked 工作。每个向量还静态循环所有 B 来找 owner 或 tail row。即使实际只生成 128 token，而 `max_generated=8192`，grid 仍覆盖八千容量；有效 store 会被 mask，发射、分支和 descriptor 读取并未消失。

这不同于 E0 的 `_map_all_layers` 使用当前 `max(lens)` 作 grid。比较时至少记录两个 arm 的 grid、有效映射数、实际 history/tail 长度，不能只看 kernel 内的 K/V 工作。

**最小优化控制：**把 immutable history 映射和 append-only tail 映射拆开。新 epoch 或页表变化时映射 history；其余步只写新 token 的物理地址。B/排列变化时更新 row identity 与必要索引，不凭地址未变推断内容有效。也可让 producer 直接读取 logical position 和实际 page table，去掉中间 index buffer；这会增加 producer 地址依赖，必须完整计时，不能预设融合更快。两者都是常规分页/增量元数据方案。

`tail_lengths.copy_(torch.tensor(lens, dtype=int32))` 每步新建 CPU tensor，默认 `non_blocking=False`；subset 变化还拷贝 tasks/row_map/merge_indices/counts 和 question_rows。可以比较固定 pinned staging＋显式生命周期控制，或使用引擎已有 GPU seq lengths 在 GPU 更新 tail ends。不能无条件复用一个正在 DMA 的 pinned buffer；不正确的“异步”会产生间歇性错长度。

Inherited graph wrapper 先调用原生 `prepare_attn` 再做候选 prepare，所以候选仍支付本机 backend 的 metadata 准备。要量化这部分是否可安全省去，必须检查它同时提供的 block table、KV slot、positions 和混合 prefill/decode 服务；不能只因最终 attention 被替换就删除全部原生准备。

## 3. 私有 suffix 一个 CTA：长尾的直接假设

每层、head、活跃 request 仅一个 suffix task；history 用 `chunk_tokens` 切分，但 suffix 的 `end` 增长到整个生成尾。N64 下，8192-token suffix 的一个 CTA 串行执行 128 次 QK/PV tile。B3、H2 只有六个 suffix CTA；history 完成以后，它们可能决定 producer 的尾部。这里的“可能”仍需实际时间验证。

| suffix 长度 | 当前每 tail CTA 串行 tile | 每层 tail CTA 数，B3/H2 | 若按 512 切分的有效 tail CTA 数 |
|---:|---:|---:|---:|
| 128 | 2 | 6 | 6 |
| 512 | 8 | 6 | 6 |
| 2048 | 32 | 6 | 24 |
| 8192 | 128 | 6 | 96 |

**具体 variant：**预留每个 request/head `ceil(max_generated/C_tail)` 个 task，固定 task 和 partial 地址；每步 live end 为 `min(start+C_tail, actual_length)`，空 task 设 active=0，不写无效 partial。merge 的有效贡献数/索引同步更新，保留固定归并顺序。先预声明 C_tail=256/512/1024 与未切分控制，分别测 0/1/127/128/129/511/512/513/2048/8192，并验证退出、排列和页边界。history kernel/配置保持不变。

这本质是标准 split-KV，PAT/FlashInfer 已有先例。它增加 partial 写读和 merge 工作；短尾可能更慢。`full - producer_only` 不能自动推导可省时间，组件隔离改变了并发和缓存。应以同图完整 attention、整步 graph、真实 EOS 请求 wall 三层证据判断。

## 4. 输出副本、LSE 和全层 workspace

owner `merge` 直接写传入的模型 output，随后 `run_layer` 又把它 `copy_` 到 `self.outputs[layer]`，供 `graph_ops` shadow/检查路径使用。当前无 shadow 的 production 仍保留每层这次 D2D copy。可以让 graph 捕获时的资格引用绑定实际输出，生产省去副本；需要确保 compiler buffer reuse 后 shadow 检查读取的仍是目标层输出，不能让迟到检查读到下一层已覆写 buffer。较稳妥的做法是资格模式保留副本、生产模式明确禁用，并把这项差异及资格覆盖写清楚。

当前 merge 也总写 LSE，实际 production O-only 不消费它。生产 variant 可不计算最终 log、不写 LSE；保留 O/LSE 资格版用于数值诊断。这个开销很小或被覆盖的可能性也存在，不能重复旧 baseline 额外 ln 转换造成的不公平比较。

partial 为每层 `task_count*32*(128+2)*4` bytes，按最大 task 数、32 行固定容量分配；不是原先 sum(active rows) 的严格 packed 容量。不同 B 图、所有 layer 都常驻一份，prepare 中还预计算 request permutations 的 metadata。B3 可行不意味着 B8 可扩展：当前 plan 显式限制 questions≤4，runner 更限制 B≤3；M16/M32 也不支持 64 个 GQA 行。不能仅更改 assert 就用于 B8。

普通控制是让串行层共享最大 partial scratch、每个图使用相容稳定地址，并在同一 stream 上保证 producer→merge→下一层顺序；不同 stream/并行 replay 需要独立 scratch。空间节省可增加容量或降低图捕获压力，但不能直接算延迟收益；它同样是常规 allocator 复用。

## 5. 最小后续实验次序

1. 保留当前资格 run，不在途中改 frozen source。先确认每层 output/logit shadow、全部 tail 和页变化资格；qualification wall 因 shadow/CPU logits/teacher forcing 无性能意义。
2. 在新独立 variant 中先只去除 production 不必要 output/LSE 工作，再分别测同步验证、metadata map、tail split，避免一次改完无法归因。每个版本都保留原始对照和完整准备费用。
3. 记录未 profile 整模型 graph 和请求 wall；独立 profile 用于定位阶段，不能把 PDL 下的 kernel-sum/span 当严格 causal fraction。统计按文档/独立请求簇，多个层/重复 event 不是新应用样本。
4. 对同支持 sparse 问题，E0 是合法数值/性能控制，但它仍是 page1 适配，不自动代表生产 dense XQA。必须同时报告同任务 dense XQA，以及 sparse 质量是否成立。给不同 arm 加同一 8192 cap 不会固定实际输出长度或计算轨迹。

## 6. 有没有超越普通 PAT 的明确候选？

本次没有。可尝试的新实现是把 **冻结稀疏 owner history、append-only 私有尾、实际页表生命周期**组成完整可部署执行合同；但组成它的 query packing、split/merge、版本化页映射和 graph 稳定地址已有明确先例。与 PAT 的“任意非层次支持、动态私有尾”语义区别值得测试，不足以独立证明创新。与 MSA 的 KV-outer worklist、PageWeaver 的 query-owned 跨页状态、已有 FlashInfer live route 也仍需直接区分。

如果单变量优化后相对真实原生路径和简单共享控制仍有稳定端到端收益，可以再寻找驱动该收益的具体调度/布局原则；如果收益主要来自移除 `.item()` 同步或多余输出副本，应诚实归为桥接修复，不能把它写成新 shared attention 原理。暂时没有充分依据启动另一轮无边界的 tile、owner 或 bitwidth 搜索。
