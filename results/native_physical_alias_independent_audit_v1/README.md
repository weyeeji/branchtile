# Native physical-alias gate：独立只读审计

2026-10-16。冻结源 `6c9c7da`；**1470 条 CPU 检查通过，未发现使本轮同算子地址干预无效的实质错误**。本审计未重跑 GPU、未改冻结源码/协议/原始结果/账本。复算命令：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_native_physical_alias.py
```

[audit.json](audit.json) 保存输入 SHA、逐项检查、独立 bootstrap、全部配对差、A/A、地址轮换和按 `name+pid` 匹配的账本终态。不是只读生产 summary 后转述。

## 资格与样本完整性

- 8/8 作业 clean exit 0，无 timeout；8 个不同 GPU UUID，所有 2 文档 × 4 预算均保留。按预声明账本计 **0.018619587832234 P1 GPUh**，未改列为 P2；独立审计额外 GPU 用量为 0。
- 48/48 arm 通过原冻结 O FP64 条件；记录最大绝对误差 **0.00228278698**，所有点甚至低于纯绝对阈值 .006。每 cell 六臂 O 和 raw LSE 均逐位一致。**raw LSE 此处只验证跨地址逐位等价，没有新增 FP64 LSE oracle。**
- 在 CPU 重建每请求/head 的完整 prefix、文档 mask、私有 tail 与 replica offset，全部 int32 indices SHA 与生产记录一致；核对真实 same-forward ready 元数据、实际原生 cache 一致收据。
- 同一 wrapper/plan、Q、索引/indptr、workspace、K/V allocation、输出及最终 copy buffer 指针不变；只更换索引内容。三个共享地址轮换和三个 private copy 映射正确。
- 384/384 地址敏感检查完整：每 cell × hot/cold × 8 graphs × 3 replicas，修改对应 V 后只有被映射请求变为 17，未映射行逐位不变，恢复后全输出逐位恢复。生产 assertions 与完整成功记录相符；不是仅以相同数值输出推断图实际使用新地址。
- 每 cell、每 regime 恰好 **60 行 = 30 blocks × random/reverse**；共 960 配对行、7680 个 timed graph samples。hot 一个 sample 为 32 次调用链的均摊，cold 为单次调用；两者不能混为同一计时精度。
- seed 随机顺序及其反序全部重现；所有 median、5 种 ratio 的 10,000 次 block-cluster bootstrap 95% CI 与原记录在 1e-12 内一致。random/reverse 同簇，没有当作 60 个独立样本。

## 全部预算结果

数值为 paired median `private/shared`，依次列 rotation 0/1/2；大于 1 表示共享物理地址较快。它们是**同 cell、同卡、同逻辑输入与同计划**的干预；不比较跨行绝对时间以推断预算阈值。

| 文档 | budget | GPU | hot，三轮换 | cold，三轮换 |
| --- | ---: | ---: | --- | --- |
| original | 4096 | 0 | 1.0005 / 1.0074 / 0.9992 | 1.2222 / 1.2222 / 1.2222 |
| original | 8192 | 1 | 1.0084 / 0.9917 / 0.9956 | 1.3636 / 1.3636 / 1.3636 |
| original | 16384 | 2 | 0.9993 / 0.9945 / 1.0052 | 1.5333 / 1.5333 / 1.5333 |
| original | 32768 | 3 | 1.7294 / 1.7171 / 1.7977 | 1.6957 / 1.7273 / 1.7727 |
| publicdev1 | 4096 | 4 | 0.9979 / 1.0037 / 1.0007 | 1.2222 / 1.2222 / 1.2222 |
| publicdev1 | 8192 | 5 | 1.0022 / 1.0174 / 0.9843 | 1.3642 / 1.3636 / 1.3642 |
| publicdev1 | 16384 | 6 | 1.0037 / 0.9880 / 1.0017 | 1.6000 / 1.5333 / 1.5333 |
| publicdev1 | 32768 | 7 | 1.8029 / 1.7534 / 1.7249 | 1.6957 / 1.6652 / 1.6957 |

**Cold 的 24/24 个轮换比较通过冻结方向门槛**：paired ratio CI 排除 1；paired median 绝对时间差至少 2.048 μs；差值大于两种 A/A 的绝对偏移。本审计同时取 paired-delta 和两个边际 median 差中的更大 A/A 偏移，结论不变。Cold A/A 的两个偏移均为 0；干预差约 4.080–34.816 μs。部分 CI 因单次事件量化退化为单点，不能据此声称无限精度。

Hot 的两个 32K cell 在三地址轮换上都呈现明显共享收益（1.717–1.803×）；4/8/16K 的结果在约 0.984–1.017×，多个 cell 会随轮换变号，不能择优称普遍正收益。保留全部 A/A：16 个 hot A/A CI 中有 1 个排除 1；该小偏移不影响 32K 的大幅配对差，但提醒不能把所有细小显著性当作稳健物理结论。CI 是逐点的，未作多重比较校正。

## 可支持的结论与边界

本轮提供了过去缺失的因果控制：**原生 E0 本身能从共享物理地址获得性能收益**，至少在这些冻结冷缓存 cell 和两个热 32K cell 上。复制保留完全相同的 BF16 KV 数值、Q、支持顺序/长度、计划、算术与总分配；因此结果不能再只归于不同 mask 的数学工作量。它不是新算子，也不意味着额外联合 selector 的准备成本得到偿还。

所有 arm 都保留三个完整副本，改变的是实际访问的地址集合。活跃 footprint、TLB、内存分区和缓存复用共同变化；不能把结果缩写成已测 DRAM 字节或 L2 hit rate，也不能声称 GPU 分配容量下降。大于 2×L2 的 flush 在 timer 外，hot 每臂先自热；复制、索引安装和 plan 在计时外，O-only 原生调用与必要 `output.copy_` 在内。

输入来自真实主模型，但这里仍是单 rank、单层 L15、step64、B3/GQA8 的连续导出 BF16 cache，使用 FlashInfer 0.7.2 FA2 page1 接口；不是生产 48 层 strided cache、FI 0.7.0 XQA、完整 TP2 step 或自然任务 JCT。八卡分配与预算一一对应，所以目前不能把 hot 差异归结为跨预算的硬件相变。需要新的独立主张时才另冻控制；本审计不申请或启动扩展实验。
