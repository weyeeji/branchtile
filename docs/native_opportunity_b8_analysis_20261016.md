# B8 长文档原生 XQA 机会测量

四个TP2作业完成：两个训练文档，每个真实8题，同时用BF16/FP8 KV分别运行。文档约120.5K和144.6K tokens；仅固定128-token完整引擎AA及单独profile，不是自然质量或候选加速结果。

| 训练文档 | KV | 固定生成AA秒 | attention/kernel累计占比，两rank | attention/trace跨度，两rank |
| --- | --- | --- | --- | --- |
| train1 / 680456 | BF16 | 2.728 / 2.719 | 35.16% / 35.67% | 52.94% / 54.12% |
| train1 / 680456 | FP8 | 2.326 / 2.320 | 31.47% / 30.99% | 44.74% / 43.78% |
| train7 / f1d5c3 | BF16 | 3.033 / 3.040 | 37.02% / 37.25% | 58.88% / 58.09% |
| train7 / f1d5c3 | FP8 | 2.643 / 2.635 | 33.29% / 33.47% | 48.77% / 49.14% |

四job每题fixed AA tokens和APC命中相等、两rank调度轨迹相等；wall max/min1.00238–1.00355。8个trace均恰32×48个XQA `kernel_mha`，全部实际元数据验证为FlashInfer/XQA。与wave26一样保留kernel重叠/PDL导致累计分母重复计时的限制，不把任何一个占比当作实际可部署端到端加速。

该场景的attention窗口占用更大，值得继续以保持O完全一致的整模型干预测边界。但B8与先前B3的文档长度、问题、配置不同，不能把差异纯归因于batch扩大；后续若要做batch消融，需同文档实际子集与相同prefix长度。

## APC 与容量

所有fixed请求有约120528–120560或144576–144608 cached tokens，说明大部分共享前缀命中。`use_cascade=False`，但这不等同于硬件没有物理缓存复用，也不能据此宣称全量重复HBM流量。

同样 `gpu_memory_utilization=0.9` 下，每rank分配的KV pool：BF16为13,444,841,472 bytes、273,536 token容量；FP8为13,445,234,688 bytes、547,088 token容量。其物理pool几乎相同，FP8容量约翻倍。不能说本次实验实际GPU显存减半。与B3的0.8内存配置也不能混比pool大小。

分析脚本从完整summary和全部trace独立重算，保留source/trace SHA及真实元数据：

```sh
.venv/bin/python scripts/summarize_native_opportunity.py \
  --root results/native_opportunity_b8_v1 \
  --output results/native_opportunity_b8_v1/analysis.json
```

本轮没有自然EOS完成率或任务质量结论；相关自然实验应单独报告。生产prefix多请求attention已有直接先例，问题空间大不意味着新机制已成立。
