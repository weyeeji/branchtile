# GQA4 原生 Q/O 地址输入：整模型开发试验

`results/gqa4_thinking_direct_quad_production_v1/train1` 完成全自然四阶段 ABBA，clean exit 0。相对同次进程的原生实现，ready比值 **1.105310×**、cold比值 **1.077776×**；计入每文档完整额外setup（包括导出）后的主净比值 **1.073101×**。原生A0/A3 generate最大/最小为1.003128。这是一个训练文档、单次进程的描述性结果，不是总体置信区间或新heldout结论。

原生cold均值70.982269秒，候选cold均值65.859950秒；完整额外setup为0.286928秒，包括离线receipt核对0.012598、配置0.132353、production建图0.137061、setup receipt导出0.004915秒。主净指标不摊销这些费用。共同初始化和输入准备不在常驻服务文档cold指标内；将共同组件相加得到首用组件估计1.050241×，它不是独立冷服务直接测量。

模型为Qwen3-4B-Thinking-2507，36层、TP2、BF16 KV，真实自然B8、共享前缀120528；候选只在预声明开发范围B4..8/private≤8192执行，其余原生回退。候选执行1611/4290个完整模型图步骤，其中B8/7/6/5/4分别为661/29/254/217/450步。计数不是GPU时间占比。所有问题自然EOS，无输出cap、无记录抢占。

四阶段的完整token/text、APC、finish/stop、采样配置和问题/位置图轨迹相同。严格和rounded质量均为0/8，2个预测解析失败，两臂相同；此前 `results/gqa4_thinking_task_quality_cpu_v1.json` 对答案单位、模型错误及文档/标注不一致的审计仍适用。该弱任务质量使它只能支持第二种模型结构上的性能和输出保持证据，不能单独充当主要应用效用证据。

## 资格与地址合同

新机制 `raw_direct_quad_native_state_continuation_grouped_hint_v1` 的direct build为 `bc26510388b35577`，copied quad base为 `28ecb045c76babdb`。生产只捕获production图，使用匹配新source/fingerprint的离线资格；每rank在B4..8各8个真实原生状态上通过检查，共80个rank状态、2880个层Q/O比较，包含native AA、hidden/logits、当前KV及输入保持等12项。

独立CPU审计核对了运行源、生成代码哈希、clean ledger终态及每rank180个B×层地址布局receipt。实际Q/O为 `[B,16,128]`，stride `[2048,128,1]`，原生地址alias为真、Q规范化copy标志为假。这个源和捕获合同不能代替独立GPU trace，更不意味着整个模型没有数据搬运。checkpoint导出、native tail/merge和当前KV写入都保留并计费。

审计入口为 `scripts/audit_gqa4_thinking_direct_quad_production.py`；完整结果保存在该运行目录的 `independent_analysis.json`。离线资格独立审计为 `results/gqa4_direct_quad_model_qualification_independent_audit_v1.json`。

## 与历史 copied quad 的关系

旧 `results/gqa4_thinking_quad_production_v1/train1` 的ready/cold/完整setup净比值为1.117702/1.088023/1.083037，新运行分别为1.105310/1.077776/1.073101。两次输入、全部四阶段输出及图轨迹相同，候选覆盖也同为1611/4290步；旧完整setup为0.306636秒。

**新单次整模型净比值低于旧运行，必须保留。**两者属于不同session，各自与本次native ABBA比较，不能据此声称移除输入搬运导致整模型增益或回退。已有standalone matched grid可以隔离算子输入搬运的成本；当前整模型单次试验不能完成同样的因果归因。对该bank若提出整模型copy消融结论，需要另行预声明同条件配对设计，而非取两个历史比值相减。
