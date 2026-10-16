# 论文实验推进台账

2026-10-16 继续探索。目标是满足 `proposal-iscas.md` 的完整论文证据要求，
不是把上一轮算子否定结论重新定义为论文实验完成。用户授权继续研究、必要时
更换 idea；普通用户权限、虚拟环境、八卡合理并行和及时提交 git 继续有效。

`FEASIBILITY.md` 是第一轮有界可行性结论：当前动态行实现没有达到推进门槛。
它的原始数据、失败和限制保持有效。后续若改变主张，必须明确说明新机制与
既有方法的区别，并重新通过强对照、质量和完整应用要求，不能只改标题。

## 完成审计矩阵

| 要求 | 当前权威证据 | 当前判定 | 尚需取得的证据 |
|---|---|---|---|
| 主模型身份、TP2 FP8 MoE、BF16 KV | `results/platform/engine_main30b_tp2_docfinqa.json`；官方权重/后端校验 | 单完整文档部署通过 | 其余长度与显存覆盖、正式数据的可服务性；当前不是全数据容量证明 |
| DocFinQA 公开来源冻结划分 | 替代 v3 manifest 和身份/泄漏审计；用户明确不采用旧数据 | 已获新协议授权，审计待完成 | 公开冻结版本、身份歧义与 label 冲突处理和适合实际独立单位的统计，不冒称旧公司分布 |
| QMSum train 开发/test35 边界 | 官方来源及冻结 manifest | 输入准备完成、评价未完成 | test35 的冻结配置自然输出、全部失败分母、质量与完整成本 |
| DocFinQA 官方答案解析和独立 fixture | 原始数据及冲突审计 | 未完成 | 数字/单位/百分比/舍入/FINAL 规则，不能事后挑数字或修 gold |
| A0 一次多问、A1 Dense+APC | `results/application_screen_v2/`，train8 两臂共64个自然结束答案 | 开发筛查完成，正式未完成 | 图优化原生强基线、三 seed/正式质量、冷暖成本及模板冻结 |
| A2 KVzip/FastKVzip | 文献与适配线索 | 未完成 | 合法 gate 或诚实兼容性判定、共享压缩实际质量与成本 |
| A3 强 RAG | `results/retrieval_preparation/`：表格/整行分块、BM25+embedding+reranker八配置及完整准备成本 | 开发生成进行中 | 自然输出质量、配置冻结、强检索对照及正式完整费用 |
| A4 query-aware 原生及共享源 | QC-SnapKV capture；无 sparse 自由生成 | 未完成 | 真正执行稀疏生成，full-budget/Dense、质量预算冻结、准备成本 |
| 自然 opportunity 与共同 ready | 1024 个 dense 轨迹快照、逻辑 tile 分析 | 部分完成 | 官方主模型实际 rank 形状、自然输出步、真实调度 ready 生命周期 |
| E0/E2 强执行 | 原生 separate/joint 和固定 union，31 组配对 | 部分完成 | 主模型自然矩阵、轮转 capture 缓存、准备摊销、第二卡/卡对复核 |
| E1 全部固定 partition | `results/partition_cascade/summary.json`，全部15划分、原生/自写、准备和第二卡 | 初始4形状执行完成 | 官方主模型与多文档自然矩阵、部署选择与事后 oracle 分开 |
| E3 及 2×2 | 当前全 canonical partial 的 48 点消融 | 部分完成 | packed-active partial、合理 split/直接输出、局部固定组融合强控制 |
| E4 精确层次共享 | 同上，15棵精确二叉树，含gather/LSE/merge/准备 | 机制移植完成 | 不是完整QUILT；主模型自然矩阵与完整应用适用性仍待验证 |
| E5/PAT/PageWeaver/Fused3S/BSA | 文献与部分源码审查 | 未完成 | 逐项适用性和强兼容实现；移植不得称作者完整复现 |
| 候选独立数据流贡献 | v1/v2/v3 没有目标性能收益 | 未满足 | 新增量的可证伪预测、相对所有适用强对照稳定约15%完整 call 信号 |
| 全 alive 子集、数值、logit、full-budget | 当前算子 FP64/内存检查；native 小形状 LSE 失败保留 | 部分完成 | 全子集和生命周期、层/logit、稀疏自然生成；开发冻结原生校准容差 |
| 引擎共享源安全与连续补入 | 仅原生 APC 部署 | 未完成 | COW、reader、取消/抢占、引用/slot generation、图失效和不等待兄弟请求 |
| 100 报告/三 seed/公司聚类质量 | 无正式评价 | 未完成 | 全分母、公司身份、10000次聚类区间、2pp 非劣界、有限样本保守界 |
| QMSum 官方 ROUGE/盲审 | 无完成回答评分 | 未完成 | 官方兼容评分、冻结 judge/rubric、事实与覆盖抽查；不能只凭 ROUGE |
| 完整应用约5%净改善 | 无候选集成 | 未完成 | 同质量、同资源、冷暖/复用摊销、TP通信、自然长度/失败计费 |
| 能耗/EDP | NVML 可读，尚无正式配对窗口 | 未完成 | 更新粒度、20–30s同工作量窗口、反序、gross板卡能量、失败分母 |
| 硬件归因 | cubin资源；NCU权限失败 | 部分完成且有平台限制 | 普通用户可得证据；硬件counter不可用时收缩归因，不使用root或虚构流量 |
| 可复现 artifact/论文图表 | 第一轮代码、raw JSON、图、台账已推送 | 部分完成 | 新实验配置/冻结协议/全部结果/最终主张一致，最终逐项审计 |

任何“部分完成”都不能在最终完成审计中当作完成。算子计时、逻辑机会、
任务质量和实际应用费用分别提供证据，不用一个范围代替另一个。

## 当前并行工作与预算（14:56 更新）

- GPU0/1：官方30B FP8真实TP2的post-RoPE Q/K/V和同forward共同ready捕获，逐位核对真实分页cache。
- GPU2：开启CUDA Graph的原生引擎A/A；GPU3：官方4B Instruct自然EOS开发筛查。
- GPU4/6/7：QMSum train8的八配置RAG开发生成，8192/题cap、无min_tokens，检索成本单列。
- GPU5：原生TopK资格及selector强基线；长CUB首次编译改为无GPU可见的CPU准备，等待期间释放GPU。
- CPU/文献：恢复或审计历史划分、核查新颖性、评分 fixture；已找到 MiniKV
  的 LSE 复用先例，因此该路径只作为强 selector baseline，不冒充新机制。

沿用首轮 P1/P2 累计预算24/8 GPUh，所有新作业、加载、编译、失败和重试
仍写入 `results/job_ledger.jsonl`。前轮台账分别0.937/3.093 GPUh，早期不完整
计时另有明确预留。到限重新用实测费用决定下一阶段投入，不把预算当完成保证。
一秒利用率和实际分配分别记录；并行的是有研究用途的作业，不用空转提高数字。

## 本次开发预注册

`scripts/screen_application.py` 只消费冻结 train8；temperature0.6、top_p0.95、
top_k20、min_p0，seed按文档/题目确定，尚不作三 seed 非劣结论。
独立回答沿用原文档前置模板；一次多问使用完整同文档和所有自然问题，严格
解析 q0…q(B−1) JSON。缺失、重复 ID、未结束 thinking 和碰 cap 均记录，
不从 reasoning 中补答案。冷臂清空 APC；暖重复单列，不能伪称 A/A。

默认 FINAL 模式的引擎逐 step 输出只反映完成，不能用来推断 token co-emission 或 attention ready 集合。
当前八卡邻近负载下记录的 NVML 总能量只用于完整费用记录，正式能耗结论
必须另做受控配对窗口。准备/初始化/JIT 与请求 JCT 分开保存。

已有负结果是下一实验的约束。候选转向优先依据自然机会、强 baseline 的
剩余时间和新颖性核查；不能靠删除不利文档、降低质量、换统计口径或
降低原门槛制造成功。当前没有已验证成功的新 idea，目标保持进行中。

2026-10-16 用户澄清：“没有，不管里面提的旧数据”。旧 manifest 不再是待用户解决的
依赖；以官方公开来源的新冻结协议推进，保留所有来源和泄漏审计。
