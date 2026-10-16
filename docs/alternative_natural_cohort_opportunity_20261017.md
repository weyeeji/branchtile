# 自然多问题长文档：CPU 场景机会核查

唯一建议进入后续实验准备的候选是 **FinanceBench 的公开自然文档组**；这不是收益结论，也不替换正在运行的 DocFinQA next8。没有运行 GPU、下载模型或读取候选模型输出。CPU 分析使用现有虚拟环境与 Qwen3 tokenizer，`CUDA_VISIBLE_DEVICES=''`，最终确认 CUDA 未初始化。

## 数据源和选择规则

官方 [FinanceBench 仓库](https://github.com/patronus-ai/financebench) 公布的是 150 个案例，不能将论文介绍的全部 10,231 题当作已可用公开数据。本次固定仓库 revision `cc39aeb4afdf33909ee1412188bf89035950c2eb`，核对两个官方 metadata 文件、原始 PDF 的 Git blob SHA 和文件 SHA256。

按原 150 行首次出现的文档顺序，保留**所有自然 distinct questions >=6 的文档及所有问题**，在读取 PDF 和 tokenization 之前写入 `data/raw/alternative_cohort_metadata_v1/financebench_cohort_selection_before_pdf.json`。没有按长度、答案或性能替换文档。84 个自然文档的题数分布为：52×B1、14×B2、12×B3、2×B4、1×B5、3×B7；没有 B8 或更大组。

| 原始文档 | 自然问题 | 全文 tokens | 最大 prompt + 8192 | 当前 163840 上限 |
|---|---:|---:|---:|---|
| AMD_2022_10K | 7 | 95,574 | 103,831 | 可行 |
| AMERICANEXPRESS_2022_10K | 7 | 195,431 | 203,691 | 超限，保留失败 |
| BOEING_2022_10K | 7 | 127,168 | 135,426 | 可行 |

全文通过系统已有 `pdftotext -layout` 提取，不截取证据页、不拼接或复制文档。长度使用一个公开记录的简单 QA 提示；这是长度检查模板，尚不是冻结的性能/评分协议。后续正式准备应保留三文档/21题分母，两个可行文档14题；不能因 AmEx 超长而换入别的文档。当前 P>=65536、B>=6 的策略在两份可行文档起始阶段有执行可能，但仅 B7，任何一个问题结束后很快接近回退边界，**不能由长度推断端到端收益**。

## 证据可核查性与评分约束

官方样本提供答案、解释、证据文本和 PDF 页号，因而可以对照原始报告检查抽取。这比只保留一个数值答案更易审计，但不等于已证明全部标注正确。

对规范化后的完整证据字符串做页内匹配：AMD 4/7 段、AmEx 2/7 段、Boeing 9/9 段在指定页完整匹配。匹配失败不能自动判缺失：表格列顺序、破折号及 PDF 阅读顺序会改变字符串。人工检查 AMD 三个未整段匹配案例的关键表格仍在原页：流动性题的现金 4835、短期投资1020、应收4126、关联应收2、流动负债6369均在第55个零基页；现金流题的经营3565、投资1999等在第57页；部门增长题的 Data Center 6043/3694、Client6201/6887、Gaming6805/5607均在第47页。这仅证明相关原表可见，不以此修改答案。

FinanceBench 含公司审计意见、定性解释、比率及分类答案，**不能直接套用 DocFinQA 数值 FINAL 评分器**。实际运行前应冻结全题任务提示及独立人工/规则评分表，报告自然 EOS、cap、所有失败和原始输出；两臂仍要求完整 token 一致。不得在看到模型答案后删题。它是最多一个候选数据集，目前没有速度或模型正确率结果。

## 未推荐的其他数据源

- [QASPER 官方论文及数据](https://aclanthology.org/2021.naacl-main.365/)：问题由只读标题/摘要的人员提出，另有回答者标证据。下载官方 v0.3 test 小档案并 CPU 统计：416 篇、1451 题，40 篇自然 B>=6；当前 tokenizer 最长完整论文30,006 tokens，B>=6 且 P>=65536 为0。它能验证中短文档效用，不能在不人为拼接的前提下测试当前长前缀策略。这里只统计 test，不外推全部 split。
- 本地 QMSum 缓存：官方 test35 manifest 最大正文33,785 tokens，29组 natural B>=6；本地 exploratory train8 最大20,896 tokens。原 manifest 每组只列4个问题，若日后使用必须回原始数据恢复自然全部问题。现有长度仍达不到当前策略门槛。
- [InfiniteBench 官方任务表](https://github.com/OpenBMB/InfiniteBench)：英文书籍 QA/MC 的上下文明确经核心实体替换，不能说成未经改写的原始真实文档；平均输入192.6k/184.4k，参考输出很短。完整英文 QA JSONL 约298MB，本次只核对官方元数据，没有下载或实测 exact-context cohort，所以不宣称其自然组达到所需 B。
- [LongBench v2 官方任务说明](https://github.com/THUDM/LongBench)：503 道多选题，人工及自动审核提高评分可核查性；上下文8k至2M词。官方条目 schema 无共享文档 ID，是否存在足够相同完整上下文的自然组必须做完整 context hash 统计；465MB 原始 JSON 不在本次小元数据检查中下载。没有证据时不能把题目总数当成自然 cohort 数。

## 可复现产物和局限

`scripts/audit_alternative_natural_cohorts_cpu.py` 产生 `results/alternative_natural_cohort_opportunity_v1.json`，记录下载 URL、文件哈希、全部 QASPER paper 长度、FinanceBench 选择及逐证据匹配。PDF/原数据留在 git 忽略的 raw 目录，未提交巨型文件。

首次 CPU 长度检查误将 Transformers 5 的 `BatchEncoding` 字段数当 token 数。失败诊断保存在 `results/alternative_natural_cohort_opportunity_invalid_token_lengths_v1.json`；修正为读取 `input_ids`、断言 token 列表及 prompt 长于正文之后重新完成上述结果。该错误未运行 GPU、未用于选择题目或排除文档。
