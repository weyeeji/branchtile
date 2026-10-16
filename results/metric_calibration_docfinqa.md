# DocFinQA 自由回答评分：来源核查与预注册 v1

日期：2026-10-16。指标 ID：`strict-final-financial-answer-v1-20261016`。本次只读取冻结的 training167/development8 标签形式、官方评测源代码，以及已冻结协议中的两个冲突标签；未读取 evaluation100 模型输出，也未运行任何模型生成或数据集自带 Python 程序。测试输入全部是独立构造的合成样例，不来自生成结果。

## 官方任务与本项目指标的区别

**FinQA 官方评测是程序评测。** 固定到作者仓库 revision `0f16e2867befa6840783e58be38c9efb9229d742` 的 [evaluate.py](https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/code/evaluate/evaluate.py) 接受 tokenized DSL 程序和表格。`eval_program` 解释 add/subtract/multiply/divide/exp/greater 及四个 table 操作；数值最终 `round(..., 5)`，百分数字面量除以 100。`evaluate_result` 将执行结果与 `qa.exe_ans` 精确比较，另通过符号化程序比较计算 program accuracy。**这不是对自然语言 `qa.answer` 抽数值后比较。** 此次只阅读代码，未执行模型程序，也未把自由回答冒充 DSL。

**DocFinQA 延续程序生成任务。** [正式 ACL 2024 论文](https://aclanthology.org/2024.acl-short.42.pdf) §3/附录 C 说明将 FinQA 派生程序转为 Python，并用于问答；[作者早期版本 §3](https://arxiv.org/html/2401.06915v2#S3) 明确要求生成 Python，而非直接生成数值。当前固定 [官方 HF revision](https://huggingface.co/datasets/kensho/DocFinQA/tree/64ebaff62f692495bcc182f45cf9a9606251b19b) 仅有 README、train/dev/test JSON；README 只有许可字段。本次没有找到可固定版本、作者发布的独立 DocFinQA 评分程序，因此其执行输出比较的精确容差仍未确认；不从 FinQA 或第三方实现推定。

当前生成约定为末行 `FINAL: numerical answer and unit`。下面实现的是**本项目自由回答的数值/布尔答案匹配**，不能命名为“官方 DocFinQA accuracy”或“FinQA execution accuracy”，也不与论文程序生成分数直接比较。程序正确性、证据出处和事实审计需另行进行。

## 预注册规则

主指标固定为 `exact_match` 的总样本平均；必须同时报告 `rounded_match`、预测解析失败、gold 解析失败及单位未验证数量。不能根据运行结果选择更高的列作为主指标。所有既定问题留在总分母；运行失败应传入空回答，不能丢弃。两项分数均为 0–1，百分制展示时乘 100。

| 项目 | 固定规则 |
| --- | --- |
| FINAL 边界 | 仅接受最后一个非空行中的大写 `FINAL:`，可有行首空白；全输出只能出现一个行首 FINAL 标记。末行之后仅允许空白。缺失、重复、额外尾文、未闭合 `<think>` 或未闭合代码围栏全部失败。 |
| 完整匹配 | FINAL 内容必须整体是一个数量或 yes/no；不从正文、多个数、代码、公式或解释中找一个接近 gold 的数字。不调用 eval/exec，也不执行 DSL。 |
| 数字 | ASCII 十进制，可有合法三位千分位、科学计数法、正负号、小数点前省略的 0。支持数学减号 U+2212；会计括号只表示负号，不能再带正负号。拒绝 NaN/Inf、任意表达式、非法逗号、Unicode 数字。量级与文本长度有界。 |
| 百分比 | `%`/`percent`/`per cent` 除以 100；`bp`/`bps`/basis point(s) 除以 10000；显式 `ratio`/times/x 或 `a:b` 为比率。`12.5%` 与 `0.125 ratio` 相等；与无单位 `0.125` 不相等。不会自动尝试 ×100/÷100 来匹配。 |
| 百分点 | pp/percentage point(s) 单独量纲，不与百分比直接互换。 |
| 货币 | `$`/US$/USD/dollar(s) 固定解释为 USD；EUR/€、GBP/£、CAD/C$、AUD/A$、JPY 分开。仅同币种转换；不进行汇率换算。`$` 具体币种在原数据中未明确的风险保留。 |
| 数量级 | thousand/million/billion/trillion 为 10³/10⁶/10⁹/10¹²；禁止含糊的 k/m/b 缩写。显式同量纲 gold 可以据此转换，例如 `$1.25 million` 与 `$1250000`。 |
| 其他单位 | 固定白名单含 share(s)、year(s)、day(s)、barrel(s)、ton(s)、tonne(s)、employee(s)、tower(s)、person/people、tbd、tpd；不自动把不同单位互换，不把任意尾部文字当单位。 |
| 无单位 gold | 只比较数值的**显示系数**，不推断上下文隐藏的美元/百万/百分比；标记 `unit_unverified=true`。例如 gold `2.5` 与 `$2.5 million` 数值一致，但**不能声称单位正确**；`$2500000` 不会被反推成 2.5。 |
| 显式单位 gold | 要求预测显式给出兼容量纲；裸预测数字失败。货币/数量级转换只在量纲匹配时发生。单位检查成功标记 `unit_verified=true`。 |
| 布尔 | yes/no 大小写无关，独立类型；不与 1/0 混同。`yes, 38.2%` 等复合 gold 不拆开挑答案。主表须注明包含布尔精确匹配。 |
| 主指标 | 经以上确定性规范化后的 Decimal 值精确相等；不使用相对误差或任意 epsilon。 |
| 舍入敏感性 | 将预测按 gold 显示精度舍入，规则 `ROUND_HALF_EVEN`，单位转换同时转换该精度。gold `12.34%` 的分辨率为 0.01 个百分点；gold `$1.23 million` 为 10000 USD。显式 `a:b` 无法唯一推断小数精度，舍入列与精确列一致。 |
| 舍入局限 | 整数 gold 的舍入列可接受 ±0.5 附近的预测，gold `0` 可接受 `0.49`；这些边界已写入合成测试。因此舍入列只作必须并列的敏感性，不替代主指标。 |
| gold 清理 | 保留原字符串；仅去首尾空白，允许一次尾部字面量 `\\n` 清理并记录。该序列在 development gold 出现，属于明确限定的序列化修复，不改数值。 |
| 不支持标签 | 含多解、自由文本、错误问题文本、不可解析单位的 gold 标记 `gold_parse_failure`，该项分数 0 且保留总分母。不能把它隐瞒成模型数值错误，也不进行有利选择。 |

训练/开发标签只用于识别格式，不用于估计可放宽的误差范围。训练集确实存在复合答案及错误文本，开发集存在缺省单位，即使规范化也不能恢复全部语义。具体覆盖率与未解析 gold 列在 [预注册审计 JSON](metric_calibration_docfinqa.json)；此处没有根据 evaluation 标签表现扩展白名单。

## 冲突标签与配对敏感性

采用 [已冻结输入协议](data_qualification/docfinqa_public_protocol_v1.json) 的 primary label，保持原始 source variants，不改 gold、不挑最接近预测的标签。该协议列出同一报告下两个 evaluation 问题，主标签分别 6% 和 22%，备选集合分别为 {6%,18%} 和 {22%,66%}。

实现 `paired_label_sensitivity(a, b, primary, variants)`：对每个候选 gold，**同时**给两种方法评分，计算 A−B，然后取共同标签下的最小/最大差。不能先独立为 A/B 挑最有利或最坏标签再相减。汇总时将两题的合法标签组合做穷举（2×2），在同一数据分母、同一报告/身份组件分组上重算主差与区间；一题变化对 371 个问题均值最多改变 1/371，一题对方法差最多改变 2/371。这是标签不确定性，不能代替采样置信区间。

## 实现、测试和调用

[实现](../src/financial_answer_metric.py) 只用 Python 标准库，CPU only，无模型、shell、网络或 Python 程序执行接口。[合成测试](../tests/test_financial_answer_metric.py) 覆盖解析边界、单位、符号、尺度、舍入 tie、代码注入、失败保留、共同冲突标签的配对差。运行于项目 `.venv`、uid 1000，没有 root 或安装系统依赖。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_financial_answer_metric.py
```

```python
from financial_answer_metric import score_pairs, paired_label_sensitivity

report = score_pairs(predictions, frozen_primary_answers)
# report['per_example'] 保留原预测/gold、两项分数、解析状态和单位标记。
# report['exact_accuracy'] 为预注册主指标；rounded_accuracy 必须并列。
```

生成阶段必须保留原输出及 cap/运行/结构解析状态；本模块不能从正确数字反推该次服务是否成功。固定流程、脚本和测试的 SHA-256 及官方源文件哈希写入审计 JSON。未取得独立官方 DocFinQA 执行 fixture，所以这次结果只证明自定义自由回答规则满足其合成契约，不证明复现官方论文指标。
