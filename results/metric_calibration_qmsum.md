# QMSum 官方公开输出的 ROUGE 校准

核验日期：2026-10-16。CPU only，uid 1000，使用 `/home/qcjz/iscas/.venv/bin/python`，未安装系统软件、未使用 root。结论：原始 Perl 路径已跑通，结果接近公开 fixture；**尚不能声称与作者发布分数逐位一致**。没有搜索参数、选择有利预处理或加拟合偏置。

| 评分路径；全部为 F1 × 100 | R-1 | R-2 | R-L / Lsum |
| --- | ---: | ---: | ---: |
| QMSum README，对应这份公开输出 | 36.51 | 11.41 | 31.60 |
| 作者指定参数 + 原 Perl 1.5.5 + HMNet 分句，本机实测 | 36.515 | 11.421 | 31.599 |
| 同一 Perl 运行的逐样本 F1 算术均值 | 36.503781 | 11.411290 | 31.595455 |
| Google rouge-score 0.1.2，stem=True，整段 rougeL | 36.092224 | 11.371163 | 22.369215 |
| Google，同分句、同 stem=True，rougeLsum | 36.092224 | 11.371163 | 31.260726 |
| Google，同分句、stem=False，rougeLsum（诊断） | 34.407800 | 10.769508 | 30.015430 |

最大残差为 **0.011 个 ROUGE 点**（R-2）。README 只给两位小数，R-2 残差不能仅由这种舍入解释；原作者 Perl 版本、实际分句脚本、fixture 生成时的文件命名与完整评分环境未发布，剩余偏差的具体来源未证实。固定当前有来源的流程，保留此限制，不继续尝试匹配数字。

## 官方依据与固定流程

1. QMSum 作者在 [issue #5 的版本说明](https://github.com/Yale-LILY/QMSum/issues/5#issuecomment-887155639) 指明 **pyrouge 0.1.3**，并在 [后续说明](https://github.com/Yale-LILY/QMSum/issues/5#issuecomment-890003212) 指向自己的 MatchSum 评分实现。因此不能直接采用 HMNet 仓库 pyrouge 的默认选项。
2. [MatchSum metrics.py](https://github.com/maszhongming/MatchSum/blob/c7754245a454d0ba3535db0e4cc1a13b3d35680d/metrics.py#L137) 的文件名为 `0.dec, 1.dec, ...` 与对应 `.ref`；参数固定为 `-c 95 -r 1000 -n 2 -m`，加 `-a` 及数据目录。pyrouge 按文件名的字典序生成 1-based EVAL ID。此处直接导入 PyPI 0.1.3 的静态转换/配置函数，未修改其代码。
3. 摘要分句采用 [HMNet 官方评测代码](https://github.com/microsoft/HMNet/blob/416966c63e3cb7a57dc59b4ce8fa76f11fd048be/Evaluation/ROUGEEval.py#L70) 的同一正则；先转义 `<`/`>`，以句号或问号后的空白作为边界，保留源码中的缩写例外，再转为逐句 SEE HTML。**这能证实是 HMNet 的实现；QMSum issue 只确认 scorer，并未单独公布它对公开 txt 的分句器。** 不使用后来 NLTK 分句来追分。
4. Perl 来自固定 [HMNet revision](https://github.com/microsoft/HMNet/tree/416966c63e3cb7a57dc59b4ce8fa76f11fd048be/ThirdParty/ROUGE/ROUGE-1.5.5)。通过 `perl -I <vendored-rouge-root>` 载入仓库随附 XML::DOM，无需 chmod 或系统安装。系统依赖为 Perl 5.38.2、XML::Parser 2.47、DB_File 1.858。
5. 执行时增加 `-d` 仅输出逐样本分数。已与未加 `-d` 的运行核对，三项 Average_F 不变。没有 `-s` 去停用词、没有截断选项、没有 PTB 再分词。

## 为什么 Python 简单替换不等价

**LCS 层级。** Google 的 `rougeL` 把全文当作一个序列；`rougeLsum` 按句计算 union-LCS。这里先使用 HMNet 正则添加换行，再调用 `rougeLsum`，L 分数由 22.369215 变为 31.260726。不能把未分句的 L 与论文的摘要级 L 混在一列。[Google 原作者的指标说明](https://github.com/google-research/google-research/tree/master/rouge)

**词元/词干。** Perl 的 `MorphStem` 先查询随附 WordNet-2.0 例外词表，再执行其 Porter 代码；Google 使用 NLTK Porter。标点/词元规范化也由不同实现完成。同分句后 R-1 与 Lsum 仍分别相差约 0.423 和 0.338 点。因此这里只把 Google 作为诊断指标，没有证明其与原 Perl 等价。[Perl 源码](https://github.com/microsoft/HMNet/blob/416966c63e3cb7a57dc59b4ce8fa76f11fd048be/ThirdParty/ROUGE/ROUGE-1.5.5/ROUGE-1.5.5.pl#L2582)

**汇总。** Perl 的 `computeAverages` 将最终 `AvgF` 赋值为 1000 次 bootstrap 的均值；逐样本算术均值另行保留在内部。bootstrap 使用排序后的字符串 EVAL ID，因此改文件命名会改变给固定抽样索引分配的样本。已做来源追踪中的诊断：保持同样的评分和分句，仅采用 HMNet 的六位补零文件名，得到 36.464 / 11.374 / 31.558；MatchSum 的不补零命名得到表中数字。这个诊断用于解释顺序敏感性，最终命名依据作者链接代码固定，并非挑选最接近的排序。[平均与 bootstrap 源码](https://github.com/microsoft/HMNet/blob/416966c63e3cb7a57dc59b4ce8fa76f11fd048be/ThirdParty/ROUGE/ROUGE-1.5.5/ROUGE-1.5.5.pl#L806)

**论文值与 fixture 值。** [QMSum 论文](https://aclanthology.org/2021.naacl-main.472.pdf) §5.1 说明论文结果为三次运行均值；表 3 的 gold-span HMNet 为 36.06 / 11.36 / 31.27。[README](https://github.com/Yale-LILY/QMSum/blob/83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e/README.md#model-outputs) 给这份输出的数值为 36.51 / 11.41 / 31.60。两者不能作为同一次运行的互相矛盾目标。

## 版本、文件和复跑

输入为官方 QMSum revision `83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e` 的 `model_output/{preds,refs}.txt`，各 279 行。原始 CRLF 字节用于计算哈希；读取时换行标准化只用于样本切分。

| 文件 | SHA-256 |
| --- | --- |
| preds.txt | `c2c4ed2cc5be7eb7ea1ee843f05570c18a2562f6908de64c661b4a8d354a3fd9` |
| refs.txt | `32ff266e626e8492ca10e9ddbef046f32621d78f385f59b9c768f368384cd467` |
| pyrouge-0.1.3.tar.gz | `364f1dd65a5aabeb40eb4cdf967bc0cb495073885c75eaf09c960fc9ea1696a3` |
| PyPI Rouge155.py | `8f0a447c9528ca53fe205371f4788657d36b8a1035eab400551791b60ae5fa8d` |
| ROUGE-1.5.5.pl | `61168737d74bea24b3e41a598cbece527c11948cbf35a702a77aacea5e40a633` |
| WordNet-2.0.exc.db | `22c96bbc6a03a8094eee569a5128dfe84dc58eb4c5faf11d3f98b7a9ab3d84ea` |

完整使用文件的哈希、版本、参数、字典序映射及 Python 对照在 [JSON](metric_calibration_qmsum.json)。原 Perl 输出在 [perl.txt](metric_calibration_qmsum.perl.txt)，按原始输入行顺序恢复的 279 条 R/P/F 在 [per_example.json](metric_calibration_qmsum.per_example.json)。每条分数源自 Perl 打印的五位小数；后续置信区间需声明该精度。

当前工作区依赖已经准备好，可以直接运行：

```bash
.venv/bin/python scripts/calibrate_qmsum_rouge.py --require-fixture
```

新环境准备（全在用户目录；不需要安装 pyrouge 到虚拟环境）：

```bash
mkdir -p cache/metric_calibration
git clone https://github.com/microsoft/HMNet.git cache/metric_calibration/HMNet
git -C cache/metric_calibration/HMNet checkout 416966c63e3cb7a57dc59b4ce8fa76f11fd048be
.venv/bin/python -m pip download --no-deps pyrouge==0.1.3 --dest cache/metric_calibration
sha256sum cache/metric_calibration/pyrouge-0.1.3.tar.gz
.venv/bin/python -c "import tarfile; t=tarfile.open('cache/metric_calibration/pyrouge-0.1.3.tar.gz'); t.extractall('cache/metric_calibration', filter='data')"
.venv/bin/python scripts/calibrate_qmsum_rouge.py --require-fixture
```

对后续实验使用 `--predictions PATH --references PATH --output-prefix PATH`，不加 `--require-fixture`。对照 google 需要本项目已有的 `rouge-score==0.1.2`、`nltk==3.10.3`；只跑 Perl 可以用 `--skip-google`。固定所有方法的样本顺序、预处理、评分器和版本。应用主表建议用上述 Perl R-1/R-2/摘要级 R-L，额外的 Google 指标写清名字；方法间置信区间用同一套逐样本分数进行按会议聚类的配对重采样，不能直接拿两个独立 Perl CI 相减。

应用代码也可以直接调用：

```python
from scripts.calibrate_qmsum_rouge import score_pairs

scores = score_pairs(preds, refs, output_prefix="results/application/qmsum_dense_rouge")
rows = scores["per_example"]  # 原输入顺序；每条 1/2/L 下为 recall/precision/f1，0–1
official_aggregate = scores["reported_average_percent"]  # 每个指标 R/P/F，0–100
macro_f1 = scores["per_case_macro_f1_percent_from_printed_5dp"]
```

不传 `output_prefix` 时仅返回结果，不保留临时文件。已用 fixture 的第 43 条单独评分核对，三项逐样本 P/R/F 与完整 corpus 中恢复到 index 42 的值一致，验证字典序 EVAL ID 映射没有打乱 bootstrap 输入。ROUGE 衡量文本重叠，不能替代事实性审计。
