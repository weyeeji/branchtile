# O-only 合同诊断的独立审计

8个单卡作业均clean exit0。CPU脚本独立重算960测量行、64组paired median/10000次按block重抽样CI，与原结果逐项一致；所有记录的源码SHA和账本terminal均通过。只使用CPU和项目虚拟环境。

固定基线是预先指定的FlashInfer tensor-core=True、page-one任意稀疏support适配，owner固定C128/N64，没有逐输入oracle。当前owner内部仍计算分段attention必需LSE。原生应用只需O时，不需要导出并转换最终LSE。

- hot：8/8 O-only原生严格快，native/owner paired ratio为0.900297–0.987847，全部CI上界<1。
- cold：6/8原生严格快，另2个CI包含1；其中一个median1。重复量化值可导致零宽CI，不代表不存在系统计时误差。
- hot最终log2→ln转换增加1.1192–1.1656µs；原生导出raw LSE本身的时间差很小。此前在O+自然logLSE合同下的局部正信号，不能据此推断部署只需O时也加速。
- eager四种合同O与FP64参考通过；hot/cold每个图及AA图重放O通过。运行脚本也断言native raw/ln LSE及owner LSE正确，但LSE逐项误差未单独序列化，不能伪造新增LSE误差统计。

这些是一个开发文档不同rank/layer/step、两KV头的相关输入；没有prepare成本，也不是生产dense XQA或整模型。结果否定的是当前固定owner实现上这组O-only速度优势，不是否定所有未来共享优化。

复现：`.venv/bin/python scripts/audit_output_contract_wave27.py`，结果 `results/operator_output_contract_wave27_audit_v1/audit.json`。
