# 联合支持集交换的可选接入合同（尚未运行自然生成）

本文件冻结最小接入设计，不改变 `FrozenQCSelector` 的默认 independent QC-SnapKV 行为，也不选择最有利的 epsilon/order。原始动态行 v1/v2/v3 的负结论不受此设计影响；支持集交换会改变 attention mask，不能作为同 mask executor 优化。

算法来源为 [CPU 协议](../results/joint_support_exchange_v1/protocol.json)，SHA256 `64ee6faf63a7f7ccec4cda0ba39d5242f464107054ce89eec3e89d754fa92e6f`。复用 [已冻结实现](../scripts/probe_joint_support_exchange.py) 的 `exchange`：每层每 KV head 独立操作，各题保持预算、first4/last32，最多两轮；仅将独占的低分原 top-k 成员换为已被他题选择的高分成员。float32 pooled scores 精确提升到 float64，最终 `math.fsum` 再验实际所选分数损失。epsilon=0 仅允许原分数精确相等的交换。该分数约束不是 O/LSE 或任务质量约束。

接入字段预定为 `joint_support_exchange=None`（默认）或一个必须完整提供的字典：

```json
{
  "protocol_id": "joint-support-exchange-cpu-v1-20261016",
  "protocol_sha256": "64ee6faf63a7f7ccec4cda0ba39d5242f464107054ce89eec3e89d754fa92e6f",
  "epsilon": "必须显式选择 0、0.001、0.005、0.01 之一",
  "request_order": "必须显式选择 ascending 或 descending",
  "max_rounds": 2,
  "protected_first": 4,
  "protected_last": 32
}
```

`epsilon` 和 `request_order` 在任何输出或 GPU probe 之前写入独立 pilot 配置并保留 hash，禁止运行中重选。每个新文档仍按原 suffix-only Q、完整 causal denominator 和 stable lower-position ties 计算 independent masks；不借用未来 decode Q。联合步骤必须等本层整个固定 cohort 的所有独立分数就绪，不把先完成题作为已改 mask 执行后再污染其他题的观察值。

最小实现是在 opt-in 时额外保存 CPU pooled scores 与 independent mask，新增显式 `finalize_joint_supports()`，在 dense selector warmup 完成、任何 candidate graph capture 之前一次调用。默认路径不保存这些额外数组。finalize 必须检查48层、两 KV head、所有 cohort 题齐全，逐层/head 应用已冻交换，完成后释放 pending scores；任何缺项或身份变化显式失败，不能静默退回 independent。独立历史与交换历史均保留可复现的有序 int64 IDs/hash；旧 `records.mask_sha256` 不被改写成另一份历史。

新记录至少包含：protocol/source SHA、rank/layer/head/request order、epsilon、全部交换事件、每题原分数/保留分数/实际损失、每题 independent/new mask SHA、交换前后 union 大小与 owner counts、score D2H wall、CPU exchange wall/process time、记录/hash wall、峰值保留 score/mask bytes、finalize 总 wall。第一次 selector 成本、Q 保存、原完整 K 临时 gather、等待其他题 ready、额外 CPU/GPU buffer 和 graph capture 都进入准备账；union 几何减少不等于物理 KV 容量已释放。

图历史身份必须由实际最终 masks 生成，并显式带 independent/new 两份身份。同一份新 mask 必须同时提供给 native E0 与 union；原 independent E0/union 不得被覆盖。用于自然开发的最低控制集合是官方 FLASHINFER/XQA full-history、同一 FLASHINFER prefill/cache 基础上的 independent native E0，以及同新 mask 的 native E0/union。若只运行某个 executor，不能把新 mask 带来的变化归因为共享执行收益。自然生成仍 cold reset + 同 q0 dense anchor + atomic enqueue，同 seed/资源；计时分别记录一次性准备、anchor、generation，完整 FINAL/错误/cap/EOS 留存。

当前门：先等待独立 actual-Q O/LSE probe；并完成官方 FLASHINFER bridge 的实际 cache/layout/full-budget qualification。随后才由 root 冻结单一有界 pilot 配置。需要 CPU 数组等价测试（原 frozen 函数逐位）、保护/预算/score-loss 边界和拒绝不完整 cohort 的测试，再做同 mask 算子资格。自然多层调用与任务质量目前未验证；没有新颖性或正式非劣结论。

CPU 模块已新增为 [joint_support_exchange.py](../src/joint_support_exchange.py)，尚未导入生产 selector。已完成 [10个边界测试](../results/joint_support_exchange_integration_v1/unit_tests.json)，以及三份冻结真实 pooled-score 数据的 [48个cohort配置/96个head配置逐位等价检查](../results/joint_support_exchange_integration_v1/cpu_qualification.json)：全部epsilon/order均保留，支持集与事件完全匹配原冻结函数，CPU wall约3.65秒、GPU费用0。独立 [接入协议](../results/joint_support_exchange_integration_v1/protocol.json)把后续开发候选32K、epsilon0.001/0.01、ascending与必需控制列明；文档/作业配置和cohort生命周期资格仍待补，未执行自然pilot。
