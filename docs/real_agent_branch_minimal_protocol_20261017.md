# 真实 agent 分支场景的最小验证方案

建议下一项应用验证采用“同一仓库、同一真实issue的并行修复尝试”，而不是继续仅用财报数字问答。共同仓库快照和issue是自然共享前缀，各尝试的推理、patch和测试反馈构成私有后缀。它能同时检验分支退出、长短输出、tool feedback导致的私有后缀增长，以及按任务成功计费。现有DocFinQA负结果仍保留；新场景不能只以更容易产生加速为筛选标准。

SWE-bench的官方评估使用真实仓库patch及测试环境，标准harness使用Docker；SWE-agent官方教程支持从GitHub issue运行修复流程。以下是我们提出的最小实验设计，不是声称官方benchmark已经定义了八分支协议。来源：[SWE-bench harness](https://github.com/SWE-bench/SWE-bench/blob/main/docs/reference/harness.md)、[SWE-agent CLI tutorial](https://github.com/SWE-agent/SWE-agent/blob/main/docs/usage/cl_tutorial.md)。当前本机有Docker二进制但尚未证明rootless配置，因此**不使用rootful Docker**；优先普通用户虚拟环境的可复现两issue工程试验，若要报告官方SWE-bench分数，需另行满足其正式harness环境合同。

## 最小可检验工作负载

先冻结两个真实Python仓库issue。按公开数据的instance_id固定hash顺序选取，环境可复现是唯一前置资格；不根据模型是否答对、输出长度、上下文长度或测得速度挑选。选择记录包含所有环境失败及原因。base commit、依赖lock、公共context文件列表/顺序/hash、issue正文、工具协议和测试名单先于native/candidate对比冻结。

公共context由issue加固定检索器选择的仓库源文件构成；检索不能利用gold patch或FAIL_TO_PASS对应的改动位置。保留自然长度，禁止复制代码、加无关文档或填充到120K来制造机会。若自然context很短、attention机会不足，这本身是场景否定证据。

每个issue启动B=8个修复分支，同一问题和代码context、不同预注册采样seed，最多两轮：第一轮输出patch；应用到各自普通用户临时checkout并运行冻结测试；第二轮把该分支测试结果追加到其私有history，允许修复一次。无效patch、测试超时、EOS未正常结束都计失败，不能从延迟分母删除。模型只能看到原有可见测试和工具结果，最终隐藏测试用于评分；gold patch只用于提前验证测试环境，绝不输入模型。

首先做2个issue × 2种实现 × 1组seed的可运行性试验。只有在native有非零可复现成功、工具环境合同成立后，才扩展更多issue和seed；不能用这两个开发issue宣称总体解决率。BF16/FP8仅在至少一种精度证明真实任务可行后扩展，不需要一开始穷举。

## 必须同时保留的测量

- 每个任务的cold wall：共同context初始化、全部实际生成、patch应用、测试与第二轮、固定选择规则；候选新增图/plan首次构建成本单列并加回首次任务。
- 已加载仓库的完整两轮session wall：真实只初始化一次共同context，工具后继续同一任务。必须实测session，不能仅把多次cold anchor算术扣除。
- 生成部分wall、GPU完整模型图区间、CPU工具wall分别记录，避免把工具耗时移出“agent端到端”标题。
- 解码每步B、共享物理页数、实际private tail、准备/切换次数与EOS/cap。第二轮追加工具结果会触发private prefill，允许native fallback，不能只测纯decode峰值。
- 每分支测试是否通过、任务是否至少一个分支成功、首次通过的时间、完成全部分支的时间及总GPU秒。预先固定“运行完所有分支”作为第一轮协议；若再研究首个成功即取消其余分支，作为独立协议且两实现相同。
- 同一seed的native/candidate采样可能分岔，保留全部实际成功/失败与长度。比较成功率和达到成功的任务成本，不能通过长度匹配挑选子集。

## 接入边界与最小工程量

原生vLLM/FlashInfer XQA仍是部署基线，现成dense cascade是成熟共享机制对照；当前算子优化的创新需要单独比较这个对照。新应用只需要新prompt/manifest适配器、普通用户patch+test工具、两轮session控制器及任务评分；不更换attention计划的数学合同。

当前DocFinQA admission identity依赖prompt hash索引，八个完全相同prompt会都被映射到question0。新场景必须在CPU admission记录明确branch_id，或者使用预先固定、极短的branch后缀使prompt hash不同；不能把hash相同误当分支相同。不同分支的采样seed必须显式传入，不能只重复同一个固定seed的请求并把八个相同结果称为八次探索。

实际部署接入还需保留不等private tails、分支重排与退出、每轮真实prefix reuse；不假定所有分支等长。先用相同真实状态完整模型图诊断区分计算机制收益，再用上述工具session检验端到端是否保留下来。如果全部成功成本主要被CPU测试或prefill支配，则应继续换机制，而不是改写端到端口径。
