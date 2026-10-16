# ISCAS 算子与片上数据流贡献：范围核对

2026-10-16。普通用户、只读文献核对，无 GPU、安装或性能试跑。本文回答会议契合度和实证范围；不据此推断审稿人偏好或录用概率。用户最新要求重新挖掘原 idea 的实现空间，早期停止判断不构成禁止继续优化的授权限制。

## 判断

**GPU 算子、片上状态组织和流水线设计可以构成与 ISCAS 相符的贡献；“ISCAS 更喜欢算子创新”不能当成已验证事实。** [ISCAS 2027 官方 CFP](https://2027.ieee-iscas.org/call-for-papers) 明列 High-Performance Computing for AI，并要求原创 circuits/systems 研究。范围未规定必须设计 ASIC/FPGA。以下已接收先例证明形式可行，不证明只改一个 kernel 就足够。

| 一手证据 | 可以支持的判断 | 不能据此声称的事 |
| --- | --- | --- |
| [AMUSD 作者主页](https://bradmcdanel.com/research/speculative-decoding/) 明确 ISCAS 2025；[论文 §III–IV](https://arxiv.org/html/2410.17375v1) 为现成 GPU 上异步 draft/verify 与回滚机制，报告生成性能及功耗/能量 | 纯 GPU 软件执行机制存在接收先例；无需额外造硬件才有资格 | 其两 GPU 方法对照单活跃 GPU 的结果不能成为本项目同资源实验标准 |
| [UbiMoE 官方录用记录](https://epapers2.org/iscas2025/ESR/paper_details.php?paper_id=1809) 标为 Accept as Lecture；[论文 §III–V](https://arxiv.org/html/2502.05602v3) 围绕 streaming attention、复用 linear kernel 和资源分配展开，在 ZCU102/U280 实测 | 少量模型、清楚的数据流和资源取舍，可以形成紧凑的硬件系统故事 | FPGA 与旧 GPU 跨平台、跨精度比较不能替代本项目同卡/同精度强对照 |
| [PRIMAL](https://arxiv.org/abs/2601.13628) 含 ISCAS 2026 正式页码和 DOI；[全文 §IV](https://arxiv.org/html/2601.13628v1) 使用 Verilog、综合布线、存储模型和 cycle-accurate 模拟 | 片上资源、映射及通信/计算流水线是明确相关的贡献类型 | 这不是流片实测，也不能用其模拟结论证明 RTX 5090 上的瓶颈 |

## 对本次实现最直接的先例边界

[UbiMoE §III-B1](https://arxiv.org/html/2502.05602v3#S3.SS2) 已让每个 Q 始终位于固定 PE，并广播 K；§III-B2 以流式 softmax 和 max 寄存器减少中间缓存。§IV 还指出，进一步全流式/双缓冲可能因挤占资源而使总延迟变差。因此，“固定 Q / 累加器、广播 KV、流水重叠”本身是已有原则。若 GQA8 register-bank 路线奏效，增量必须具体落在**动态稀疏活跃行变化时，怎样在既定 MMA 片段约束下避免状态往返，以及为此支付多少寄存器和同步成本**。它仍须与 PAT、PageWeaver、Cascade 等更直接的 GPU 先例区分，不能仅因 FPGA/GPU 平台不同宣称新颖。

2026 年的相关一手研究进一步支持“先减少非矩阵成本，再讨论更高算力”的研究方法：[FlashAttention-4](https://arxiv.org/abs/2603.05451) 围绕 Blackwell 的非对称资源增长，联合修改流水线和 softmax 执行；但其论文主要证据在 B200，不能把 tcgen05/TMEM/2-CTA 路线直接迁移成 SM120 可用能力。[DATE 2026 的 3D-Flow](https://arxiv.org/abs/2602.11016) 则以特制 3D NPU 的寄存器间传输减少片上 SRAM 往返，属于相近动机、不同硬件机制；它不能证明现成 GPU 上能得到相同收益。

## 本项目需要的实证（研究判断，并非官方清单）

1. **机制可辨识：**保存编译资源和指令证据，量化各版本每个 owner 转换的状态搬运、同步、寄存器/shared 使用，以及由此允许的驻留 CTA。无权限读取的 counter 如实记为未测，不能猜测 spill、bank conflict 或 GDDR 流量。
2. **有因果关系的消融：**仅改变状态驻留方式；再单独加入异步搬运/双缓冲、metadata 准备复用。输入 Q/K/V、mask、输出合同一致，比较全部准备、merge、copy 的完整调用成本，不能只截取内层 MMA。
3. **对照覆盖已有共享：**原生实现必须使用真实共同物理 KV 地址；同时比较最佳固定分组/union、层次共享等已冻结强对照。独立副本实验只能解释共享收益来源，不能冒充部署基线。
4. **边界足够明确：**真实多层/多头与冷热工作集、共享度、批大小、预算共同报告；合成极端用于解释机制，真实 trace 用于确认相关性。固定 token 轨迹测纯执行，真实 EOS 任务测完成时间与质量，二者分开。

若新实现优于旧 v3，却仍慢于上述强对照，结论是修复了实现低效；要成为主要加速贡献，还需给出可复现的新优势区域和可解释的失败边界。若主张仅为保持同一数学操作的算子加速，固定真实轨迹的证据最直接；若进一步主张稀疏选择、量化或自然回答效率，仍需独立的质量与 EOS 证据。
