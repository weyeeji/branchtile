# 动态 owner 解码与寄存器状态：2026-10-16 一手文献增量

本轮在等待算子资格实验时只做浏览与 CPU 源码阅读，新增 GPU 用量为零；不读取或判断 wave3 结果。
已先读仓库的 register-bank、register-routes 与 ISCAS contribution 审查，下面不重复已有共享前缀、固定 Q 和 streaming-softmax 基础文献。核查截止日为 **2026-10-16**。

本次新增三项可用先例和一项撤稿排除项。没有找到一份已核查资料明确给出：**同 CTA 的稀疏 owner 集合变化时，按固定请求身份保留同 lane 的 FP32 输出 banks，并在 M16/M32 MMA 之间完成无 shared-U 往返的映射**。这是本次检索范围内未覆盖的具体实现点，不是“全领域首次”的证明，也不是性能结论。

## 1. JSSA：最新的原布局、逐 KV-head、gather-free 稀疏 GQA

**日期/状态：**首稿 2026-10-13，arXiv v1；官方仓库也按 arXiv preprint 引用，未据此声称会议录用。[arXiv 元数据](https://arxiv.org/abs/2610.19026)、[论文 §3.3](https://arxiv.org/html/2610.19026v1#S3.SS3)、[官方仓库](https://github.com/jiwonsong-dev/JSSA)。

**作者实际贡献：**共享 Q/K 低维评分子空间与闭式校准；执行上直接索引标准 GQA paged KV，每个 KV head 可有独立 token 集，省去 gather staging 和逐 head 启动。这里的“共享子空间”不是多请求 KV owner 合并。

**源码读到的边界：**已读官方补丁中的 `triton_jssa_indexed_attend.py`：program 由 request、KV-head 以及可选 split 索引，FP32 输出 accumulator 固定对应本 program 的 GQA 行并跨 KV tile 累积，之后合并 split。未见跨请求动态 owner 区域和 canonical-request row 重排。因此不能再宽泛主张“首次原布局、无 gather 的细粒度稀疏 GQA”；但这份源码也没有直接实现本轮拟验证的动态 bank 路由。[固定 commit 补丁](https://github.com/jiwonsong-dev/JSSA/blob/34ce5abd82453b16ca6d5f16a1a8ae50fac5b920/vllm_patch/jssa_vllm.patch)

读取版本：commit `34ce5abd82453b16ca6d5f16a1a8ae50fac5b920`，committer 时间 2026-10-15 07:24:17 UTC；补丁 142524 bytes，SHA256 `4edb5adc5900c4445868e414e13571d51db52be0e8cb6565f37853bbb72902bd`。只在内存阅读，没有安装、应用 patch 或运行外部代码。其 B200 测量不能外推为 SM120 或本项目应用收益。

## 2. AMD Gluon decode：匹配两次 MMA 的布局、避免转换是已有工程原则

**日期/状态：**2026-07-27，AMD 官方工程文章及开源例程链接，非同行评审论文。[官方原文](https://rocm.blogs.amd.com/software-tools-optimization/gluon-attention-decode-mi450/README.html)、[官方日期索引](https://rocm.blogs.amd.com/hpc-software-tools-and-optimizations.html)。

**公开机制：**文中明确指出 QK 的 WMMA 输出若不匹配 PV 输入布局，热循环要付 register/LDS 转换成本；通过转置 WMMA 输出布局并匹配 PV 的 K-width 消除这项转换。同时根据 softmax reduction 方向选择 wave 分布，避免跨 wave 通信。

**对本工作的约束：**“选择一致的 MMA 布局以少搬状态”本身不能独立声称创新。其目标是 MI450 WMMA、单 request/KV-head 的 decode，不是 NVIDIA mma.m16n8、变动 owner 或多请求 canonical U。我们需要单独比较 emptyguard 与仅换 FullCol 的控制，再讨论额外 U 驻留是否有独立收益；不能把 FullCol 带来的全部变化归因于 state transport elimination。本文未复现该 AMD 例程。

## 3. TyphoonMLA：2026 录用的 shared-prefix 表示与执行协同先例

**日期/状态：**arXiv 首稿 2025-09-25，v2 更新 2026-02-12；ICLR 2026 官方列表收录为 Poster，不能把首稿日期写成 2026。[版本历史](https://arxiv.org/abs/2509.21081)、[ICLR 官方列表](https://iclr.cc/Downloads/2026)、[官方 Poster 条目](https://iclr.cc/virtual/2026/poster/10008790)。

**作者实际贡献：**MLA 的 shared 部分使用 naive 计算与展开 KV，private 部分保留 absorb/latent 表示，以不同计算/带宽需求选择等价公式并合并输出；还给出依赖共享批量的切换条件。[论文 §3](https://arxiv.org/html/2509.21081v2#S3)

**关系与边界：**这是共享程度驱动“表示＋执行”协同的近邻，而不是动态 GQA row-map 消除；MLA 的等价变换不能直接用于当前 Qwen GQA。其价值在于提醒：共享并不只影响读取量，还影响算术强度与最合适的公式。它没有为本轮 lane-local banks 的新颖性或速度提供正证据。

## 4. GraniKV：必须排除其已撤稿性能，不能只读仍可访问的 v1

**日期/状态：**v1 2026-08-16；v2 2026-08-23 **已撤稿**。arXiv 最新页明确说明吞吐测量方法存在错误，论文细节废止。[最新状态与撤稿原因](https://arxiv.org/abs/2608.15584)。

旧 v1 曾描述 shared-prefix 连续 HOT pool、细粒度 private COLD pool 和双 backend 逐步选择。[仅供排重的历史 v1](https://arxiv.org/html/2608.15584v1)。这些描述可作为进一步查找存储布局先例的线索；本审查不采用其吞吐、访存、计数器或应用改善数值，也不把它算成已成立的新进展。未下载或运行其匿名仓库。

## 对候选定位的更新

- **已占据的宽泛主张：**无 gather 原布局稀疏 GQA；MMA 生产/消费布局匹配；固定 program 输出 accumulator 跨 tile 累积；共享/私有区域使用不同执行表示。不能只换场景或 policy 名字作为贡献。
- **尚待验证的窄问题：**面对多个请求的任意稀疏 owner 区域，固定请求的 U 元素能否在活跃行集合和 M16/M32 执行形状变化后仍落在同 lane，并减少必要 shared-U 搬运；代价是否转移为额外寄存器、代码大小、低占用或同步。这仍只是实现假设。
- **能支撑机制归因的证据：**CPU route 双射/同 lane 检查；编译后 PTX/CUDA 的状态路径与资源；v2、emptyguard、FullCol-only、register-bank 四条件的同输入数值资格和完整 producer+merge 配对计时。图内算子收益之后仍需完整模型、准备成本及自然质量验证。

检索包含 dynamic-owner/shared-prefix/multi-query sparse decode，以及 register remapping、MMA warp allocation、layout conversion 等组合。一手来源优先 arXiv、作者官方 repo、会议官方列表和硬件厂商原文；检索到的第三方综述和二次博客不作为技术结论依据。本轮没有确立新论文贡献，也没有因“未找到完全相同实现”宣布首次。
