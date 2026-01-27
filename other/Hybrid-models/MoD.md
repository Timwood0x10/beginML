# 深度稀疏化：Mixture of Depths (MoD) 与架构效率大对决

> **摘要**
> 传统 Transformer 对每个 Token 投入等量计算，无论其是复杂的“逻辑词”还是简单的“标点符号”。
> **MoD (Mixture of Depths)** 由 Google DeepMind 提出（arXiv:2404.02258），通过动态路由让部分 Token 跳过计算层，实现深度稀疏化（40–50% FLOPs 减小，无明显性能损失）。
> 本章解构 MoD 的容量路由机制（含数学论证），并从“计算预算”视角对比 Jamba 与 MLA 的工程权衡。

## 一、MoD 的数学本质：按需分配的“层预算”

MoD 核心在于**固定容量路由**：不是随机跳层，而是基于容量阈值 $C$ 决定哪些 Token 进入计算。

### 1. 路由逻辑：从概率到排名

Router（轻量 MLP + sigmoid）为每个 Token 计算标量得分 $s_i$。

$$
y_i =
\begin{cases}
x_i + \text{Layer}(x_i) & \text{if } s_i \in \text{Top-k}(S, C) \\
x_i & \text{if } s_i \notin \text{Top-k}(S, C)
\end{cases}
$$

**为什么用 Top-k 而非固定阈值？**

- 固定阈值会导致批次内激活 Token 数波动，Tensor 形状动态变化，GPU 并行效率下降 10–20%。
- Top-k 强制每批激活 Token 数恒定为 $C \times \text{batch\_size}$，计算图稳定。
- **训练辅助损失**：$L_{\text{aux}} = (\sum \text{top-k} - C)^2 + \sum \text{var}(\text{top-k})$，防止路由坍缩（DeepMind 实验：无 $L_{\text{aux}}$ 时性能降 3–5%）。

### 2. 计算开销缩减（量化）

设 $C = 0.5$（每层计算 50% Token）：

- 理论 FLOPs 减 50%（跳过层不计算 Attention/FFN）。
- 实际减 40–45%（vLLM 实测）：KV Cache 填充 + Router 开销 ≈5–10%。
- 推理吞吐提升 1.5–2x（DeepMind 报告，batch=32 时）。

## 二、架构之争：Jamba vs. MLA vs. MoD

三种路径分别解决 Transformer 的不同瓶颈。

### 1. 维度对比表

| 维度                 | **传统 MHA** | **Jamba (Hybrid)**  | **MLA (DeepSeek)**      | **MoD (DeepMind)** |
| -------------------- | ------------------ | ------------------------- | ----------------------------- | ------------------------ |
| **优化目标**   | /                  | Context Length (长文本)   | Inference Cost (显存/带宽)    | Total FLOPs (计算量)     |
| **核心手段**   | 暴力全算           | 物理替换 (SSM 替代 Attn)  | 矩阵低秩分解 (KV Compression) | 动态跳层 (稀疏计算)      |
| **KV Cache**   | 100%               | ~12.5% (仅剩少量 Attn 层) | ~6.7% (94.1% 减小)            | 95–98% (需填充)         |
| **推理复杂度** | $O(L^2)$         | $O(L)$ (Linear)         | $O(L^2)$ 但系数极小         | $O(L^2)$ 但总量减半    |
| **推理速度**   | 慢（带宽瓶颈）     | 极快（线性）              | 极快（带宽降 85%）            | 较快（算力降）           |
| **能力损耗**   | 基准               | 无损/略强                 | 几乎无损                      | 轻微（$C=0.5$ 时 <2%） |

### 2. 路径深度解析

- **Jamba (Hybrid)**：**“空间换时间”**。通过引入 Mamba 层彻底消灭大部分 KV Cache。适合 128k+ 超长文，因为内存占用不随 $L$ 线性爆炸。
- **MLA (DeepSeek)**：**“数学换空间”**。通过训练时引入低秩投影矩阵（Down/Up 投影），KV Cache 压缩 94.1% 而几乎无损。**为什么能力无损**：KV 矩阵内在低秩（SVD 验证 rank ≈512），信息损失 <5%，全局注意力完整。
- **MoD (DeepMind)**：**“按需分配算力”**。
  **核心发现**：DeepMind 实验表明，对于 "the", "a", "," 等停用词，深层计算完全冗余。MoD 允许这些词在深层“坐直通车”。
  **效果**：通过增加深度（如 64 层 MoD 等效 32 层 Dense），将脑力集中在困难 Token 上。

## 三、工程现实与落地挑战

尽管 MoD 理论优美，但在实际推理框架中落地主要面临两大挑战：

### 1. 内存访问非对齐 + KV 一致性

- **问题**：Token A 选择计算第 5 层，Token B 跳过第 5 层。
  - Batch 内参与计算的 Token 稀疏且不连续。
  - CUDA Kernel 喜欢连续内存访问（Coalesced Access）。MoD 需要 Gather/Scatter 重组 Tensor，开销 ≈10%（vLLM 源码分析）。
- **KV 一致性**：跳过计算的 Token 仍需 KV Cache 更新（通常复制上一层或 mask）。
- **当前状态**：vLLM 0.6.x 已支持，但 Batch 非对齐仍是瓶颈。

### 2. MoE + MoD：终极稀疏形态

目前最前沿的架构是 **双重稀疏**：

- **横向稀疏 (MoE)**：每次只用 $1/N$ 的参数。
- **纵向稀疏 (MoD)**：每次只走 $1/M$ 的层数。

$$
\text{Total FLOPs} \approx \text{Dense} \times (1 - \text{Sparsity}_{MoE}) \times (1 - C) \approx 1 \times 0.25 \times 0.5 = 12.5\%
$$

这意味着我们可能用 12.5% 的算力驱动万亿参数模型。

## 总结：计算效率的“不可能三角”

| 技术          | 解决什么痛点？           | 代价是什么？              | 一句话总结                 |
| ------------- | ------------------------ | ------------------------- | -------------------------- |
| **MoE** | 训练慢、参数利用率低     | 显存需求大                | 专业的事交给专业专家       |
| **MLA** | 推理显存 (KV Cache) 爆炸 | 架构固定，无法后加        | 用最少内存存最久历史       |
| **MoD** | 推理计算量 (FLOPs) 太高  | CUDA 优化极难，生态未成熟 | 重要的词多算，不重要的少算 |

> **结语**：MLA 帮显存“瘦身”，MoE 帮大脑“分工”，MoD 帮计算“偷懒”。三者融合是 Agentic LLM 实现端侧部署与实时推理的必经之路。
