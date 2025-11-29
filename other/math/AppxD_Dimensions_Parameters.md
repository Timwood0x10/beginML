# 工程速查：维度分析与参数估算 (Dimensions & Parameters Cheat Sheet)

**摘要**：深度学习不仅是数学推导，更是显存与算力的博弈。理解**维度 (Dimensions)** 流动和**参数 (Parameters)** 计数是设计高效模型的基础。本章提供主流架构（Linear, CNN, RNN, Transformer, Mamba）的参数量计算公式与维度变换逻辑，是实施 **Scaling Laws** 和 **显存优化** 的前置手册。

---

## 1. 全连接层 (Fully Connected / Linear)

这是最基础的仿射变换 $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$。

### 1.1 维度定义
*   $D_{in}$：输入特征维度。
*   $D_{out}$：输出特征维度。

### 1.2 参数量公式
$$ N_{linear} = (D_{in} \times D_{out}) + D_{out} $$
*   **权重**：矩阵形状 $[D_{out}, D_{in}]$。
*   **偏置**：向量形状 $[D_{out}]$。

### 1.3 关键特性
*   参数量与输入数据的 Batch Size 无关。
*   参数量随维度呈**二次方**增长（若 $D_{in} \approx D_{out}$）。

---

## 2. 卷积层 (Convolutional Layer)

CNN 通过**权重共享 (Weight Sharing)** 实现了参数量与空间尺寸 $(H, W)$ 的解耦。

### 2.1 维度定义
*   $C_{in}$：输入通道数。
*   $C_{out}$：输出通道数（卷积核数量）。
*   $K$：卷积核尺寸 (Kernel Size，如 3x3)。

### 2.2 参数量公式 (2D Conv)
$$ N_{conv} = (C_{in} \times K \times K \times C_{out}) + C_{out} $$
*   **权重**：形状 $[C_{out}, C_{in}, K, K]$。
*   **偏置**：形状 $[C_{out}]$。

### 2.3 关键特性
*   **平移不变性**：参数量完全不依赖于输入图像的高宽 $(H, W)$。处理 $224 \times 224$ 和 $1024 \times 1024$ 的图，模型大小不变（但在显存中的 Activation 会变）。
*   **分组卷积 (Groups)**：若设置 `groups=g`，参数量除以 $g$。深度可分离卷积 (Depthwise Separable) 利用了这一点大幅压缩模型。

---

## 3. 循环神经网络 (RNN: LSTM/GRU)

RNN 在时间步之间共享权重，处理序列数据。

### 3.1 维度定义
*   $D_{in}$：输入特征维度。
*   $D_{h}$：隐藏层维度 (Hidden Size)。

### 3.2 参数量公式 (LSTM)
LSTM 内部有 4 个门 (Input, Forget, Cell, Output)，每个门涉及两个变换（输入->隐层，隐层->隐层）。
$$ N_{lstm} \approx 4 \times \left( (D_{in} \times D_{h}) + (D_{h} \times D_{h}) + D_{h} + D_{h} \right) $$
简化版（忽略偏置细节）：
$$ N_{lstm} \approx 4 \times (D_{in} + D_{h}) \times D_{h} $$

### 3.3 关键特性
*   **平方级增长**：参数量主要由 $D_{h} \times D_{h}$ 主导。将 Hidden Size 翻倍，参数量增加约 4 倍。
*   **双向 (Bidirectional)**：参数量 $\times 2$。

---

## 4. Transformer (Self-Attention)

Transformer 的参数量主要集中在 QKV 投影和 FFN 层。

### 4.1 维度定义
*   $D_{model}$：嵌入维度 (Embedding Dim, 如 768, 1024)。
*   $D_{ff}$：前馈网络维度 (通常为 $4 \times D_{model}$)。
*   $L$：层数 (Layers)。

### 4.2 单层 Block 参数量估算
1.  **Multi-Head Attention (MHA)**:
    *   $W_Q, W_K, W_V$：三个 $[D_{model}, D_{model}]$ 矩阵。
    *   $W_O$：输出投影 $[D_{model}, D_{model}]$。
    *   合计：$4 \times D_{model}^2$。
2.  **Feed-Forward Network (FFN)**:
    *   第一层 (升维)：$[D_{model}, D_{ff}]$。
    *   第二层 (降维)：$[D_{ff}, D_{model}]$。
    *   合计：$2 \times D_{model} \times D_{ff}$。若 $D_{ff}=4D_{model}$，则为 $8 \times D_{model}^2$。

**Transformer 总公式 (近似)**：
$$ N_{transformer} \approx 12 \times D_{model}^2 \times L $$

### 4.3 关键特性
*   **平方统治**：参数量完全由 $D_{model}$ 的平方决定。
*   **注意力头数 (Heads)**：改变 Head 数量**不改变**参数总量（因为每个 Head 的维度 $d_k = D_{model} / Heads$，总和不变）。
*   **序列长度**：参数量与序列长度 $T$ **无关**（但推理时的 KV Cache 显存占用与 $T$ 成线性/二次关系）。

---

## 5. Mamba (State Space Model)

Mamba 引入了选择性扫描 (Selective Scan)，其参数结构与 Transformer 不同。

### 5.1 维度定义
*   $D$：模型维度 ($D_{model}$)。
*   $N$：SSM 状态维度 (State dimension, 通常较小如 16)。
*   $E$：扩展因子 (Expand factor, 通常为 2)。内部维度 $D_{in} = E \times D$。
*   $K_{conv}$：局部 1D 卷积核大小 (如 4)。

### 5.2 参数来源
1.  **输入投影 (Input Projections)**：将 $x$ 映射到 $z$ 和 $x'$。两个 $[D, D_{in}]$。
    $\approx 2 \times E \times D^2$。
2.  **1D 卷积**：$[D_{in}, 1, K_{conv}]$ (Depthwise)。参数极少。
3.  **SSM 参数投影 (Project to $\Delta, B, C$)**：
    这是 Mamba 的特殊之处。$\Delta, B, C$ 是输入依赖的。
    需要从 $D_{in}$ 投影到参数空间。
    *   $\Delta$: $[D_{in}, D_{in}]$ (Rank-1) 或类似小投影。
    *   $B, C$: $[D_{in}, N]$。
4.  **输出投影**：$[D_{in}, D]$。
    $\approx E \times D^2$。

### 5.3 关键特性
*   **线性复杂度**：虽然参数量与 $D$ 有关，但其推理时的计算复杂度相对于序列长度是线性的 $O(T)$，且推理显存是恒定的（状态 $N$ 固定）。
*   相比 Transformer，Mamba 在处理超长序列时具有巨大的**推理效率优势**。

---

## 6. 代码实战：通用参数计数器

编写一个能够自动分析 PyTorch 模型参数分布的工具。

```python
import torch
import torch.nn as nn

def count_parameters(model, name="Model"):
    """
    打印模型每一层的参数量，并计算总量
    """
    print(f"\n--- Analyzing {name} ---")
    total_params = 0
    trainable_params = 0
    
    # 打印每一层的摘要
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        # 简单打印前几层用于示例
        if total_params == 0: 
            print(f"Layer: {name} | Size: {parameter.size()} | Count: {param}")
        
        total_params += param
        if parameter.requires_grad:
            trainable_params += param
            
    print(f"Total Parameters: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    # 计算模型大小 (FP32 = 4 bytes)
    size_mb = total_params * 4 / (1024 ** 2)
    print(f"Model Size (FP32): {size_mb:.2f} MB")
    return total_params

# === 示例 1: Transformer Block ===
d_model = 768
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=12, dim_feedforward=4*d_model)
count_parameters(encoder_layer, "Transformer Block (Bert-Base Size)")

# 计算验证: 12 * d_model^2
expected = 12 * (d_model ** 2)
print(f"Theoretical Approximation (12*D^2): {expected:,}")

# === 示例 2: LSTM Layer ===
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1)
count_parameters(lstm, "LSTM Layer")

# === 示例 3: CNN Layer ===
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
count_parameters(conv, "Conv2d Layer")
```


## 7. 进阶：训练显存解剖学 (Training Memory Anatomy)

我们在设计模型时算的 $N$ 只是**静态显存**。在训练时，显存占用通常是参数量的 **3-4 倍**甚至更多。显存主要被以下四部分瓜分：

### 7.1 显存四巨头
1.  **模型参数 (Model Weights)**:
    *   $N$ 个参数。FP32 占用 $4N$ 字节，FP16/BF16 占用 $2N$ 字节。
    *   *占比*：较小。
2.  **优化器状态 (Optimizer States)**:
    *   **AdamW** 需要维护一阶矩 $m$ 和二阶矩 $v$（通常是 FP32）。
    *   **公式**：$8N$ (Adam states) + $4N$ (Master weights in Mixed Precision) $\approx 12N$ 字节。
    *   *占比*：**巨大**。这是为什么 LoRA 省显存的核心原因（冻结了主干参数，不需要维护它们的优化器状态）。
3.  **梯度 (Gradients)**:
    *   与参数量一致。FP32 占用 $4N$，FP16 占用 $2N$。
    *   *占比*：中等。
4.  **激活值 (Activations)**:
    *   前向传播时保存的中间变量，用于反向传播计算梯度。
    *   **公式**：$B \times L \times (H \times W \text{ or } T) \times C$。
    *   *占比*：**动态且巨大**。与 Batch Size 和序列长度 $T$ 成正比。
    *   *优化*：**梯度检查点 (Gradient Checkpointing)** 通过“用时间换空间”（重算前向）来消除这部分开销。

### 7.2 显存估算公式 (Mixed Precision Training)
训练一个 $N$ 参数的模型，Batch Size 为 $B$：

$$ \text{Total Memory} \approx \underbrace{16N}_{\text{Static (Weights+Opt+Grad)}} + \underbrace{\text{Activations}(B, T)}_{\text{Dynamic}} + \underbrace{\text{Fragmentation}}_{\text{Overhead}} $$

*   **16N 的来源**：
    *   Weights (FP16): $2N$
    *   Gradients (FP16): $2N$
    *   Optimizer (FP32 Adam): $12N$ (Copy of weights + Momentum + Variance)

### 7.3 实战推演：为什么 7B 模型需要 >24G 显存？
*   **模型参数 $N \approx 7 \times 10^9$**。
*   **静态需求 (16N)**：$16 \times 7 \text{GB} \approx 112 \text{GB}$！
*   这就是为什么单卡 24G/40G 根本跑不动 7B 的全量微调。
*   **LoRA 的救赎**：
    *   冻结主干 -> 优化器状态降为 0 (针对主干)。
    *   只训练 LoRA 参数 ($r \ll D$) -> 优化器状态极小。
    *   显存大头只剩下：**Weights (14GB, FP16) + Activations**。
    *   配合 4-bit 量化 (QLoRA)，Weights 降为 3.5GB，单卡 24G 轻松拿下。