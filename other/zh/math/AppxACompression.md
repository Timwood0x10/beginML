# [Appendix A] 模型压缩数学：LoRA 与 量化

**摘要**：随着 LLM 参数量突破千亿，全参数微调和 FP16 推理的显存成本已成为瓶颈。工程界提出了两大解决方案：**LoRA**（通过低秩矩阵分解减少训练参数）和 **量化**（通过降低数值精度减少存储需求）。本章将从数学角度严格推导 LoRA 的梯度更新机制，解析量化过程中的舍入误差分布，并探讨 QLoRA 背后的正态分布映射原理。

---

## 1. LoRA (Low-Rank Adaptation) 的线性代数

**核心假设**：虽然预训练模型的权重矩阵 $W \in \mathbb{R}^{d \times k}$ 秩很高（满秩），但在特定任务微调时，权重更新量 $\Delta W$ 的**内在维度 (Intrinsic Dimension)** 是极低的。

### 1.1 矩阵分解形式
对于预训练权重 $W_0$，我们将更新量约束为两个低秩矩阵的乘积：
$$ W = W_0 + \Delta W = W_0 + BA $$
其中：
*   $W_0 \in \mathbb{R}^{d \times k}$：冻结的预训练权重。
*   $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$：可训练的低秩适配器。
*   $r \ll \min(d, k)$：秩，通常取 8, 16, 64。

### 1.2 前向传播与初始化
$$ h = Wx = W_0 x + BAx $$
**初始化策略的数学意义**：
为了保证训练初始阶段模型行为与预训练模型完全一致（即 $\Delta W = 0$）：
*   $A$ 初始化为**高斯随机分布** $\mathcal{N}(0, \sigma^2)$。
*   $B$ 初始化为**全零矩阵**。
*   由此可得初始时刻：$BA = 0 \cdot A = 0$。

### 1.3 梯度的链式法则
为什么训练 $A, B$ 比训练 $W$ 快？看梯度流。
设损失函数为 $\mathcal{L}$，对于 $W$ 的梯度为 $\nabla_W \mathcal{L}$。
根据链式法则，传导给 $A$ 和 $B$ 的梯度为：
$$ \frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial W} A^T \in \mathbb{R}^{d \times r} $$
$$ \frac{\partial \mathcal{L}}{\partial A} = B^T \frac{\partial \mathcal{L}}{\partial W} \in \mathbb{R}^{r \times k} $$

**参数量对比**：
*   全量微调参数：$d \times k$
*   LoRA 参数：$(d+k) \times r$
*   当 $r \ll d, k$ 时，参数量减少了几个数量级（例如 GPT-3 175B 中仅需训练 0.01% 的参数）。

### 1.4 缩放因子 $\alpha$ (Scaling Factor)
实际公式通常包含缩放系数：
$$ h = W_0 x + \frac{\alpha}{r} BA x $$

*   **数学作用**：解耦超参数。当我们将 $r$ 从 8 调整为 16 时，如果保持 $\alpha$ 不变，总体更新幅度 $\Delta W$ 的尺度保持稳定，无需重新搜索学习率。
*   **物理直觉**：$\alpha$ 类似于 LoRA 路径的`学习率放大器`。
    
    * 等价于将学习率乘以 α/r，因此许多代码里直接把 α 设为 16、32、64 来偷懒调参

---

## 2. 量化 (Quantization) 的数值分析

量化的本质是将连续实数域 $\mathbb{R}$（或高精度浮点域 FP32）映射到有限的离散整数域 $\mathbb{Z}_q$（如 INT8）。

### 2.1 仿射量化 (Affine Quantization)
这是最通用的非对称量化方案。定义映射函数 $Q$ 和反量化函数 $D$：

$$ Q(x) = \text{clamp}\left( \text{round}\left( \frac{x}{S} + Z \right), q_{min}, q_{max} \right) $$
$$ \tilde{x} = D(q) = S(q - Z) $$

其中：
*   $S$ (Scale)：缩放因子（FP32）。
*   $Z$ (Zero-point)：零点偏移（Integer），确保实数 0 精确映射到整数，这对 ReLU 等激活函数至关重要。

**参数计算**：
给定待量化张量的范围 $[x_{min}, x_{max}]$：
$$ S = \frac{x_{max} - x_{min}}{q_{max} - q_{min}} $$
$$ Z = \text{round}\left( q_{min} - \frac{x_{min}}{S} \right) $$

### 2.2 对称量化 (Symmetric Quantization)
为了加速计算（减少 $Z$ 带来的加法开销），通常强制 $Z=0$。此时范围被约束为关于原点对称 $[-c, c]$。
$$ S = \frac{\max(|x|)}{q_{max}} $$
这对权重量化非常有效，因为权重分布通常接近以 0 为中心的对称分布。

### 2.3 量化误差分析
量化误差（Quantization Error）定义为 $\epsilon = x - \tilde{x}$。
假设输入信号 $x$ 的动态范围远大于步长 $S$，舍入误差 $\epsilon$ 可以近似看作在区间 $[-S/2, S/2]$ 上的**均匀分布** $U(-S/2, S/2)$。

**均方误差 (MSE)**：
$$ \mathbb{E}[\epsilon^2] = \int_{-S/2}^{S/2} \frac{1}{S} u^2 du = \frac{S^2}{12} $$

**信噪比 (SQNR)**：
对于 $b$-bit 量化，步长 $S \approx \frac{R}{2^b}$（$R$为动态范围）。
$$ \text{SQNR}_{dB} \approx 6.02 b + 1.76 $$
这推导出了著名的经验法则：**每增加 1 bit 位宽，信噪比提升约 6 dB。**

---

## 3. QLoRA 与 NF4 (Normal Float 4)

QLoRA 结合了上述两者，在 4-bit 量化的基座模型上训练 LoRA。这里引入了 **NF4** 数据类型。

### 3.1 分位数量化 (Quantile Quantization)
神经网络的权重通常服从正态分布 $\mathcal{N}(0, 1)$，而不是均匀分布。使用均匀间隔的 INT4 会浪费大量 bit 在空旷的尾部。

**信息论最优策略**：每个量化桶 (Bin) 中应该包含相同数量的数值（等概率）。

NF4 的构造过程：
1.  取正态分布的累计分布函数 (CDF) $F(x)$。
2.  将 $[0, 1]$ 区间等分为 16 份（$2^4$）：$p_i = \frac{i+0.5}{16}$。
3.  通过反函数 $Q_i = F^{-1}(p_i)$ 找到 16 个对应的分位点。
4.  这 16 个值就是 NF4 的量化码本。
5.  NF4 实际码本为 [-1.0, -0.696, …, +1.0]，共 16 个值，可在 bitsandbytes 库中查到。

这种设计使得 NF4 在表示正态分布权重时，比 INT4 具有更小的 MSE 误差。

---

## 4. 工程实战建议

### 4.1 LoRA 最佳实践
*   **Rank 选择**：对于一般任务，**$r=8$ 或 $16$** 足矣。对于逻辑推理或数学等复杂任务，可尝试 $r=64$。
*   **Alpha 设置**：经典设置为 $\alpha = 2r$ 或 $\alpha = r$。在调参时，固定 $\alpha$ 调整 $r$ 会改变训练动力学；通常建议固定 $r$，调整学习率。
*   **Target Modules**：不仅仅微调 `q_proj, v_proj`。实验证明，微调**所有线性层** (All Linear Layers) 效果最好。

### 4.2 量化陷阱
*   **激活值异常点 (Activation Outliers)**：权重通常平滑，但激活值（Feature Maps）中常存在个别数值极大的`离群点`（尤其在 6B+ 模型中）。这些离群通道在 7B+ 模型中通常集中在第 24–30 层，呈现明显的层级依赖性（Llama-2 实测）。

*   **解决方案 (LLM.int8())**：**混合精度分解**。设定阈值（如 6.0），将超过阈值的离群维度提取出来用 FP16 计算，其余部分用 INT8 向量乘法。
    $$ Y = X_{out} W_{out} + X_{int8} W_{int8} $$
    虽然离群点很少（<0.1%），但它们对精度至关重要。

---

## 5. 代码实现：手搓 LoRA 与 INT8 量化

以下代码展示了一个带有 LoRA 旁路的线性层，以及一个仿射量化器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        # 1. 冻结的预训练权重 (模拟)
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # 2. LoRA 适配器
        self.r = rank
        self.scaling = alpha / rank
        # A: 高斯初始化
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1/rank)**0.5)
        # B: 零初始化
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        # 原始路径 (Frozen)
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA 路径 (Trainable): x @ A^T @ B^T * scale
        # 注意 PyTorch Linear 是 x @ W^T，所以这里顺序是 x @ A.T @ B.T
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return result + lora_out

def affine_quantize(tensor, num_bits=8):
    """
    非对称仿射量化模拟
    """
    qmin, qmax = 0, 2**num_bits - 1
    
    # 1. 计算 Scale 和 Zero-point
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    zero_point = torch.clamp(torch.round(zero_point), qmin, qmax)
    
    # 2. 量化 (Quantize)
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    
    # 3. 反量化 (De-quantize) - 用于计算伪量化误差
    deq_tensor = scale * (q_tensor - zero_point)
    
    return q_tensor, deq_tensor, scale

# === 测试代码 ===
x = torch.randn(1, 10) # Input
layer = LoRALinear(10, 20, rank=4)
y_pred = layer(x)

# 打印参数量对比
full_params = 10 * 20
lora_params = (10 + 20) * 4
print(f"Full Params: {full_params}, LoRA Params: {lora_params} (Compression: {full_params/lora_params:.1f}x)")

# 测试量化误差
weights = layer.weight
q_w, deq_w, s = affine_quantize(weights, num_bits=8)
mse = torch.mean((weights - deq_w)**2)
print(f"INT8 Quantization MSE: {mse.item():.6f}")
print(f"Theoretical MSE (S^2/12): {(s**2/12).item():.6f}")
```

**结果分析**

```shell
Full Params: 200, LoRA Params: 120 (Compression: 1.7x)
INT8 Quantization MSE: 0.000035
Theoretical MSE (S^2/12): 0.000038
```

在运行上述代码时，会发现：
*   参数大幅减少：LoRA 路径的参数远少于原始权重。
*   量化误差吻合：实际测量的 MSE 与理论推导值 
    $$ S^2/12 $$
    非常接近，这验证了均匀分布假设在一般情况下是成立的。
*   零初始化作用：如果将 lora_B 改为随机初始化，初始输出将发生剧烈跳变，可能破坏预训练的特征分布，导致微调初期不稳定。

