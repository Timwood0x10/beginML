# Engineering Cheat Sheet: Dimension Analysis & Parameter Estimation

**Abstract**: Deep learning isn't just mathematical derivation — it's a battle between memory and compute. Understanding **Dimensions** flow and **Parameters** counting is the foundation of designing efficient models. This chapter provides parameter-count formulas and dimension-transformation logic for mainstream architectures (Linear, CNN, RNN, Transformer, Mamba) — a prerequisite handbook for implementing **Scaling Laws** and **memory optimization**.

---

## 1. Fully Connected Layer (Linear)

This is the most basic affine transformation $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

### 1.1 Dimension Definitions
*   $D_{in}$: input feature dimension.
*   $D_{out}$: output feature dimension.

### 1.2 Parameter Count Formula
$$ N_{linear} = (D_{in} \times D_{out}) + D_{out} $$
*   **Weights**: matrix shape $[D_{out}, D_{in}]$.
*   **Bias**: vector shape $[D_{out}]$.

### 1.3 Key Properties
*   The parameter count is independent of the input Batch Size.
*   The parameter count grows **quadratically** with dimension (if $D_{in} \approx D_{out}$).

---

## 2. Convolutional Layer

CNN decouples the parameter count from the spatial size $(H, W)$ through **Weight Sharing**.

### 2.1 Dimension Definitions
*   $C_{in}$: number of input channels.
*   $C_{out}$: number of output channels (number of kernels).
*   $K$: kernel size (e.g., 3x3).

### 2.2 Parameter Count Formula (2D Conv)
$$ N_{conv} = (C_{in} \times K \times K \times C_{out}) + C_{out} $$
*   **Weights**: shape $[C_{out}, C_{in}, K, K]$.
*   **Bias**: shape $[C_{out}]$.

### 2.3 Key Properties
*   **Translation invariance**: the parameter count is completely independent of the input image's height/width $(H, W)$. Processing $224 \times 224$ and $1024 \times 1024$ images leaves the model size unchanged (though the Activations in memory change).
*   **Grouped convolution (Groups)**: if you set `groups=g`, the parameter count divides by $g$. Depthwise Separable convolution exploits this to greatly compress models.

---

## 3. Recurrent Neural Networks (RNN: LSTM/GRU)

RNNs share weights across time steps to process sequential data.

### 3.1 Dimension Definitions
*   $D_{in}$: input feature dimension.
*   $D_{h}$: hidden layer dimension (Hidden Size).

### 3.2 Parameter Count Formula (LSTM)
LSTM has 4 gates (Input, Forget, Cell, Output), each involving two transformations (input->hidden, hidden->hidden).
$$ N_{lstm} \approx 4 \times \left( (D_{in} \times D_{h}) + (D_{h} \times D_{h}) + D_{h} + D_{h} \right) $$
Simplified (ignoring bias details):
$$ N_{lstm} \approx 4 \times (D_{in} + D_{h}) \times D_{h} $$

### 3.3 Key Properties
*   **Quadratic growth**: the parameter count is dominated by $D_{h} \times D_{h}$. Doubling the Hidden Size increases parameters by about 4x.
*   **Bidirectional**: parameter count $\times 2$.

---

## 4. Transformer (Self-Attention)

The Transformer's parameters concentrate in the QKV projections and FFN layers.

### 4.1 Dimension Definitions
*   $D_{model}$: embedding dimension (e.g., 768, 1024).
*   $D_{ff}$: feed-forward network dimension (usually $4 \times D_{model}$).
*   $L$: number of layers.

### 4.2 Single-Block Parameter Estimation
1.  **Multi-Head Attention (MHA)**:
    *   $W_Q, W_K, W_V$: three $[D_{model}, D_{model}]$ matrices.
    *   $W_O$: output projection $[D_{model}, D_{model}]$.
    *   Total: $4 \times D_{model}^2$.
2.  **Feed-Forward Network (FFN)**:
    *   First layer (expand): $[D_{model}, D_{ff}]$.
    *   Second layer (reduce): $[D_{ff}, D_{model}]$.
    *   Total: $2 \times D_{model} \times D_{ff}$. If $D_{ff}=4D_{model}$, that's $8 \times D_{model}^2$.

**Transformer total formula (approximate)**:
$$ N_{transformer} \approx 12 \times D_{model}^2 \times L $$

### 4.3 Key Properties
*   **Quadratic dominance**: the parameter count is entirely determined by the square of $D_{model}$.
*   **Number of heads (Heads)**: changing the Head count **doesn't change** the total parameters (because each head's dimension $d_k = D_{model} / Heads$, so the sum is unchanged).
*   **Sequence length**: the parameter count is **independent** of sequence length $T$ (but the KV Cache memory at inference grows linearly/quadratically with $T$).

---

## 5. Mamba (State Space Model)

Mamba introduces the Selective Scan; its parameter structure differs from the Transformer.

### 5.1 Dimension Definitions
*   $D$: model dimension ($D_{model}$).
*   $N$: SSM state dimension (usually small, like 16).
*   $E$: expand factor (usually 2). Internal dimension $D_{in} = E \times D$.
*   $K_{conv}$: local 1D convolution kernel size (e.g., 4).

### 5.2 Sources of Parameters
1.  **Input Projections**: map $x$ to $z$ and $x'$. Two $[D, D_{in}]$ matrices.
    $\approx 2 \times E \times D^2$.
2.  **1D convolution**: $[D_{in}, 1, K_{conv}]$ (Depthwise). Very few parameters.
3.  **SSM parameter projections (Project to $\Delta, B, C$)**:
    This is Mamba's special feature. $\Delta, B, C$ are input-dependent.
    They must be projected from $D_{in}$ into parameter space.
    *   $\Delta$: $[D_{in}, D_{in}]$ (Rank-1) or similar small projection.
    *   $B, C$: $[D_{in}, N]$.
4.  **Output projection**: $[D_{in}, D]$.
    $\approx E \times D^2$.

### 5.3 Key Properties
*   **Linear complexity**: although the parameter count depends on $D$, the inference compute is linear in the sequence length $O(T)$, and the inference memory is constant (the state $N$ is fixed).
*   Compared to the Transformer, Mamba has a huge **inference efficiency advantage** when processing ultra-long sequences.

---

## 6. Code Practice: A Universal Parameter Counter

Write a tool that automatically analyzes the parameter distribution of a PyTorch model.

```python
import torch
import torch.nn as nn

def count_parameters(model, name="Model"):
    """
    print the parameter count of each layer in the model and compute the total
    """
    print(f"\n--- Analyzing {name} ---")
    total_params = 0
    trainable_params = 0
    
    # print a summary of each layer
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        # simply print the first few layers as an example
        if total_params == 0: 
            print(f"Layer: {name} | Size: {parameter.size()} | Count: {param}")
        
        total_params += param
        if parameter.requires_grad:
            trainable_params += param
            
    print(f"Total Parameters: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    # compute the model size (FP32 = 4 bytes)
    size_mb = total_params * 4 / (1024 ** 2)
    print(f"Model Size (FP32): {size_mb:.2f} MB")
    return total_params

# === Example 1: Transformer Block ===
d_model = 768
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=12, dim_feedforward=4*d_model)
count_parameters(encoder_layer, "Transformer Block (Bert-Base Size)")

# verify the computation: 12 * d_model^2
expected = 12 * (d_model ** 2)
print(f"Theoretical Approximation (12*D^2): {expected:,}")

# === Example 2: LSTM Layer ===
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1)
count_parameters(lstm, "LSTM Layer")

# === Example 3: CNN Layer ===
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
count_parameters(conv, "Conv2d Layer")
```


## 7. Advanced: Training Memory Anatomy

The $N$ we compute when designing a model is only the **static memory**. During training, memory usage is usually **3-4x** the parameter count or even more. Memory is mainly divided among the following four parts:

### 7.1 The Four Memory Giants
1.  **Model Weights**:
    *   $N$ parameters. FP32 takes $4N$ bytes; FP16/BF16 takes $2N$ bytes.
    *   *Share*: small.
2.  **Optimizer States**:
    *   **AdamW** needs to maintain the first moment $m$ and second moment $v$ (usually FP32).
    *   **Formula**: $8N$ (Adam states) + $4N$ (Master weights in Mixed Precision) $\approx 12N$ bytes.
    *   *Share*: **huge**. This is the core reason LoRA saves memory (it freezes the backbone parameters, so it doesn't need to maintain their optimizer states).
3.  **Gradients**:
    *   Same as the parameter count. FP32 takes $4N$; FP16 takes $2N$.
    *   *Share*: medium.
4.  **Activations**:
    *   Intermediate variables saved during the forward pass, used to compute gradients in backprop.
    *   **Formula**: $B \times L \times (H \times W \text{ or } T) \times C$.
    *   *Share*: **dynamic and huge**. Proportional to Batch Size and sequence length $T$.
    *   *Optimization*: **Gradient Checkpointing** eliminates this overhead by "trading time for space" (recomputing the forward pass).

### 7.2 Memory Estimation Formula (Mixed Precision Training)
Training an $N$-parameter model with Batch Size $B$:

$$ \text{Total Memory} \approx \underbrace{16N}_{\text{Static (Weights+Opt+Grad)}} + \underbrace{\text{Activations}(B, T)}_{\text{Dynamic}} + \underbrace{\text{Fragmentation}}_{\text{Overhead}} $$

*   **The origin of 16N**:
    *   Weights (FP16): $2N$
    *   Gradients (FP16): $2N$
    *   Optimizer (FP32 Adam): $12N$ (Copy of weights + Momentum + Variance)

### 7.3 Practical Reasoning: Why Does a 7B Model Need >24G of Memory?
*   **Model parameters $N \approx 7 \times 10^9$**.
*   **Static requirement (16N)**: $16 \times 7 \text{GB} \approx 112 \text{GB}$!
*   This is why a single 24G/40G card simply can't do full fine-tuning of a 7B model.
*   **LoRA to the rescue**:
    *   Freeze the backbone -> optimizer states drop to 0 (for the backbone).
    *   Only train the LoRA parameters ($r \ll D$) -> optimizer states are tiny.
    *   The remaining memory head is just: **Weights (14GB, FP16) + Activations**.
    *   Combined with 4-bit quantization (QLoRA), Weights drop to 3.5GB, and a single 24G card handles it easily.
