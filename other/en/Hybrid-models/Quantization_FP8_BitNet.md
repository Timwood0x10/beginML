# The Quantization Revolution: From DeepSeek's FP8 to BitNet's Ternary Logic

> **Abstract**
> As parameters break past the trillion mark, the Memory Wall becomes the biggest bottleneck.
> DeepSeek-V3, through FP8 mixed-precision training + block-wise quantization, compresses training cost to the extreme (memory/bandwidth down 50%, compute utilization up 2x).
> BitNet b1.58 proposes an even more radical paradigm shift: compressing weights to the ternary set \(\{-1, 0, 1\}\), completely abandoning floating-point multiplication (theoretical information entropy 1.58 bits, ~10x volume compression, 70–90% energy reduction).
> This chapter deeply dissects the internal mechanisms of both technologies from three dimensions — mathematical principles, engineering implementation, and deployment trade-offs (data from the DeepSeek-V3 report arXiv:2412.19437, the BitNet b1.58 paper arXiv:2310.11453, and 2026 community reproductions).

## 1. The Physics of Numerical Precision

The numerical representation on a GPU directly determines compute, memory, and energy.

### 1. Format Comparison: The E and M Trade-off

Floating-point format: $(-1)^S \times 2^{E - \text{bias}} \times (1 + M)$.

| Format | Sign | Exponent | Mantissa | Total bits | Dynamic range | Precision | Typical use | H100 Tensor Core throughput |
| -------- | ---- | -------- | -------- | ------ | ------------ | ---- | ------------------- | --------------------- |
| FP32 | 1 | 8 | 23 | 32 | Extremely large | High | Baseline | Baseline |
| BF16 | 1 | 8 | 7 | 16 | Same as FP32 | Medium | Mainstream training | Baseline |
| FP8 E4M3 | 1 | 4 | 3 | 8 | Medium | High | Weights/activations | 2x BF16 |
| FP8 E5M2 | 1 | 5 | 2 | 8 | Large | Low | Gradients (prevents overflow) | 2x BF16 |

**Why is FP8 Tensor Core throughput 2x BF16?**

- FP8's vector width is 256 (BF16's is 128), doubling GEMM throughput (H100 FP8 3958 TFLOPS vs BF16 1979 TFLOPS).
- DeepSeek-V3's goal: maximize use of the FP8 Tensor Core's "free" 2x compute while preventing overflow/precision loss.

## 2. DeepSeek-V3's FP8 Mixed-Precision Training

Going all-FP8 directly causes Loss Divergence. DeepSeek uses **fine-grained block-wise quantization + Master Weights**.

### 1. Block-wise Quantization

Traditional per-tensor quantization is heavily affected by outliers (error +15–20%). DeepSeek splits the matrix into small blocks (128×128), computing a scale factor independently per block.

**Quantization formula** (round-to-nearest + clip):

$$
x_q = \text{clamp}\left( \lfloor x / s \rceil, -Q_{\max}, Q_{\max} \right)
$$

The scale factor:

$$
s = \frac{\max(|x|)}{Q_{\max}}
$$

**Dequantization**:

$$
x \approx x_q \times s
$$

**Why is block-wise better than per-tensor?**

- per-tensor: a global outlier causes precision loss in normal values (error +15–20%).
- Block-wise: each block scales independently; quantization error <5% (V3 ablation).
- **Why round-to-nearest + clip**: round-to-nearest minimizes rounding bias; clip prevents overflow (FP8 E4M3 range [-448, 448]).

### 2. Engineering Tricks: Master Weights + High-Precision Key Layers

- **Master Weights**: always kept in BF16; only converted to FP8 in real time before entering the Tensor Core (avoiding accumulated quantization error).
- **High-precision key layers**: Embedding, Output Head, and Attention QKV stay in BF16/FP32 (extremely precision-sensitive).
- **Result**: training Loss matches BF16, memory/bandwidth down 50%, compute utilization up 2x (V3 report Section 3.2).

## 3. BitNet b1.58: Ending Floating-Point Multiplication

BitNet b1.58 compresses weights to the ternary set \(\{-1, 0, 1\}\), completely abandoning floating-point multiplication.

### 1. Why 1.58 bits?

3 weight states; the information entropy:

$$
\log_2(3) \approx 1.58 \text{ bits}
$$

Compared to FP16 (16 bits), theoretical volume compression ≈10x (engineering requires a 2-bit packing Kernel).

### 2. Mathematical Derivation: AbsMean Quantization

BitNet uses the average absolute value (AbsMean) rather than Min-Max scaling, preserving weight energy.

**Scale factor**:

$$
\gamma = \frac{1}{NM} \sum_{i,j} |W_{ij}|
$$

**Weight ternarization**:

$$
W_{\text{quant}} = \text{clamp}\left( \lfloor W / \gamma \rceil, -1, 1 \right)
$$

**Why is AbsMean better than Min-Max?**

- Min-Max is heavily affected by outliers (an outlier inflates s, losing precision in normal values).
- AbsMean preserves overall energy, reducing variance error 30% (BitNet paper ablation).
- **Why variance matters more**: quantization error mainly comes from energy loss; AbsMean keeps \(\|W\|^2\) closer to the original value.

**Activation quantization**: usually 8-bit (clip + scale) to prevent activation overflow.

### 3. The Core Magic: Matrix Multiplication Becomes Addition

Standard matrix multiplication: \(Y = W \cdot X = \sum W_i X_i\). When \(W_i \in \{-1, 0, 1\}\):

- \(W_i = 1\): \(Y \leftarrow Y + X_i\)
- \(W_i = -1\): \(Y \leftarrow Y - X_i\)
- \(W_i = 0\): skip

**Why 70–90% energy reduction?**

- Floating-point multipliers (Multiplier) take 70% of the FP16 unit area and dominate energy.
- BitNet only needs adders (Adder), cutting hardware energy/area 70–90% (BitNet paper Section 4.2).
- **Engineering status**: in 2026, dedicated Kernels supporting 2-bit packing are still required; commercial deployment is limited.

## 4. Inference Optimization: KV Cache and AWQ

In long-text (128k context) scenarios, the KV Cache exceeds the model weights themselves.

### 1. AWQ (Activation-aware Weight Quantization)

Quantizing purely by weight magnitude ignores activations. AWQ protects **salient weights** (weights with large activation values).

**Core finding**: weight distributions follow a power law; 1% salient weights contribute 50% of activation energy.
AWQ keeps high precision or special scaling for these weights, and harshly quantizes the other 99%.

**Why does it work?**

- Weights with large activations strongly affect output; quantization error is amplified.
- AWQ keeps error within 1–2% (AWQ paper ablation).

### 2. KV Cache Quantization

**RoPE-aware quantization**:

- Problem: directly quantizing after RoPE rotation causes huge errors.
- Solution:
  1. First inverse-rotate back to standard space.
  2. Quantize for storage (e.g., Int8/FP8).
  3. On access, dequantize + rotate.
- **Why does it work**: after inverse rotation the distribution is more uniform; quantization error <3% (KVCache community implementation).

## Summary: The Game Between Precision and Efficiency

| Feature | **BF16 (standard)** | **FP8 (DeepSeek-V3)** | **BitNet b1.58** |
| ------------------ | --------------------- | --------------------------- | ----------------------------------- |
| **Numerical space** | Continuous reals | Sparse reals | \(\{-1, 0, 1\}\) |
| **Core operator** | Fused Multiply-Add | FP8 Tensor Core | Integer Add |
| **Memory usage** | 1x | 0.5x | 0.1x |
| **Training difficulty** | Low | High (must prevent overflow) | Extremely high (needs QAT) |
| **Inference energy** | Baseline | 0.5x | 0.1–0.3x |
| **Status** | Industry standard | Current SOTA | Future trend (experimental; needs dedicated Kernels) |

> **Conclusion**
> DeepSeek-V3 proves FP8 is the current ultimate way to squeeze GPU performance.
> BitNet foreshadows a future that needs no floating-point multiplication — only integer addition.
> **The endgame of large models may not be bigger GPUs, but simpler mathematics.**
