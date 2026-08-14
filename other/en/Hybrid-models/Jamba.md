# The Hybrid Architecture Explained: Deep Fusion of Transformer and Mamba (Using Jamba as an Example)

> **Abstract**:
> The Transformer architecture is limited by $O(L^2)$ computational complexity and a KV Cache that grows linearly with sequence length $L$.
> **Jamba** (AI21 Labs) achieves a performance breakthrough on long text by mixing **Mamba (Selective SSM)**, **Transformer**, and **MoE (Mixture-of-Experts)**.
> This chapter deconstructs Jamba's mathematical essence and explores how it balances memory precision and computational efficiency through the "hybrid mechanism."

## 1. The Mathematical Foundation: From Generic SSM to Mamba's Evolution

Mamba's core lies in the **Selective State Space Model (Selective SSM)**. Unlike generic SSMs, Mamba makes key simplifications in its implementation for computational efficiency.

### 1. Discretization Engineering Approximation

In continuous systems, $\dot{h}(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$. In Mamba's engineering implementation, $\mathbf{A}$ is assumed to be a **diagonal matrix**; using zero-order hold (ZOH) for discretization, the simplified form is:

$$
\bar{A}_k = \exp(\Delta_k \cdot A_{kk}), \quad \bar{B}_k = (\Delta_k \cdot B_k)
$$

> **Note**: real code often uses the approximation $\bar{B} = \Delta \odot B$. This diagonalized structure is the prerequisite for Fast Scan and Fused CUDA Kernels.
> Since A is a diagonal matrix, its eigenvalues are exactly the diagonal elements Akk, so exp(Δk⋅Akk) can be computed element-wise, avoiding the expensive matrix exponential — this is the mathematical basis for Mamba's linear complexity.

### 2. Complexity Analysis of Parallel Scan

The key to Mamba escaping the RNN serial limitation is **Associative Scan**.

- **Total Work Complexity**: $O(L)$, guaranteeing linear scaling with sequence length.
- **Parallel Depth**: $O(\log L)$, determining the number of steps for parallel execution on the GPU.

---

## 2. Architecture Design: Why Do We Need "Hybrid"?

Jamba isn't pure Mamba; it combines the advantages of both by **interleaving** Attention layers and Mamba layers.

### Macro Structure: Interleaved Stack

Jamba is built by repeating multiple "Jamba Groups," each containing a specific ratio of Attention to Mamba (e.g., 1:7).

![img](./image/Jamba/struct.png)


### **Micro Structure: Inside a Single Hybrid Block**

**Each Block follows the principle of "sequence mixing first, expert mixing second."**

![block](./image/Jamba/block.png)

### Deep Reading of the Diagram:

* **Sequence Mixer layer**: this is Jamba's core variable. Most layers (7/8) use **Mamba** for efficient linear sequence processing; only a few layers (1/8) use **Attention** for global K-V retrieval, ensuring the model doesn't get "lost" in ultra-long text.
* **Sparse MoE MLP layer**: in the second half of each Block, Jamba discards the traditional dense MLP in favor of MoE.

  * **Router**: dynamically decides which experts to activate based on the input Token's features.
* **Computational efficiency**: this design lets the model hold a huge 52B "knowledge capacity" (total parameters), yet each Token only passes through a 12B computation path (activated parameters).
* **Residual Stream**: the **+** symbol in the figure represents residual connections. It ensures gradients flow effectively even in deep networks and allows each layer to learn only "incremental" information on top of the original input.


### 1. Mechanism Comparison and Complementarity

| Mechanism | Strengths (Pros) | Weaknesses (Cons) |
| --------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Attention** | **Precise retrieval**: can locate any exact coordinate in history via "pointers." | **Non-compressive**: compute and storage grow quadratically/linearly with $L$, limiting capability. |
| **SSM (Mamba)** | **Progressive compression**: efficiently handles streaming information; inference cost is constant. | **Compression loss**: the state $h_t$ is an **exponentially decaying weighted sum** of history $h_t = \mathbf{\bar{A}}_t \, h_0 \;+\; \sum_{i=1}^{t} \mathbf{\bar{A}}_{\,t-i} \, \mathbf{\bar{B}} \, x_i$; in long sequences, early tokens' weights decay sharply, making precise backtracking hard. |

**Jamba's strategy**: use a few Attention layers (like a 1:7 ratio) as "global precise indexing anchors," and use many Mamba layers for "efficient semantic compression and transport." This ratio is a tunable hyperparameter, traded off between recall precision and inference speed.

### 2. Positioning of MoE Layers

**MoE as a general efficiency tool**: Jamba uses MoE in the MLP stage — a **parameter-efficiency** strategy **independent of** the SSM/Attention mix. It lets the model expand total capacity and knowledge storage without significantly increasing inference compute (FLOPS), further improving the **semantic richness** of long-text processing.

---

## 3. Engineering Practice: The Truth About Memory Overhead

In long-text inference, Jamba's memory advantage comes from classified management of different states.

### 1. Memory-Overhead Formula Comparison

- **Mamba layer state memory (Fixed)**: doesn't grow with $L$, but is affected by Batch Size ($B$) and hidden-state dimension.

  $$
  \text{Mem}_{\text{Mamba}} \propto N_{\text{mamba}} \cdot B \cdot d_{\text{model}} \cdot d_{\text{state}}
  $$
- **Attention layer KV Cache (Dynamic)**: grows linearly with $L$.

  $$
  \text{Mem}_{\text{KV}} \propto N_{\text{attn}} \cdot B \cdot L \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}}
  $$

### 2. The Revolutionary Reduction of the KV Cache

**Correcting a misconception**: Jamba's Attention layers **still need** the KV Cache.

Jamba usually uses **Grouped Query Attention (GQA)** rather than standard Multi-Head Attention (MHA). GQA is itself a KV-Cache optimization: it reduces the number of Key/Value heads, further lowering cache memory, forming a **dual optimization** with Mamba's efficient design.

The advantage: since Attention layers make up only about 12.5% of total layers (1/8) and are usually combined with **GQA**, the overall KV-Cache growth slope is far lower than a pure Transformer. This makes 128k-context inference possible on a single card.

> Take 128K tokens, Batch Size=1, d_model=4096, and a typical GQA config (num_kv_heads=2–8) as an example:
>
> * Pure Transformer: KV Cache needs about **800–1500 MB** (depending on layers, heads, precision).
> * Jamba (1:7 ratio + GQA): Attention layers take only ~12.5%, so the KV Cache is about **100–300 MB** (230 MB is a reasonable example value).
>   **Example values vary slightly with GQA head count, precision (bf16/float16), and specific layer count; verify with actual vLLM / mamba-ssm measurements.**
>   **Conclusion**: with the same memory, Jamba can support 6x larger Batch Size, or roughly 6x longer context (single-card 80GB A100/H100 scenarios).

---

## 4. Performance and Bottlenecks: Prefill vs. Decode

Jamba's computational advantage shows differently at different stages:

1. **Prefill**: the Mamba part is $O(L)$; the Attention layer uses **global self-attention** (no sliding window, no sparse attention), theoretically still $O(L^2)$. But since Attention layers make up only about 1/8 of total layers (typical 1:7 ratio), the $O(L^2)$ part in the Prefill stage is greatly diluted. In actual tests, Prefill throughput for 128K–256K contexts is still far higher than a pure Transformer of the same scale (officially 3x+ throughput vs Mixtral 8x7B).
   1. This design shifts **the Prefill compute bottleneck from the traditional Attention quadratic term to the linear-complexity Mamba part**, achieving a qualitative leap in ultra-long-context scenarios.
2. **Decode (generation)**: Mamba behaves as $O(1)$ state recurrence, with extremely high inference throughput.
   Overall, the hybrid architecture gives Jamba a significant end-to-end throughput lead over traditional Transformers on long contexts.

---

## 5. Challenges and Limitations

Despite Jamba's excellent performance, it still faces challenges in engineering deployment:

1. **Training stability**: Mamba and Attention have significantly different gradient scales; **layer-wise learning rates** and special initialization (like HiPPO) and RMSNorm strategies are usually needed to prevent divergence.
2. **Ecosystem maturity**:

   - Transformer has perfect support from vLLM, TensorRT-LLM, etc.
   - Mamba/Jamba relies on dedicated **Fused Kernels**; migration costs are high on non-NVIDIA hardware or older framework versions.
3. **Task differences**: in extremely high-precision In-context Learning or complex logical reasoning tasks, pure Transformer solutions currently retain a slight performance-ceiling advantage.
4. **Practical tips**: when training Jamba, it's recommended to use **Layer-wise Adaptive Learning Rates (LR)**: Mamba layer LR = 1.0 × base LR, Attention layer LR = 0.5 × base LR. Additionally, Mamba parameters often use a `Softplus` activation plus a small bias (like 0.1) to avoid $\delta = 0$ vanishing.
5. **Quantization and deployment optimization**: early versions relied on dedicated Fused Kernels with limited quantization options. But **Jamba-1.5 has introduced ExpertsInt8 quantization** (an INT8 weight quantization designed for MoE), quantizing only the MoE/MLP layers (85%+ of model weights), enabling efficient INT8 inference in vLLM with almost no quality loss.
   This method needs no calibration (takes minutes), supports BF16 activations, letting Jamba-1.5-Large (398B total / 94B active) easily handle 220K–256K contexts on 8×80GB GPUs, further lowering the deployment threshold.

---

## 6. Summary: Entering the Hybrid-Architecture Era

| Feature | Transformer | Pure Mamba | **Jamba (Hybrid)** |
| -------------------- | ----------------------------------- | --------------- | ----------------------------------------------- |
| **Computational complexity** | $O(L^2)$ | $O(L)$ | **Close to $O(L)$** |
| **Memory growth** | Fast linear growth (slope $\approx$ 1.0) | Constant | **Slow linear growth (slope $\approx$ 0.125)** |
| **Memory mode** | Non-compressive (precise) | Compressive (lossy) | **Precise and compressive combined** |
| **Long-text backtracking** | Extremely strong | Weak | **Strong** |

> **Conclusion**: Jamba's arrival doesn't announce the end of the Transformer; it marks LLM architectures officially entering a new era of **"Hybrid-by-Design."** It proves that deeply fusing **Attention's precise retrieval ability**, **SSM's efficient compression**, and **MoE's parameter scalability** is a viable path to breaking the long-text bottleneck. Future large-model architectures will likely dynamically and intelligently allocate these building blocks around task requirements, building stronger, more efficient complex intelligent systems.
