# Engram — Conditional Memory: The "Second Axis" of Large-Model Sparsification

> **Abstract**:
> Existing sparse models (MoE) mainly focus on **Conditional Computation**. DeepSeek's **Engram** introduces **Conditional Memory**.
> Its core logic: free the Transformer from the heavy burden of "simulating retrieval through computation," achieving native reading of static knowledge through $O(1)$ hash table lookups.
> This chapter analyzes, from the three dimensions of mathematical modeling, architecture design, and systems engineering, how Engram breaks through the parameter-scale and long-text bottlenecks without increasing compute (FLOPs).

## 1. Design Philosophy: Separating Knowledge Retrieval from Compositional Reasoning

The DeepSeek team believes language modeling involves two essentially different tasks:

1. **Compositional Reasoning**: requires deep dynamic computation (the Transformer's strength).
2. **Knowledge Retrieval**: handles static, stereotyped patterns (like entity names, fixed collocations).

**The Transformer's limitation**: lacking a native lookup-table primitive, it forces the model to "reconstruct" static knowledge in early layers using precious Attention and FFN, wasting effective depth.

**Engram's logic**: build a parallel "memory axis" that directly fetches Embeddings via $N$-gram indexing, letting the backbone focus on higher-order reasoning.

---

## 2. Engram's Core Architecture Design

Engram's execution logic consists of three key modules: **tokenizer compression**, **multi-head hash retrieval**, and **context-aware fusion**.

### 1. Tokenizer Compression

To handle the combinatorial explosion of $N$-grams, Engram performs a vocabulary mapping $P: V \to V'$:

* **Operation**: NFKC normalization, lowercasing, merging equivalent Tokens (like `Apple` and `apple`).
* **Effect**: achieves a **23%** effective size reduction on a 128k vocabulary, significantly improving the semantic density of $N$-gram storage.

### 2. Multi-Head Hashing

To achieve $O(1)$-level addressing of massive parameters (100B+):

$$
z_{t,n,k} \triangleq \phi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]
$$

* **Mechanism**: for each $N$-gram order (n=2, 3...), use $K$ independent hash heads.
* **Logic**: use the Multi-head mechanism to mitigate hash collisions. The final retrieved vector $\mathbf{e}_t$ is the concatenation of all hash heads' results.

### 3. Context-aware Gating — The Core Point

Static lookup can't handle ambiguity (polysemy). Engram introduces a dynamic gating mechanism:

* **Query**: the current layer's hidden state $\mathbf{h}_t$ (aggregating global context).
* **Key/Value**: the static embedding $\mathbf{e}_t$ retrieved from the Engram table.
* **Gating coefficient $\alpha_t$ computation**:
  $$
  \alpha_t = \sigma \left( \frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}} \right)
  $$
* **Semantic alignment**: if the retrieved knowledge conflicts with the context, $\alpha_t \to 0$, automatically filtering the noise.

---

## 3. Scaling Laws and Sparsity Allocation

The paper raises a key engineering question: with a fixed total parameter budget, how should parameters be allocated between MoE experts and Engram storage?

### 1. The U-shaped Scaling Law

Experiments (Figure 3) prove:

* **Pure MoE ($\rho = 100\%$)**: the model lacks dedicated storage, forced to simulate memory with compute — suboptimal efficiency.
* **Pure Engram ($\rho = 0\%$)**: the model loses dynamic reasoning ability.
* **Best practice**: allocate about **20%~25%** of sparse parameters to Engram.

### 2. Increasing "Effective Depth"

Via LogitLens analysis, Engram performs best when intervening in early layers (like Layer 2). It shares the burden of low-level static modeling, so Layer 5's prediction accuracy already matches the baseline model's Layer 12.

---

## 4. Systems Engineering: Breaking Through GPU HBM's Physical Limits

Engram's system advantage lies in its **Deterministic** access pattern.

### 1. Runtime Prefetching

* **MoE's pain point**: routing depends on dynamic hidden states; you must finish computing the previous layer before knowing where to read experts, making communication latency hard to hide.
* **Engram's advantage**: the index depends only on the Token ID sequence.
* **Implementation**: while computing Layer 0, the system can already know the hash IDs of all subsequent Engram layers, prefetching embeddings stored in **Host Memory (CPU RAM)** asynchronously via PCIe.

### 2. Infrastructure-aware Efficiency

* **Offload storage**: supports offloading 100B-level parameter tables to CPU memory or even NVMe SSDs.
* **Performance cost**: mounting 100B parameters on an 8B model loses **< 3%** inference throughput — almost perfectly hiding cross-device communication latency.

---

## 5. Core Benchmark Comparison (Engram-27B vs MoE-27B)

| Task dimension | Improvement | Key data observations |
| :------------------------------ | :-------------- | :------------------------------------------------------ |
| **Knowledge retrieval** | **+3.4** | MMLU significantly improves, showing the intuitive advantage of conditional memory. |
| **General reasoning** | **+5.0** | BBH improves even more than knowledge tasks, proving the compute layers are freed up. |
| **Long context** | **+12.8** | NIAH rises from 84.2 to 97.0, freeing Attention's global bandwidth. |

## 6. The Spatial Game: Where to Insert the Engram Module

The paper reveals Engram's "layer sensitivity" in the Transformer's vertical structure through large-scale ablation experiments (Figure 5).

### 1. Core Conclusion: Early Intervention Beats Deep Injection

* **Experimental data**: inserting a single Engram layer at different positions, **Layer 2** shows the best Validation Loss (1.770), with effectiveness linearly degrading as depth increases.
* **Physical logic**:
  * **Layer 0-1 (too early)**: hidden states haven't been aggregated by Attention yet, lacking enough global context to generate a precise "gating signal ($\alpha_t$)," causing high retrieval noise.
  * **Layer 2 (the golden position)**: by now one round of global modeling has happened, enough to generate high-quality Queries, and instant retrieval frees the backbone from "low-level pattern reconstruction."
  * **Deep layers (too late)**: the backbone has already painstakingly completed pattern recognition by wasting compute (Attention/FFN); giving Engram at this point is "icing on the cake" rather than "fuel in the snow" — extremely low ROI.

### 2. Advanced Strategy: Layered Insertion

* **Best practice**: the paper suggests splitting the 1.6B memory budget and injecting into **Layer 2** and **Layer 6** separately.
* **Gain**: balances "early knowledge offloading" with "late fine-grained gating modulation," more robust than a single large-capacity module.

---

## 7. Memory Evolution: How Does Engram Update and Expand?

This is the most easily misunderstood point: is Engram a "frozen weight" or a "pluggable plugin"?

### 1. The "Infinite Memory Regime" That Needs No Retraining

Engram achieves complete decoupling of **Parameters** and **Compute (FLOPs)**:

* **Horizontal scaling**: you can brute-force expand the Engram table by increasing the number of hash buckets ($M$) (e.g., from 2.7B to 100B+).
* **Experimental verification**: the paper proves that with a fixed MoE backbone (3B) unchanged, simply increasing Engram's Slot count makes the validation-set Loss follow a strict **Power Law** decline.
* **Engineering significance**: this means you can keep squeezing model performance by adding storage rather than compute.

### 2. Three Paths for Dynamic Updates

When the knowledge base needs updating (e.g., new $N$-gram facts appear), Engram offers:

* **Path A: Incremental Embedding Learning (no backbone retraining)**
  * Since retrieval logic is deterministic (based on $N$-gram hashing), once the backbone's $W_{up}/W_{down}$ matrices are trained, their semantic-alignment space is fixed.
  * **Operation**: just generate $N$-grams for new documents and update/initialize only the corresponding Embedding Slots. This is essentially ultra-low-cost **Parameter-Efficient Fine-Tuning (PEFT)**.
* **Path B: Hot-Pluggable Offloading**
  * Engram supports storing the huge parameter table (100B+) in CPU RAM or even NVMe SSDs.
  * **Operation**: through the asynchronous prefetch mechanism, new "memory shards" can be dynamically loaded without interrupting inference.
* **Path C: Hash-Redundancy Exploitation**
  * Leverage the redundancy of "multi-head hashing," fine-tuning the Router (gating layer) lightly so the model learns to attend to newly written Slots.

---

## 8. Clarifying Key Technical Misconceptions (FAQ)

### Q: Since it's hash retrieval, what about hash collisions?

**A (pragmatic reading)**: Engram uses **Multi-head Hashing**. Even if Head 1 collides, Heads 2 and 3 have a very high probability of producing different IDs. The final vector is a concatenation of multiple heads, and the model, through learning, can automatically filter out dimensions with collision noise in the fusion layer.

### Q: Will the Embeddings retrieved by Engram crash the model's output?

**A (rigorous reading)**: No. Because of **Context-aware Gating** (formula 4). If the retrieved Engram vector has extremely low cosine similarity with the current context $\mathbf{h}_t$, the gate $\alpha_t$ approaches 0, physically blocking this signal from entering the residual stream.

---

## Summary: The Evolution Direction of Sparse Architectures

| Dimension | Traditional MoE (compute sparsity) | **Engram (memory sparsity)** |
| :----------------- | :----------------------- | :--------------------------------------------- |
| **Position decision** | Spread across every layer | **Mainly concentrated in early layers (Layer 2/6)** |
| **Update cost** | Extremely high (requires retraining routing and experts) | **Extremely low (can update only Embedding Slots)** |
| **Hardware bottleneck** | Limited by GPU memory (HBM) | **Breaks through memory; can use CPU memory/SSD** |
| **Applicable scenarios** | Handling logic, reasoning, dynamic transformations | **Storing facts, entities, long-tail knowledge, long-text anchors** |

1. **Architecture evolution**: from pure "compute sparsity" to "compute + memory" dual-axis sparsity.
2. **Hierarchical storage**: exploiting the Zipf distribution, frequent $N$-grams reside in GPU HBM while long-tail $N$-grams reside in Host DRAM.
3. **Compute decoupling**: Engram proves that through "external memory," we can keep expanding the model's knowledge boundary via Scaling Laws without increasing training/inference FLOPs.

**One-sentence summary**: if MoE gives the large model a "multi-core brain," then Engram gives it an "encyclopedia" that can be instantly retrieved without consuming brainpower.

---
