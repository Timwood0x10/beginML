# Pretraining Dynamics: The Mathematical Essence of Scaling Laws and Data Engineering

> **Abstract**:
> By 2026, pretraining has shifted from Kaplan's "parameters decide everything" to Chinchilla's "compute-optimal" ($N_{opt} \propto D^{0.5}$), and then to Llama-3/DeepSeek-V3's "inference-optimal + data quality first."
> This chapter deconstructs the core math that determines a model's intelligence ceiling: from the complete derivation of the **Chinchilla Scaling Laws**, to the pseudocode implementation of the **MinHash LSH** deduplication algorithm, and finally a deep dive into how **Three-Stage LR Annealing** lets the model converge to a better optimum.

## 1. Scaling Laws: The Mathematical Art of Resource Allocation

With a fixed total compute budget C = 6ND (single forward pass FLOPs ≈ 6N D), how do we allocate parameters N and data D to minimize loss L?

### 1. Kaplan's Law → The Chinchilla Optimal Derivation

**Kaplan (2020, arXiv:2001.08361)**: assume $L(N) = E + A / N^α$ (infinite data), $\alpha \approx 0.76$. The optimum is "bigger is better," but it ignores data finiteness, causing GPT-3 to be over-parameterized (N >> D) with 15% higher training Loss (ablation measurement).

**Why derive a power law**: from Transformer activation statistics assumptions (the power-law distribution arises from high-dimensional space compression), empirically fit $L \propto \frac{1}{N^{\alpha}}$.

**Chinchilla (2022, arXiv:2203.15556)**: introduces the data constraint; the loss function:
$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_0
$$

Measured α ≈ 0.34, β ≈ 0.28 (fitted from 200+ model training curves).

**Why optimal? Complete derivative proof**:

- Constraint: C = 6 N D = constant → D = C / (6 N)
- Objective: $min L(N) = E + A/N^{\alpha} + B / (C/(6N))^{\beta} + L_0$
- Derivative: $\partial L/ \partial N = -α A / N^{\alpha+1} + \beta B (6N / C)^{\beta+1} / N = 0$
- Solve: $N_opt \approx (\alpha A / (\beta B))^{1/(\alpha+\beta)} · (C/6)^{β/(α+β)}$
- Simplified: $N_{opt} \propto C^{0.5}, D_{opt} \propto C^{0.5}$, with **ratio N:D ≈ 1:20** (the measured golden ratio).

**Why do this**: Kaplan ignores the data bottleneck, wasting compute on ineffective parameters; Chinchilla proves "balanced allocation" lowers loss 20-30% (ablation plot: N:D=1:1 has 15% higher loss). Because the power-law exponent $\alpha < \beta$, data has higher marginal returns.

**DeepSeek-V3 measurement**: a 67B model with 1.4T Tokens (20:1) reaches 85.2% on MMLU (reported data).

### 2. Llama-3's "Inference-Optimal" Rebellion

Llama-3 8B uses **15T Tokens** (ratio ~1900:1, arXiv:2407.21783), far beyond Chinchilla.

**Why do this? Mathematical argument**:

- Chinchilla optimizes **training Loss**, but total industry cost = C_train + k C_infer ($k\approx 10$; inference is 10x training).
- Oversaturated training ($D \gg N_{opt}$): early loss drops slowly, but later activation distributions are smoother, giving +5-10% generalization at inference (Llama-3 ablation: 8B +15T > 70B +1.4T).
- Corrected formula: $min (L + k·Infer_{Cost})$, with $Infer_{Cost} \propto N$; at the oversaturation minimum, $D_{opt} \gg N_{opt}$.

**Why optimal**: small models are expensive to train but cheap to infer; "overdata" buys a "strong small model" (inference FLOPs saved 10x).

## 2. Data Engineering: Cleaning Is Training

"Garbage In, Garbage Out." The core moat of pretraining isn't model architecture (everyone uses Transformer); it's industrial-scale data processing.

### 1. Fuzzy Deduplication: MinHash + LSH

**Why needed**: exact matching misses "rewritten/paraphrased" content; fuzzy deduplication is the bottleneck (PB-scale data makes O(N^2) infeasible).

**Jaccard similarity**:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**MinHash unbiased-estimator derivation**:

- Randomly permute document shingles (k-grams) and take the min hash value.
- Proof: P(min_hash(A) = min_hash(B)) = J(A,B) (under a random permutation, the probability that the minimum elements are equal equals the intersection-over-union ratio).

**LSH reduces complexity**:

- Generate k hashes split into b bands (r=k/b); only if a whole band matches do we collide; P(collision) ≈ J^r.
- Why optimal? P(collision) decays exponentially with r: low-J documents collide <0.01% of the time, high-J (>0.8) >0.99%; complexity is O(N + M), with $M \ll N^2$.

**Engineering pseudocode (Python + datatrove style)**:

```python
def minhash_lsh(docs, k=128, b=16, r=8, threshold=0.8):
    signatures = [min_hash(doc, k) for doc in docs]  # [N, k] signature matrix
    buckets = [[] for _ in range(b)]  # b buckets
    for i, sig in enumerate(signatures):
        for band in range(b):
            band_hash = hash(sig[band*r:(band+1)*r])  # all r hashes equal
            buckets[band_hash % len(buckets[0])].append(i)
    clusters = connected_components(buckets)  # Union-Find clustering
    return dedup_clusters(clusters, threshold)  # dedup when J > threshold
```

**Why optimal**: after DeepSeek-V3 / Llama-3 deduplicate with this method, effective data quality improves 20-40%, and final performance improves 2-5% (reported data).

### 2. Data Mixing: Why Does Code/Math Improve Reasoning?

- **Code**: provides structured logic chains (if-else, loops), strengthening CoT (+4.1% GSM8K).
- **Synthetic data**: GPT-4o generates "textbook-grade" samples, compensating for natural-corpus noise (Llama-3 +10% data quality).

## 3. Two-Stage Pretraining: Context Scaling and Annealing

### 1. Long-Context Extension

First 90% with a short window (4k) + last 10% with a long window (128k), saving 80% of compute.
**RoPE scaling derivation**:

$\theta'_i = \theta_i \cdot \lambda, \quad \lambda = \frac{\text{max\_pos}_{new}}{\text{max\_pos}_{old}}$

**Why**: YaRN dynamic interpolation keeps relative positions unchanged, avoiding the O(L^2) explosion of long-window training.

### 2. Learning Rate Annealing

DeepSeek-V3's two-stage annealing: the formula is the "two-stage + Cosine Decay + Linear Tail" combination strategy most common in modern large-model pretraining, adopted (or in variants) by almost all 2024-2026 mainstream models including Llama-3, DeepSeek-V3, Qwen2, and Mistral.

### The Original Formula (written piecewise)

$$
lr(t) = 
\begin{cases} 
lr_{\max} \cdot \dfrac{t}{t_{\text{warmup}}} & \text{Stage 1: } t \le t_{\text{warmup}} \\[1em]
lr_{\min} + 0.5(lr_{\max} - lr_{\min})\left(1 + \cos\left(\pi \dfrac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right) & \text{Stage 2: } t_{\text{warmup}} < t < T \\[1em]
lr_{\min} \cdot \dfrac{T - t}{T - t_{\text{cossin}}} & \text{Stage 3: } t \ge T
\end{cases}
$$

where:

- $t$: the current training step (global step)
- $T$: total training steps
- $t_{\text{warmup}}$: the step where warmup ends (usually 1%~5% of total steps, like 1000–5000 steps)
- $t_{\text{cossin}}$: the step where Cosine Decay ends (usually 90%~95% of T)
- $lr_{\max}$: the peak learning rate (e.g., 4e-4 ~ 1e-3)
- $lr_{\min}$: the minimum learning rate (usually 1/10 ~ 1/100 of lr_max)

### Piecewise Detailed Breakdown + Plain Explanation

#### Stage 1: Warmup (linear rise, t ≤ t_warmup)

**Formula**:

$$
lr(t) = lr_{\max} \cdot \frac{t}{t_{\text{warmup}}}
$$

**Plain explanation**:

- Starts at lr=0 at step 0 and rises linearly to lr_max at step t_warmup.
- Like starting a car: gently press the accelerator first, avoiding revving a cold engine (gradient explosion).

**Why do this** (core reason):

- At the start of training, weights are randomly initialized and gradient directions are extremely unstable (huge variance).
- If we used a high learning rate immediately, parameters would "jump around," easily exploding gradients or landing in bad local optima.
- Linear warmup lets the model "probe" the gradient direction with small steps first, then accelerate (lr_max) once gradients stabilize.

**Life analogy**:
When you start running in the morning, warm up with a slow 5-minute walk (warmup), then sprint (lr_max) once your body is warm.

**Typical parameters**: t_warmup = 1%~5% of total steps (e.g., 1M total steps → warmup 10k~50k steps).

#### Stage 2: Cosine Decay (t_warmup < t < T)

**Formula**:

$$
lr(t) = lr_{\min} + 0.5(lr_{\max} - lr_{\min})\left(1 + \cos\left(\pi \dfrac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right)
$$

**Plain explanation**:

- Smoothly decays from lr_max to lr_min (usually lr_min = lr_max / 10 ~ 1/100).
- The decay curve is a cosine: slow at the start (keeping high lr for exploration), accelerating in the middle, and slowing again at the end (fine convergence).
- The (t - t_warmup)/(T - t_warmup) inside the parentheses is normalized progress (from 0 to 1); multiplied by π, cos goes from 1 → -1.

**Why Cosine rather than linear/exponential decay?** (mathematical reasons):

- The cosine curve has a **small slope** at the start and end (derivative near 0) and a large slope in the middle.
- Small start slope → keeps high lr, large exploration space, unlikely to fall into local optima early.
- Small end slope → in the last few thousand steps lr changes extremely slowly, letting the model converge finely to a sharp optimum (better generalization).
- Linear decay changes too fast at the end, easily "skipping" the optimum; exponential decay drops to tiny lr too early, under-exploring.

**Life analogy**:
Like losing weight:

- Start (high lr): slowly burn fat (explore a large space)
- Middle: accelerate weight loss (fast descent)
- Last few weeks: fine-tune the diet (lr changes extremely slowly), avoiding rebound and locking in a good figure.

**Typical parameters**: lr_min = lr_max / 10 ~ 1/100; T - t_warmup takes up 90%~95% of total steps.

#### Stage 3: Linear Decay (linear tail, t ≥ T)

**Formula**:

$$
lr(t) = lr_{\min} \cdot \frac{T - t}{T - t_{\text{cossin}}}
$$

**Plain explanation**:

- After the Cosine stage ends (t ≥ T), lr decays linearly from lr_min to 0 (or near 0).
- The decay slope is constant until training fully ends.

**Why need this tail?** (engineering reasons):

- At the end of Cosine Decay, lr approaches lr_min but never equals 0; the model may keep "fine-tuning" without truly converging.
- The linear tail forces lr → 0, letting parameters "set" in the last few thousand steps, avoiding late-stage oscillation.
- Many experiments show: models with a linear tail have lower final Loss and better generalization (e.g., DeepSeek-V3 ablation: +0.8% MMLU after adding the tail).

**Life analogy**:
In the last week of dieting, cut carbs completely (lr → 0), locking in the weight with no more fluctuation.

### Full Curve Diagram (text description)

The whole lr(t) curve looks like this:

- 0 → t_warmup: straight line up (warmup)
- t_warmup → T: cosine-shaped descent (main decay)
- T → end: straight line down to 0 (setting)

![Learning Rate Annealing Curve](./image/lr_annealing_curve.png)

![Learning Rate Annealing](./image/LearningRateAnnealing.png)

**Why is the three-stage form optimal?** (engineering consensus):

- Warmup prevents explosion
- Cosine balances exploration and convergence
- Linear Tail locks in the optimum


## 4. Training Stability: μ-Transfer

**Why needed**: hyperparameters tuned on 1B can't be directly applied to 100B (gradient explosion/vanishing).

**μP's complete 4 Scaling Rules** (Muennighoff et al., arXiv:2205.11916):

1. **Output Variance**: $Var(Wx) \propto 1/width$ (initialization scaling).
2. **Input Variance**: $Var(x)\propto width$ before LayerNorm.
3. **Attention Scale**: $QKV scale \propto 1/ \sqrt{width}$.
4. **Learning Rate**: $lr \propto 1/ \sqrt{width}$.

**Why can it zero-shot transfer? Proof**: μP makes gradient variance independent of width (derived from $Var(ΔW) = Var(\partial L/ \partial W) \propto 1/width$), so hyperparameters tuned on a small model transfer directly (1B → 100B Loss curves overlap, saving 90% of trial-and-error compute).

**Python pseudocode**:

```python
def mu_init(weight, fan_in, fan_out, mu=True):
    if mu:
        std = 1 / sqrt(fan_in) * sqrt(fan_out / width)  # Rule 1 & 2
    else:
        std = sqrt(2 / fan_in)  # He Init
    return nn.init.normal_(weight, mean=0, std=std)
```

## Summary

> **Conclusion**: Pretraining determines the genes (capacity ceiling). SFT/RL only change the personality; they can't break through this ceiling.
