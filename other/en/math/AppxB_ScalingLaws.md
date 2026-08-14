# [Appendix B] Scaling Laws: The Mathematics of Foresight

**Abstract**: Deep learning was once considered an "alchemy," until the arrival of **Scaling Laws**. They reveal a stunning physical fact: a model's performance (Loss) follows a strict **power-law relationship** with compute, parameters, and data. This chapter derives the famous **Chinchilla Scaling Laws**, explains the "compute-optimal" frontier, and discusses why, in the 2024+ Llama era, we've begun pursuing "inference-optimal" rather than "training-optimal."

---

## 1. The Power Law

### 1.1 The Empirical Formula
OpenAI (Kaplan et al., 2020) and DeepMind (Hoffmann et al., 2022) found that the cross-entropy loss $L$ on the test set follows a power-law relationship with a variable $X$:

$$ L(X) \approx C + \frac{A}{X^\alpha} $$

Or on a Log-Log Scale:

$$ \log(L(X) - C) \approx \log A - \alpha \log X $$

This appears as a **straight line**. This means: **as long as you keep increasing compute/data/parameters, Loss predictably decreases, with no end (until hitting the irreducible error $C$).**

### 1.2 Why a Power Law?
From the manifold hypothesis (Chapter 26):
*   The high-dimensional data manifold is wrapped in complex curved surfaces.
*   Increasing the parameter count $N$ or data amount $D$ is like approximating the manifold with a finer grid.
*   The approximation error usually scales inversely with a power of the grid density (like the remainder term of a Taylor expansion).

---

## 2. Compute Budget and Optimization Constraints

Suppose we have a fixed compute budget $C$ (Compute, in FLOPs); we need to decide how much to allocate to model size $N$ (Parameters) and how much to training data $D$ (Tokens).

### 2.1 The FLOPs Approximation Formula
The compute for training a Transformer model is approximately:
$$ C \approx 6 N D $$
**Derivation**:
*   **Forward pass**: each parameter involves one multiply and one add (2 FLOPs). $C_{fwd} \approx 2N$ per token.
*   **Backward pass**: computing gradients and updating weights costs about 2x the forward. $C_{bwd} \approx 4N$ per token.
*   **Total**: $6N$ FLOPs per token.

### 2.2 The Parameterized Loss Function
DeepMind proposed in the Chinchilla paper that the joint loss can be modeled as:
$$ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} $$
where:
*   $E$: the irreducible loss (like the entropy of natural language itself).
*   $\frac{A}{N^\alpha}$: the bias caused by the model being too small.
*   $\frac{B}{D^\beta}$: the variance caused by too little data.
*   $\alpha, \beta$: fitted constants, usually between $0.3 \sim 0.5$.

---

## 3. The Chinchilla Optimal Frontier: A Lagrange Derivation

Our goal is to minimize $L(N, D)$ under the constraint $C = 6ND$.

### 3.1 Constructing the Lagrange Function
$$ \mathcal{L}(N, D, \lambda) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} - \lambda (6ND - C) $$
*(ignoring the constant $E$, which doesn't affect the extremum)*

### 3.2 Solving the Partial Derivatives
Take partial derivatives w.r.t. $N$ and $D$ and set them to 0:

1.  $\frac{\partial \mathcal{L}}{\partial N} = -\alpha A N^{-\alpha - 1} - 6\lambda D = 0 \implies \lambda = -\frac{\alpha A}{6 D N^{\alpha + 1}}$
2.  $\frac{\partial \mathcal{L}}{\partial D} = -\beta B D^{-\beta - 1} - 6\lambda N = 0 \implies \lambda = -\frac{\beta B}{6 N D^{\beta + 1}}$

### 3.3 Solving Simultaneously
Set the two $\lambda$'s equal:
$$ \frac{\alpha A}{6 D N^{\alpha + 1}} = \frac{\beta B}{6 N D^{\beta + 1}} $$
Simplifying:
$$ \frac{\alpha A}{N^\alpha} = \frac{\beta B}{D^\beta} $$
This reveals a deep **equilibrium condition**: at the optimum, the error contribution from model size should be proportional to the error contribution from data size.

Rearranging gives the relationship between $N$ and $D$:
$$ N_{opt}^\alpha \propto D_{opt}^\beta \implies N_{opt} \propto D_{opt}^{\frac{\beta}{\alpha}} $$

### 3.4 How Does It Grow with Compute $C$?
Using $C = 6ND$ and the above relationship, we can solve for $N_{opt}$ and $D_{opt}$ as functions of $C$:

$$ N_{opt} \propto C^{\frac{\beta}{\alpha + \beta}}, \quad D_{opt} \propto C^{\frac{\alpha}{\alpha + \beta}} $$

**Chinchilla's finding (Hoffmann et al., 2022)**:
Through extensive experiments, they fitted $\alpha \approx 0.50$, $\beta \approx 0.50$.
This means:
$$ \frac{\beta}{\alpha + \beta} \approx 0.5, \quad \frac{\alpha}{\alpha + \beta} \approx 0.5 $$

**Conclusion**:
$$ N_{opt} \propto \sqrt{C}, \quad D_{opt} \propto \sqrt{C} $$
That is, when the compute budget grows 10x, you should **simultaneously** scale the model parameters by 3.16x and the data by 3.16x.
Kaplan's law ($\alpha \approx 0.74, \beta \approx 0.28$) holds in the data-limited era; Chinchilla's law ($\alpha \approx \beta \approx 0.5$) holds in the data-abundant era.

---

## 4. Engineering Practice: The Golden Ratio and the Llama Era

### 4.1 The Chinchilla Golden Rule (training-optimal)
Based on the derivation above, the compute-optimal configuration usually follows:
$$ D_{opt} \approx 20 \times N_{opt} $$
i.e., **every 1 parameter corresponds to 20 training tokens.**

*   **10B model**: needs 200B tokens.
*   **70B model**: needs 1.4T tokens.

Llama 3 70B actually used 15T tokens, about 214×N — an extreme inference-optimal strategy.

### 4.2 "Inference-Optimality" in the Llama Era
In 2024-2025, we found that models like Llama 3 don't follow the Chinchilla law.
*   **Llama 3 8B** was trained on **15T tokens**.
*   Per Chinchilla, an 8B model only needs 160B tokens. Did Llama 3 "overfit" by nearly 100x?

**Explanation**: Chinchilla optimizes **training cost** (Training Compute Optimal). But in real applications, **inference cost** matters more.
*   Inference cost only depends on the parameter count $N$, not the training data amount $D$.
*   To get an extremely small but extremely smart model (easy to deploy on phones), we're willing to spend far more compute than Chinchilla suggests to train a small model.
*   **New trend**: under a fixed inference budget (fixed $N$), more data $D$ is always better (until marginal returns hit 0).

---

## 5. Analysis of the Experimental Results

![Chinchilla Scaling Laws](../images/chinchilla_scaling.png)


The figure above shows simulated Loss curves under different compute budgets ($10^{18}$ to $10^{21}$ FLOPs).

### Reading the figure
1.  **The U-shaped curve (The Basin of Optimization)**:
    *   For each colored curve (representing a fixed compute budget), there's a clear minimum point.
    *   **Left rising region**: the model parameter count $N$ is too small, forcing you to make up compute with huge data $D$. But small models are under-parameterized; no matter how much data, they can't learn it well, so Loss is high.
    *   **Right rising region**: the model parameter count $N$ is too large, so the budget $C$ is eaten up by the model, leaving very little data $D$. The large model is under-trained, generalizes poorly, and Loss is also high.

2.  **The red star (The Optimal Frontier)**:
    *   The red star in the figure marks the **optimal model size** for each budget.
    *   The trajectory connecting these red stars is the **Compute-Optimal Frontier**.
    *   You can see that as the budget grows from $10^{18}$ to $10^{21}$ (1000x), the optimal parameter count grows from 0.1B to 1.8B (about 18x), roughly consistent with the $\sqrt{1000} \approx 31.6$ order of magnitude (accounting for small deviations in the measured $\alpha, \beta$).

3.  **Engineering insights**:
    *   If you have 1000 H100 cards, you should check this table first to determine whether your optimal parameter count is 70B or 130B, instead of guessing.

---

## 6. Code Implementation: Finding the Bottom of the Loss Valley

```python
import numpy as np
import matplotlib.pyplot as plt

def estimated_loss(N, D):
    """
    the parameterized Loss function from the Chinchilla paper
    coefficients from the paper's appendix (Table A4)
    """
    E = 1.69  # irreducible error (Entropy of natural text)
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28
    
    term_N = A / (N ** alpha)
    term_D = B / (D ** beta)
    
    return E + term_N + term_D

# set the compute budget C (FLOPs)
# e.g., training a 1B model + 20B Tokens -> C = 6 * 1e9 * 2e10 = 1.2e20
compute_budgets = [1e18, 1e19, 1e20, 1e21] 

plt.figure(figsize=(12, 7))

for C in compute_budgets:
    # under a fixed budget, scan different N (Parameters)
    # N range: from 10M to 100B
    Ns = np.logspace(7, 11, 200)
    
    # the corresponding D is determined by the constraint C = 6ND
    Ds = C / (6 * Ns)
    
    losses = estimated_loss(Ns, Ds)
    
    # find the optimal N under this budget
    min_idx = np.argmin(losses)
    opt_N = Ns[min_idx]
    opt_loss = losses[min_idx]
    
    # plot
    plt.plot(Ns, losses, linewidth=2, label=f'Budget {C:.0e} FLOPs')
    plt.scatter(opt_N, opt_loss, c='red', s=100, marker='*', zorder=10)
    
    # annotate the optimal parameter count
    plt.text(opt_N, opt_loss + 0.05, f"{opt_N/1e9:.1f}B Params", 
             ha='center', fontsize=12, color='black')

plt.xscale('log')
# plt.yscale('log') # Loss doesn't vary much; a linear axis is more intuitive
plt.xlabel('Model Parameters (N)', fontsize=12)
plt.ylabel('Loss (L)', fontsize=12)
plt.title('Chinchilla Scaling Laws: Finding the Optimal Model Size', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=12)
plt.ylim(2, 9.5) # focus on the effective region
plt.savefig("chinchilla_scaling.png", dpi=300, bbox_inches='tight')
plt.show()

···
