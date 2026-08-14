# [Appendix C] Safety Math: Differential Privacy and Adversarial Robustness

**Abstract**: As AI moves toward real-world deployment, safety has become a mathematical problem that can't be ignored. This chapter explores two core areas: **Differential Privacy (DP)** — how to protect data privacy with mathematically defined noise; and **Adversarial Robustness** — why invisible tiny perturbations in high-dimensional space can destroy deep neural network judgments, and the geometric principles behind it.

---

## 1. Differential Privacy

Privacy isn't a feeling; it's a strict mathematical definition. DP's core idea: **an attacker cannot tell, by observing the algorithm's output, whether a specific sample exists in the dataset.**

### 1.1 The $(\epsilon, \delta)$-DP Definition
Suppose there are two **adjacent datasets** $D$ and $D'$ differing by only one record (e.g., $D$ contains your data, $D'$ doesn't).
A randomized algorithm $\mathcal{M}$ satisfies $(\epsilon, \delta)$-differential privacy if and only if for every possible output set $S \subseteq \text{Range}(\mathcal{M})$:

$$ P[\mathcal{M}(D) \in S] \le e^\epsilon \cdot P[\mathcal{M}(D') \in S] + \delta $$

*   **$\epsilon$ (Privacy Budget)**: the privacy budget. Smaller $\epsilon$ means stronger privacy protection (the two probability distributions are closer), but worse data utility.
*   **$\delta$**: the failure probability. Usually required to satisfy $\delta \ll 1/|D|$ (less than the reciprocal of the sample size).

### 1.2 Sensitivity
To hide the influence of a single record, we need to know how much a single record can maximally change the output.
For a query function $f$, its $L_1$-sensitivity is defined as:
$$ \Delta f = \max_{D, D'} ||f(D) - f(D')||_1 $$
For example: if $f$ is "count the number of sick people," then removing one person changes the result by at most 1, so $\Delta f = 1$.

### 1.3 The Laplace Mechanism
For a real-valued function $f$, the Laplace mechanism achieves $\epsilon$-DP (with $\delta=0$) by adding noise following a Laplace distribution:

$$ \mathcal{M}(D) = f(D) + \eta, \quad \eta \sim \text{Lap}\left( \frac{\Delta f}{\epsilon} \right) $$

**Mathematical proof (sketch)**:
The Laplace distribution's probability density function is $p(x) = \frac{1}{2b} e^{-|x|/b}$, where $b = \Delta f / \epsilon$.
Consider the worst case $|f(D) - f(D')| = \Delta f$:
$$ \frac{P(x|D)}{P(x|D')} = \frac{e^{-|x - f(D)|/b}}{e^{-|x - f(D')|/b}} = e^{\frac{|x - f(D')| - |x - f(D)|}{b}} \le e^{\frac{|f(D) - f(D')|}{b}} = e^{\frac{\Delta f}{\Delta f / \epsilon}} = e^\epsilon $$
QED.

### 1.4 DP-SGD in Deep Learning
When training an LLM, we can't add noise directly to the Loss; we add it to the **gradients**.
**DP-SGD algorithm steps**:
1.  **Compute gradients**: $g_i = \nabla_\theta \mathcal{L}(x_i)$.
2.  **Gradient Clipping**: bound each sample gradient's norm, forcing an upper bound $C$ on sensitivity.
    $$ \bar{g}_i = g_i / \max(1, \frac{||g_i||_2}{C}) $$
3.  **Noise Injection**: use the Gaussian mechanism (because the Gaussian follows the central limit theorem, better suited to multi-round iterative composition).
    $$ \tilde{g} = \frac{1}{B} \sum \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I) $$

---

## 2. The High-Dimensional Geometry of Adversarial Attacks

Why does a photo of a panda, with a tiny amount of noise invisible to the human eye, make a machine think with 99% confidence it's a gibbon?

### 2.1 Formalizing the Attack
An adversarial attack looks for the point of maximum loss inside an $\epsilon$-ball:
$$ \max_{||\delta||_\infty \le \epsilon} \mathcal{L}(f_\theta(x + \delta), y) $$

### 2.2 The Linearity Hypothesis
Ian Goodfellow proposed an extremely intuitive explanation: **this fragility doesn't come from nonlinearity — it comes precisely from the highly linear behavior of neural networks.**

Consider a linear model (or one layer of a neural network): $w^T x$.
Add a perturbation $\eta$ with $||\eta||_\infty \le \epsilon$ (i.e., each pixel changes by at most $\epsilon$).
To maximize the change in the activation, the attacker should set $\eta = \epsilon \cdot \text{sign}(w)$.

The activation change is then:
$$ \Delta = w^T (x + \eta) - w^T x = w^T \eta = \epsilon \sum_{i=1}^n |w_i| $$

**The curse of dimensionality**:
Suppose the input dimension $n$ (e.g., image pixels) is very large. Even if $\epsilon$ is tiny (imperceptible), if $n$ is 1,000,000 and the mean weight is $m$, then the change $\Delta \approx \epsilon \cdot n \cdot m$.
**Tiny pixel-level perturbations, accumulated across high dimensions, are enough to cross the decision boundary.**

### 2.3 FGSM (Fast Gradient Sign Method)
Based on the linearity hypothesis above, the FGSM attack directly uses the gradient direction to generate adversarial examples:
$$ x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y)) $$
This is essentially taking a big step along the loss function's surface in the direction opposite the gradient (the ascending direction for an attacker).

### 2.4 The "Robustness-Accuracy Tradeoff" from a Geometric View
*   **Manifold distribution**: real data lives on a low-dimensional manifold.
*   **Decision boundary**: the boundary the model learns tries to separate manifolds of different classes.
*   **Adversarial examples**: tiny offsets in the Normal Direction of the manifold. The model fits well on the manifold (high accuracy), but it's never trained in regions off the manifold, so the decision boundary is extremely fragile in the normal direction.
*   **Conclusion**: to improve robustness (via adversarial training), we often need to smooth the decision boundary, which can reduce accuracy on clean data.

---

## 3. Code Practice: The Double-Edged Sword of Noise

The following code demonstrates two extreme applications of noise:
1.  **Laplace Noise**: intentionally adding noise to protect privacy.
2.  **FGSM Attack**: maliciously adding noise to fool the model.

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ================================
# Part 1: differential privacy (Laplace mechanism)
# ================================
def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    f(x) + Lap(sensitivity / epsilon)
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

# simulate a query: count of sick patients (True count = 100)
true_count = 100
sensitivity = 1 # removing one person changes it by at most 1
epsilons = [0.1, 0.5, 1.0, 5.0]

results = []
for eps in epsilons:
    # simulate 1000 queries
    noisy_counts = [laplace_mechanism(true_count, sensitivity, eps) for _ in range(1000)]
    results.append(noisy_counts)

# ================================
# Part 2: adversarial attack (FGSM math demo)
# ================================
def fgsm_linear_demo(dim=1000, epsilon=0.01):
    """
    demonstrate the linear accumulation effect in high-dimensional space
    """
    # simulate a linear weight w (random)
    w = torch.randn(dim)
    # simulate an input x
    x = torch.randn(dim)
    
    # original output
    original_output = torch.dot(w, x).item()
    
    # construct the adversarial perturbation: eta = epsilon * sign(w)
    # purpose: maximize w^T (x + eta)
    perturbation = epsilon * torch.sign(w)
    
    # adversarial output
    adv_x = x + perturbation
    adv_output = torch.dot(w, adv_x).item()
    
    # the change
    delta = adv_output - original_output
    theoretical_delta = epsilon * torch.sum(torch.abs(w)).item()
    
    return original_output, adv_output, delta, theoretical_delta

# ================================
# visualization
# ================================
plt.figure(figsize=(12, 5))

# Plot 1: DP noise distribution
plt.subplot(1, 2, 1)
for i, eps in enumerate(epsilons):
    # Fix: use rf'' (raw f-string) to prevent \epsilon from being misread as an escape character
    plt.hist(results[i], bins=30, alpha=0.5, density=True, label=rf'$\epsilon={eps}$')
plt.axvline(true_count, color='r', linestyle='--', label='True Value')
plt.title("Differential Privacy: Privacy vs. Utility")
plt.xlabel("Query Result")
plt.ylabel("Probability Density")
plt.legend()

# Plot 2: adversarial dimension effect
plt.subplot(1, 2, 2)
dims = [10, 100, 1000, 10000]
deltas = []
for d in dims:
    _, _, delta, _ = fgsm_linear_demo(dim=d, epsilon=0.01)
    deltas.append(delta)

plt.plot(dims, deltas, 'o-', color='purple')
plt.xscale('log')
plt.title("Adversarial Linear Hypothesis")
plt.xlabel("Input Dimension (Log Scale)")
plt.ylabel("Output Change (w/ epsilon=0.01)")
plt.grid(True)
plt.text(100, deltas[1], "Small noise accumulates\nin high dimensions!", fontsize=10)

plt.tight_layout()
plt.show()
```

**Result analysis**

![safety math](../images/safety_math.png)

* Left plot (DP):
    * When $\varepsilon=0.1$ (blue), the distribution is extremely wide (large variance); attackers can hardly guess the true value is 100, but data utility is poor.
    * When $\varepsilon=5.0$ (red), the distribution sharply concentrates near 100; the data is accurate, but the privacy-leak risk is high.
    This is the intuitive manifestation of the Privacy-Utility Tradeoff.
* Right plot (Adversarial):
    * The x-axis is the input dimension $n$ (log scale). The y-axis is the output change.
    * Note $\varepsilon=0.01$ (tiny perturbation).
    * As the dimension $n$ grows from 10 to 10000, the output change explodes linearly.
    * This is the mathematical essence of why ImageNet ($224x224x3 \approx 150k$ dimensions) models are so fragile — dimension itself is the attacker's weapon.

This appendix reveals the defensive nature of mathematics. Whether masking individuals with Laplace noise or patching high-dimensional boundaries with adversarial training, safety math is the cornerstone of the modern AI trust system.
