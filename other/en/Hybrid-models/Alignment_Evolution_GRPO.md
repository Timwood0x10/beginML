# The Evolution of Alignment Algorithms: From the Closed-Form Derivation of DPO to the Decentralization of GRPO

> **Abstract**
> The essence of Alignment is maximizing human-preference reward while preserving model diversity.
> This chapter uses rigorous mathematical derivation to prove how **DPO (Direct Preference Optimization)** eliminates the Reward Model through convex-optimization duality.
> Then we deconstruct **GRPO**, the core algorithm of **DeepSeek-R1**, showing how "group relative advantage" enables the emergence of System 2 reasoning without a Critic model.

## 1. The First Principles of Alignment: A KL-Regularized Objective

Whether PPO, DPO, or GRPO, they all originate from the same optimization objective. We need to find a policy \(\pi_\theta\) that maximizes reward \(r(x,y)\) while not deviating too far from the reference model \(\pi_{\text{ref}}\) (KL regularization prevents mode collapse).

### 1. The General Objective Function

$$
\max_{\pi} \mathcal{J}(\pi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

- \(r(x,y)\): the reward function (human preference or rules).
- \(\beta\): the KL coefficient (controlling alignment strength and diversity).
- \(\log \frac{\pi}{\pi_{\text{ref}}}\): per-Token KL divergence.

**Why do we need KL regularization?**

- Without the KL term, the model overfits the preference data, causing mode collapse (diversity disappears).
- The larger β, the closer the model stays to ref — conservative but safe; the smaller β, the more aggressive the model, prone to learning high-reward but low-probability behaviors.

### 2. The Closed Form of the Optimal Solution

This is a variational optimization problem. By the properties of the Gibbs distribution, the optimal policy \(\pi^*\) is:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

where the partition function:

$$
Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

**Physical meaning**: the optimal policy weights the reference model exponentially by reward. The higher the reward, the exponentially higher the probability.

## 2. DPO: The Complete Mathematical Derivation

PPO approximates \(\pi^*\) through policy gradients; DPO's genius lies in: **directly solving the reward function from the closed form, eliminating the Reward Model**.

### Step 1: Solving for the Reward Function

Take the log of the optimal solution:

$$
\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)
$$

Rearranging:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**Key**: Z(x) depends only on x, not on y.

### Step 2: The Bradley-Terry Preference Model

The probability that humans choose \(y_w\) (win) over \(y_l\) (loss):

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

### Step 3: Substituting and Eliminating Z(x)

Substitute r into the difference term:

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

Z(x) cancels out perfectly.

### Step 4: The Final DPO Loss

Replace \(\pi^*\) with \(\pi_\theta\) and maximize the likelihood of the preference data:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

**Gradient analysis**: let \(\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}\), then:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \sigma(\hat{r}_l - \hat{r}_w) \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]
$$

**Analysis**:

- Boost Winner: increase the probability of \(y_w\).
- Suppress Loser: decrease the probability of \(y_l\).
- Error Weight \(\sigma(\hat{r}_l - \hat{r}_w)\): adjusts dynamically. When confidence is high, \(\sigma \to 0\), gradients are small, preventing over-alignment; when confidence is low, σ is large, providing strong correction.

**Why is DPO stable?**

- σ automatically "brakes": stops learning when confident, avoiding mode collapse.

## 3. GRPO: The Reasoning Engine of DeepSeek-R1

GRPO (Group Relative Policy Optimization) was first proposed in DeepSeekMath (arXiv:2402.03300) and applied at scale in DeepSeek-R1 (arXiv:2501.12948). It's a **decentralized PPO** variant that needs no Critic model.

### 1. Algorithm Logic: Group Sampling

For each prompt \(x\), sample a group of responses from the current policy \(\pi_{\theta_{\text{old}}}\):

$$
\{o_1, o_2, \dots, o_G\}, \quad G \text{ usually } 64-128
$$

Compute each response's reward \(\{r_1, r_2, \dots, r_G\}\) (pure rules for R1-Zero, hybrid for R1).

### 2. Advantage Estimation (No Critic)

Traditional PPO advantage is A = r + γV(s') - V(s); GRPO replaces the baseline with within-group statistics:

$$
A_i = \frac{r_i - \mu}{\sigma + \epsilon}, \quad \mu = \frac{1}{G} \sum r_j, \quad \sigma = \sqrt{\frac{1}{G} \sum (r_j - \mu)^2}
$$

**Why can an in-group z-score replace the Critic?**

- The Critic V(s) estimates the expected reward E[r|s].
- Group sampling is a Monte Carlo estimate: \(\mu \approx E[r|x]\), with \(\sigma\) measuring variance.
- The z-score is an unbiased relative-advantage estimate (statistical centering/normalization), avoiding the extra training and memory cost of a Critic.

### 3. The GRPO Objective Function

Reusing the PPO Clip mechanism, the KL divergence is added directly to the objective as a penalty term (computed per-sample, then averaged):

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x,\{o_i\}} \left[ \frac{1}{G} \sum_i \min\left( r(\theta) A_i, \, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \cdot \frac{1}{G} \sum_i \mathbb{D}_{\text{KL}}(\pi_\theta(o_i|x) || \pi_{\text{ref}}(o_i|x)) \right]
$$

where \(r(\theta) = \frac{\pi_\theta(o_i|x)}{\pi_{\theta_{\text{old}}}(o_i|x)}\) and \(\mathbb{D}_{\text{KL}}\) is the per-sample KL.

**Why is GRPO suited to reasoning?**

- Memory liberation: only Actor + Ref are needed; the Critic is offloaded.
- Self-play: the model competes with "its past self"; as long as it generates a solution better than the group average, it improves.
- R1 practice: R1-Zero uses pure RL focused on reasoning; R1 uses multi-stage (cold-start SFT → reasoning RL → rejection sampling → general RL) to balance generality.
- **`<think>` tag Format Reward**: R1 introduces a `<think>` tag to enforce format rewards, significantly improving reasoning-chain completeness (R1 reports: without the `<think>` Reward, reasoning ability drops 15–20%).

## 4. Gradient Analysis: What Does the Model Actually Learn?

The DPO loss gradient:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \sigma(\hat{r}_l - \hat{r}_w) \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]
$$

where:

$$
\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}, \quad \hat{r}_l = \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

**Analysis**:

- Boost Winner: increase the probability of \(y_w\).
- Suppress Loser: decrease the probability of \(y_l\).
- Error Weight \(\sigma(\hat{r}_l - \hat{r}_w)\): adjusts dynamically.
  - When the model is confident (\(\hat{r}_w \gg \hat{r}_l\)), \(\sigma \to 0\), gradients are small, preventing over-alignment.
  - When the model has it backwards, σ is large, providing strong correction.

**Why is DPO stable?**

- σ automatically "brakes": stops learning when confident, avoiding mode collapse.

## Summary: From Complex Back to Simple

| Algorithm | **Core mathematical mechanism** | **Reward Model** | **Critic Model** | **Complexity** | **Applicable scenarios** |
| -------------- | ---------------------- | ---------------------- | ---------------------- | -------------------- | ------------------ |
| **PPO** | Policy gradient + Critic | ✅ Explicit | ✅ Explicit | Extremely high (4 Models) | Early general RLHF |
| **DPO** | Closed form + convex optimization | ❌ Implicitly eliminated | ❌ Not needed | Low (2 Models) | General dialogue |
| **GRPO** | In-group relative advantage | ✅ Explicit/rule-based | ❌ Replaced by the mean | Medium (2 Models + sampling) | **Reasoning tasks** |

> **One-sentence summary**:
> DPO uses mathematics to eliminate the Reward Model, solving the **stability** problem.
> GRPO uses statistics to eliminate the Critic Model, solving the **memory** problem.
> Every step mathematics simplifies, engineering productivity leaps forward.
