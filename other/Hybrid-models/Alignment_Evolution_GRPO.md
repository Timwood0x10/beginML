# 对齐算法的演进：从 DPO 的解析解推导到 GRPO 的去中心化

> **摘要**
> 对齐 (Alignment) 的本质是在保持模型多样性的前提下，最大化人类偏好奖励。
> 本章通过严谨的数学推导，证明 **DPO (Direct Preference Optimization)** 如何利用凸优化对偶性消除 Reward Model。
> 随后，我们解构 **DeepSeek-R1** 的核心算法 **GRPO**，展示如何通过“群组相对优势”在无需 Critic 模型的情况下实现 System 2 推理能力的涌现。
> 来源声明：DPO 原论文 (arXiv:2305.18290)，DeepSeekMath GRPO 首提 (arXiv:2402.03300)，DeepSeek-R1 应用 (arXiv:2501.12948)。所有公式严格对齐论文表述。

## 一、对齐的数学第一性原理：KL 正则化目标

无论是 PPO、DPO 还是 GRPO，其根源都来自同一个优化目标。我们需要找到策略 \($\pi_\theta$)，最大化奖励 \(r(x,y)\)，同时不偏离参考模型 \($\pi_{\text{ref}}$\) 太远（KL 正则化防止模式坍缩）。

### 1. 通用目标函数

$$
\max_{\pi} \mathcal{J}(\pi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

- \(r(x,y)\)：奖励函数（人类偏好或规则）。
- \($\beta$\)：KL 系数（控制对齐强度与多样性）。
- \($\log \frac{\pi}{\pi_{\text{ref}}}$\)：逐 Token KL 散度。

**为什么需要 KL 正则化？**

- 无 KL 项，模型会过度拟合偏好数据，导致模式坍缩（diversity 消失）。
- β 越大，模型越贴近 ref，保守但安全；β 越小，模型更激进，易学到高奖励但低概率行为。

### 2. 最优解的封闭形式

这是一个变分优化问题。根据 Gibbs 分布性质，最优策略 \($\pi^*$\) 为：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

其中配分函数：

$$
Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

**物理含义**：最优策略在参考模型基础上，根据奖励指数加权。奖励越高，概率指数级上升。

## 二、DPO：完整的数学推导

PPO 通过策略梯度逼近 ($\pi^*$\)，DPO 的天才之处在于：**直接从解析解反解奖励函数，消除 Reward Model**。

### Step 1: 反解奖励函数

对最优解取对数：

$$
\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)
$$

重排：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**关键**：Z(x) 只依赖 x，与 y 无关。

### Step 2: Bradley-Terry 偏好模型

人类在 \($y_w$\) (win) 和 \($y_l$) (loss) 中选择 ($y_w$) 的概率：

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

### Step 3: 代入并消去 Z(x)

将 r 代入差值项：

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

Z(x) 被完美抵消。

### Step 4: DPO 最终损失

替换 ($\pi^*$) 为 \($\pi_\theta$\)，最大化偏好数据似然：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

**梯度分析**：令 $(\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)})$，则：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \sigma(\hat{r}_l - \hat{r}_w) \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]
$$

**解析**：

- Boost Winner：增大 \($y_w$\) 概率。
- Suppress Loser：减小 \($y_l$\) 概率。
- Error Weight \($\sigma(\hat{r}_l - \hat{r}_w)$\)：动态调节。信心高时 $\sigma \to 0$ ，梯度小，防止过度对齐；信心低时 σ 大，强力修正。

**为什么 DPO 稳定？**

- σ 自动“刹车”：信心高时停止学习，避免模式坍缩。

## 三、GRPO：DeepSeek-R1 的推理引擎

GRPO（Group Relative Policy Optimization）首提于 DeepSeekMath (arXiv:2402.03300)，在 DeepSeek-R1 (arXiv:2501.12948) 中大规模应用，是**去中心化 PPO**变体，无需 Critic 模型。

### 1. 算法逻辑：Group Sampling

对于每个 Prompt \(x\)，从当前策略 \($\pi_{\theta_{\text{old}}}$\) 采样一组回答：

$$
\{o_1, o_2, \dots, o_G\}, \quad G \text{ 通常 } 64-128
$$

计算每个回答奖励 \(${r_1, r_2, \dots, r_G}$\)（R1-Zero 为纯规则，R1 为 hybrid）。

### 2. 优势函数估计（无 Critic）

传统 PPO 优势 A = r + γV(s') - V(s)，GRPO 用组内统计替代基线：

$$
A_i = \frac{r_i - \mu}{\sigma + \epsilon}, \quad \mu = \frac{1}{G} \sum r_j, \quad \sigma = \sqrt{\frac{1}{G} \sum (r_j - \mu)^2}
$$

**为什么组内 z-score 能替代 Critic？**

- Critic V(s) 估计期望奖励 E[r|s]。
- 组采样是蒙特卡洛估计：($\mu \approx E[r|x]$\)，\($\sigma$\) 衡量方差。
- z-score 是无偏相对优势估计（统计学中心化归一化），避免 Critic 额外训练和显存开销。

### 3. GRPO 目标函数

复用 PPO Clip 机制，KL 散度作为惩罚项直接加在目标函数中（逐样本计算后平均）：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x,\{o_i\}} \left[ \frac{1}{G} \sum_i \min\left( r(\theta) A_i, \, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \cdot \frac{1}{G} \sum_i \mathbb{D}_{\text{KL}}(\pi_\theta(o_i|x) || \pi_{\text{ref}}(o_i|x)) \right]
$$

其中 \( $r(\theta) = \frac{\pi_\theta(o_i|x)}{\pi_{\theta_{\text{old}}}(o_i|x)}$ \)，\($\mathbb{D}_{\text{KL}}$) 为逐样本 KL。

**为什么 GRPO 适合推理？**

- 显存解放：只需 Actor + Ref，Critic 卸载。
- 自博弈：模型和“过去的自己”比赛，只要生成比组平均好的解，就进步。
- R1 实践：R1-Zero 纯 RL 聚焦推理，R1 通过 multi-stage（cold-start SFT → reasoning RL → rejection sampling → general RL）平衡通用性。
- **`<think>` 标签 Format Reward**：R1 中引入 `<think>` 标签强制格式奖励，显著提升推理链完整性（R1 报告：无 `<think>` Reward 时 reasoning 能力下降 15–20%）。

## 四、梯度分析：模型到底学到了什么？

DPO 损失梯度：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \sigma(\hat{r}_l - \hat{r}_w) \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]
$$

其中：

$$
\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}, \quad \hat{r}_l = \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

**解析**：

- Boost Winner：增大 \($y_w$\) 概率。
- Suppress Loser：减小 \($y_l$\) 概率。
- Error Weight \($\sigma(\hat{r}_l - \hat{r}_w)$\)：动态调节。
  - 模型信心高（$\hat{r}_w \gg \hat{r}_l）时 $$\sigma \to 0$ ，梯度小，防止过度对齐。
  - 模型搞反时 σ 大，强力修正。

**为什么 DPO 稳定？**

- σ 自动“刹车”：信心高时停止学习，避免模式坍缩。

## 总结：从复杂回归简单

| 算法           | **核心数学机制** | **Reward Model** | **Critic Model** | **复杂度**     | **适用场景** |
| -------------- | ---------------------- | ---------------------- | ---------------------- | -------------------- | ------------------ |
| **PPO**  | 策略梯度 + Critic      | ✅ 显式                | ✅ 显式                | 极高 (4 Models)      | 早期通用 RLHF      |
| **DPO**  | 解析解 + 凸优化        | ❌ 隐式消除            | ❌ 不需要              | 低 (2 Models)        | 通用对话           |
| **GRPO** | 组内相对优势           | ✅ 显式/规则           | ❌ 用均值代替          | 中 (2 Models + 采样) | **推理任务** |

> **一句话总结**：
> DPO 用数学消灭 Reward Model，解决了**稳定性**问题。
> GRPO 用统计消灭 Critic Model，解决了**显存**问题。
> 数学每简化一步，工程生产力飞跃一步。
