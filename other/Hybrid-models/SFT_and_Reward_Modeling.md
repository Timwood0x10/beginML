# 后训练 SFT 与 Reward Modeling 的数学与工程

> **摘要**：
> 预训练模型 (Base Model) 只是一个概率预测机，它不懂对话，甚至可能补全有害内容。
> **Post-training (后训练)** 旨在将模型与人类意图对齐。
> 本章解构这一过程的两个监督学习阶段：**SFT (监督微调)** 利用行为克隆赋予模型对话能力，而 **RM (奖励建模)** 利用 Bradley-Terry 模型将人类的主观偏好转化为可微的数学函数。


SFT 是对齐的第一步，俗称“教会模型说人话”。

### 1. 数学本质：有监督的 Next Token Prediction

SFT 的训练目标与预训练完全一致（最大化似然估计），但有一个关键的工程区别：**Loss Masking (损失掩码)**。

给定指令 $x$ (Prompt) 和回答 $y$ (Response)，我们只计算 $y$ 的生成概率，不计算 $x$ 的。

$$
\mathcal{L}_{SFT}(\theta) = - \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t})
$$

* **Prompt**: `User: 解释量子力学。` $\to$ **Loss = 0** (不通过反向传播更新权重)。
* **Response**: `Assistant: 量子力学是...` $\to$ **Loss > 0** (模型学习这部分)。

### 2. 核心假设：LIMA (Less Is More for Alignment)

Meta 的论文 *LIMA* 提出了一个震撼的观点：**SFT 并没有让模型学到新知识，它只是在学习“格式”与“风格”。**

> **表面形式对齐假设 (Surface Form Alignment Hypothesis)**：
> 模型的知识与能力主要来自预训练。SFT 只是激活了特定的子空间，学会了与其训练数据相似的交互风格。

**工程推论**：

* **数据量**：不需要海量数据。1,000 条**高质量**、**多样化**的人类手写数据，效果远好于 100,000 条由 ChatGPT 生成的平庸数据。
* **多样性**：必须覆盖问答、摘要、逻辑推理、代码等多种任务类型。

---

## ⚖️ 二、Reward Modeling (RM)：打造数字裁判

人类无法为每个回答精准打分（例如 88.5 分 vs 88.9 分），但人类非常擅长**比较**（A 比 B 好）。RM 的目标是训练一个神经网络，来模拟人类的排序偏好。

### 1. 训练数据：Pairwise Preference

数据格式不是 `(Question, Answer, Score)`，而是三元组：

$$
\mathcal{D} = \{ (x, y_w, y_l) \}
$$

* $x$: Prompt
* $y_w$: Winner (人类觉得更好的回答)
* $y_l$: Loser (较差的回答)

### 2. 数学模型：Bradley-Terry Model 

我们将偏好关系转化为概率分布。假设奖励函数 $r_\theta(x, y)$ 输出一个标量分数，则 $y_w$ 优于 $y_l$ 的概率建模为 Sigmoid 函数：

$$
P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) = \frac{1}{1 + e^{-(r_w - r_l)}}
$$

### 3. 损失函数：Ranking Loss

最大化人类偏好数据的似然概率，等价于最小化以下对数损失：

$$
\mathcal{L}_{RM}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]
$$

**直观理解**：

* 我们希望 $r(y_w)$ 尽可能大，且 $r(y_l)$ 尽可能小。
* **差值越大，Loss 越小**。

### 4. RM 的架构设计

RM 通常基于 SFT 后的模型初始化，但做了一些修改：

* **去掉 Unembedding Layer**：不再输出词表概率 (Vocab Size)。
* **添加 Scalar Head**：在最后一层添加一个线性层 $(d_{model} \to 1)$，输出一个实数 Logit。
* **Padding Token**：通常取最后一个有效 token 的 hidden state 作为整个句子的奖励表征。

---

## 🔗 三、从 SFT 到 RM 的连接与局限

为什么有了 SFT 还需要 RM？为什么不直接用 RM 训练生成？

| 特性               | SFT 模型                                                   | Reward Model (RM)                                                |
| :----------------- | :--------------------------------------------------------- | :--------------------------------------------------------------- |
| **角色**     | **演员 (Actor)**                                     | **裁判 (Critic/Judge)**                                    |
| **任务**     | 生成下一个 Token                                           | 给整个句子打分                                                   |
| **训练目标** | 模仿人类说话                                               | 模仿人类评价                                                     |
| **局限性**   | **幻觉与平庸**：只能模仿，无法超越数据中的人类水平。 | **代理谬误**：它只是人类偏好的拟合，容易被Hack（见下文）。 |

### 关键挑战：Reward Hacking (古德哈特定律)

一旦我们将 RM 作为一个固定的优化目标（在后续 RL 中），模型非常容易找到 RM 的漏洞。

* *例子*：RM 可能认为“越长的回答越好”。
* *结果*：RL 模型开始输出无限重复的废话，只为骗取高分。

这就是为什么在下一章的 RLHF/PPO 中，我们必须引入 **KL 散度 (KL Divergence)** 作为约束。

---

## 🛠️ 四、工程实战代码片段

### 1. SFT Loss Masking (PyTorch)

```python
def sft_loss(logits, labels, label_mask):
    """
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
    label_mask: [batch, seq_len] (1 for response, 0 for prompt)
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = label_mask[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1))
  
    # Apply Mask: 只计算 Response 部分的 Loss
    loss = loss.view(shift_labels.size()) * shift_mask
    return loss.sum() / shift_mask.sum()
```

## 总结

**SFT 和 RM 是现代大模型对齐的“左右手”：**

* **SFT** **建立了模型遵循指令的基本能力（Format & Instruction）。**
* **RM** **将人类难以言喻的价值观量化为了数学指标（Proxy for Human Preference）。**
