# The Mathematics and Engineering of Post-Training SFT and Reward Modeling

> **Abstract**:
> A pretrained model (Base Model) is just a probability predictor; it doesn't understand dialogue and may even complete harmful content.
> **Post-training** aims to align the model with human intent.
> This chapter deconstructs the two supervised-learning stages of this process: **SFT (Supervised Fine-Tuning)** uses behavioral cloning to give the model conversational ability, while **RM (Reward Modeling)** uses the Bradley-Terry model to turn human subjective preferences into a differentiable mathematical function.


SFT is the first step of alignment, colloquially known as "teaching the model to speak human."

### 1. Mathematical Essence: Supervised Next Token Prediction

SFT's training objective is identical to pretraining (maximum likelihood estimation), but with one key engineering difference: **Loss Masking**.

Given an instruction $x$ (Prompt) and a response $y$ (Response), we only compute the generation probability of $y$, not of $x$.

$$
\mathcal{L}_{SFT}(\theta) = - \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t})
$$

* **Prompt**: `User: Explain quantum mechanics.` $\to$ **Loss = 0** (no weight updates via backprop).
* **Response**: `Assistant: Quantum mechanics is...` $\to$ **Loss > 0** (the model learns this part).

### 2. Core Assumption: LIMA (Less Is More for Alignment)

Meta's paper *LIMA* proposed a striking viewpoint: **SFT doesn't make the model learn new knowledge; it's just learning "format" and "style."**

> **Surface Form Alignment Hypothesis**:
> The model's knowledge and abilities mainly come from pretraining. SFT just activates specific subspaces, learning an interaction style similar to its training data.

**Engineering implications**:

* **Data volume**: no need for massive data. 1,000 **high-quality**, **diverse** human-written examples work far better than 100,000 mediocre ChatGPT-generated ones.
* **Diversity**: must cover multiple task types — Q&A, summarization, logical reasoning, code, etc.

---

## ⚖️ 2. Reward Modeling (RM): Building a Digital Judge

Humans can't precisely score every response (e.g., 88.5 vs 88.9), but humans are very good at **comparing** (A is better than B). RM's goal is to train a neural network that simulates human ranking preferences.

### 1. Training Data: Pairwise Preference

The data format isn't `(Question, Answer, Score)` but a triple:

$$
\mathcal{D} = \{ (x, y_w, y_l) \}
$$

* $x$: the Prompt
* $y_w$: the Winner (the response humans think is better)
* $y_l$: the Loser (the worse response)

### 2. Mathematical Model: The Bradley-Terry Model

We convert the preference relation into a probability distribution. Assuming the reward function $r_\theta(x, y)$ outputs a scalar score, the probability that $y_w$ is preferred over $y_l$ is modeled with a Sigmoid function:

$$
P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) = \frac{1}{1 + e^{-(r_w - r_l)}}
$$

### 3. Loss Function: Ranking Loss

Maximizing the likelihood of the human preference data is equivalent to minimizing the following log loss:

$$
\mathcal{L}_{RM}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]
$$

**Intuition**:

* We want $r(y_w)$ to be as large as possible and $r(y_l)$ as small as possible.
* **The larger the difference, the smaller the Loss**.

### 4. RM Architecture Design

RM is usually initialized from an SFT'd model, but with some modifications:

* **Remove the Unembedding Layer**: no longer outputs vocabulary probabilities (Vocab Size).
* **Add a Scalar Head**: add a linear layer $(d_{model} \to 1)$ on the last layer, outputting a real-valued Logit.
* **Padding Token**: usually take the hidden state of the last valid token as the sentence's reward representation.

---

## 🔗 3. Connections and Limitations from SFT to RM

Why do we need RM if we already have SFT? Why not train generation directly with RM?

| Feature | SFT model | Reward Model (RM) |
| :----------------- | :--------------------------------------------------------- | :--------------------------------------------------------------- |
| **Role** | **Actor** | **Critic/Judge** |
| **Task** | Generate the next Token | Score the whole sentence |
| **Training objective** | Imitate how humans speak | Imitate how humans evaluate |
| **Limitation** | **Hallucination and mediocrity**: can only imitate; can't surpass the human level in the data. | **Proxy fallacy**: it's just a fit of human preference; easily hacked (see below). |

### Key Challenge: Reward Hacking (Goodhart's Law)

Once we treat RM as a fixed optimization objective (in subsequent RL), the model very easily finds holes in the RM.

* *Example*: RM might believe "longer responses are better."
* *Result*: the RL model starts outputting infinitely repeating nonsense, just to scam high scores.

This is why, in the next chapter's RLHF/PPO, we must introduce **KL Divergence** as a constraint.

---

## 🛠️ 4. Engineering Practice Code Snippets

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
  
    # Apply Mask: only compute the Loss of the Response part
    loss = loss.view(shift_labels.size()) * shift_mask
    return loss.sum() / shift_mask.sum()
```

## Summary

**SFT and RM are the "left and right hands" of modern large-model alignment:**

* **SFT** establishes the model's basic ability to follow instructions (Format & Instruction).
* **RM** quantifies humans' inexpressible values into a mathematical metric (Proxy for Human Preference).
