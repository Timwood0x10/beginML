# Inference-Time Compute: The o1 Paradigm and the System 2 Revolution

> **Abstract**:
> The release of OpenAI's o1 marks a major paradigm shift in LLMs from "parameter-driven" to "compute-driven."
> Unlike traditional models' "fast but shallow" per-Token prediction, o1 introduces **System 2 Reasoning** (slow thinking), allocating extra compute during the inference phase for path search, self-play, and error correction.
> This chapter deconstructs o1's core mechanisms: **inference-time Scaling Laws**, **Process Reward Models (PRM)**, and **search algorithms**, and explores how they synergize with architectures like Jamba/MLA.

## 1. The Mathematical Foundation: Inference-Time Scaling Laws

In the past we followed the training-time Scaling Law: performance grows with parameter count and data volume. o1 proves the existence of a third dimension: **performance grows linearly/exponentially with Test-time Compute.**

### 1. From "Probability Sampling" to "Path Planning"

Traditional model objective: $P(y|x) = \prod P(y_t | x, y_{<t})$.
o1 model objective: among all possible reasoning-chain paths $\pi \in \Pi$, find the path that maximizes the probability of a correct answer:

$$
\hat{y} = \text{argmax}_{y} \sum_{\pi \in \Pi} P(y|\pi, Q) \cdot P(\pi|Q)
$$


o1 must, among all possible "thinking paths," find the one where the final answer y has the highest probability of appearing (via a weighted average over all paths).

**Plain explanation**:

* $\pi$ is a "thinking chain" inside the model (e.g., "first assume A, then verify B, finally conclude C").
* $\Pi$ is the set of all possible thinking chains.
* $P(\pi|Q)$ is the probability of this thinking chain itself appearing (does the model think this line of reasoning is plausible).
* $P(y|\pi, Q)$ is the probability of reaching answer y after following this thinking chain.
* o1 doesn't directly pick one path; it weight-averages the answer probabilities of all plausible paths and picks the y with the highest value.



### 2. The Conversion Formula Between Compute and Capability

Let $C_{test}$ be the FLOPs consumed at inference (determined by the number of inference Tokens) and $E$ the task error rate. Experiments show:

$$
E \propto \frac{1}{\log(C_{test})}
$$

This means: **as long as you give the model enough "thinking time," a medium-sized model can beat an ultra-large model on logical tasks.**

Measured results: when inference compute rises from 1x to 128x, accuracy on the AIME math competition improves from 74.4% to 96.7%, approximately satisfying:

$$
\text{Accuracy} \approx 1 - e^{-k \cdot C_{\text{test}}^{0.1}}
$$

($C_{\text{test}}$ is the inference-time compute; the exponent is only 0.1, showing extremely high returns)

---

## 2. Core Mechanisms: The Engineering Implementation of System 2

o1 internalizes human "slow thinking" into three key components of the model weights:

### 1. Reinforcement Learning on Chain-of-Thought (RL on CoT)

o1 isn't simply triggered into CoT by a Prompt; it's trained through large-scale **reinforcement learning (like PPO/DPO)**. Through self-play, the model discovers that generating words like "Wait..." and "Let me re-check" significantly boosts reward scores.

**Key training paradigm**:

- Use a **Process Reward Model (PRM)** to score every step of the reasoning chain, not just the final result.
- Combine **Monte Carlo Tree Search (MCTS)**-style sampling and backtracking to explore multiple reasoning paths.
- **Mathematical essence**: transforms Token generation from traditional "probability sampling" into "path search" — no longer relying on a single maximum-likelihood path, but searching the optimal reasoning tree.

### 2. Process-Based Reward Model (PRM)

This is the "core judge" that lets o1 self-correct.

- **ORM (Outcome-based)**: only looks at the final result (0 or 1).
- **PRM (Process-based)**: scores every intermediate step of the reasoning.

**Mathematical essence**:
Estimate a value function for the intermediate state $s_i$:

$$
V(s_i) = \text{PRM}(s_i, \pi_{<i})
$$

The model uses $V(s_i)$'s feedback to decide whether to continue the current path or **Backtrack**.

### 3. Search Algorithm: An Internalized Tree of Thoughts (ToT)

o1 runs an internal mechanism similar to **MCTS (Monte Carlo Tree Search)**:

1. **Selection**: based on PRM scores, pick the most promising thinking branch.
2. **Expansion**: generate the next reasoning step.
3. **Simulation/Evaluation**: assess that step's contribution to the final answer.
4. **Backpropagation**: update path weights, correcting previous logic errors.

---

## 3. Architecture Comparison: o1 vs. the Traditional Paradigm

| Dimension | **System 1 (GPT-4o)** | **System 2 (o1-preview)** | **Paradigm significance** |
| :-------------------- | :-------------------------- | :------------------------------ | :--------------------- |
| **Thinking mode** | Intuition, pattern recognition (fast thinking) | Logic, planning, verification (slow thinking) | From "guessing words" to "solving problems" |
| **Compute distribution** | $O(1)$, fixed compute per generation | **Dynamic compute** (on-demand allocation) | Compute scales with task difficulty |
| **Error handling** | Error all the way (hallucination) | **Self-detection and backtracking** | Significantly reduces hallucination in complex logic |
| **AIME math score** | 13.4% | **83.3%** | A leap in capability |

---

## 4. Engineering Practice: How Does o1 Synergize with Jamba/MLA/MoD?

o1 isn't an isolated technology; it's the "commander" of existing architecture-optimization techniques.

### 1. MLA + o1: The Cure for Memory Anxiety

At inference, o1 generates a flood of **Hidden Thought Tokens**, causing severe KV-Cache pressure.

* **Synergy point**: using **MLA (DeepSeek)** to compress the KV Cache is the only engineering solution to let o1-like models support ultra-long reasoning chains (minutes of thinking).

### 2. MoD + o1: Precision Deployment of Compute Resources

* **Synergy point**: **MoD (Mixture of Depths)** decides "which Tokens deserve deep thinking," while the **o1 mechanism** decides "how long to think."
* **Effect**: for a simple "hello," the Router guides Tokens to skip complex layers (System 1); for "optimize this kernel algorithm," it triggers full-layer execution for search (System 2).

### 3. Completely Solving the "Strawberry" Counting Problem

* **Traditional model**: directly sees the ID `[berry]` and can't count the `r`s.
* **o1 model**: in its "thinking process," it explicitly generates steps: "1. split the letters s-t-r-a-w-b-e-r-r-y; 2. check and count one by one." **Making the reasoning path explicit compensates for the semantic collapse at the Token layer.**

---

## 5. Cost and Limitations (The "O1" Tax)

1. **Inference latency**: not suited to real-time dialogue. Users must accept the "thinking..." wait time.
2. **Token cost**: the user only sees one line of answer, but the model may have burned 5000 "thinking Tokens" in the background.
3. **Black-box risk**: OpenAI hides the thinking chain. This "invisibility" increases alignment risk and debugging difficulty.

---

## 6. Summary: Entering the "Post-Parameter Era"

| Era | Core metric | Core philosophy |
| :----------------------- | :--------------------- | :------------------------------- |
| **Pre-training** | Parameter count (V) | Memorize the knowledge of the entire internet |
| **Post-training** | Instruction alignment (RLHF) | Learn to speak like a human |
| **Inference-time** | **Thinking time (T)** | **Learn to reason logically like a human** |

> Engineering value: the small model's "trade time for space"

> o1 proves: **a small model (like the 7B-70B scale), by greatly increasing inference-time compute (thinking tokens from a few hundred to tens of thousands or even hundreds of thousands), can reach or even surpass the level of hundred-billion-parameter models on math, logic, programming, and other tasks.**

> This directly challenges the traditional "parameters equal capability" Scaling Law, opening a new era of "**inference-time scaling laws**":
> **Capability no longer depends only on model size, but on how long you're willing to let it "think."**

---
