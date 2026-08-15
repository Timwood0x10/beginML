# ML Experiment Lab — 开发计划 v2.1（Architecture Freeze）

> 核心命题（保留，作为产品 DNA）：
> **不是给每篇笔记配一个可视化，而是从每个知识点里找出一个可以亲手操纵的核心机制，把它做成实验。**
>
> 产品 slogan：**Don't watch the answer. Run the experiment.**
> 浓缩版：**The computer runs the experiment. You make the discovery.**
>
> 两轮 review 评级：产品 9.5 / 学习机制 9.5 / 工程 9 / 视觉 8.5。
> 不引入新框架，现有 `modules.py → compute → router → React Lab` 链路足够。
> **开工前冻结：三项契约（§3-§5）+ Question 层 + Discovery Artifact + Next Question（§9）。**
> **MVP = 一个完整认知闭环，不是 N 个实验。**

---

## 〇、No-LLM Principle（最高级约束，产品 DNA）

**ML Experiment Lab 禁止 LLM 参与核心实验闭环。**

不是「暂时不用 LLM」，而是架构约束：实验结果、证据、判定、发现均由**确定性程序**产生。
LLM 不参与实验运行、结果解释、正确性判定或 Discovery 生成。

```
User → Question → Predict → Controls → Simulation → Observation → Evidence
     → Deterministic Explanation → Discovery Artifact → Next Question
                     （全链路 deterministic / numerical / rule-based）
```

- 无 Agent / 无 LLM / 无 RAG / 无 MCP / 无 Vector DB / 无 Graph DB / 无自动 Discovery Engine
- 不是让 AI 告诉你答案，而是让程序给你证据，让你自己发现答案。

---

## 一、学习循环：Predict → Experiment → Observe → Explain

这是产品的核心，不是 UI feature。

| 传统 | 本平台 |
|---|---|
| Read → Understand → Remember | Predict → Experiment → Observe → Explain |

- **Predict**：用户在不知道答案时先做判断（假设）。
- **Experiment**：操纵变量，运行实验。
- **Observe**：看观察窗口，收集证据。
- **Explain**：证据落回解释，接笔记锚点。

循环是可重复的，且以 **Question** 为起点、以 **Next Question** 为出口：

```
Question → Predict → Experiment → Observe → Evidence → Explain → Next Question
```

例如 Sampling Machine：Question「为什么调高温度会让输出更随机？」
→ Predict → 实验 → Observe（entropy ↑, diversity ↑）→ Evidence
→ Explain → Next Question「什么是熵？」→ 跳转信息论实验。

> 这不只是 visualization，而是 **Scientific Method for Learning**。

关键深化——**Evidence 层**：实验不是「动画结束 → 给答案」，而是
`Prediction → Observation → Evidence → Explanation` 四段式。例如 Double Descent：

```
YOUR PREDICTION
❌ Bigger models should overfit more.
OBSERVED
Test error: 0.31 → 0.87 → 0.19
EVIDENCE
Training error  ↓↓↓
Test error      ↑↓
Interpolation   crossed
EXPLANATION
...
```

这使实验区别于普通教育动画。

---

## 二、Experiment = 可重复的实验协议（不是页面数据结构）

```
Experiment
├── question       ← 驱动实验的问题（Question 层，实验的起点）
├── hypothesis     ← 用户实验前的假设（Predict 的产物，进入实验状态）
├── controls       ← 可操作变量
├── simulation     ← 后端数值模拟（见 Simulation Contract）
├── observations   ← 观察窗口
├── evidence       ← 证据（Observe 的产物）
├── explanation    ← 解释（Explain 的产物）
├── notes          ← 笔记锚点
├── provenance     ← 结果怎么来的（realtime/cached/seeded + 参数快照）
└── next_question  ← 下一个问题（Discovery Graph 的边）
```

每次运行是一个 `ExperimentRun`：

```ts
type ExperimentRun = {
  params: Params
  hypothesis?: Hypothesis    // 用户预测（可选，但鼓励）
  observation?: Observation
  evidence?: Evidence
  result?: Result
}
```

这样「你刚才预测错了 3 次」成为可能，并沉淀为 **Experiment History**（见 §12 Journal）：

```
RUN #01  Temperature ↑  →  Prediction: more concentrated ❌
RUN #02  Top-p ↓        →  Prediction: diversity increases ❌
RUN #03  ...
```

**Evidence = simulation 输出中的可验证事实（计算结果，不搞 Evidence Engine）**：

```ts
type Evidence = {
  metrics: Record<string, number>   // 如 entropy_before / entropy_after
  comparisons: Comparison[]          // 前后对比
  highlights: string[]               // 关键可验证事实
}
```

Sampling 示例：`{ entropy_before: 1.42, entropy_after: 1.91, temperature: 1.8, diversity_before: 0.31, diversity_after: 0.47 }`。Phase 1 不做 Evidence Engine / Aggregator / Graph / Scorer——前端直接展示 compute 返回的可验证事实。

**Explanation = 人工撰写的静态要点（不是生成系统）**：

```ts
// 例如 Sampling：
explanation = [
  "Temperature scales logits before softmax.",
  "Higher temperature produces a flatter probability distribution.",
  "A flatter distribution increases sampling diversity.",
]
```

不做 `experiment → LLM → "根据你的实验，我认为……"`。Discovery 也不由模型生成。

**Discovery Artifact（Insight 是用户的，不是 AI 的）**：

```ts
type DiscoveryArtifact = {
  experimentId: string
  runId: number
  prediction: string
  observation: string
  evidence: Evidence
  userInsight: string   // 用户自己写的发现
  params: Params
  createdAt: string
}
```

---

## 三、冻结契约 #1 — Experiment Protocol

每个实验（新 lab 模块）必须声明以下字段，缺一不可：

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` | string | 驱动实验的问题（Question 层，实验的起点） |
| `next_question` | string | 引导的下一个问题（Discovery Graph 的边） |
| `id` | string | 模块 id（如 `sampling-machine`） |
| `title` | string | 隐喻标题（如 `Sampling Machine`） |
| `subject` | string | 实验对象（如 `logits → probability → sampling`） |
| `controls` | LabControl[] | 可操作变量（复用现有控件类型） |
| `simulation` | `realtime \| cached \| seeded` | 见 Simulation Contract |
| `observations` | string[] | 观察窗口清单 |
| `challenge` | ChallengeSpec | 见 Challenge Contract |
| `explanation` | string[] | 结论解释要点 |
| `notes` | string[] | 笔记锚点（`Self-Attention/xxx.md` 等） |
| `provenance` | bool | 结果携带参数快照 + 来源标注 |

---

## 四、冻结契约 #2 — Simulation Contract

`compute(params)` 不再只有一种形态，按实验性质分三类：

```
Simulation
├── realtime    compute(params)              —— 每次请求即时算（Sampling / RoPE）
├── cached      lookup(params)               —— 预计算 / 缓存（Double Descent）
└── stochastic  seeded_compute(params, seed) —— 带种子可复现 + 可缓存（Bias-Variance）
```

- **realtime**：数值便宜、交互需要即时反馈的（Sampling、RoPE）。
- **cached**：数值重、曲线形状依赖大量采样/训练的（Double Descent）。**允许预计算实验数据**——仍属纯数值模拟，不违反「不介入真模型」原则，且比每次 slider 重新训练更可靠。
- **stochastic**：需要随机性的（Bias-Variance 的 bootstrap 重采样），**必须带 seed**，保证同一参数可复现、可进缓存。

任何随机实验必须有 `seed` 参数；`provenance` 记录参数快照 + 模拟类型。

---

## 五、冻结契约 #3 — Challenge Contract（独立于 ControlRow）

**Control 与 Challenge 是两个语义，视觉上必须分离。**

- Control = 我要改变实验（左侧控件面板）。
- Challenge = 我在不知道答案时做判断（独立预测层）。

```
┌──────────────────────────────────────┐
│ EXPERIMENT                           │
│   controls                           │
│   temperature ─────●                 │
│   top-p       ───────●               │
├──────────────────────────────────────┤
│ 🔍 MAKE A PREDICTION                 │
│   What happens when temperature ↑ ?  │
│   ○ More concentrated   ○ More diverse   ○ No change │
│                    [ RUN EXPERIMENT ] │
└──────────────────────────────────────┘
```

Challenge 是**独立层组件**（如 `ChallengePanel`），不是 ControlRow 的一种控件类型。未来可扩展题型：

- multiple choice（单选）
- prediction curve（画预测曲线）
- drag prediction（拖拽预测）
- ranking（排序）
- estimate（估数值）
- explain why（开放解释）

**Challenge 三层演进（冻结方向）**——避免变成考试软件：

| 层 | 形式 | 例子 |
|---|---|---|
| L1 Predict | 选择题 | 「temperature ↑ 会发生什么？」○ A ○ B ○ C → RUN |
| L2 Manipulate | 不告诉参数，给目标 | 「让模型生成更发散的分布」——用户自己调 temperature / top-p / top-k 到成功 ✓ |
| L3 Explain | 解释为什么 | 「为什么你的配置有效？」多选 + 反馈 |

目标交互：`Predict → Manipulate → Observe → Explain`。Phase 1 只做 L1，L2/L3 随实验迭代加入。

后端 `ExperimentRun.hypothesis` 持久化用户预测；RUN 后与 `observation`/`evidence` 对比，判定对错，进入 History。

---

## 六、与现有架构的映射（零新框架）

| Protocol 字段 | 现有落点 |
|---|---|
| `id / title / controls` | 后端 `lab/modules.py` 的 MODULES 元数据 |
| `simulation` | 后端 `lab/*.py` 的 `compute(params)` + `router.py` 的 `_COMPUTERS` |
| `visualization` | 前端 `src/lab/modules/*.tsx` + `LabPage.tsx` 的 `LAB_COMPONENTS` |
| `observations` | 组件内数值面板 / canvas 标注 |
| `challenge` | **新增**独立 `ChallengePanel` 组件（不属于 ControlRow） |
| `notes` | 组件内 Related Notes 面板 |
| `provenance` | compute 返回参数快照 + simulation 类型标注 |

**新增模块标准 6 步**：① modules.py 元数据 → ② lab/<id>.py compute → ③ router 注册 → ④ 前端组件 → ⑤ LabPage 注册 + i18n → ⑥ ChallengePanel + Related Notes 挂载。

---

## 七、世界观、材质与配色

- **EXPLORE（实验）** 与 **BUILD（建造）** 并列：Agent Forge 是「搭东西」，Transformer Lab 是「观察神经系统」。
- **统一「材质」，不统一「配色」**：不要找一个万能主色。统一实验室材质——背景浅暖灰/雾白、文字深墨灰、边框低对比银灰、控件半哑光、阴影极轻、图形低饱和。**强调色每个实验自己拥有一个 muted accent**：

| 实验 | Accent |
|---|---|
| Sampling | 柔和珊瑚橙 |
| RoPE | 雾紫 |
| Double Descent | 苔绿 |
| Bias-Variance | 柔和靛蓝 |
| Quantization | 竹青 |
| MDP | 沙黄 |
| Diffusion | 雾蓝 |

- Transformer Lab 通用作用域 `data-scope="tlab"`（CSS 变量 `--tlab-*`），与 6 主题潮汐切换零冲突；每个实验在作用域内再定自己的 accent。
- **实验命名本身是教学**：Rotary Observatory、Sampling Machine、Dangerous Mountain、Weight Freezer、Shooting Range……

---

## 八、Lab 导航重组（按「我要探索什么」组织）

**不是文件目录，是探索意图分组：**

```
🔬 EXPERIMENT LAB — Predict. Experiment. Discover.

NOW EXPERIMENTING
──────────────────
🎰 Sampling Machine

ROTARY OBSERVATORY
──────────────────
🌀 RoPE

MODEL BEHAVIOR
──────────────
👥 Token Society
🕵️ Transformer Detective
🌊 Representation River

LEARNING DYNAMICS
─────────────────
🏔️ Dangerous Mountain
🎯 Shooting Range

MODEL EFFICIENCY
────────────────
🧊 Weight Freezer
```

实现：后端 MODULES 元数据增加 `group` 字段（如 `"learning-dynamics"`），前端侧栏按 group 分组渲染。

---

## 九、产品三层 + 最终目标：Discovery Graph / Curiosity Graph

### 9.1 产品三层：Playground → Experiment → Discovery

| 层 | 用户状态 | 交互 | 用户心理 |
|---|---|---|---|
| Layer 1 Playground | 玩 | sliders / drag / click / animate | 「这是什么东西？」 |
| Layer 2 Experiment | 做实验 | Predict / Manipulate / Observe / Evidence | 「原来它是这么工作的。」 |
| Layer 3 Discovery | 形成发现 | Insight / Journal / Graph / Next Question | 「这是我自己发现的。」 |

```
                 ML KNOWLEDGE
                      │
                      ▼
              ┌──────────────┐
              │  PLAYGROUND  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ EXPERIMENT   │
              │ Predict      │
              │ Manipulate   │
              │ Observe      │
              │ Evidence     │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  DISCOVERY   │
              │ Insight      │
              │ Journal      │
              │ Graph        │
              │ Next Question│
              └──────┬───────┘
                     ▼
                  KNOWLEDGE
```

### 9.2 Discovery Graph / Curiosity Graph（不是 Experiment List）

最大风险是做成 10 个漂亮的 mini-app。**不是知识节点图，而是「问题 → 实验 → 发现 → 新问题」的认知路径**：

```
                 QUESTION
                    │
                    ▼
              ┌───────────┐
              │ EXPERIMENT│
              └─────┬─────┘
                    │
             ┌──────┴──────┐
             ▼             ▼
        OBSERVATION     SURPRISE
             │             │
             ▼             ▼
         EVIDENCE       NEW QUESTION
             │             │
             └──────┬──────┘
                    ▼
                EXPERIMENT
```

例：「为什么调高温度会更随机？」→ Sampling Machine → entropy ↑ → Next Question「什么是熵？」→ 信息论实验。知识节点图只是静态底图，**Question 边才是活的东西**。

**静态知识底图（连接关系，非核心）**：实验之间存在知识连接——`Sampling → Temperature → Entropy`、`Bias → Double Descent → Regularization`、`RoPE → Attention → Token Society → Detective → Residual River`。这些只是底图，**产品价值在 Question 边**：每个实验带「连接出口」，例如 Dangerous Mountain 跨过危险区后弹出 `[ ENTER SHOOTING RANGE ]`——「想理解经典 bias-variance 在这里预言了什么？」用户不是在浏览实验，而是在知识空间里从一个实验发现另一个问题。

**Phase 1 实现（不做 Graph Engine）**：Discovery Graph 首版只是元数据里的**静态 `next_question` 字段**，不做图数据库 / Graph Traversal / Graph Store / Graph Algorithm：

```py
MODULES = {
    "sampling-machine": {
        ...
        "next_question": "What is entropy?",
        "next_experiment": "entropy-playground",
    }
}
```

前端 DISCOVERY 面板直接渲染：

```
DISCOVERY
────────────────────
You discovered:
Temperature changes distribution sharpness.
NEXT QUESTION
Why does entropy measure this?
[ EXPLORE → ]
```

未来实验数量真起来了，再把这些静态边组织成 Graph。

---

## 十、Phase 1 — 招牌实验（冻结详设）

**MVP 定义 = 一个完整认知闭环，不是三个模块。** 三个实验各自验证一种**学习模式**，而不是「做三个模块」：

| Lab | 验证的学习模式 | 闭环 |
|---|---|---|
| Sampling Machine | 操纵机制 | Predict → Manipulate → Observe → Evidence → Explain → Next Question |
| Rotary Observatory | 发现机制 | Intuition → Manipulation → Unexpected relationship → Aha moment → Formula |
| Dangerous Mountain | 推翻直觉 | Prediction → Contradiction → Evidence → Surprise → New question → Graph transition |

先做 **Sampling Machine 一个**，完整跑通认知闭环；跑通后再把 RoPE、Dangerous Mountain 当作同一认知引擎的不同「实验仪器」接入。**禁止按实验数量推进。**

### 10.1 🎰 Sampling Machine — 先做，验证整个 Engine

**笔记锚**：`Self-Attention/9.Inference_Sampling.md`。

**⚠ 教学修正（冻结）**：Temperature 不改变 logits 本身，它改变 `softmax(logits / T)`。UI 必须明确管线，禁止「temperature 把 logits 压平」的错误表达：

```
┌────────┐   ┌─────────────┐   ┌───────────┐   ┌─────────────┐   ┌──────┐
│ LOGITS │ → │ TEMPERATURE │ → │  SOFTMAX  │ → │ FILTER GATE │ → │ 🎲   │
│ (raw)  │   │ (/T, 不改动) │   │ (prob)    │   │ top-k / top-p│   │SAMPLE│
└────────┘   └─────────────┘   └───────────┘   └─────────────┘   └──────┘
```

- **question**：「为什么调高温度会让输出更随机？」（Question 层驱动整个实验）
- **controls**：`logits_mode`（select：多峰/平坦/尖峰）、`temperature`（range 0.1..3）、`gate`（select：none / top-k / top-p / both）、`top_k`（range 1..10）、`top_p`（range 0.05..1.0）、`samples`（action：SAMPLE ×20）。
- **observations**：raw logits 柱；temperature 后（softmax 前）柱——**明确标注「logits 不变」**；softmax 概率柱；gate 前后对比（BEFORE → GATE → AFTER）；采样球落入 token 槽动画 + 直方图；熵值显示。
- **challenge**：L1 Predict「temperature ↑ 之后采样分布会怎样？」A 更集中 / B 更发散 / C 不变 → RUN → 对比证据 → 解释（熵与锐化）。
- **next_question**：「什么是熵？」→ 跳转信息论 / 分布实验。
- **simulation**：realtime。
- **后端**：`lab/sampling.py` 纯 numpy：logits → logits/T → softmax → gate 裁剪重归一 → 采样计数。

### 10.2 🌀 Rotary Observatory — 视觉旗舰，精雕

**笔记锚**：`Self-Attention/11.RoPE.md`、`7.Advanced.md`。

**⚠ 数学终点（冻结）**：单点旋转只是第一阶段的 intuition。**最终要让用户发现的是内积依赖相对位置**：

```
(R(m)q)ᵀ(R(n)k)  只依赖  m - n
```

阶段递进（不给一次全给）：① 单点随 position 旋转（位置 = 旋转角）→ ② 双 token ROTATE TOGETHER（绝对角 vs 相对相位 Δθ = θ(m-n)）→ ③ FIND THE DISTANCE（拖两位置，相似度 vs 距离曲线实时出现，REVEAL 才给公式）→ ④ Frequency Lens（高低频维度分工）。

**「啊哈时刻」**：最后出现一句极简结语——

> **Absolute rotation disappears. Relative rotation remains.**

再展开数学。

- **controls**：`position`（0..32）、`frequency`（low..high）、`pair`（toggle）、`distance_mode`（toggle）。
- **challenge**：「距离从 4 增到 8，similarity 变高/低/不变？」→ RUN → 揭晓。
- **simulation**：realtime。

### 10.3 🏔️ Dangerous Mountain — 验证「实验产生反直觉发现」

**笔记锚**：`math/10.Important_Curves.md`、`9.noise.md`。

**⚠ 最大风险 + 对策（冻结）**：random Fourier features 不一定稳定产生经典 double descent。**写任何前端之前，先做独立 spike**：

```
backend/lab/spikes/double_descent.py
```

目标只有一个：在选定的参数范围内，**稳定产生教学上漂亮且数学上合理的 double descent**。依赖变量：数据生成方式、feature distribution、regularization、interpolation regime、sample/feature 比、噪声、solver。spike 通过后固化参数，正式 Lab **使用预计算实验数据（cached 模式）**——比每次 slider 重新训练可靠，仍属纯数值模拟。

- **观察窗口**：测试误差 vs 容量山形 + ⚠ DANGER ZONE；UNDERFIT → DANGER → OVERPARAMETERIZED 状态卡；CLASSICAL vs MODERN 分屏 + THEORY→REALITY 融合滑块。
- **challenge**：「更大模型应该过拟合更严重——你信吗？」→ 跨过临界点后弹出「Wait. Why did the bigger model get better?」。
- **连接出口**：`[ ENTER SHOOTING RANGE ]` → Phase 2 的 Bias-Variance。
- **simulation**：cached（预计算）。

---

## 十一、Phase 2

### 11.1 🎯 Shooting Range（Bias-Variance）— 提前设计，与 Dangerous Mountain 互链

**笔记锚**：`math/9.noise.md`、`10.Important_Curves.md`。

知识叙事线：**Bias-Variance → Regularization → Double Descent**。与 Dangerous Mountain 一起设计（同属 LEARNING DYNAMICS group），不必一起发布；Double Descent 页内嵌 `[ ENTER SHOOTING RANGE ]` 出口。

- 靶场隐喻：弹孔 = 预测分布，靶心 = 真值。四象限状态直观呈现。
- `MSE = Bias² + Variance + Noise²` 三段消长条；DATASET 10→500 看 Variance 缩水。
- **simulation**：stochastic（seed）+ cached。

### 11.2 🧊 Weight Freezer（Quantization）— 做成「破坏实验」

**笔记锚**：`Hybrid-models/Quantization_FP8_BitNet.md`。

在 FREEZE 基础上加 **BREAK IT**：

```
FP32 → INT8 → INT4 → INT2 → TERNARY
Memory ↓↓↓↓        Error ↑↑↑↑
```

用户不是在「看量化误差」，而是在**寻找 Pareto 甜蜜点**：「Where does compression stop being worth it?」加 TERNARY MODE（{-1,0,+1} 三簇，「1.58 bit」不再是论文名词）。

### 11.3 🌊 Representation River（Residual Stream）— 明确仿真边界

**笔记锚**：`Self-Attention/3.ResidualConnection.md`。

**⚠ 真模型边界（冻结）**：没有真实 Transformer activation，就**不要说「Paris 在模型里逐渐变成了什么」说得太实**（避免伪 mech interp）。UI 明确标注：

```
SIMULATION MODE
This experiment models representation dynamics.
It does not inspect a real model.
```

- **Mode A（本次）**：Synthetic Representation River——模拟 representation 演化（PCA 投影 2D 轨迹）。
- **Mode B（未来）**：Model Trace——接入真实模型 activation，UI 标注 TRACE MODE。

---

## 十二、Experiment Journal + Artifact（高价值小功能）

用户做实验时沉淀自己的认知轨迹：

```
EXPERIMENT JOURNAL
My prediction:  "Temperature ↑ makes ..."
Observed:       "..."
What surprised me: "..."
[SAVE OBSERVATION]

MY EXPERIMENTS
RoPE             ✓ discovered relative position
Sampling         ✓ temperature / entropy
Double Descent   ✓ interpolation threshold
```

平台从「灌知识」变成「记录用户的探索路径」，与 Memory Distillation / cognition 思路契合。MVP 阶段 Journal 可先做最简版（预测 + 结果 + 对错记录），随 History 演进。

### Experiment Artifact（比 Journal 更优先）

用户做完一次实验后，不只留下 `RUN #3 prediction wrong`，而是可以生成一张 **Experiment Card**：

```
┌──────────────────────────────────────────┐
│ 🏔️ DANGEROUS MOUNTAIN                    │
│ MY DISCOVERY                             │
│ "Bigger models don't necessarily         │
│  generalize worse."                      │
│ ───────────────────────────────────────  │
│ Capacity 512 · Samples 128 · Noise 0.10  │
│ Prediction ❌ · Observation: test error ↓ │
│ Evidence: train ────↓ / test ──╱╲──↓      │
│ [ SAVE DISCOVERY ]                       │
└──────────────────────────────────────────┘
```

**Save Discovery** 形成个人 **MY DISCOVERIES** 学习档案：

```
MY DISCOVERIES
🌀 I discovered why relative position works
🎰 I discovered what temperature really changes
🏔️ I discovered double descent
🧊 I found the quantization trade-off
```

**与 Memory Distillation 契合**：Journal 不是普通学习笔记，它天然是
`Raw Experience → Observation → Surprise → Insight → Distilled Memory`：

```
RUN #01 ❌ → RUN #02 ❌ → RUN #03 ✅ → Pattern detected
→ "温度改变的是 softmax 分布，不是 logits"
→ DISTILLED INSIGHT
```

未来接 ARES Memory Distillation 时，这里是天然实验场。MVP 做最简版（预测 + 结果 + 对错 + Card 雏形）。

---

## 十三、Phase 3 — 空白带扩展（Token Society 最高优先级）

**Token Society 是 Transformer Lab 的入口世界（母实验）**：

```
TOKEN SOCIETY
 ├── Who looks at whom?   → Attention
 ├── Why this token?      → Detective
 ├── How does info move?  → Residual River
 └── How does position matter? → RoPE
```

| 优先级 | 实验 | 笔记锚 | 隐喻 | 状态 |
|---|---|---|---|---|
| P0 | 👥 Token Society（母实验） | `Self-Attention/1.math.md` `2.multi-headed.md` | 谁在看谁 | Transformer 入口 |
| P0 | 🕵️ Transformer Detective | `6.CausalMask.md` `1.math.md` | 破案 | 证据 = 算法生成 + 人工复核案例库 |
| P1 | 🩻 Transformer MRI（简化版） | `10.Training_Essentials.md` | 逐层扫描 | 先挂 transformer-training 的 loss/entropy |
| P1 | 🧬 Feature Hunt | 补 mech interp 笔记 | 找 feature | 算法模拟合成激活，标注 SIMULATION MODE |
| P1 | 🐍 Mamba Memory Race | `Appendix_E_Mamba_vs_Transformer.md` | O(L) vs O(L²) | 复杂度对比 |
| P1 | 🚪 MoE Expert Routing | `Hybrid-models/MoE.md` | 专家分诊 | 路由门控可视化 |
| P2 | 🌫️ Noise Rewind | `math/15.DiffusionModel.md` | 从噪声倒放 | SDE 模拟 |
| P2 | 🗺️ Decision World | `math/16.MDP.md` | 改 reward 看策略翻转 | 网格世界 + 值迭代 |
| P2 | 🪢 Constraint Playground | `math/4.Lagrange_Multiplier.md` | 拖约束边界 | 与 regularization 呼应 |
| P2 | 🧭 Token Compass | `11.RoPE.md` | 交换 token 看变化 | RoPE 姊妹实验 |

---

## 十四、技术落点（逐文件）

| 层 | 文件 | 改动 |
|---|---|---|
| 后端 | `lab/modules.py` | 新实验 MODULES 元数据（含 `group` 字段） |
| 后端 | `lab/<id>.py`（新建） | `compute(params)` / `lookup(params)` / `seeded_compute(params, seed)` |
| 后端 | `lab/router.py` | `_COMPUTERS` 注册；cached 实验走预计算数据加载 |
| 后端 | `lab/spikes/`（新建） | Double Descent 等重数值实验的独立 spike 脚本 |
| 前端 | `lab/modules/<Id>Lab.tsx`（新建） | 实验组件（canvas/svg） |
| 前端 | `lab/ChallengePanel.tsx`（新建） | **独立 Challenge 层**（冻结契约 #3） |
| 前端 | `lab/LabPage.tsx` | `LAB_COMPONENTS` 注册 + 侧栏按 group 分组 |
| 前端 | `lab/Controls.tsx` | **不改**（Challenge 不进入 ControlRow） |
| 前端 | `i18n/lab.ts` | 中英 meta + group 标签 |
| 前端 | `index.css` | `[data-scope='tlab']` 配色作用域 |
| 全局 | `types.ts` | `ExperimentRun` / `Hypothesis` / `Evidence` / `DiscoveryArtifact` 类型 |

**笔记锚点约定**：每个实验组件侧栏固定 Related Notes 区；笔记库（`ml/other/{zh,en}`）是理论依据，实验是操作入口，互相跳转。

---

## 十五、里程碑（阶段化，禁止「功能已定义 = 产品已完成」）

每个实验按以下阶段推进，**逐项勾选**，不整段打 [x]：

```
[ ] 1. Contract frozen      —— Protocol / Simulation / Challenge 契约声明齐
[ ] 2. Backend simulation   —— compute 跑通，数值正确
[ ] 3. Visual prototype     —— 实验仪器渲染可用
[ ] 4. Challenge            —— Predict 层接入，对错判定
[ ] 5. Notes                —— Related Notes 挂载
[ ] 6. Accessibility        —— 键盘/读屏基础可用
[ ] 7. i18n                 —— 中英双语
[ ] 8. Performance          —— 动画不卡顿，cached 生效
[ ] 9. tsc --noEmit         —— 类型检查通过
```

### M1（MVP = Sampling Machine 一个完整认知闭环）
- Question → Predict → Manipulate → Run → Observe → Evidence → Explain → Save Discovery → Next Question 全链路跑通
- 验证 Experiment Engine（三项契约落地）
- Rotary Observatory / Dangerous Mountain 依次补齐（不按数量推进，每个都验证一种学习模式）

### M2（Phase 2 三实验 + Journal / Artifact 最简版）

### M3（Phase 3，Token Society / Detective 优先）

### 全局验收
- 每个实验遵循冻结的三项契约
- 无 torch / transformers 运行时推理
- Discovery Graph 有真实 Question 连接出口，非孤立 mini-app
- 中英双语

---

## 十六、风险与待定决策（更新）

1. **DD spike 先行**（已冻结）：random Fourier features 若不稳定，退路是预计算 + 人工校验曲线，正式 Lab 用 cached 数据。
2. **Challenge 题型 v1 范围**：Phase 1 只做 L1 Predict（multiple choice），L2 Manipulate / L3 Explain 随实验迭代加入。
3. **Journal / Artifact 范围**：MVP 只做「预测 + 结果 + 对错 + Experiment Card 雏形」最简版，完整 Journal + MY DISCOVERIES 进 M2。
4. **Related Notes 跳转**：笔记在仓库外部，Phase 1 先站内 `/note/:id` 占位，Phase 3 定外链。
5. **tlab 作用域配色与潮汐切换入口**：配色已定 CSS 作用域；进入 tlab 是否保留潮汐切换入口，M1 评审定。
6. **Detective / Feature Hunt 证据**：算法生成 + 人工复核，10 案例起步（与真模型无关）。
7. **导航 group 字段**：后端元数据加 `group`，前端侧栏分组渲染——M1 前与 Sampling Machine 一起落地。
