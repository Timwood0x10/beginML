// verification.ts — multi-source knowledge verification (No-LLM).
//
// EVERY knowledge point of EVERY experiment is cross-checked here, one
// claim per point: the question, each explanation line and the aha
// takeaway each carry their own authoritative sources (papers / official
// docs), verified via web search. VerificationPanel renders them so a
// learner can confirm the knowledge before trusting it ("查证挑战").
// Copy is bilingual; URLs point to primary sources (arXiv / publisher /
// official documentation).

export type VerificationSourceKind = 'paper' | 'docs'

export interface VerificationSource {
  kind: VerificationSourceKind
  title: string
  titleEn: string
  url: string
}

export interface VerificationClaim {
  zh: string
  en: string
  sources: VerificationSource[]
}

export interface VerificationEntry {
  id: string
  claims: VerificationClaim[]
}

// ---- shared authoritative sources (cross-checked via web search) --------

const HF_TEXTGEN: VerificationSource = {
  kind: 'docs',
  title: 'HuggingFace 文本生成文档（temperature / top-k / top-p）',
  titleEn: 'HuggingFace text generation docs (temperature / top-k / top-p)',
  url: 'https://huggingface.co/docs/transformers/en/main_classes/text_generation',
}
const HF_LLM: VerificationSource = {
  kind: 'docs',
  title: 'HuggingFace LLM 教程：生成策略',
  titleEn: 'HuggingFace LLM tutorial: generation strategies',
  url: 'https://huggingface.co/docs/transformers/en/llm_tutorial',
}

const ROFORMER: VerificationSource = {
  kind: 'paper',
  title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding（Su et al., 2021, arXiv:2104.09864）',
  titleEn: 'RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021, arXiv:2104.09864)',
  url: 'https://arxiv.org/abs/2104.09864',
}
const HF_ROFORMER: VerificationSource = {
  kind: 'docs',
  title: 'HuggingFace RoFormer 模型文档',
  titleEn: 'HuggingFace RoFormer model documentation',
  url: 'https://huggingface.co/docs/transformers/model_doc/roformer',
}

const BELKIN: VerificationSource = {
  kind: 'paper',
  title: 'Reconciling modern machine-learning practice and the classical bias–variance trade-off（Belkin et al., 2019, arXiv:1812.11118）',
  titleEn: 'Reconciling modern machine-learning practice and the classical bias–variance trade-off (Belkin et al., 2019, arXiv:1812.11118)',
  url: 'https://arxiv.org/abs/1812.11118',
}
const NAKKIRAN: VerificationSource = {
  kind: 'paper',
  title: 'Deep Double Descent: Where Bigger Models and More Data Hurt（Nakkiran et al., 2019, arXiv:1912.02292）',
  titleEn: 'Deep Double Descent: Where Bigger Models and More Data Hurt (Nakkiran et al., 2019, arXiv:1912.02292)',
  url: 'https://arxiv.org/abs/1912.02292',
}

const DOMINGOS: VerificationSource = {
  kind: 'paper',
  title: 'A Unified Bias-Variance Decomposition for Zero-One and Squared Loss（Domingos, AAAI 2000）',
  titleEn: 'A Unified Bias-Variance Decomposition for Zero-One and Squared Loss (Domingos, AAAI 2000)',
  url: 'https://homes.cs.washington.edu/~pedrod/papers/aaai00.pdf',
}
const SKLEARN_BV: VerificationSource = {
  kind: 'docs',
  title: 'scikit-learn: Single estimator versus bagging — bias-variance decomposition',
  titleEn: 'scikit-learn: Single estimator versus bagging — bias-variance decomposition',
  url: 'https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html',
}

const BITNET: VerificationSource = {
  kind: 'paper',
  title: 'BitNet: Scaling 1-bit Transformers for Large Language Models（Wang et al., 2023, arXiv:2310.11453）',
  titleEn: 'BitNet: Scaling 1-bit Transformers for Large Language Models (Wang et al., 2023, arXiv:2310.11453)',
  url: 'https://arxiv.org/abs/2310.11453',
}
const BITNET_158: VerificationSource = {
  kind: 'paper',
  title: 'The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits（Ma et al., 2024, arXiv:2402.17764）',
  titleEn: 'The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (Ma et al., 2024, arXiv:2402.17764)',
  url: 'https://arxiv.org/abs/2402.17764',
}
const GPTQ: VerificationSource = {
  kind: 'paper',
  title: 'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers（Frantar et al., 2022, arXiv:2210.17323）',
  titleEn: 'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022, arXiv:2210.17323)',
  url: 'https://arxiv.org/abs/2210.17323',
}

const TC_FRAMEWORK: VerificationSource = {
  kind: 'docs',
  title: 'A Mathematical Framework for Transformer Circuits（Anthropic, 2021）',
  titleEn: 'A Mathematical Framework for Transformer Circuits (Anthropic, 2021)',
  url: 'https://transformer-circuits.pub/2021/framework/index.html',
}
const RESNET: VerificationSource = {
  kind: 'paper',
  title: 'Deep Residual Learning for Image Recognition（He et al., 2016, arXiv:1512.03385）',
  titleEn: 'Deep Residual Learning for Image Recognition (He et al., 2016, arXiv:1512.03385)',
  url: 'https://arxiv.org/abs/1512.03385',
}

const INDUCTION: VerificationSource = {
  kind: 'paper',
  title: 'In-context Learning and Induction Heads（Olsson et al., 2022, arXiv:2209.11895）',
  titleEn: 'In-context Learning and Induction Heads (Olsson et al., 2022, arXiv:2209.11895)',
  url: 'https://arxiv.org/abs/2209.11895',
}

const SHAZEER: VerificationSource = {
  kind: 'paper',
  title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer（Shazeer et al., 2017, arXiv:1701.06538）',
  titleEn: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017, arXiv:1701.06538)',
  url: 'https://arxiv.org/abs/1701.06538',
}
const SWITCH: VerificationSource = {
  kind: 'paper',
  title: 'Switch Transformers: Scaling to Trillion Parameter Models（Fedus, Zoph & Shazeer, 2021, arXiv:2101.03961）',
  titleEn: 'Switch Transformers: Scaling to Trillion Parameter Models (Fedus, Zoph & Shazeer, 2021, arXiv:2101.03961)',
  url: 'https://arxiv.org/abs/2101.03961',
}

const AIAYN: VerificationSource = {
  kind: 'paper',
  title: 'Attention Is All You Need（Vaswani et al., 2017, arXiv:1706.03762）',
  titleEn: 'Attention Is All You Need (Vaswani et al., 2017, arXiv:1706.03762)',
  url: 'https://arxiv.org/abs/1706.03762',
}
const MAMBA: VerificationSource = {
  kind: 'paper',
  title: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces（Gu & Dao, 2023, arXiv:2312.00752）',
  titleEn: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023, arXiv:2312.00752)',
  url: 'https://arxiv.org/abs/2312.00752',
}
const MAMBA_GITHUB: VerificationSource = {
  kind: 'docs',
  title: 'Mamba 官方仓库（state-spaces/mamba）',
  titleEn: 'Mamba official repository (state-spaces/mamba)',
  url: 'https://github.com/state-spaces/mamba',
}

const SUPERPOSITION: VerificationSource = {
  kind: 'paper',
  title: 'Toy Models of Superposition（Elhage et al., 2022, arXiv:2209.10652）',
  titleEn: 'Toy Models of Superposition (Elhage et al., 2022, arXiv:2209.10652)',
  url: 'https://arxiv.org/abs/2209.10652',
}
const TC_TOY: VerificationSource = {
  kind: 'docs',
  title: 'Toy Models of Superposition（Anthropic, 2022）',
  titleEn: 'Toy Models of Superposition (Anthropic, 2022)',
  url: 'https://transformer-circuits.pub/2022/toy_model/',
}

// ---- per-experiment knowledge points: ONE claim PER knowledge point -------
// Order matches the experiment UI: question, explanation lines, aha.

export const verificationData: Record<string, VerificationEntry> = {
  'sampling-machine': {
    id: 'sampling-machine',
    claims: [
      {
        zh: '为什么调高温度会让输出更随机？——因为温度缩放 softmax 分布。',
        en: 'Why does raising the temperature make outputs more random? — because temperature rescales the softmax distribution.',
        sources: [HF_TEXTGEN, HF_LLM],
      },
      {
        zh: 'temperature 作用于 softmax(logits / T)，并不改变 logits 本身。',
        en: 'Temperature acts on softmax(logits / T) — it never touches the logits themselves.',
        sources: [HF_TEXTGEN],
      },
      {
        zh: 'T 增大 → logits/T 变小 → softmax 分布更平坦。',
        en: 'Larger T → smaller logits/T → a flatter softmax distribution.',
        sources: [HF_TEXTGEN],
      },
      {
        zh: '更平坦的分布 → 熵更高 → 采样结果更多样。',
        en: 'A flatter distribution → higher entropy → more diverse samples.',
        sources: [HF_LLM],
      },
    ],
  },

  'rotary-observatory': {
    id: 'rotary-observatory',
    claims: [
      {
        zh: '为什么位置信息能被编码成旋转角度？——RoPE 用旋转矩阵编码绝对位置并融入相对位置。',
        en: 'Why can position be encoded as a rotation angle? — RoPE encodes absolute position via a rotation matrix and folds in relative position.',
        sources: [ROFORMER, HF_ROFORMER],
      },
      {
        zh: 'similarity = cos((m−n)·θ₀)，只依赖相对位置。',
        en: 'similarity = cos((m−n)·θ₀) — it depends only on the relative position.',
        sources: [ROFORMER],
      },
      {
        zh: '距离从 4 增到 8：相似度按 cos 曲线变化（随相对距离衰减）。',
        en: 'As the distance grows 4 → 8, similarity follows the cosine curve (decaying with relative distance).',
        sources: [ROFORMER],
      },
      {
        zh: '绝对旋转消失，相对旋转保留——这就是 RoPE。',
        en: 'Absolute rotation disappears, relative rotation remains — that is RoPE.',
        sources: [ROFORMER, HF_ROFORMER],
      },
      {
        zh: 'aha：RoPE 的关键特性——位置信息的乘法式编码与长序列扩展能力。',
        en: 'aha: the key property of RoPE — multiplicative position encoding and extension to long sequences.',
        sources: [ROFORMER],
      },
    ],
  },

  'dangerous-mountain': {
    id: 'dangerous-mountain',
    claims: [
      {
        zh: '更大的模型应该过拟合更严重——经典 bias-variance 的预言是真的吗？',
        en: 'Bigger models should overfit more — is the classical bias-variance prediction really true?',
        sources: [BELKIN, NAKKIRAN],
      },
      {
        zh: '经典 bias-variance 预言：模型越大 → 方差越大 → 测试误差只升不降（U 型曲线）。',
        en: 'Classical bias-variance predicts: bigger model → more variance → test error only grows (U-shaped curve).',
        sources: [BELKIN],
      },
      {
        zh: '现代观察：插值阈值处误差尖峰，过参数化区反而回落（双下降）。',
        en: 'Modern observation: error spikes at the interpolation threshold, then falls again in the overparameterized regime (double descent).',
        sources: [NAKKIRAN, BELKIN],
      },
      {
        zh: '最小范数解提供隐式正则化——更大的模型可以泛化得更好。',
        en: 'The minimum-norm solution acts as implicit regularization — bigger models can generalize better.',
        sources: [BELKIN],
      },
      {
        zh: 'aha：为什么更大的模型反而更好？——双下降统一了 U 型曲线与现代实践。',
        en: 'aha: why did the bigger model get better? — double descent reconciles the U-shaped curve with modern practice.',
        sources: [BELKIN, NAKKIRAN],
      },
    ],
  },

  'shooting-range': {
    id: 'shooting-range',
    claims: [
      {
        zh: '模型表现差——是偏差还是方差在捣鬼？',
        en: 'When a model performs badly — is it bias or variance?',
        sources: [DOMINGOS, SKLEARN_BV],
      },
      {
        zh: '低复杂度 → 模型太简单 → 偏差主导（欠拟合）。',
        en: 'Low complexity → model too simple → bias dominates (underfit).',
        sources: [DOMINGOS, SKLEARN_BV],
      },
      {
        zh: '高复杂度 + 样本少 → 模型记忆噪声 → 方差主导（过拟合）。',
        en: 'High complexity + few samples → model memorizes noise → variance dominates (overfit).',
        sources: [SKLEARN_BV],
      },
      {
        zh: 'MSE = Bias² + Variance + Noise²——拆开三股绳子，才知道该加容量、加数据还是接受噪声。',
        en: 'MSE = Bias² + Variance + Noise² — take the three strands apart to know whether to add capacity, add data, or accept the noise.',
        sources: [DOMINGOS, SKLEARN_BV],
      },
    ],
  },

  'weight-freezer': {
    id: 'weight-freezer',
    claims: [
      {
        zh: '压缩多少还值得？——取决于位宽与误差的权衡。',
        en: 'Where does compression stop being worth it? — it depends on the bit-width vs error trade-off.',
        sources: [GPTQ, BITNET],
      },
      {
        zh: 'FP32 → INT8 几乎无损，INT4 才开始可见误差。',
        en: 'FP32 → INT8 is nearly lossless; INT4 is where error becomes visible.',
        sources: [GPTQ],
      },
      {
        zh: 'INT2 时网格只剩 4 级，误差陡增——压缩的甜点通常在 INT4/INT8。',
        en: 'At INT2 the grid has only 4 levels and error jumps — the sweet spot is usually INT4/INT8.',
        sources: [GPTQ],
      },
      {
        zh: 'BitNet 1.58b 把权重钉在 {-1,0,+1}：内存降到 1.58 bit，用模型重训练换精度。',
        en: 'BitNet 1.58b pins weights to {-1, 0, +1}: memory drops to 1.58 bit, precision is bought back by retraining the model.',
        sources: [BITNET, BITNET_158],
      },
    ],
  },

  'representation-river': {
    id: 'representation-river',
    claims: [
      {
        zh: '一个 token 在模型里逐渐变成了什么？——沿着残差流累积各层注入。',
        en: 'What does a token gradually become inside the model? — it accumulates injections along the residual stream.',
        sources: [TC_FRAMEWORK, RESNET],
      },
      {
        zh: '残差流是一条河：每一层 x + Attn(x) + FFN(x) 往河里注入信息。',
        en: 'The residual stream is a river: every layer injects x + Attn(x) + FFN(x) into it.',
        sources: [TC_FRAMEWORK, RESNET],
      },
      {
        zh: 'Attention 支流把 token 彼此混合（谁关注谁），FFN 支流对每个 token 单独变换。',
        en: 'The Attention tributary mixes tokens (who attends to whom); the FFN tributary transforms each token alone.',
        sources: [TC_FRAMEWORK],
      },
      {
        zh: '本实验模拟表示演化（残差流注入机制），不检查真实模型。',
        en: 'This experiment models representation dynamics (residual-stream injection); it does not inspect a real model.',
        sources: [TC_FRAMEWORK],
      },
    ],
  },

  'token-society': {
    id: 'token-society',
    claims: [
      {
        zh: '一个句子里，token 如何互相注视？——通过多头注意力。',
        en: 'In a sentence, how do tokens look at each other? — through multi-head attention.',
        sources: [TC_FRAMEWORK, INDUCTION],
      },
      {
        zh: '每个 head 是一个不同的观察者：有的看近邻、有的看远处、有的回看一步。',
        en: 'Each head is a different kind of observer: some watch neighbours, some scan far, some look one step back.',
        sources: [TC_FRAMEWORK, INDUCTION],
      },
      {
        zh: '行为指纹从 attention 模式算出：avg_dist（平均距离）、local_ratio（近邻比例）、diag_ratio（回看一步强度）。',
        en: 'The fingerprint is computed from the attention pattern: avg_dist, local_ratio, diag_ratio.',
        sources: [TC_FRAMEWORK],
      },
      {
        zh: 'induction head 实现 [A][B] … [A] → [B] 补全，被认为是 in-context learning 的主要机制。',
        en: 'Induction heads implement [A][B] … [A] → [B] completion, hypothesized to be the main mechanism of in-context learning.',
        sources: [INDUCTION],
      },
    ],
  },

  'transformer-detective': {
    id: 'transformer-detective',
    claims: [
      {
        zh: '为什么模型在这里预测了这个？——通过注意力与特征的归因分析。',
        en: 'Why did the model predict this here? — via attribution over attention and features.',
        sources: [TC_FRAMEWORK],
      },
      {
        zh: '证据由确定性程序生成：attention 权重与特征强度加权排序（不检查真实模型）。',
        en: 'Evidence is algorithmically generated: attention weights and feature strength are ranked (no real model inspected).',
        sources: [TC_FRAMEWORK],
      },
      {
        zh: '最有力的嫌疑人是排名第一的证据——它把目标 token 抬进候选池。',
        en: 'The strongest suspect is the top-ranked evidence — it lifts the target token into the candidate pool.',
        sources: [TC_FRAMEWORK],
      },
      {
        zh: 'CASE CLOSED 展示影响力排序：head/feature 按得分降序。',
        en: 'CASE CLOSED shows the influence ranking: heads/features sorted by score descending.',
        sources: [TC_FRAMEWORK, INDUCTION],
      },
    ],
  },

  'moe-expert-routing': {
    id: 'moe-expert-routing',
    claims: [
      {
        zh: '一个 token 会被哪些专家接走？——由门控网络的路由权重决定。',
        en: 'Which experts pick up this token? — decided by the gating network\'s routing weights.',
        sources: [SHAZEER, SWITCH],
      },
      {
        zh: '门控网络 softmax(E·W_gate) 决定每个 token 的路由权重。',
        en: 'A gate network softmax(E·W_gate) decides each token\'s routing weights.',
        sources: [SHAZEER],
      },
      {
        zh: 'top-k 路由只保留最强的 k 条连接，其余置零并重归一化——路由变得稀疏。',
        en: 'Top-k routing keeps only the strongest k connections, zeroes the rest and renormalizes — routing becomes sparse.',
        sources: [SHAZEER, SWITCH],
      },
      {
        zh: '每个专家的负载 = 它接收的路由权重之和；负载差异大时专家忙闲不均（可用辅助损失均衡）。',
        en: 'Each expert\'s load is the routing weight it receives; big gaps mean busy vs idle experts (an auxiliary loss can balance them).',
        sources: [SWITCH],
      },
    ],
  },

  'mamba-memory-race': {
    id: 'mamba-memory-race',
    claims: [
      {
        zh: '序列变长时，为什么 Transformer 会比 Mamba 慢这么多？——复杂度不同。',
        en: 'Why does a Transformer slow down so much more than Mamba as the sequence grows? — different complexity.',
        sources: [AIAYN, MAMBA],
      },
      {
        zh: 'Transformer attention 每层做 O(L²) 工作：每个 token 都要回看所有前序 token。',
        en: 'Transformer attention does O(L²) work per layer: every token looks back at every previous token.',
        sources: [AIAYN],
      },
      {
        zh: 'Mamba 式 SSM 保持固定大小的状态，每个 token 只触碰状态一次：O(L)。',
        en: 'A Mamba-style SSM keeps a fixed-size state; each token touches it once: O(L).',
        sources: [MAMBA, MAMBA_GITHUB],
      },
      {
        zh: 'L 翻 8 倍时，Transformer 的 FLOPs 翻 64 倍，Mamba 只翻 8 倍——这就是二次 vs 线性。',
        en: 'When L grows 8×, Transformer FLOPs grow 64× while Mamba grows only 8× — quadratic vs linear.',
        sources: [AIAYN, MAMBA],
      },
    ],
  },

  'transformer-mri': {
    id: 'transformer-mri',
    claims: [
      {
        zh: '扫描仪下，模型内部是健康的吗？——看 loss / 熵 / 梯度范数三个通道。',
        en: 'Under the scanner, is the model healthy? — watch the loss / entropy / gradient-norm channels.',
        sources: [RESNET, TC_FRAMEWORK],
      },
      {
        zh: '扫描仪扫过三个通道：loss、表示 entropy、梯度范数。',
        en: 'The scanner sweeps three channels: loss, representation entropy, gradient norm.',
        sources: [TC_FRAMEWORK],
      },
      {
        zh: '健康信号：loss 收敛、entropy 上升（信息在积累）、梯度保持有界。',
        en: 'Healthy signs: loss converging, entropy rising (information accumulating), gradients bounded.',
        sources: [TC_FRAMEWORK],
      },
      {
        zh: '梯度消失病理：深层梯度范数趋近于零——深层表示不再被更新。',
        en: 'Vanishing-gradient pathology: deep-layer gradient norms collapse to near zero — deep representations stop updating.',
        sources: [RESNET],
      },
    ],
  },

  'feature-hunt': {
    id: 'feature-hunt',
    claims: [
      {
        zh: '噪声堆里，哪些神经元是真正的特征？——看激活模式是否语义一致。',
        en: 'In the noise pile, which neurons are real features? — look for semantically consistent activation.',
        sources: [SUPERPOSITION, TC_TOY],
      },
      {
        zh: '真 feature 对某个语义组的 token 强烈且一致地激活（> 0.75）。',
        en: 'A real feature fires strongly and consistently on one semantic group of tokens (> 0.75).',
        sources: [SUPERPOSITION, TC_TOY],
      },
      {
        zh: '噪声神经元在所有 token 上只有随机低激活（< 0.2）——这就是"稀疏激活"。',
        en: 'Noise neurons have only random low activations across all tokens (< 0.2) — that is sparse activation.',
        sources: [TC_TOY, SUPERPOSITION],
      },
      {
        zh: '神经元可能是多义的（polysemantic）：稀疏特征以 superposition 形式叠加存储，代价是干扰。',
        en: 'Neurons can be polysemantic: sparse features are stored in superposition at the cost of interference.',
        sources: [SUPERPOSITION],
      },
    ],
  },
}
