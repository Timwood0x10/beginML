// Math Lab module metadata translations — keyed by backend module id and
// control key. Backend returns English metadata; the UI overlays these.

export interface LabMeta {
  title: string
  subtitle: string
  blurb: string
}

export const labModulesZh: Record<string, LabMeta> = {
  'gradient-descent': {
    title: '梯度下降',
    subtitle: 'SGD、动量与 Adam 的轨迹',
    blurb: '观察优化器在损失曲面上的行进。切换目标函数（Sphere / Rosenbrock / Rastrigin），调整学习率与动量，在等高线图上实时对比轨迹。',
  },
  activations: {
    title: '激活函数',
    subtitle: 'Sigmoid、tanh、ReLU、GELU 等',
    blurb: '对比常见激活函数及其导数。拖动探针点观察切线（局部梯度）沿曲线移动。',
  },
  losses: {
    title: '损失函数',
    subtitle: 'MSE、MAE、Huber、交叉熵、合页',
    blurb: '可视化回归与分类损失。移动探针点检查梯度，理解每种损失的独特行为。',
  },
  convolution: {
    title: '卷积',
    subtitle: '在 1D 信号上滑动核',
    blurb: '让核沿输入信号滑动，检查每个窗口的点积。可播放动画或手动拖动位置。',
  },
  'matrix-transform': {
    title: '矩阵变换',
    subtitle: '特征向量、行列式与线性映射',
    blurb: '把 2x2 矩阵作用到网格、单位正方形与圆上。实时查看特征向量、行列式、迹与秩。',
  },
  distributions: {
    title: '概率分布',
    subtitle: 'PDF、PMF 与采样直方图',
    blurb: '探索高斯、均匀、指数、拉普拉斯、二项与泊松分布。每条曲线都实时采样并绘制。',
  },
  entropy: {
    title: '熵与 KL 散度',
    subtitle: 'H(P)、H(P,Q) 与 KL(P||Q)',
    blurb: '观察熵、交叉熵与 KL 散度在两个伯努利分布靠近或远离时的变化。',
  },
  'neural-net': {
    title: '神经网络游乐场',
    subtitle: '在 2D 数据上训练两层 MLP',
    blurb: '在月亮、圆、螺旋或 XOR 数据上训练一个小型 numpy 神经网络。改变隐藏层大小、学习率与数据集，观察决策面与损失曲线实时更新。',
  },
  attention: {
    title: '自注意力',
    subtitle: 'Q·Kᵀ/√d 与注意力热力图',
    blurb: '一步步构建注意力权重矩阵。调整温度、切换因果掩码，观察 softmax 如何锐化或平滑每个查询的注意力分布。',
  },
  'transformer-training': {
    title: 'Transformer 训练',
    subtitle: '训练迷你 Transformer：损失曲线与注意力',
    blurb: '从零训练一个小型 decoder-only Transformer（预测前一个 token）。观察损失收敛、各头注意力热图逐步锐化成可解释的"回看一步"对角线模式。',
  },
  'agent-builder': {
    title: 'Agent 构建器',
    subtitle: '用积木搭建一个 Agent',
    blurb: '在记忆、工具、规划与多智能体协作中各选一个组件，看它们拼成一个完整 Agent：分层架构图、通俗的架构说明与可直接编辑的 YAML 配置。',
  },
  'agent-forge': {
    title: 'ARES Agent Lab',
    subtitle: 'Build. Break. Evolve.',
    blurb: '像乐高一样搭一个 Agent：拖认知积木、吸附语义端口、展开 Skill 积木盒、接入恢复积木。然后 RUN 观察它思考，用混沌弄坏它，看它如何自愈——编译成 YAML 或对比架构。',
  },
  pca: {
    title: 'PCA 与特征向量',
    subtitle: '主成分的几何',
    blurb: '生成相关 2D 数据，投影到主成分上。观察特征向量随数据的旋转。',
  },
  regularization: {
    title: 'L1 与 L2 正则化',
    subtitle: '约束几何与正则化路径',
    blurb: '可视化为什么 L1（Lasso）产生稀疏权重，而 L2（Ridge）平滑收缩。',
  },
  svm: {
    title: 'SVM 决策边界',
    subtitle: '最大间隔分类与核',
    blurb: '点击放置两类点，然后拟合支持向量机。切换线性 / RBF / 多项式核，滑动软间隔 C。',
  },
}

export const labModulesEn: Record<string, LabMeta> = {
  'gradient-descent': {
    title: 'Gradient Descent',
    subtitle: 'Trajectories of SGD, Momentum & Adam',
    blurb: 'Watch optimizers walk down a loss surface. Switch the objective (Sphere / Rosenbrock / Rastrigin), tune learning rate & momentum, and compare trajectories live on the contour map.',
  },
  activations: {
    title: 'Activation Functions',
    subtitle: 'Sigmoid, tanh, ReLU, GELU and friends',
    blurb: 'Compare common activation functions and their derivatives. Drag the evaluation point to see the tangent line (local gradient) move along the curve.',
  },
  losses: {
    title: 'Loss Functions',
    subtitle: 'MSE, MAE, Huber, cross-entropy, hinge',
    blurb: 'Visualize regression and classification losses. Move the probe point to inspect the gradient and see why each loss behaves the way it does.',
  },
  convolution: {
    title: 'Convolution',
    subtitle: 'Sliding kernels on 1D signals',
    blurb: 'Slide a kernel across an input signal and inspect each windowed dot product. Animate the kernel or scrub the position manually.',
  },
  'matrix-transform': {
    title: 'Matrix Transforms',
    subtitle: 'Eigenvectors, determinants and linear maps',
    blurb: 'Apply a 2x2 matrix to a grid, unit square and circle. Inspect eigenvectors, determinant, trace and rank in real time.',
  },
  distributions: {
    title: 'Distributions',
    subtitle: 'PDFs, PMFs and sampled histograms',
    blurb: 'Explore Gaussian, uniform, exponential, Laplace, binomial and Poisson distributions. Every curve is sampled and plotted live.',
  },
  entropy: {
    title: 'Entropy & KL Divergence',
    subtitle: 'H(P), H(P,Q) and KL(P||Q)',
    blurb: 'See how entropy, cross-entropy and KL divergence behave as two Bernoulli distributions approach or diverge from each other.',
  },
  'neural-net': {
    title: 'Neural Network Playground',
    subtitle: 'Train a 2-layer MLP on 2D data',
    blurb: 'Train a small numpy neural network on moons, circles, spirals or XOR data. Watch the decision surface and loss curve update as you change the hidden size, learning rate and dataset.',
  },
  attention: {
    title: 'Self-Attention',
    subtitle: 'Q*K^T / sqrt(d) and the attention heatmap',
    blurb: 'Build an attention weight matrix step by step. Tweak temperature, toggle causal masking and see how softmax sharpens the distribution.',
  },
  'transformer-training': {
    title: 'Transformer Training',
    subtitle: 'Train a tiny transformer: loss curve & attention',
    blurb: 'Train a small decoder-only transformer from scratch on a predict-the-previous-token task. Watch the loss converge and the per-head attention heatmaps sharpen into an interpretable one-step-back diagonal pattern.',
  },
  'agent-builder': {
    title: 'Agent Builder',
    subtitle: 'Assemble an agent from building blocks',
    blurb: 'Pick one option from memory, tools, planning and multi-agent coordination, and watch the pieces snap into a full agent: a layered diagram, a plain-English architecture summary and a ready-to-edit YAML config.',
  },
  'agent-forge': {
    title: 'ARES Agent Lab',
    subtitle: 'Build. Break. Evolve.',
    blurb: 'Build an agent like LEGO: drag cognitive bricks, snap their semantic ports, expand skill boxes, attach recovery bricks. Then RUN to watch it think, BREAK it with chaos, and let it recover — compile the result to YAML, or compare architectures.',
  },
  pca: {
    title: 'PCA & Eigenvectors',
    subtitle: 'The geometry of principal components',
    blurb: 'Generate correlated 2D data, then project it onto its principal components. See eigenvectors rotate with the data.',
  },
  regularization: {
    title: 'L1 vs L2 Regularization',
    subtitle: 'Constraint geometry & the regularization path',
    blurb: 'Visualize why L1 (Lasso) produces sparse weights and L2 (Ridge) shrinks them smoothly.',
  },
  svm: {
    title: 'SVM Decision Boundary',
    subtitle: 'Max-margin classification & kernels',
    blurb: 'Click to place two classes, then fit a support vector machine. Toggle linear / RBF / polynomial kernels and slide soft-margin C.',
  },
}

// Control labels keyed by control key (shared across modules).
export const controlLabelsZh: Record<string, string> = {
  objective: '目标函数', optimizer: '优化器', lr: '学习率', momentum: '动量',
  steps: '步数', start: '起点', function: '函数', point: '探针点 x',
  xMin: 'X 最小值', xMax: 'X 最大值', loss: '损失', target: '目标', probe: '探针',
  kernel: '核', length: '信号长度', preset: '预设', a: 'M[0,0]', b: 'M[0,1]',
  c: 'M[1,0]', d: 'M[1,1]', distribution: '分布', mu: '均值 (mu)',
  sigma: '标准差 (sigma)', rate: '速率 / lambda', n: '试验次数 (n)', p: '概率 (p)',
  seed: '随机种子', mode: '模式', k: '类别数 k', temperature: '温度',
  dataset: '数据集', hidden: '隐藏单元', epochs: '轮数', samples: '样本数',
  noise: '噪声', penalty: '惩罚', angle: '损失倾斜', optX: '最优点 x',
  optY: '最优点 y', gamma: 'RBF γ', degree: '多项式次数', reset: '重置数据点',
  causal: '因果掩码', components: '保留成分', correlation: '相关性', spread: '扩展比',
  tokens: '序列长度', layers: '层数', heads: '注意力头数',
  memory: '记忆', tools: '工具', planning: '规划', multi: '多智能体协作',
  task: '任务提示词',
  chaos_memory: '记忆不可用', chaos_tool: '工具超时', chaos_mcp: 'MCP 故障',
  chaos_llm: 'LLM 重试', chaos_context: '上下文溢出', compare: '与基线对比',
}

export const controlLabelsEn: Record<string, string> = {
  objective: 'Objective', optimizer: 'Optimizer', lr: 'Learning rate', momentum: 'Momentum',
  steps: 'Steps', start: 'Start point', function: 'Function', point: 'Probe point x',
  xMin: 'X min', xMax: 'X max', loss: 'Loss', target: 'Target', probe: 'Probe',
  kernel: 'Kernel', length: 'Signal length', preset: 'Preset', a: 'M[0,0]', b: 'M[0,1]',
  c: 'M[1,0]', d: 'M[1,1]', distribution: 'Distribution', mu: 'Mean (mu)',
  sigma: 'Std (sigma)', rate: 'Rate / lambda', n: 'Trials (n)', p: 'Prob (p)',
  seed: 'Seed', mode: 'Mode', k: 'Categories k', temperature: 'Temperature',
  dataset: 'Dataset', hidden: 'Hidden units', epochs: 'Epochs', samples: 'Samples',
  noise: 'Noise', penalty: 'Penalty', angle: 'Loss tilt', optX: 'Optimum x',
  optY: 'Optimum y', gamma: 'RBF gamma', degree: 'Poly degree', reset: 'Reset points',
  causal: 'Causal mask', components: 'Components kept', correlation: 'Correlation', spread: 'Spread ratio',
  tokens: 'Sequence length', layers: 'Layers', heads: 'Attention heads',
  memory: 'Memory', tools: 'Tools', planning: 'Planning', multi: 'Multi-agent',
  task: 'Task prompt',
  chaos_memory: 'Memory unavailable', chaos_tool: 'Tool timeout', chaos_mcp: 'MCP failure',
  chaos_llm: 'LLM retry', chaos_context: 'Context overflow', compare: 'Compare with baseline',
}
