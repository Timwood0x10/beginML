"""Metadata for every interactive Math Lab module (drives the sidebar)."""

from typing import Any

MODULES: list[dict[str, Any]] = [
    {
        "id": "gradient-descent",
        "title": "Gradient Descent",
        "subtitle": "Trajectories of SGD, Momentum & Adam",
        "icon": "trending_down",
        "category": "Optimization",
        "blurb": (
            "Watch optimizers walk down a loss surface. Switch the objective "
            "(Sphere / Rosenbrock / Rastrigin), tune learning rate & momentum, "
            "and compare trajectories live on the contour map."
        ),
        "controls": [
            {
                "key": "objective",
                "label": "Objective",
                "type": "select",
                "options": ["sphere", "rosenbrock", "rastrigin", "beale"],
            },
            {
                "key": "optimizer",
                "label": "Optimizer",
                "type": "select",
                "options": [
                    "sgd",
                    "momentum",
                    "adam",
                    "nesterov",
                    "rmsprop",
                    "adagrad",
                ],
            },
            {
                "key": "lr",
                "label": "Learning rate",
                "type": "range",
                "min": 0.0005,
                "max": 0.2,
                "step": 0.0005,
                "default": 0.02,
            },
            {
                "key": "momentum",
                "label": "Momentum",
                "type": "range",
                "min": 0.0,
                "max": 0.99,
                "step": 0.01,
                "default": 0.9,
            },
            {
                "key": "steps",
                "label": "Steps",
                "type": "range",
                "min": 10,
                "max": 200,
                "step": 1,
                "default": 80,
            },
            {
                "key": "start",
                "label": "Start point",
                "type": "select",
                "options": ["random", "far", "corner"],
            },
        ],
    },
    {
        "id": "activations",
        "title": "Activation Functions",
        "subtitle": "Sigmoid, tanh, ReLU, GELU and friends",
        "icon": "show_chart",
        "category": "Neural Networks",
        "blurb": (
            "Compare common activation functions and their derivatives. Drag "
            "the evaluation point to see the tangent line (local gradient) "
            "move along the curve."
        ),
        "controls": [
            {
                "key": "function",
                "label": "Function",
                "type": "select",
                "options": [
                    "relu",
                    "sigmoid",
                    "tanh",
                    "leaky-relu",
                    "gelu",
                    "swish",
                    "softplus",
                ],
            },
            {
                "key": "point",
                "label": "Probe point x",
                "type": "range",
                "min": -5.0,
                "max": 5.0,
                "step": 0.1,
                "default": 1.0,
            },
            {
                "key": "xMin",
                "label": "X min",
                "type": "range",
                "min": -10.0,
                "max": 0.0,
                "step": 0.5,
                "default": -6.0,
            },
            {
                "key": "xMax",
                "label": "X max",
                "type": "range",
                "min": 0.0,
                "max": 10.0,
                "step": 0.5,
                "default": 6.0,
            },
        ],
    },
    {
        "id": "losses",
        "title": "Loss Functions",
        "subtitle": "MSE, MAE, Huber, cross-entropy, hinge",
        "icon": "water_drop",
        "category": "Optimization",
        "blurb": (
            "Visualize regression and classification losses. Move the probe "
            "point to inspect the gradient and see why each loss behaves the "
            "way it does."
        ),
        "controls": [
            {
                "key": "loss",
                "label": "Loss",
                "type": "select",
                "options": ["mse", "mae", "huber", "bce", "hinge", "cross-entropy"],
            },
            {
                "key": "target",
                "label": "Target",
                "type": "range",
                "min": -2.0,
                "max": 2.0,
                "step": 0.1,
                "default": 0.0,
            },
            {
                "key": "probe",
                "label": "Probe",
                "type": "range",
                "min": -3.0,
                "max": 3.0,
                "step": 0.1,
                "default": 0.8,
            },
        ],
    },
    {
        "id": "convolution",
        "title": "Convolution",
        "subtitle": "Sliding kernels on 1D signals",
        "icon": "stacked_bar_chart",
        "category": "Signal Processing",
        "blurb": (
            "Slide a kernel across an input signal and inspect each windowed "
            "dot product. Animate the kernel or scrub the position manually."
        ),
        "controls": [
            {
                "key": "kernel",
                "label": "Kernel",
                "type": "select",
                "options": [
                    "edge-detect",
                    "identity",
                    "gaussian",
                    "box-blur",
                    "sharpen",
                    "emboss",
                ],
            },
            {
                "key": "length",
                "label": "Signal length",
                "type": "range",
                "min": 16,
                "max": 64,
                "step": 1,
                "default": 32,
            },
        ],
    },
    {
        "id": "matrix-transform",
        "title": "Matrix Transforms",
        "subtitle": "Eigenvectors, determinants and linear maps",
        "icon": "grid_view",
        "category": "Linear Algebra",
        "blurb": (
            "Apply a 2x2 matrix to a grid, unit square and circle. Inspect "
            "eigenvectors, determinant, trace and rank in real time."
        ),
        "controls": [
            {
                "key": "preset",
                "label": "Preset",
                "type": "select",
                "options": [
                    "identity",
                    "rotation-45",
                    "rotation-90",
                    "scale-2x",
                    "scale-aniso",
                    "shear",
                    "reflection-x",
                    "projection",
                    "custom",
                ],
            },
            {
                "key": "a",
                "label": "M[0,0]",
                "type": "range",
                "min": -2.0,
                "max": 2.0,
                "step": 0.05,
                "default": 1.0,
            },
            {
                "key": "b",
                "label": "M[0,1]",
                "type": "range",
                "min": -2.0,
                "max": 2.0,
                "step": 0.05,
                "default": 0.0,
            },
            {
                "key": "c",
                "label": "M[1,0]",
                "type": "range",
                "min": -2.0,
                "max": 2.0,
                "step": 0.05,
                "default": 0.0,
            },
            {
                "key": "d",
                "label": "M[1,1]",
                "type": "range",
                "min": -2.0,
                "max": 2.0,
                "step": 0.05,
                "default": 1.0,
            },
        ],
    },
    {
        "id": "distributions",
        "title": "Distributions",
        "subtitle": "PDFs, PMFs and sampled histograms",
        "icon": "bar_chart",
        "category": "Probability",
        "blurb": (
            "Explore Gaussian, uniform, exponential, Laplace, binomial and "
            "Poisson distributions. Every curve is sampled and plotted live."
        ),
        "controls": [
            {
                "key": "distribution",
                "label": "Distribution",
                "type": "select",
                "options": [
                    "gaussian",
                    "uniform",
                    "exponential",
                    "laplace",
                    "binomial",
                    "poisson",
                ],
            },
            {
                "key": "mu",
                "label": "Mean (mu)",
                "type": "range",
                "min": -3.0,
                "max": 3.0,
                "step": 0.1,
                "default": 0.0,
            },
            {
                "key": "sigma",
                "label": "Std (sigma)",
                "type": "range",
                "min": 0.1,
                "max": 3.0,
                "step": 0.05,
                "default": 1.0,
            },
            {
                "key": "rate",
                "label": "Rate / lambda",
                "type": "range",
                "min": 0.1,
                "max": 5.0,
                "step": 0.1,
                "default": 1.0,
            },
            {
                "key": "n",
                "label": "Trials (n)",
                "type": "range",
                "min": 1,
                "max": 40,
                "step": 1,
                "default": 20,
            },
            {
                "key": "p",
                "label": "Prob (p)",
                "type": "range",
                "min": 0.01,
                "max": 0.99,
                "step": 0.01,
                "default": 0.5,
            },
            {
                "key": "seed",
                "label": "Seed",
                "type": "range",
                "min": 0,
                "max": 99,
                "step": 1,
                "default": 0,
            },
        ],
    },
    {
        "id": "entropy",
        "title": "Entropy & KL Divergence",
        "subtitle": "H(P), H(P,Q) and KL(P||Q)",
        "icon": "waves",
        "category": "Information Theory",
        "blurb": (
            "See how entropy, cross-entropy and KL divergence behave as two "
            "Bernoulli distributions approach or diverge from each other."
        ),
        "controls": [
            {
                "key": "mode",
                "label": "Mode",
                "type": "select",
                "options": ["bernoulli", "categorical"],
            },
            {
                "key": "p",
                "label": "P (true)",
                "type": "range",
                "min": 0.01,
                "max": 0.99,
                "step": 0.01,
                "default": 0.7,
            },
            {
                "key": "k",
                "label": "Categories k",
                "type": "range",
                "min": 2,
                "max": 10,
                "step": 1,
                "default": 5,
            },
            {
                "key": "temperature",
                "label": "Q temperature",
                "type": "range",
                "min": 0.1,
                "max": 3.0,
                "step": 0.05,
                "default": 1.0,
            },
            {
                "key": "seed",
                "label": "Seed",
                "type": "range",
                "min": 0,
                "max": 99,
                "step": 1,
                "default": 1,
            },
        ],
    },
    {
        "id": "neural-net",
        "title": "Neural Network Playground",
        "subtitle": "Train a 2-layer MLP on 2D data",
        "icon": "bubble_chart",
        "category": "Neural Networks",
        "blurb": (
            "Train a small numpy neural network on moons, circles, spirals or "
            "XOR data. Watch the decision surface and loss curve update as you "
            "change the hidden size, learning rate and dataset."
        ),
        "controls": [
            {
                "key": "dataset",
                "label": "Dataset",
                "type": "select",
                "options": ["moons", "circles", "spiral", "xor", "blobs"],
            },
            {
                "key": "hidden",
                "label": "Hidden units",
                "type": "range",
                "min": 2,
                "max": 32,
                "step": 1,
                "default": 8,
            },
            {
                "key": "lr",
                "label": "Learning rate",
                "type": "range",
                "min": 0.05,
                "max": 2.0,
                "step": 0.05,
                "default": 0.5,
            },
            {
                "key": "epochs",
                "label": "Epochs",
                "type": "range",
                "min": 20,
                "max": 500,
                "step": 10,
                "default": 200,
            },
            {
                "key": "samples",
                "label": "Samples",
                "type": "range",
                "min": 40,
                "max": 300,
                "step": 10,
                "default": 120,
            },
            {
                "key": "noise",
                "label": "Noise",
                "type": "range",
                "min": 0.0,
                "max": 0.5,
                "step": 0.02,
                "default": 0.0,
            },
            {
                "key": "seed",
                "label": "Seed",
                "type": "range",
                "min": 0,
                "max": 99,
                "step": 1,
                "default": 1,
            },
        ],
    },
    {
        "id": "attention",
        "title": "Self-Attention",
        "subtitle": "Q*K^T / sqrt(d) and the attention heatmap",
        "icon": "grid_on",
        "category": "Architectures",
        "blurb": (
            "Build an attention weight matrix step by step. Tweak temperature, "
            "toggle causal masking and see how softmax sharpens the distribution."
        ),
        "controls": [
            {
                "key": "tokens",
                "label": "Tokens",
                "type": "range",
                "min": 3,
                "max": 10,
                "step": 1,
                "default": 6,
            },
            {
                "key": "temperature",
                "label": "Temperature (1/sqrt d)",
                "type": "range",
                "min": 0.1,
                "max": 4.0,
                "step": 0.05,
                "default": 1.0,
            },
            {
                "key": "causal",
                "label": "Causal mask",
                "type": "toggle",
                "default": True,
            },
            {
                "key": "seed",
                "label": "Random seed",
                "type": "range",
                "min": 0,
                "max": 99,
                "step": 1,
                "default": 7,
            },
        ],
    },
    {
        "id": "transformer-training",
        "title": "Transformer Training",
        "subtitle": "Train a tiny transformer: loss curve & attention",
        "icon": "model_training",
        "category": "Architectures",
        "blurb": (
            "Train a small decoder-only transformer from scratch on a "
            "predict-the-previous-token task. The model has d_model=16, so the "
            "number of heads must divide 16 (1/2/4/8). Watch the loss converge "
            "and the per-head attention heatmaps sharpen into an interpretable "
            "pattern."
        ),
        "controls": [
            {
                "key": "tokens",
                "label": "Sequence length",
                "type": "range",
                "min": 4,
                "max": 12,
                "step": 1,
                "default": 8,
            },
            {
                "key": "layers",
                "label": "Layers",
                "type": "range",
                "min": 1,
                "max": 3,
                "step": 1,
                "default": 2,
            },
            {
                "key": "heads",
                "label": "Attention heads (divisor of 16)",
                "type": "select",
                "options": ["1", "2", "4", "8"],
                "default": "2",
            },
            {
                "key": "lr",
                "label": "Learning rate",
                "type": "range",
                "min": 0.005,
                "max": 0.3,
                "step": 0.005,
                "default": 0.05,
            },
            {
                "key": "epochs",
                "label": "Training steps",
                "type": "range",
                "min": 20,
                "max": 250,
                "step": 10,
                "default": 100,
            },
            {
                "key": "causal",
                "label": "Causal mask",
                "type": "toggle",
                "default": True,
            },
            {
                "key": "seed",
                "label": "Random seed",
                "type": "range",
                "min": 0,
                "max": 99,
                "step": 1,
                "default": 3,
            },
        ],
    },
    {
        "id": "agent-forge",
        "title": "ARES Agent Lab",
        "subtitle": "Build. Break. Evolve.",
        "icon": "construction",
        "category": "Architectures",
        "blurb": (
            "Build an agent like LEGO: drag cognitive bricks, snap their "
            "semantic ports, expand skill boxes, attach recovery bricks. Then "
            "RUN to watch it think, BREAK it with chaos, and let it recover — "
            "compile the result to YAML, or compare architectures."
        ),
        "controls": [
            {"key": "preset", "label": "Preset architecture", "type": "select",
             "options": ["simple", "rag", "tiered", "multi-agent", "empty"],
             "default": "simple"},
            {"key": "task", "label": "Task prompt", "type": "select",
             "options": [
                 "分析这个项目的 FFI 安全问题",
                 "审查这段代码的并发缺陷",
                 "调研多模态 Agent 的最新进展",
                 "为一款产品设计评测方案",
             ],
             "default": "分析这个项目的 FFI 安全问题"},
            {"key": "chaos_memory", "label": "Memory unavailable", "type": "toggle", "default": False},
            {"key": "chaos_tool", "label": "Tool timeout", "type": "toggle", "default": False},
            {"key": "chaos_mcp", "label": "MCP failure", "type": "toggle", "default": False},
            {"key": "chaos_llm", "label": "LLM retry", "type": "toggle", "default": False},
            {"key": "chaos_context", "label": "Context overflow", "type": "toggle", "default": False},
            {"key": "compare", "label": "Compare with baseline", "type": "toggle", "default": False},
        ],
    },
    {
        "id": "pca",
        "title": "PCA & Eigenvectors",
        "subtitle": "The geometry of principal components",
        "icon": "center_focus_strong",
        "category": "Linear Algebra",
        "blurb": (
            "Generate correlated 2D data, then project it onto its principal "
            "components. See eigenvectors rotate with the data."
        ),
        "controls": [
            {
                "key": "samples",
                "label": "Samples",
                "type": "range",
                "min": 30,
                "max": 400,
                "step": 10,
                "default": 150,
            },
            {
                "key": "correlation",
                "label": "Correlation",
                "type": "range",
                "min": -0.95,
                "max": 0.95,
                "step": 0.05,
                "default": 0.75,
            },
            {
                "key": "spread",
                "label": "Spread ratio",
                "type": "range",
                "min": 0.1,
                "max": 3.0,
                "step": 0.1,
                "default": 1.6,
            },
            {
                "key": "components",
                "label": "Components kept",
                "type": "range",
                "min": 0,
                "max": 2,
                "step": 1,
                "default": 1,
            },
            {
                "key": "seed",
                "label": "Random seed",
                "type": "range",
                "min": 0,
                "max": 99,
                "step": 1,
                "default": 3,
            },
        ],
    },
    {
        "id": "regularization",
        "title": "L1 vs L2 Regularization",
        "subtitle": "Constraint geometry & the regularization path",
        "icon": "circle",
        "category": "Optimization",
        "blurb": (
            "Visualize why L1 (Lasso) produces sparse weights and L2 (Ridge) "
            "shrinks them smoothly."
        ),
        "controls": [
            {
                "key": "penalty",
                "label": "Penalty",
                "type": "select",
                "options": ["l1", "l2", "both"],
            },
            {
                "key": "c",
                "label": "Constraint radius C",
                "type": "range",
                "min": 0.1,
                "max": 3.0,
                "step": 0.05,
                "default": 1.2,
            },
            {
                "key": "angle",
                "label": "Loss tilt",
                "type": "range",
                "min": 0,
                "max": 3.14,
                "step": 0.02,
                "default": 0.6,
            },
            {
                "key": "optX",
                "label": "Optimum x",
                "type": "range",
                "min": 0.2,
                "max": 2.8,
                "step": 0.05,
                "default": 2.0,
            },
            {
                "key": "optY",
                "label": "Optimum y",
                "type": "range",
                "min": 0.2,
                "max": 2.8,
                "step": 0.05,
                "default": 1.6,
            },
        ],
    },
    {
        "id": "svm",
        "title": "SVM Decision Boundary",
        "subtitle": "Max-margin classification & kernels",
        "icon": "linear_scale",
        "category": "Classification",
        "blurb": (
            "Click to place two classes, then fit a support vector machine. "
            "Toggle linear / RBF / polynomial kernels and slide soft-margin C."
        ),
        "controls": [
            {
                "key": "kernel",
                "label": "Kernel",
                "type": "select",
                "options": ["linear", "rbf", "poly"],
            },
            {
                "key": "c",
                "label": "Soft-margin C",
                "type": "range",
                "min": 0.05,
                "max": 50.0,
                "step": 0.05,
                "default": 1.0,
            },
            {
                "key": "gamma",
                "label": "RBF gamma",
                "type": "range",
                "min": 0.05,
                "max": 5.0,
                "step": 0.05,
                "default": 0.8,
            },
            {
                "key": "degree",
                "label": "Poly degree",
                "type": "range",
                "min": 2,
                "max": 5,
                "step": 1,
                "default": 3,
            },
            {"key": "reset", "label": "Reset points", "type": "action"},
        ],
    },
    {
        "id": "sampling-machine",
        "title": "Sampling Machine",
        "subtitle": "Temperature, Top-k & Top-p decode logits",
        "icon": "casino",
        "category": "Inference",
        "group": "now-experimenting",
        "blurb": (
            "A probability slot machine. Watch how temperature rescales the "
            "softmax input (logits themselves stay untouched), how the filter "
            "gate admits only top-k or top-p tokens, and how the sampler "
            "draws balls into token slots. Predict first, then run."
        ),
        "question": "为什么调高温度会让输出更随机？",
        "next_question": "什么是熵？",
        "next_experiment": "entropy",
        "controls": [
            {
                "key": "logits_mode",
                "label": "Logits shape",
                "type": "select",
                "options": ["multi", "spiky", "flat"],
            },
            {
                "key": "temperature",
                "label": "Temperature",
                "type": "range",
                "min": 0.1,
                "max": 3.0,
                "step": 0.05,
                "default": 1.0,
            },
            {
                "key": "gate",
                "label": "Filter gate",
                "type": "select",
                "options": ["none", "top-k", "top-p", "both"],
            },
            {
                "key": "top_k",
                "label": "Top-k",
                "type": "range",
                "min": 1,
                "max": 8,
                "step": 1,
                "default": 3,
            },
            {
                "key": "top_p",
                "label": "Top-p",
                "type": "range",
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
                "default": 0.9,
            },
            {"key": "sample", "label": "SAMPLE ×20", "type": "action"},
        ],
    },
    {
        "id": "rotary-observatory",
        "title": "Rotary Observatory",
        "subtitle": "RoPE: positions become rotation phases",
        "icon": "rotate_right",
        "category": "Positional Encoding",
        "group": "rotary-observatory",
        "blurb": (
            "Watch a token star rotate on the complex plane as its position "
            "grows — then add a second star and discover that only the "
            "relative phase matters. Rotate together, find the distance, and "
            "inspect the frequency lens. Predict first, then run."
        ),
        "question": "为什么位置信息能被编码成旋转角度？",
        "next_question": "RoPE 如何注入 Transformer？",
        "next_experiment": "token-society",
        "controls": [
            {
                "key": "position",
                "label": "Position",
                "type": "range",
                "min": 0,
                "max": 32,
                "step": 1,
                "default": 8,
            },
            {
                "key": "position_b",
                "label": "Position B",
                "type": "range",
                "min": 0,
                "max": 32,
                "step": 1,
                "default": 4,
            },
            {
                "key": "frequency",
                "label": "Frequency",
                "type": "range",
                "min": 0.25,
                "max": 4.0,
                "step": 0.05,
                "default": 1.0,
            },
            {
                "key": "dims",
                "label": "Dimensions",
                "type": "range",
                "min": 4,
                "max": 32,
                "step": 4,
                "default": 8,
            },
            {
                "key": "pair",
                "label": "Two tokens",
                "type": "toggle",
                "default": True,
            },
            {
                "key": "distance_mode",
                "label": "Find the distance",
                "type": "toggle",
                "default": True,
            },
        ],
    },
    {
        "id": "dangerous-mountain",
        "title": "Dangerous Mountain",
        "subtitle": "Double descent: bigger models can generalize better",
        "icon": "terrain",
        "category": "Learning Dynamics",
        "group": "learning-dynamics",
        "blurb": (
            "A mountain of model capacity. Drag the capacity slider across "
            "the interpolation threshold and watch test error spike into the "
            "danger zone — then fall again as the model gets bigger. "
            "Classical bias-variance says the mountain only grows; the "
            "modern view says the opposite. Predict first, then run."
        ),
        "question": "更大的模型应该过拟合更严重——你信吗？",
        "next_question": "经典 bias-variance 在这里预言了什么？",
        "next_experiment": "shooting-range",
        "controls": [
            {
                "key": "capacity",
                "label": "Model capacity",
                "type": "range",
                "min": 2,
                "max": 300,
                "step": 2,
                "default": 64,
            },
            {
                "key": "samples",
                "label": "Samples",
                "type": "select",
                "options": ["32", "64", "96"],
            },
            {
                "key": "noise",
                "label": "Label noise",
                "type": "range",
                "min": 0.2,
                "max": 0.6,
                "step": 0.05,
                "default": 0.4,
            },
        ],
    },
    {
        "id": "shooting-range",
        "title": "Shooting Range",
        "subtitle": "Bias² + Variance + Noise², taken apart",
        "icon": "track_changes",
        "category": "Learning Dynamics",
        "group": "learning-dynamics",
        "blurb": (
            "A firing range for models. Each bootstrap fit is one shot; the "
            "bullseye is the true function. Scatter is variance, offset is "
            "bias, the irreducible ring is noise. Drag complexity and sample "
            "count to watch the three terms trade off — then feed 500 "
            "samples and watch variance shrink."
        ),
        "question": "模型表现差——是偏差还是方差在捣鬼？",
        "next_question": "正则化如何权衡偏差与方差？",
        "next_experiment": "regularization",
        "controls": [
            {
                "key": "complexity",
                "label": "Model complexity",
                "type": "range",
                "min": 1,
                "max": 15,
                "step": 1,
                "default": 3,
            },
            {
                "key": "samples",
                "label": "Samples",
                "type": "select",
                "options": ["10", "50", "500"],
            },
            {
                "key": "noise",
                "label": "Label noise",
                "type": "range",
                "min": 0.1,
                "max": 0.4,
                "step": 0.05,
                "default": 0.2,
            },
        ],
    },
    {
        "id": "weight-freezer",
        "title": "Weight Freezer",
        "subtitle": "Quantization: compression vs error",
        "icon": "ac_unit",
        "category": "Model Efficiency",
        "group": "model-efficiency",
        "blurb": (
            "A freezer for weights. Float them free, then FREEZE them onto a "
            "coarser and coarser grid — FP32 → INT8 → INT4 → INT2 → TERNARY. "
            "Memory drops, error climbs. Shake the cloud and find the Pareto "
            "sweet spot: where does compression stop being worth it?"
        ),
        "question": "压缩多少还值得？",
        "next_question": "BitNet 的 1.58 bit 为什么可行？",
        "next_experiment": "mamba-memory-race",
        "controls": [
            {
                "key": "bit_width",
                "label": "Bit width",
                "type": "range",
                "min": 2,
                "max": 16,
                "step": 2,
                "default": 8,
            },
            {
                "key": "ternary",
                "label": "BitNet 1.58b mode",
                "type": "toggle",
                "default": False,
            },
            {"key": "reshuffle", "label": "SHAKE WEIGHTS", "type": "action"},
        ],
    },
    {
        "id": "representation-river",
        "title": "Representation River",
        "subtitle": "The residual stream: how a token's meaning flows",
        "icon": "water",
        "category": "Model Behavior",
        "group": "model-behavior",
        "blurb": (
            "A river runs through the Transformer: the residual stream. "
            "Attention and FFN are tributaries injecting information at "
            "every layer. Pick a token and watch its representation drift "
            "downstream — SIMULATION MODE: this models dynamics, it does not "
            "inspect a real model."
        ),
        "question": "一个 token 在模型里逐渐变成了什么？",
        "next_question": "注意力如何决定谁影响谁？",
        "next_experiment": "token-society",
        "controls": [
            {
                "key": "tokens",
                "label": "Tokens",
                "type": "range",
                "min": 4,
                "max": 10,
                "step": 1,
                "default": 6,
            },
            {
                "key": "layer",
                "label": "Layers",
                "type": "range",
                "min": 1,
                "max": 12,
                "step": 1,
                "default": 12,
            },
            {
                "key": "show",
                "label": "Injection",
                "type": "select",
                "options": ["all", "attention", "ffn"],
            },
        ],
    },
    {
        "id": "token-society",
        "title": "Token Society",
        "subtitle": "Who looks at whom in a sentence",
        "icon": "diversity_3",
        "category": "Model Behavior",
        "group": "model-behavior",
        "blurb": (
            "A sentence is a small society. Each token is a member; each "
            "attention head is a different kind of observer — the Repeater, "
            "the Long-Distance Scout, the Nearby One. Click a token to see "
            "who it listens to, and meet the observer behind each pattern. "
            "SIMULATION MODE: synthetic, not a real model."
        ),
        "question": "一个句子里，token 如何互相注视？",
        "next_question": "为什么这个 token 看那个 token？",
        "next_experiment": "transformer-detective",
        "controls": [
            {
                "key": "sentence",
                "label": "Sentence",
                "type": "select",
                "options": ["0", "1", "2", "3"],
            },
            {
                "key": "heads",
                "label": "Observers (heads)",
                "type": "range",
                "min": 2,
                "max": 8,
                "step": 1,
                "default": 6,
            },
            {"key": "reshuffle", "label": "NEW SOCIETY", "type": "action"},
        ],
    },
    {
        "id": "transformer-detective",
        "title": "Transformer Detective",
        "subtitle": "Why did the model predict this?",
        "icon": "search",
        "category": "Model Behavior",
        "group": "model-behavior",
        "blurb": (
            "A case file. The model made a prediction — your job is to find "
            "out why. Interrogate the suspects (heads, features, FFN) and "
            "close the case with the most influential evidence. Evidence is "
            "algorithmically generated from a curated case library — no real "
            "model is inspected."
        ),
        "question": "为什么模型在这里预测了这个？",
        "next_question": "残差流如何运送这些证据？",
        "next_experiment": "representation-river",
        "controls": [
            {
                "key": "case",
                "label": "Case file",
                "type": "select",
                "options": ["0", "1", "2"],
            },
        ],
    },
    {
        "id": "moe-expert-routing",
        "title": "Expert Routing Room",
        "subtitle": "MoE: who picks up this token?",
        "icon": "call_split",
        "category": "Model Efficiency",
        "group": "model-efficiency",
        "blurb": (
            "A triage room for tokens. A gate network routes each token to "
            "the experts most suited to it; top-k routing keeps the strongest "
            "connections sparse. Watch loads build up per expert and see "
            "which specialist takes each token. SIMULATION MODE: synthetic "
            "gate, not a real MoE model."
        ),
        "question": "一个 token 会被哪些专家接走？",
        "next_question": "路由负载均衡为什么重要？",
        "next_experiment": "mamba-memory-race",
        "controls": [
            {
                "key": "experts",
                "label": "Experts",
                "type": "range",
                "min": 2,
                "max": 8,
                "step": 1,
                "default": 4,
            },
            {
                "key": "top_k",
                "label": "Top-k routing",
                "type": "range",
                "min": 1,
                "max": 4,
                "step": 1,
                "default": 2,
            },
            {
                "key": "tokens",
                "label": "Tokens",
                "type": "range",
                "min": 4,
                "max": 12,
                "step": 1,
                "default": 8,
            },
            {
                "key": "temperature",
                "label": "Gate temperature",
                "type": "range",
                "min": 0.2,
                "max": 3.0,
                "step": 0.05,
                "default": 1.0,
            },
            {"key": "reshuffle", "label": "NEW TOKENS", "type": "action"},
        ],
    },
    {
        "id": "mamba-memory-race",
        "title": "Mamba Memory Race",
        "subtitle": "O(L) vs O(L²): why sequence length matters",
        "icon": "speed",
        "category": "Model Efficiency",
        "group": "model-efficiency",
        "blurb": (
            "A memory race down the sequence. A Transformer attention layer "
            "does O(L²) work — every token looks back at every previous one. "
            "A Mamba-style SSM keeps a fixed-size state and only touches it "
            "per token: O(L). Drag the sequence length and watch the "
            "quadratic curve pull ahead."
        ),
        "question": "序列变长时，为什么 Transformer 会比 Mamba 慢这么多？",
        "next_question": "KV Cache 如何缓解 Transformer 的平方开销？",
        "next_experiment": "weight-freezer",
        "controls": [
            {
                "key": "length",
                "label": "Sequence length",
                "type": "range",
                "min": 4,
                "max": 256,
                "step": 4,
                "default": 64,
            },
        ],
    },
    {
        "id": "transformer-mri",
        "title": "Transformer MRI",
        "subtitle": "Scan the model's training dynamics",
        "icon": "monitor_heart",
        "category": "Model Behavior",
        "group": "model-behavior",
        "blurb": (
            "An MRI scan of the Transformer. Sweep the slice plane across "
            "training dynamics — loss, representation entropy, gradient norm "
            "— through layers (the body) and steps (the time axis). Healthy "
            "signs: loss converging, entropy rising, gradients bounded. A "
            "vanishing-gradient pathology shows up as near-zero norms in "
            "deep layers. SIMULATION MODE: synthetic scan, not a real run."
        ),
        "question": "扫描仪下，模型内部是健康的吗？",
        "next_question": "梯度消失时，深层的表示还在演化吗？",
        "next_experiment": "representation-river",
        "controls": [
            {
                "key": "scan",
                "label": "Scan channel",
                "type": "select",
                "options": ["loss", "entropy", "grad_norm"],
            },
            {
                "key": "layer",
                "label": "Slice layer",
                "type": "range",
                "min": 0,
                "max": 11,
                "step": 1,
                "default": 6,
            },
            {
                "key": "step",
                "label": "Slice step",
                "type": "range",
                "min": 0,
                "max": 99,
                "step": 1,
                "default": 99,
            },
        ],
    },
    {
        "id": "feature-hunt",
        "title": "Feature Hunt",
        "subtitle": "Find the real features in an activation zoo",
        "icon": "track_changes",
        "category": "Model Behavior",
        "group": "model-behavior",
        "blurb": (
            "A feature zoo. N candidate neurons fire over a set of tokens; "
            "most are noise, a few are REAL features — they fire strongly and "
            "consistently on one semantic group. Scan the activation heatmap, "
            "click a neuron to see what it fires on, and bet which ones are "
            "real. SIMULATION MODE: synthetic sparse activations, not a real "
            "model."
        ),
        "question": "噪声堆里，哪些神经元是真正的特征？",
        "next_question": "稀疏激活为什么让特征可解释？",
        "next_experiment": "transformer-mri",
        "controls": [
            {
                "key": "features",
                "label": "Candidate neurons",
                "type": "range",
                "min": 8,
                "max": 24,
                "step": 2,
                "default": 16,
            },
            {
                "key": "tokens",
                "label": "Tokens",
                "type": "range",
                "min": 6,
                "max": 12,
                "step": 2,
                "default": 12,
            },
            {"key": "reshuffle", "label": "NEW ZOO", "type": "action"},
        ],
    },
]


def get_module(module_id: str) -> dict[str, Any] | None:
    for m in MODULES:
        if m["id"] == module_id:
            return m
    return None
