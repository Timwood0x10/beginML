"""Metadata for every interactive Math Lab module (drives the sidebar)."""

from typing import Any

from . import agent_builder

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
        "id": "agent-builder",
        "title": "Agent Builder",
        "subtitle": "Assemble an agent from building blocks",
        "icon": "widgets",
        "category": "Architectures",
        "blurb": (
            "Pick one option from memory, tools, planning and multi-agent "
            "coordination, and watch the pieces snap into a full agent: a "
            "layered diagram, a plain-English architecture summary and a "
            "ready-to-edit YAML config."
        ),
        "controls": [
            {"key": cat, "label": agent_builder.COMPONENTS[cat]["label"],
             "type": "select",
             "options": agent_builder.option_labels(cat),
             "default": agent_builder.option_labels(cat)[0]}
            for cat in agent_builder.CATEGORY_ORDER
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
]


def get_module(module_id: str) -> dict[str, Any] | None:
    for m in MODULES:
        if m["id"] == module_id:
            return m
    return None
