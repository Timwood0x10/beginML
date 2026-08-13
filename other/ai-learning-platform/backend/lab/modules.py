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
            {"key": "objective", "label": "Objective", "type": "select",
             "options": ["sphere", "rosenbrock", "rastrigin", "beale"]},
            {"key": "optimizer", "label": "Optimizer", "type": "select",
             "options": ["sgd", "momentum", "adam", "nesterov", "rmsprop", "adagrad"]},
            {"key": "lr", "label": "Learning rate", "type": "range",
             "min": 0.0005, "max": 0.2, "step": 0.0005, "default": 0.02},
            {"key": "momentum", "label": "Momentum", "type": "range",
             "min": 0.0, "max": 0.99, "step": 0.01, "default": 0.9},
            {"key": "steps", "label": "Steps", "type": "range",
             "min": 10, "max": 200, "step": 1, "default": 80},
            {"key": "start", "label": "Start point", "type": "select",
             "options": ["random", "far", "corner"]},
        ],
    },
    {
        "id": "attention",
        "title": "Self-Attention",
        "subtitle": "Q·Kᵀ / √d and the attention heatmap",
        "icon": "grid_on",
        "category": "Architectures",
        "blurb": (
            "Build an attention weight matrix step by step. Tweak the temperature "
            "√d scale, toggle causal masking and see how softmax sharpens or "
            "smooths the distribution each query attends over."
        ),
        "controls": [
            {"key": "tokens", "label": "Tokens", "type": "range",
             "min": 3, "max": 10, "step": 1, "default": 6},
            {"key": "temperature", "label": "Temperature (1/√d)", "type": "range",
             "min": 0.1, "max": 4.0, "step": 0.05, "default": 1.0},
            {"key": "causal", "label": "Causal mask", "type": "toggle", "default": True},
            {"key": "seed", "label": "Random seed", "type": "range",
             "min": 0, "max": 99, "step": 1, "default": 7},
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
            "components. See the eigenvectors of the covariance matrix rotate "
            "with the data and reconstruction error shrink as you add components."
        ),
        "controls": [
            {"key": "samples", "label": "Samples", "type": "range",
             "min": 30, "max": 400, "step": 10, "default": 150},
            {"key": "correlation", "label": "Correlation", "type": "range",
             "min": -0.95, "max": 0.95, "step": 0.05, "default": 0.75},
            {"key": "spread", "label": "Spread ratio", "type": "range",
             "min": 0.1, "max": 3.0, "step": 0.1, "default": 1.6},
            {"key": "components", "label": "Components kept", "type": "range",
             "min": 0, "max": 2, "step": 1, "default": 1},
            {"key": "seed", "label": "Random seed", "type": "range",
             "min": 0, "max": 99, "step": 1, "default": 3},
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
            "shrinks them smoothly. Drag the unconstrained optimum and watch "
            "where each constraint ball touches the loss contours."
        ),
        "controls": [
            {"key": "penalty", "label": "Penalty", "type": "select",
             "options": ["l1", "l2", "both"]},
            {"key": "c", "label": "Constraint radius C", "type": "range",
             "min": 0.1, "max": 3.0, "step": 0.05, "default": 1.2},
            {"key": "angle", "label": "Loss tilt", "type": "range",
             "min": 0, "max": 3.14, "step": 0.02, "default": 0.6},
            {"key": "optX", "label": "Optimum x", "type": "range",
             "min": 0.2, "max": 2.8, "step": 0.05, "default": 2.0},
            {"key": "optY", "label": "Optimum y", "type": "range",
             "min": 0.2, "max": 2.8, "step": 0.05, "default": 1.6},
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
            "Toggle linear / RBF / polynomial kernels and slide the soft-margin "
            "C to see the margin, support vectors and decision surface change."
        ),
        "controls": [
            {"key": "kernel", "label": "Kernel", "type": "select",
             "options": ["linear", "rbf", "poly"]},
            {"key": "c", "label": "Soft-margin C", "type": "range",
             "min": 0.05, "max": 50.0, "step": 0.05, "default": 1.0},
            {"key": "gamma", "label": "RBF γ", "type": "range",
             "min": 0.05, "max": 5.0, "step": 0.05, "default": 0.8},
            {"key": "degree", "label": "Poly degree", "type": "range",
             "min": 2, "max": 5, "step": 1, "default": 3},
            {"key": "reset", "label": "reset", "type": "action"},
        ],
    },
]


def get_module(module_id: str) -> dict[str, Any] | None:
    for m in MODULES:
        if m["id"] == module_id:
            return m
    return None
