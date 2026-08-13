"""
Loss functions — compute curves and gradients for regression/classification.
The frontend renders the curves; no formula is duplicated client-side.
"""

from typing import Any
import numpy as np


def _mse(y_true: float, y_pred: np.ndarray) -> np.ndarray:
    return (y_pred - y_true) ** 2


def _mae(y_true: float, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true)


def _huber(y_true: float, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    err = y_pred - y_true
    abs_err = np.abs(err)
    quad = np.where(abs_err <= delta, 0.5 * err ** 2, delta * (abs_err - 0.5 * delta))
    return quad


def _bce(y_true: float, y_pred: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def _hinge(y_true: int, y_score: np.ndarray) -> np.ndarray:
    # y_true in {0,1}; map to {-1,+1}
    t = 1.0 if y_true == 1 else -1.0
    return np.maximum(0, 1 - t * y_score)


def _cross_entropy_loss(y_true_idx: int, logits: np.ndarray) -> np.ndarray:
    """Multi-class cross entropy as one logit varies, others fixed at 0."""
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=0))
    return -shifted[y_true_idx] + log_z


LOSS_REGISTRY: dict[str, dict[str, Any]] = {
    "mse": {
        "fn": _mse,
        "name": "Mean Squared Error",
        "formula": "L = (y_pred - y)^2",
        "xLabel": "prediction",
        "yRange": (-0.2, 6.0),
        "classification": False,
    },
    "mae": {
        "fn": _mae,
        "name": "Mean Absolute Error",
        "formula": "L = |y_pred - y|",
        "xLabel": "prediction",
        "yRange": (-0.2, 4.0),
        "classification": False,
    },
    "huber": {
        "fn": _huber,
        "name": "Huber Loss",
        "formula": "L = 0.5e^2 if |e|<=d else d(|e|-0.5d)",
        "xLabel": "prediction",
        "yRange": (-0.2, 4.0),
        "classification": False,
    },
    "bce": {
        "fn": _bce,
        "name": "Binary Cross-Entropy",
        "formula": "L = -[y log(p) + (1-y) log(1-p)]",
        "xLabel": "predicted probability",
        "yRange": (-0.2, 5.0),
        "classification": True,
    },
    "hinge": {
        "fn": _hinge,
        "name": "Hinge Loss",
        "formula": "L = max(0, 1 - t * score)",
        "xLabel": "decision score",
        "yRange": (-0.2, 4.0),
        "classification": True,
    },
    "cross-entropy": {
        "fn": _cross_entropy_loss,
        "name": "Cross-Entropy (3-class)",
        "formula": "L = -log(softmax(z)_y)",
        "xLabel": "logit for true class",
        "yRange": (-0.2, 5.0),
        "classification": True,
    },
}


def _curve(loss_id: str, target: float, n: int = 300) -> tuple[np.ndarray, np.ndarray, list[float]]:
    spec = LOSS_REGISTRY[loss_id]
    if loss_id == "cross-entropy":
        x = np.linspace(-4, 4, n)
        # 3 classes: true class logit varies, other two fixed at 0
        logits = np.vstack([x, np.zeros(n), np.zeros(n)])
        y = spec["fn"](int(target), logits)
        return x, y, [-4.0, 4.0]
    if spec["classification"] and loss_id in ("bce",):
        x = np.linspace(0.01, 0.99, n)
    elif spec["classification"] and loss_id == "hinge":
        x = np.linspace(-3, 3, n)
    else:
        lo = target - 3
        hi = target + 3
        x = np.linspace(lo, hi, n)
    y = spec["fn"](target, x)
    return x, y, [float(x.min()), float(x.max())]


def compute(params: dict[str, Any]) -> dict[str, Any]:
    loss_id = params.get("loss", "mse")
    if loss_id not in LOSS_REGISTRY:
        loss_id = "mse"
    target = float(params.get("target", 0.0 if loss_id != "bce" else 1.0))
    if loss_id == "bce":
        target = 1.0 if target >= 0.5 else 0.0
    if loss_id == "cross-entropy":
        target = int(np.clip(round(target), 0, 2))
    if loss_id == "hinge":
        target = 1 if target >= 0.5 else 0

    x, y, x_range = _curve(loss_id, target)
    spec = LOSS_REGISTRY[loss_id]

    # Find minimum
    min_idx = int(np.argmin(y))
    minimum = {"x": round(float(x[min_idx]), 4), "y": round(float(y[min_idx]), 4)}

    # Gradient at a probe point (numeric, robust across all losses)
    probe = float(params.get("probe", target + 0.8))
    probe = max(float(x.min()), min(float(x.max()), probe))
    dx = 1e-5
    if loss_id == "cross-entropy":
        logits_lo = np.array([[probe - dx], [0], [0]])
        logits_hi = np.array([[probe + dx], [0], [0]])
        g = (float(spec["fn"](int(target), logits_hi)[0]) - float(spec["fn"](int(target), logits_lo)[0])) / (2 * dx)
        fp = float(spec["fn"](int(target), np.array([[probe], [0], [0]]))[0])
    else:
        fp = float(spec["fn"](target, probe))
        fp_hi = float(spec["fn"](target, probe + dx))
        fp_lo = float(spec["fn"](target, probe - dx))
        g = (fp_hi - fp_lo) / (2 * dx)

    # tangent line at the probe point
    tan_x = np.array([probe - 1.0, probe + 1.0])
    tan_y = fp + g * (tan_x - probe)

    return {
        "x": x.round(6).tolist(),
        "y": y.round(6).tolist(),
        "domain": {"x": x_range, "y": list(spec["yRange"])},
        "loss": loss_id,
        "name": spec["name"],
        "formula": spec["formula"],
        "xLabel": spec["xLabel"],
        "target": round(float(target), 4),
        "minimum": minimum,
        "probe": {"x": round(probe, 4), "y": round(fp, 4), "dy": round(g, 4)},
        "tangent": {"x": tan_x.round(4).tolist(), "y": tan_y.round(4).tolist()},
        "available": [{"id": k, "name": v["name"]} for k, v in LOSS_REGISTRY.items()],
    }
