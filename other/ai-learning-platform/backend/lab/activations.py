"""
Activation functions — compute curves and derivatives server-side.
The frontend only renders the returned arrays; no formula is hardcoded there.
"""

from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _swish(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


# name -> (forward fn, derivative fn, formula, y-range for display)
FUNCTIONS: dict[str, tuple] = {
    "sigmoid": (
        _sigmoid,
        lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)),
        "sigma(x) = 1 / (1 + e^-x)",
        (-0.1, 1.1),
    ),
    "tanh": (
        np.tanh,
        lambda x: 1.0 - np.tanh(x) ** 2,
        "tanh(x) = (e^x - e^-x) / (e^x + e^-x)",
        (-1.1, 1.1),
    ),
    "relu": (
        lambda x: np.maximum(0, x),
        lambda x: np.where(x > 0, 1.0, 0.0),
        "ReLU(x) = max(0, x)",
        (-0.5, 6.0),
    ),
    "leaky-relu": (
        lambda x: np.where(x > 0, x, 0.01 * x),
        lambda x: np.where(x > 0, 1.0, 0.01),
        "LReLU(x) = x if x>0 else 0.01x",
        (-0.5, 6.0),
    ),
    "gelu": (
        _gelu,
        lambda x: (
            0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
            + 0.5
            * x
            * (1.0 - np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)) ** 2)
            * np.sqrt(2.0 / np.pi)
            * (1.0 + 3.0 * 0.044715 * x**2)
        ),
        "GELU(x) = x * Phi(x)",
        (-0.5, 6.0),
    ),
    "swish": (
        _swish,
        lambda x: _sigmoid(x) + x * _sigmoid(x) * (1.0 - _sigmoid(x)),
        "swish(x) = x * sigma(x)",
        (-0.5, 6.0),
    ),
    "softplus": (
        _softplus,
        _sigmoid,
        "softplus(x) = ln(1 + e^x)",
        (-0.5, 6.0),
    ),
}


def compute(params: dict[str, Any]) -> dict[str, Any]:
    fn_id = params.get("function", "relu")
    x_min = float(params.get("xMin", -6.0))
    x_max = float(params.get("xMax", 6.0))
    n = 200

    if fn_id not in FUNCTIONS:
        fn_id = "relu"

    fn, dfn, formula, (y_min, y_max) = FUNCTIONS[fn_id]
    x = np.linspace(x_min, x_max, n)
    y = fn(x)
    dy = dfn(x)

    # Evaluate at the user-chosen point so the frontend can draw a tangent.
    point = float(params.get("point", 1.0))
    point = max(x_min, min(x_max, point))
    fp = float(fn(point))
    dfp = float(dfn(point))

    # Tangent line around the chosen point
    tangent_x = np.array([point - 1.5, point + 1.5])
    tangent_y = fp + dfp * (tangent_x - point)

    return {
        "x": x.round(6).tolist(),
        "y": y.round(6).tolist(),
        "dy": dy.round(6).tolist(),
        "domain": {"x": [x_min, x_max], "y": [y_min, y_max]},
        "point": {"x": round(point, 4), "y": round(fp, 4), "dy": round(dfp, 4)},
        "tangent": {
            "x": tangent_x.round(4).tolist(),
            "y": tangent_y.round(4).tolist(),
        },
        "formula": formula,
        "function": fn_id,
        "available": [
            {"id": k, "name": k.replace("-", " ").title()} for k in FUNCTIONS
        ],
    }
