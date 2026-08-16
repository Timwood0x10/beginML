"""
Gradient-descent trajectories on classic test objectives.

Returns:
  contour: {x, y, z} grid for a filled contour plot
  trajectories: {name, color, points: [{x,y,loss}]} for each optimizer
  minimum: {x,y} known global minimum (for plotting a marker)
"""

from typing import Any

import numpy as np

# Objectives --------------------------------------------------------------


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def _sphere_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * x


def _rosenbrock(x: np.ndarray) -> float:
    a, b = 1.0, 100.0
    return float((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)


def _rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    a, b = 1.0, 100.0
    dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    dy = 2 * b * (x[1] - x[0] ** 2)
    return np.array([dx, dy], dtype=float)


def _rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * len(x) + np.sum(x * x - A * np.cos(2 * np.pi * x)))


def _rastrigin_grad(x: np.ndarray) -> np.ndarray:
    A = 10.0
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


def _beale(x: np.ndarray) -> float:
    xx, yy = x[0], x[1]
    return float(
        (1.5 - xx + xx * yy) ** 2
        + (2.25 - xx + xx * yy**2) ** 2
        + (2.625 - xx + xx * yy**3) ** 2
    )


def _beale_grad(x: np.ndarray) -> np.ndarray:
    xx, yy = x[0], x[1]
    f1 = 1.5 - xx + xx * yy
    f2 = 2.25 - xx + xx * yy**2
    f3 = 2.625 - xx + xx * yy**3
    dxx = f1 * (-1 + yy) + f2 * (-1 + yy**2) + f3 * (-1 + yy**3)
    dyy = f1 * xx + f2 * 2 * xx * yy + f3 * 3 * xx * yy**2
    return 2 * np.array([dxx, dyy], dtype=float)


OBJECTIVES = {
    "sphere": {
        "fn": _sphere,
        "grad": _sphere_grad,
        "lim": (-3.0, 3.0),
        "minimum": (0.0, 0.0),
    },
    "rosenbrock": {
        "fn": _rosenbrock,
        "grad": _rosenbrock_grad,
        "lim": (-2.0, 2.0),
        "minimum": (1.0, 1.0),
    },
    "rastrigin": {
        "fn": _rastrigin,
        "grad": _rastrigin_grad,
        "lim": (-4.5, 4.5),
        "minimum": (0.0, 0.0),
    },
    "beale": {
        "fn": _beale,
        "grad": _beale_grad,
        "lim": (-4.0, 4.0),
        "minimum": (3.0, 0.5),
    },
}

# Optimizers --------------------------------------------------------------


def _start_point(rng: np.random.Generator, name: str, lim: tuple[float, float]):
    if name == "corner":
        return np.array([lim[0] + 0.4, lim[1] - 0.4])
    if name == "far":
        return np.array([lim[0] + 0.6, lim[0] + 0.6])
    # random, but keep it visibly away from the optimum
    for _ in range(20):
        p = rng.uniform(lim[0] + 0.5, lim[1] - 0.5, size=2)
        if np.linalg.norm(p) > 0.8:
            return p
    return np.array([lim[0] + 0.8, lim[1] - 0.8])


def _run(
    fn,
    grad,
    start: np.ndarray,
    optimizer: str,
    lr: float,
    momentum: float,
    steps: int,
):
    x = start.astype(float).copy()
    vel = np.zeros_like(x)
    sq = np.zeros_like(x)  # adagrad/rmsprop accumulator
    m = np.zeros_like(x)  # adam first moment
    v = np.zeros_like(x)  # adam second moment
    eps = 1e-8
    path = [{"x": float(x[0]), "y": float(x[1]), "loss": float(fn(x))}]

    for t in range(1, steps + 1):
        g = grad(x)
        if np.any(np.isnan(g)) or np.linalg.norm(g) > 1e8:
            break

        if optimizer == "sgd":
            x = x - lr * g
        elif optimizer == "momentum":
            vel = momentum * vel - lr * g
            x = x + vel
        elif optimizer == "nesterov":
            look = x + momentum * vel
            vel = momentum * vel - lr * grad(look)
            x = x + vel
        elif optimizer == "adagrad":
            sq = sq + g * g
            x = x - lr * g / (np.sqrt(sq) + eps)
        elif optimizer == "rmsprop":
            sq = 0.9 * sq + 0.1 * g * g
            x = x - lr * g / (np.sqrt(sq) + eps)
        elif optimizer == "adam":
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mhat = m / (1 - 0.9**t)
            vhat = v / (1 - 0.999**t)
            x = x - lr * mhat / (np.sqrt(vhat) + eps)
        else:
            raise ValueError(f"unknown optimizer {optimizer}")

        if np.any(np.isnan(x)) or np.linalg.norm(x) > 1e6:
            break
        path.append({"x": float(x[0]), "y": float(x[1]), "loss": float(fn(x))})

    return path


# Public API --------------------------------------------------------------


def compute(params: dict[str, Any]) -> dict[str, Any]:
    objective = params.get("objective", "sphere")
    optimizer = params.get("optimizer", "momentum")
    lr = float(params.get("lr", 0.02))
    momentum = float(params.get("momentum", 0.9))
    steps = int(params.get("steps", 80))
    start = params.get("start", "random")
    seed = int(params.get("seed", 0))
    rng = np.random.default_rng(seed)

    obj = OBJECTIVES[objective]
    fn, grad, lim = obj["fn"], obj["grad"], obj["lim"]

    # A custom start point can be supplied directly (e.g. user clicked the
    # canvas); otherwise pick from the preset strategies.
    start_x = params.get("startX")
    start_y = params.get("startY")
    if isinstance(start_x, (int, float)) and isinstance(start_y, (int, float)):
        sx = float(np.clip(start_x, lim[0], lim[1]))
        sy = float(np.clip(start_y, lim[0], lim[1]))
        start_pt = np.array([sx, sy])
    else:
        start_pt = _start_point(rng, start, lim)

    # Contour grid (log-scaled levels so non-convex functions stay readable)
    grid_n = 60
    xs = np.linspace(lim[0], lim[1], grid_n)
    ys = np.linspace(lim[0], lim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(grid_n):
        for j in range(grid_n):
            Z[i, j] = fn(np.array([X[i, j], Y[i, j]]))
    zmax = float(np.percentile(Z, 92))
    zmin = float(np.min(Z))

    trajectory = _run(fn, grad, start_pt, optimizer, lr, momentum, steps)

    # A reference SGD path at a known-stable lr for comparison
    ref_path = _run(fn, grad, start_pt, "sgd", min(lr, 0.01), 0.0, min(steps, 120))

    mx, my = obj["minimum"]
    return {
        "domain": {"x": [lim[0], lim[1]], "y": [lim[0], lim[1]]},
        "contour": {
            "x": xs.round(5).tolist(),
            "y": ys.round(5).tolist(),
            "z": np.clip(Z, zmin, zmax).round(6).tolist(),
            "zmin": round(zmin, 6),
            "zmax": round(zmax, 6),
        },
        "minimum": {"x": float(mx), "y": float(my)},
        "start": {"x": float(start_pt[0]), "y": float(start_pt[1])},
        "trajectories": [
            {"name": optimizer, "color": "#C8604A", "points": trajectory},
            {"name": "sgd (ref)", "color": "#7d766d", "points": ref_path},
        ],
        "finalLoss": trajectory[-1]["loss"] if trajectory else None,
    }
