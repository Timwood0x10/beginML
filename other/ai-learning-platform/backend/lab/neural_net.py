"""
Neural network playground — train a 2-layer MLP on a 2D dataset and return
the decision-surface grid plus the training points. All math runs here in
numpy; the frontend only renders.
"""

from typing import Any

import numpy as np


def _make_dataset(kind: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if kind == "moons":
        t = rng.uniform(0, np.pi, n // 2)
        x0 = np.stack([np.cos(t), np.sin(t)], axis=1)
        x1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    elif kind == "circles":
        r0 = rng.normal(0.5, 0.08, n // 2)
        r1 = rng.normal(1.0, 0.08, n // 2)
        a0 = rng.uniform(0, 2 * np.pi, n // 2)
        a1 = rng.uniform(0, 2 * np.pi, n // 2)
        x0 = np.stack([r0 * np.cos(a0), r0 * np.sin(a0)], axis=1)
        x1 = np.stack([r1 * np.cos(a1), r1 * np.sin(a1)], axis=1)
    elif kind == "xor":
        pts = rng.uniform(-1, 1, size=(n, 2))
        y = ((pts[:, 0] > 0) ^ (pts[:, 1] > 0)).astype(float)
        return pts, y
    elif kind == "spiral":
        n2 = n // 2
        t = np.linspace(0, 3 * np.pi, n2) + rng.normal(0, 0.1, n2)
        x0 = np.stack([0.4 * t * np.cos(t) / 3, 0.4 * t * np.sin(t) / 3], axis=1)
        x1 = np.stack([-0.4 * t * np.cos(t) / 3, -0.4 * t * np.sin(t) / 3], axis=1)
    else:  # gaussian blobs
        x0 = rng.normal([-0.7, -0.7], 0.4, size=(n // 2, 2))
        x1 = rng.normal([0.7, 0.7], 0.4, size=(n // 2, 2))

    X = np.vstack([x0, x1]) + rng.normal(0, 0.04, size=(n, 2))
    y = np.array([0] * len(x0) + [1] * len(x1), dtype=float)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _train(
    X: np.ndarray, y: np.ndarray, hidden: int, lr: float, epochs: int, seed: int
) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, np.sqrt(2.0 / 2), size=(2, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, 1))
    b2 = np.zeros(1)

    losses: list[float] = []
    n = len(X)
    for epoch in range(epochs):
        z1 = X @ w1 + b1
        a1 = _relu(z1)
        z2 = a1 @ w2 + b2
        p = _sigmoid(z2).ravel()

        eps = 1e-7
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        losses.append(round(float(loss), 5))

        dz2 = (p - y).reshape(-1, 1) / n
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}, losses


def _predict(params: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    a1 = _relu(X @ params["w1"] + params["b1"])
    return _sigmoid((a1 @ params["w2"] + params["b2"]).ravel())


def compute(params: dict[str, Any]) -> dict[str, Any]:
    dataset = params.get("dataset", "moons")
    n = int(params.get("samples", 120))
    hidden = int(params.get("hidden", 8))
    lr = float(params.get("lr", 0.5))
    epochs = int(params.get("epochs", 200))
    seed = int(params.get("seed", 1))
    noise = float(params.get("noise", 0.0))

    # Custom user-placed points take priority over the generated dataset.
    raw = params.get("points")
    if isinstance(raw, list) and len(raw) >= 4:
        X = np.array([[float(p["x"]), float(p["y"])] for p in raw])
        y = np.array([int(p["cls"]) for p in raw], dtype=float)
        if len(set(y.tolist())) < 2:
            X, y = _make_dataset(dataset, n, seed)
    else:
        X, y = _make_dataset(dataset, n, seed)
        if noise > 0:
            rng = np.random.default_rng(seed + 7)
            X = X + rng.normal(0, noise, size=X.shape)

    trained, losses = _train(X, y, hidden, lr, epochs, seed)
    acc = float(((_predict(trained, X) > 0.5) == y).mean())

    # Decision-surface grid
    pad = 0.6
    x_min, x_max = float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad
    y_min, y_max = float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad
    grid_n = 50
    gx = np.linspace(x_min, x_max, grid_n)
    gy = np.linspace(y_min, y_max, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.c_[GX.ravel(), GY.ravel()]
    probs = _predict(trained, grid_pts).reshape(GX.shape)

    points = [
        {
            "x": round(float(X[i, 0]), 4),
            "y": round(float(X[i, 1]), 4),
            "cls": int(y[i]),
            "prob": round(float(_predict(trained, X[i : i + 1])[0]), 3),
        }
        for i in range(len(X))
    ]

    return {
        "dataset": dataset,
        "points": points,
        "grid": {
            "x": gx.round(4).tolist(),
            "y": gy.round(4).tolist(),
            "z": probs.round(4).tolist(),
        },
        "domain": {
            "x": [round(x_min, 3), round(x_max, 3)],
            "y": [round(y_min, 3), round(y_max, 3)],
        },
        "losses": losses,
        "finalLoss": losses[-1] if losses else None,
        "accuracy": round(acc, 4),
        "architecture": f"2-{hidden}-1",
        "epochs": epochs,
        "learningRate": lr,
    }
