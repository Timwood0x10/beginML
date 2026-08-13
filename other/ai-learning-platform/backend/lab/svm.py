"""
SVM decision boundary: fit an SVM on user-placed points and return a decision
grid plus support vectors so the frontend can draw the margin.

If the user hasn't supplied enough points, we synthesize a small linearly
separable demo set so the module is populated on first load.
"""

from typing import Any
import numpy as np
from sklearn.svm import SVC


def _default_points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    pos = rng.normal([-1.0, 0.6], 0.5, size=(8, 2))
    neg = rng.normal([1.0, -0.6], 0.5, size=(8, 2))
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    return X, y


def compute(params: dict[str, Any]) -> dict[str, Any]:
    kernel = params.get("kernel", "linear")
    C = float(params.get("c", 1.0))
    gamma = float(params.get("gamma", 0.8))
    degree = int(params.get("degree", 3))

    raw = params.get("points")
    if raw and len(raw) >= 2:
        X = np.array([[float(p["x"]), float(p["y"])] for p in raw])
        y = np.array([int(p["cls"]) for p in raw])
        if len(set(y.tolist())) < 2:
            X, y = _default_points()
    else:
        X, y = _default_points()

    if len(X) < 2:
        X, y = _default_points()

    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    clf.fit(X, y)

    # Build a decision-value grid over the data range (+padding)
    xmin, xmax = float(X[:, 0].min()) - 1.0, float(X[:, 0].max()) + 1.0
    ymin, ymax = float(X[:, 1].min()) - 1.0, float(X[:, 1].max()) + 1.0
    n = 70
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.c_[XX.ravel(), YY.ravel()]
    Z = clf.decision_function(grid).reshape(XX.shape)

    support_idx = set(getattr(clf, "support_", []).tolist())
    points_out = []
    for i, (px, py) in enumerate(X):
        points_out.append({
            "x": round(float(px), 4), "y": round(float(py), 4),
            "cls": int(y[i]), "support": i in support_idx,
        })

    # Coefficient vector for linear kernel (draws the margin lines explicitly)
    coef = None
    intercept = float(clf.intercept_[0])
    if kernel == "linear":
        c = clf.coef_[0]
        coef = [round(float(c[0]), 4), round(float(c[1]), 4)]

    return {
        "domain": {"x": [round(xmin, 3), round(xmax, 3)], "y": [round(ymin, 3), round(ymax, 3)]},
        "points": points_out,
        "grid": {
            "x": xs.round(4).tolist(),
            "y": ys.round(4).tolist(),
            "z": Z.round(4).tolist(),
        },
        "coef": coef,
        "intercept": round(intercept, 4),
        "nSupport": int(clf.n_support_.sum()),
        "accuracy": round(float(clf.score(X, y)), 4),
        "kernel": kernel,
    }
