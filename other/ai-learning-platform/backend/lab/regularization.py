"""
L1 vs L2 regularization: constraint-set geometry and the contact point.

We visualize a quadratic loss centered at an (unconstrained) optimum and find
where the L1 (diamond) or L2 (circle) constraint ball first touches the loss
contours. For L2 this is closed form; for L1 we evaluate the four diamond
vertices and pick the one minimizing the quadratic (the contact always lies on
a vertex for a tilted quadratic, illustrating sparsity).
"""

from typing import Any
import numpy as np


def _loss_grid(center: np.ndarray, tilt: float, lim: tuple[float, float], n: int = 60):
    # A rotated anisotropic quadratic:  f = u^T A u  with u = x - center
    xs = np.linspace(lim[0], lim[1], n)
    ys = np.linspace(lim[0], lim[1], n)
    X, Y = np.meshgrid(xs, ys)
    c, s = np.cos(tilt), np.sin(tilt)
    R = np.array([[c, -s], [s, c]])
    A = R @ np.diag([3.0, 0.8]) @ R.T
    Z = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            d = np.array([X[i, j], Y[i, j]]) - center
            Z[i, j] = d @ A @ d
    return xs, ys, Z, A


def _l2_contact(center: np.ndarray, C: float, A: np.ndarray) -> np.ndarray:
    """Point on the circle of radius C minimizing (x-c)^T A (x-c).

    On ||x||=C with c fixed: minimize  x^T A x - 2 c^T A x + c^T A c.
    For a positive-definite A the minimizer is along A^{-1} c rescaled to the
    circle — a clean geometric result.
    """
    direction = np.linalg.solve(A, center)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array([C, 0.0])
    return C * direction / norm


def _l1_contact(center: np.ndarray, C: float, A: np.ndarray) -> tuple[np.ndarray, bool]:
    """Minimize quadratic over the L1 diamond. For a strictly convex quadratic
    the contact with the diamond lies either at a vertex or on an edge; we check
    vertices and the midpoint of each edge and pick the minimum. Returns the
    point and whether it landed on a vertex (sparsity)."""
    candidates = [
        np.array([C, 0.0]),
        np.array([-C, 0.0]),
        np.array([0.0, C]),
        np.array([0.0, -C]),
    ]
    # edge midpoints (projection of center onto each edge, clipped to the edge)
    edges = [
        (np.array([C, 0.0]), np.array([0.0, C])),
        (np.array([0.0, C]), np.array([-C, 0.0])),
        (np.array([-C, 0.0]), np.array([0.0, -C])),
        (np.array([0.0, -C]), np.array([C, 0.0])),
    ]
    for a, b in edges:
        # project center onto line a + t(b-a), t in [0,1]
        ab = b - a
        t = float(np.dot(center - a, ab) / (np.dot(ab, ab) + 1e-9))
        t = max(0.0, min(1.0, t))
        candidates.append(a + t * ab)

    def f(p: np.ndarray) -> float:
        d = p - center
        return float(d @ A @ d)

    best = min(candidates, key=f)
    on_vertex = any(np.allclose(best, v) for v in candidates[:4])
    return best, on_vertex


def _circle(C: float, n: int = 120) -> list[list[float]]:
    t = np.linspace(0, 2 * np.pi, n)
    return [[round(float(C * np.cos(a)), 4), round(float(C * np.sin(a)), 4)] for a in t]


def _diamond(C: float) -> list[list[float]]:
    return [
        [float(C), 0.0],
        [0.0, float(C)],
        [-float(C), 0.0],
        [0.0, -float(C)],
        [float(C), 0.0],
    ]


def compute(params: dict[str, Any]) -> dict[str, Any]:
    penalty = params.get("penalty", "l2")
    C = float(params.get("c", 1.2))
    tilt = float(params.get("angle", 0.6))
    opt = np.array([float(params.get("optX", 2.0)), float(params.get("optY", 1.6))])

    lim = (-3.2, 3.2)
    xs, ys, Z, A = _loss_grid(opt, tilt, lim)

    shapes: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []

    if penalty in ("l2", "both"):
        p2 = _l2_contact(opt, C, A)
        shapes.append({"type": "l2", "points": _circle(C)})
        contacts.append(
            {
                "penalty": "l2",
                "point": [round(float(p2[0]), 4), round(float(p2[1]), 4)],
                "sparse": bool(abs(p2[0]) < 1e-6 or abs(p2[1]) < 1e-6),
            }
        )
    if penalty in ("l1", "both"):
        p1, sparse = _l1_contact(opt, C, A)
        shapes.append({"type": "l1", "points": _diamond(C)})
        contacts.append(
            {
                "penalty": "l1",
                "point": [round(float(p1[0]), 4), round(float(p1[1]), 4)],
                "sparse": sparse,
            }
        )

    zmax = float(np.percentile(Z, 95))
    return {
        "domain": {"x": list(lim), "y": list(lim)},
        "contour": {
            "x": xs.round(4).tolist(),
            "y": ys.round(4).tolist(),
            "z": np.clip(Z, 0, zmax).round(4).tolist(),
            "zmax": round(zmax, 4),
        },
        "optimum": [round(float(opt[0]), 4), round(float(opt[1]), 4)],
        "constraint": C,
        "shapes": shapes,
        "contacts": contacts,
    }
