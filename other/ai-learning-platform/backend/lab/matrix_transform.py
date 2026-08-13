"""
2D matrix-transform visualizer.

Given a 2x2 matrix, compute:
  * eigenvectors / eigenvalues
  * determinant, trace, rank
  * transformed grid lines and basis vectors
  * a small set of test shapes (unit square, circle)

The user can pick a preset (rotation, scaling, shear, projection) or enter
each of the four entries directly. All math runs here in numpy.
"""

from typing import Any
import numpy as np


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _shear(k: float) -> np.ndarray:
    return np.array([[1.0, k], [0.0, 1.0]])


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0], [0.0, sy]])


PRESETS: dict[str, np.ndarray] = {
    "identity": np.eye(2),
    "rotation-45": _rotation(np.pi / 4),
    "rotation-90": _rotation(np.pi / 2),
    "scale-2x": _scale(2.0, 2.0),
    "scale-aniso": _scale(2.0, 0.5),
    "shear": _shear(1.0),
    "projection": np.array([[1.0, 1.0], [1.0, 1.0]]),
    "reflection-x": np.array([[1.0, 0.0], [0.0, -1.0]]),
}


def _transform_grid(M: np.ndarray, lo: float = -2, hi: float = 2, step: float = 0.5):
    """Return original + transformed grid line segments."""
    lines_orig: list[list[list[float]]] = []
    lines_tx: list[list[list[float]]] = []
    vals = np.arange(lo, hi + 1e-9, step)
    for v in vals:
        # horizontal line (x varies, y=v)
        a = np.array([lo, v]); b = np.array([hi, v])
        lines_orig.append([a.tolist(), b.tolist()])
        lines_tx.append([(M @ a).tolist(), (M @ b).tolist()])
        # vertical line (x=v, y varies)
        a = np.array([v, lo]); b = np.array([v, hi])
        lines_orig.append([a.tolist(), b.tolist()])
        lines_tx.append([(M @ a).tolist(), (M @ b).tolist()])
    return lines_orig, lines_tx


def _unit_square() -> np.ndarray:
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)


def _circle_points(n: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def compute(params: dict[str, Any]) -> dict[str, Any]:
    preset = params.get("preset", "identity")
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    c = float(params.get("c", 0.0))
    d = float(params.get("d", 1.0))

    if preset and preset in PRESETS:
        M = PRESETS[preset].copy()
    else:
        M = np.array([[a, b], [c, d]], dtype=float)

    det = float(np.linalg.det(M))
    trace = float(np.trace(M))
    rank = int(np.linalg.matrix_rank(M))

    eigvals, eigvecs = np.linalg.eig(M)

    # Keep only real eigenpairs (2D rotations have complex eigenvalues).
    eigen = []
    for i in range(2):
        v = eigvals[i]
        if np.isreal(v):
            eigen.append(
                {
                    "value": round(float(np.real(v)), 4),
                    "vector": [round(float(eigvecs[0, i].real), 4), round(float(eigvecs[1, i].real), 4)],
                }
            )

    grid_orig, grid_tx = _transform_grid(M)
    square = _unit_square()
    square_tx = (M @ square.T).T
    circle = _circle_points()
    circle_tx = (M @ circle.T).T

    # basis vectors
    ex = M @ np.array([1.0, 0.0])
    ey = M @ np.array([0.0, 1.0])

    # Bounds that contain both original and transformed content
    all_pts = np.vstack(
        [
            np.array(grid_orig).reshape(-1, 2),
            np.array(grid_tx).reshape(-1, 2),
            circle_tx,
            square_tx,
        ]
    )
    lo = float(all_pts.min() - 0.5)
    hi = float(all_pts.max() + 0.5)

    return {
        "matrix": M.round(4).tolist(),
        "det": round(det, 4),
        "trace": round(trace, 4),
        "rank": rank,
        "eigen": eigen,
        "basis": {
            "ex": ex.round(4).tolist(),
            "ey": ey.round(4).tolist(),
        },
        "gridOriginal": np.round(grid_orig, 4).tolist(),
        "gridTransformed": np.round(grid_tx, 4).tolist(),
        "squareOriginal": square.round(4).tolist(),
        "squareTransformed": square_tx.round(4).tolist(),
        "circleTransformed": circle_tx.round(4).tolist(),
        "domain": {"x": [lo, hi], "y": [lo, hi]},
        "available": [{"id": k, "name": k.replace("-", " ").title()} for k in PRESETS],
        "preset": preset,
    }
