"""
Shooting Range — bias / variance / noise decomposition as an experiment
(plan §11.1).

Metaphor: a firing range. Each bootstrap model is one shot; the bullseye is
the true function value; the scatter of shots is variance, the offset of
their centre is bias, and the irreducible ring around the target is noise.

    MSE = Bias² + Variance + Noise²

Simulation contract: stochastic + seeded. The same (complexity, samples,
noise, seed) always reproduces the same shots and decomposition. The
frontend advances `seed` to draw a fresh volley.

Numerical care: x is scaled to [-1, 1] and fits are solved via the
normal equations with a tiny ridge — high-degree polynomials stay
well-conditioned (no RankWarning / exploding coefficients).

No-LLM principle: every number is computed; the quadrant label is a
deterministic rule over the computed decomposition.
"""

from typing import Any

import numpy as np

X_MIN, X_MAX = -2.0, 2.0
GRID_N = 64  # points on the x grid used for curves
FIT_N = 12  # bootstrap fits sent to the frontend (subsampled)
SHOT_N = 40  # shots (bootstrap predictions) at the bullseye point
RIDGE = 1e-6  # regularizer for the normal equations


def _true_fn(x: np.ndarray) -> np.ndarray:
    """Nonlinear teacher so polynomial fits cannot win at every complexity."""
    return np.sin(2.0 * x) + 0.5 * np.cos(3.0 * x) - 0.3 * x * x


def _fit(xn: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    """Well-conditioned polynomial fit on x scaled to [-1, 1] (tiny ridge)."""
    V = np.vander(xn, deg + 1, increasing=True)
    A = V.T @ V + RIDGE * np.eye(deg + 1)
    b = V.T @ y
    return np.linalg.solve(A, b)


def _eval(coef: np.ndarray, xn: np.ndarray) -> np.ndarray:
    V = np.vander(xn, len(coef), increasing=True)
    return V @ coef


def _quadrant(bias2: float, variance: float, noise2: float) -> str:
    """Deterministic four-quadrant rule over the computed decomposition.

    Compares the two model terms against each other (×1.5 margin) and
    against the irreducible noise floor:
      - variance clearly larger  -> variance-dominated (overfit)
      - bias clearly larger      -> bias-dominated (underfit)
      - both above noise but even -> high-high (scarce data, both bad)
      - otherwise                -> balanced (healthy)
    """
    if variance > noise2 and variance > bias2 * 1.5:
        return "variance-dominated"
    if bias2 > noise2 and bias2 > variance * 1.5:
        return "bias-dominated"
    if bias2 > noise2 and variance > noise2:
        return "high-high"
    return "balanced"


def compute(params: dict[str, Any]) -> dict[str, Any]:
    complexity = int(params.get("complexity", 3))
    samples = int(params.get("samples", 50))
    noise = float(params.get("noise", 0.2))
    seed = int(params.get("seed", 7))

    rng = np.random.default_rng(seed)

    # --- data -----------------------------------------------------------
    xs = np.linspace(X_MIN, X_MAX, samples)
    ys = _true_fn(xs) + noise * rng.normal(0, 1, size=samples)

    grid = np.linspace(X_MIN, X_MAX, GRID_N)
    f_grid = _true_fn(grid)

    xs_n = xs / X_MAX
    grid_n = grid / X_MAX

    # --- bootstrap fits --------------------------------------------------
    deg = max(1, min(complexity, samples - 1))
    coefs: list[np.ndarray] = []
    for _ in range(SHOT_N):
        idx = rng.integers(0, samples, size=samples)  # resample with replacement
        try:
            coefs.append(_fit(xs_n[idx], ys[idx], deg))
        except np.linalg.LinAlgError:
            continue

    fits: list[np.ndarray] = []
    shots: list[float] = []
    for coef in coefs:
        pred = _eval(coef, grid_n)
        shot = float(_eval(coef, np.array([0.0]))[0])
        if np.all(np.isfinite(pred)) and np.isfinite(shot):
            fits.append(pred)
            shots.append(shot)

    if not fits:  # degenerate — fall back to a single degree-1 fit
        coef = _fit(xs_n, ys, 1)
        fits = [_eval(coef, grid_n)]
        shots = [float(_eval(coef, np.array([0.0]))[0])]

    x_star = 0.0
    true_at_star = float(_true_fn(np.array([x_star]))[0])

    shots_arr = np.array(shots)
    mean_shot = float(shots_arr.mean())

    # Decomposition over the whole grid (the honest bias-variance trade-off):
    # bias² = mean over x of (mean fit − true)², variance = mean over x of
    # the fit variance. The single-point `shots` stay for the bullseye view.
    fits_mat = np.array(fits)
    bias2 = float(((fits_mat.mean(axis=0) - f_grid) ** 2).mean())
    variance = float(fits_mat.var(axis=0).mean())
    noise2 = float(noise * noise)
    mse = bias2 + variance + noise2

    # subsample fits for the wireframe (keep them ordered by index)
    step = max(1, len(fits) // FIT_N)
    fits_out = [np.round(fits[i], 4).tolist() for i in range(0, len(fits), step)][
        :FIT_N
    ]

    return {
        "x": np.round(grid, 4).tolist(),
        "f": np.round(f_grid, 4).tolist(),
        "xs": np.round(xs, 4).tolist(),
        "ys": np.round(ys, 4).tolist(),
        "fits": fits_out,
        "x_star": x_star,
        "true_at_star": round(true_at_star, 4),
        "shots": [round(s, 4) for s in shots_arr[:SHOT_N]],
        "mean_shot": round(mean_shot, 4),
        "bias2": round(bias2, 4),
        "variance": round(variance, 4),
        "noise2": round(noise2, 4),
        "mse": round(mse, 4),
        "quadrant": _quadrant(bias2, variance, noise2),
        "complexity": deg,
        "samples": samples,
        "noise": noise,
        "seed": seed,
        "provenance": f"seeded(bootstrap ×{len(shots)}, seed {seed})",
    }
