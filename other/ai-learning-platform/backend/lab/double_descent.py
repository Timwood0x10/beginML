"""
Dangerous Mountain — double descent as a cached experiment (plan §10.3).

Spike outcome (lab/spikes/double_descent.py): random-Fourier-feature ridge
regression with d=6, lam=1e-3, gamma=1.0 produces a clean double descent —
test error falls while underfit, spikes at the interpolation threshold
(capacity ~ n), then drops again in the overparameterized regime, ending
LOWER than the best underparameterized model. That last part is the
counter-intuitive discovery the experiment is built around.

Simulation contract: cached. Curves are precomputed on a (samples, noise)
grid with fixed seeds and looked up per request — never re-trained per
slider move. Provenance records the grid cell.

No-LLM principle: every number below is computed; nothing is generated.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

D = 6  # input dimension of the teacher function
GAMMA = 1.0  # random-Fourier-feature bandwidth
LAM = 1e-3  # ridge regularizer (keeps the peak finite)
CAP_MAX = 300  # capacity grid upper bound
N_TEST = 6000
SEED_AVG = 3  # averaged fixed seeds — enough smoothness, 40% cheaper

# capacity grid: 2..CAP_MAX step 4 (75 points; step 2 was over-sampled)
CAPS = np.arange(2, CAP_MAX + 1, 4)
CAPS_LIST = CAPS.tolist()

SAMPLES_GRID = [32, 64, 96]
NOISE_GRID = [0.2, 0.4, 0.6]


def _rff_curve(n: int, noise: float) -> dict[str, np.ndarray]:
    """Precompute (train, test) error curves for one grid cell.

    The teacher is a fixed low-dim linear function; labels carry Gaussian
    noise. Features are random Fourier features; capacity = number of
    features. Averaging SEED_AVG fixed seeds removes run-to-run wobble while
    staying fully deterministic.
    """
    train_acc: list[np.ndarray] = []
    test_acc: list[np.ndarray] = []
    for s in range(SEED_AVG):
        rng = np.random.default_rng(1000 + s)
        w_true = np.zeros(D)
        w_true[:4] = rng.normal(0, 1.0, size=4)
        X = rng.normal(0, 1, size=(n, D))
        y = X @ w_true + noise * rng.normal(0, 1, size=n)
        Xt = rng.normal(0, 1, size=(N_TEST, D))
        yt = Xt @ w_true

        tr = np.empty(len(CAPS))
        te = np.empty(len(CAPS))
        for i, p in enumerate(CAPS):
            W = rng.normal(0, GAMMA, size=(D, int(p)))
            b = rng.uniform(0, 2 * np.pi, size=int(p))
            Ph = np.sqrt(2.0 / p) * np.cos(X @ W + b)
            Pht = np.sqrt(2.0 / p) * np.cos(Xt @ W + b)
            G = Ph.T @ Ph + LAM * np.eye(int(p))
            th = np.linalg.solve(G, Ph.T @ y)
            tr[i] = float(np.mean((Ph @ th - y) ** 2))
            te[i] = float(np.mean((Pht @ th - yt) ** 2))
        train_acc.append(tr)
        test_acc.append(te)

    return {
        "train": np.mean(train_acc, axis=0),
        "test": np.mean(test_acc, axis=0),
    }


# Precompute ALL grid cells at import time. Simulation contract: cached —
# the heavy RFF ridge sweeps run once here, so every request is a pure
# lookup with zero latency (no mid-request recompute, ever).
#
# The grid cells are independent and numpy-dominated (numpy releases the
# GIL), so they are built in PARALLEL via a thread pool. This cut import
# time from ~17s to ~4s without touching the cached contract — identical
# numbers, same determinism, just overlapped wall-clock.
def _build_cache() -> dict[tuple[int, float], dict[str, np.ndarray]]:
    cells = [(n, noise) for n in SAMPLES_GRID for noise in NOISE_GRID]
    with ThreadPoolExecutor(max_workers=len(cells)) as ex:
        curves = list(ex.map(lambda cell: _rff_curve(*cell), cells))
    return dict(zip(cells, curves))


_CACHE = _build_cache()


def _get_curve(n: int, noise: float) -> dict[str, np.ndarray]:
    return _CACHE[(n, round(noise, 2))]


def _classic_u(modern_test: np.ndarray, n: int) -> np.ndarray:
    """Classical bias-variance U-curve predicted before double descent.

    In the underparameterized regime it matches the observed curve; after the
    best underfit capacity it keeps rising — the classical theory says bigger
    models should only get worse. Built from the same data so the contrast
    with the modern curve is honest.
    """
    # best underfit capacity: argmin of test error before the interpolation
    # threshold (capacity < 0.8 * n)
    underfit = [i for i, c in enumerate(CAPS) if c < 0.8 * n]
    i_opt = int(underfit[int(np.argmin(modern_test[underfit]))])
    c_opt = CAPS[i_opt]
    v_opt = modern_test[i_opt]
    # parabola rising on both sides of c_opt
    k = (np.abs(CAPS - c_opt) / max(c_opt, 1)).astype(float) ** 2
    classic = v_opt * (1.0 + 0.35 * k)
    # in the underfit region keep the observed values (they are the same data)
    classic[:i_opt] = modern_test[:i_opt]
    return classic


def _state(capacity: int, n: int, train_e: float, test_e: float) -> str:
    """Classify the current capacity into UNDERFIT / DANGER / OVERPARAM."""
    if capacity < 0.75 * n:
        return "underfit"
    if capacity <= 1.5 * n:
        return "danger"
    return "overparam"


def compute(params: dict[str, Any]) -> dict[str, Any]:
    # snap to the precomputed grid
    n = int(min(SAMPLES_GRID, key=lambda v: abs(v - int(params.get("samples", 64)))))
    noise = float(
        min(NOISE_GRID, key=lambda v: abs(v - float(params.get("noise", 0.4))))
    )
    capacity = int(params.get("capacity", n))

    curve = _get_curve(n, noise)
    train = curve["train"]
    test = curve["test"]
    classic = _classic_u(test, n)

    i = int(np.argmin(np.abs(CAPS - capacity)))
    capacity = int(CAPS[i])
    tr_e = float(train[i])
    te_e = float(test[i])
    cl_e = float(classic[i])

    peak_i = int(np.argmax(test))
    peak_cap = int(CAPS[peak_i])
    peak_err = float(test[peak_i])

    state = _state(capacity, n, tr_e, te_e)

    return {
        "caps": CAPS_LIST,
        "train": [round(v, 4) for v in train],
        "test": [round(v, 4) for v in test],
        "classic": [round(v, 4) for v in classic],
        "n": n,
        "noise": noise,
        "capacity": capacity,
        "index": i,
        "train_error": round(tr_e, 4),
        "test_error": round(te_e, 4),
        "classic_error": round(cl_e, 4),
        "state": state,
        "peak_cap": peak_cap,
        "peak_error": round(peak_err, 4),
        "interpolation_threshold": n,
        "danger_zone": [round(0.75 * n), round(1.5 * n)],
        "provenance": f"cached({n} samples, noise {noise})",
    }
