"""
Weight Freezer — quantization as a "break it" experiment (plan §11.2).

Metaphor: a freezer. Weights float free in continuous space; FREEZE snaps
them onto a grid. Each step down FP32 → INT8 → INT4 → INT2 → TERNARY costs
memory but grows error — the user hunts the Pareto sweet spot ("where does
compression stop being worth it?").

Simulation contract: realtime + seeded. The weight cloud is generated from a
seed; quantization itself is deterministic. The same (seed, bit) always
reproduces the same cloud and error.

No-LLM principle: every number is computed; nothing is generated.
"""

from typing import Any

import numpy as np

N_POINTS = 120
DIMS = 2


def _weight_cloud(seed: int) -> np.ndarray:
    """A structured 2D cloud so the grid snap is visible (not pure noise)."""
    rng = np.random.default_rng(seed)
    n = N_POINTS
    # two clusters + a ring, so quantization shows both collapse and spread
    c1 = rng.normal([-1.2, -1.2], 0.35, size=(n // 3, 2))
    c2 = rng.normal([1.2, 1.2], 0.35, size=(n // 3, 2))
    ang = rng.uniform(0, 2 * np.pi, size=n - 2 * (n // 3))
    ring = np.stack([np.cos(ang) * 2.0, np.sin(ang) * 2.0], axis=1)
    cloud = np.vstack([c1, c2, ring])
    # jitter so points aren't exactly on the grid before freezing
    cloud = cloud + rng.normal(0, 0.06, size=cloud.shape)
    return cloud


def _quantize(cloud: np.ndarray, bits: int) -> np.ndarray:
    """Per-dimension uniform quantization onto a 2^bits grid."""
    out = np.empty_like(cloud)
    for d in range(cloud.shape[1]):
        col = cloud[:, d]
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-9:
            out[:, d] = col
            continue
        n_levels = max(2, 2 ** bits)
        step = (hi - lo) / (n_levels - 1)
        q = np.round((col - lo) / step) * step + lo
        out[:, d] = q
    return out


def _quantize_ternary(cloud: np.ndarray) -> np.ndarray:
    """BitNet 1.58b mode: every weight snaps to {-1, 0, +1} per dimension."""
    out = np.zeros_like(cloud)
    for d in range(cloud.shape[1]):
        col = cloud[:, d]
        thresh = 0.5 * float(np.abs(col).mean()) or 1e-9
        out[:, d] = np.where(col > thresh, 1.0, np.where(col < -thresh, -1.0, 0.0))
    return out


def _metrics(orig: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """(relative error, memory bits per param)."""
    var = float(orig.var()) or 1e-9
    mse = float(np.mean((orig - q) ** 2))
    return mse / var, mse


# (label, bits-per-param) for the Pareto curve; ternary ≈ 1.58
LEVELS: list[tuple[str, int, float]] = [
    ("fp32", 32, 32.0),
    ("int8", 8, 8.0),
    ("int4", 4, 4.0),
    ("int2", 2, 2.0),
    ("ternary", 0, 1.58),
]


def compute(params: dict[str, Any]) -> dict[str, Any]:
    seed = int(params.get("seed", 7))
    bit_width = int(params.get("bit_width", 8))  # 2 | 4 | 8 | 16
    ternary = bool(params.get("ternary", False))

    cloud = _weight_cloud(seed)

    # current quantization
    if ternary:
        cur = _quantize_ternary(cloud)
        cur_label = "ternary"
        cur_bits = 1.58
    else:
        cur = _quantize(cloud, bit_width)
        cur_label = f"int{bit_width}"
        cur_bits = float(bit_width)

    cur_err, cur_mse = _metrics(cloud, cur)

    # full Pareto curve across levels
    levels = []
    for label, b, mem in LEVELS:
        if label == "ternary":
            q = _quantize_ternary(cloud)
        else:
            q = _quantize(cloud, b)
        rel, mse = _metrics(cloud, q)
        levels.append({
            "label": label,
            "bits": mem,
            "error": round(rel, 5),
            "mse": round(mse, 5),
            "memory_mb": round(mem * N_POINTS * DIMS / 8 / 1e6, 6),
        })

    return {
        "points": np.round(cloud, 4).tolist(),
        "quantized": np.round(cur, 4).tolist(),
        "current_label": cur_label,
        "current_bits": cur_bits,
        "current_error": round(cur_err, 5),
        "levels": levels,
        "bit_width": bit_width,
        "ternary": ternary,
        "n_points": N_POINTS,
        "seed": seed,
        "provenance": f"seeded(weight cloud, seed {seed})",
    }
