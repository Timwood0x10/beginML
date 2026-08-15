"""
Rotary Observatory — RoPE as a repeatable experiment (plan §10.2).

Teaching endpoint (frozen): a single point rotating on the complex plane is
only the first intuition. The real discovery is that the inner product

    (R(m)q)ᵀ (R(n)k)   depends only on  m - n

— absolute rotation disappears, relative rotation remains.

Simulation contract: realtime. Everything is deterministic given params; the
frontend advances `position` / `position_b` and reads back angles, relative
phase, a similarity-vs-distance curve and a per-dimension frequency lens.

No-LLM principle: all numbers below are computed, nothing is generated.
"""

from typing import Any

import numpy as np

MAX_DIST = 16  # similarity-vs-distance curve length


def _rot2(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def compute(params: dict[str, Any]) -> dict[str, Any]:
    m = int(params.get("position", 8))
    n = int(params.get("position_b", 4))
    freq = float(params.get("frequency", 1.0))          # 0.25 .. 4 (low..high)
    dims = int(params.get("dims", 8))
    pair = bool(params.get("pair", False))
    distance_mode = bool(params.get("distance_mode", False))

    # Base rotation per unit position; `frequency` scales it like a speed dial.
    theta0 = freq * 0.25  # radians per position at dim 0

    angle_m = m * theta0
    angle_n = n * theta0
    rel_phase = angle_m - angle_n  # = (m - n) * theta0

    # Single token: a unit vector q = (1, 0) rotated by R(m) — the "star".
    q = np.array([1.0, 0.0])
    qm = _rot2(angle_m) @ q
    qn = _rot2(angle_n) @ q

    # Similarity for aligned unit q/k: (R(m)q)ᵀ(R(n)k) = cos((m-n)·θ).
    similarity = float(np.cos(rel_phase))
    sim_at_4 = float(np.cos(4 * theta0))
    sim_at_8 = float(np.cos(8 * theta0))

    # Similarity vs distance curve (the FIND THE DISTANCE plot).
    distances = list(range(0, MAX_DIST + 1))
    similarities = [round(float(np.cos(d * theta0)), 4) for d in distances]

    # Frequency lens: dim k rotates at theta0 * rate^(-k), so high-index dims
    # (low frequency) turn slowly and capture long-range structure.
    rate = 1.6
    lens = []
    for k in range(dims):
        theta_k = theta0 * (rate ** -k)
        lens.append({
            "dim": k,
            "theta": round(float(theta_k), 5),
            "angle_at_m": round(float(m * theta_k), 4),
            "angle_at_n": round(float(n * theta_k), 4),
        })

    return {
        "position": m,
        "position_b": n,
        "frequency": round(freq, 2),
        "theta0": round(theta0, 5),
        "dims": dims,
        "pair": pair,
        "distance_mode": distance_mode,
        "angle_m": round(angle_m, 4),
        "angle_n": round(angle_n, 4),
        "relative_phase": round(rel_phase, 4),
        "point_m": [round(float(qm[0]), 4), round(float(qm[1]), 4)],
        "point_n": [round(float(qn[0]), 4), round(float(qn[1]), 4)],
        "similarity": round(similarity, 4),
        "sim_at_4": round(sim_at_4, 4),
        "sim_at_8": round(sim_at_8, 4),
        "distance_curve": {"distances": distances, "similarities": similarities},
        "frequency_lens": lens,
        "provenance": "realtime",
    }
