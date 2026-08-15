"""
Sampling Machine — faithful logits → probability → sampling pipeline.

  LOGITS → TEMPERATURE (÷T) → SOFTMAX → FILTER GATE (top-k / top-p) → SAMPLE

Teaching point (frozen in the plan): temperature rescales the *input to
softmax* as logits/T; it does NOT touch the raw logits themselves. The UI
shows raw logits fixed while the softmax input changes with T.

Simulation contract: realtime — the distribution is cheap and deterministic;
the sampler is seeded so each SAMPLE run is reproducible (the frontend
increments `seed` per run). Provenance records how the result was produced.
"""

from typing import Any

import numpy as np

TOKENS = ["Paris", "Rome", "Tokyo", "Berlin", "London", "Madrid", "Oslo", "Kyoto"]


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _raw_logits(mode: str) -> np.ndarray:
    """Synthetic logits shaped by the selected mode (deterministic)."""
    if mode == "spiky":
        return np.array([4.2, 1.3, 0.9, 0.5, 0.3, 0.1, -0.2, -0.6])
    if mode == "multi":
        return np.array([3.4, 3.1, 2.7, 1.0, 0.8, 0.4, 0.1, -0.4])
    # flat
    return np.array([0.7, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, -0.05])


def _apply_gate(
    probs: np.ndarray, gate: str, top_k: int, top_p: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (kept-mask, renormalized filtered probs).

    - top-k: keep the k highest-probability tokens.
    - top-p: nucleus sampling — keep the smallest set whose cumulative
      probability reaches p.
    - both: apply top-k first, then top-p on the survivors.
    """
    n = len(probs)
    mask = np.ones(n, dtype=bool)

    if gate in ("top-k", "both"):
        k = min(max(1, int(top_k)), n)
        cutoff = np.sort(probs)[-k]
        mask &= probs >= cutoff

    if gate in ("top-p", "both"):
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        cut = int(np.searchsorted(cum, top_p, side="left")) + 1
        cut = min(max(1, cut), n)
        keep = np.zeros(n, dtype=bool)
        keep[order[:cut]] = True
        mask &= keep

    filtered = probs * mask
    total = float(filtered.sum())
    if total <= 0:  # degenerate gate — fall back to the full distribution
        return np.ones(n, dtype=bool), probs
    return mask, filtered / total


def compute(params: dict[str, Any]) -> dict[str, Any]:
    mode = str(params.get("logits_mode", "multi"))
    temperature = float(params.get("temperature", 1.0))
    gate = str(params.get("gate", "none"))
    top_k = int(params.get("top_k", 3))
    top_p = float(params.get("top_p", 0.9))
    sample_count = int(params.get("sample_count", 20))
    seed = int(params.get("seed", 1))

    raw = _raw_logits(mode)
    # Temperature step: the softmax input becomes logits / T (raw logits
    # themselves are untouched — T < 1 sharpens, T > 1 flattens).
    scaled = raw / max(temperature, 1e-6)
    probs = _stable_softmax(scaled)
    mask, filtered = _apply_gate(probs, gate, top_k, top_p)

    rng = np.random.default_rng(seed)
    draws = rng.choice(len(TOKENS), size=sample_count, p=filtered)
    counts = np.bincount(draws, minlength=len(TOKENS)).tolist()

    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    return {
        "tokens": TOKENS,
        "raw_logits": raw.round(3).tolist(),
        "scaled_logits": scaled.round(3).tolist(),
        "probs": probs.round(4).tolist(),
        "mask": mask.tolist(),
        "filtered_probs": filtered.round(4).tolist(),
        "counts": counts,
        "samples": sample_count,
        "temperature": round(temperature, 2),
        "gate": gate,
        "top_k": top_k,
        "top_p": round(top_p, 2),
        "seed": seed,
        "entropy": round(entropy, 3),
        "max_entropy": round(float(np.log2(len(TOKENS))), 3),
        "provenance": "realtime",
    }
