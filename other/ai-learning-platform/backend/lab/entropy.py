"""
Entropy, cross-entropy and KL divergence.

Builds two Bernoulli / categorical distributions P and Q from user-controlled
parameters and returns:
  * H(P), H(Q) — entropy of each
  * H(P,Q)     — cross-entropy
  * KL(P||Q)   — KL divergence (with visualization-friendly curves)
All math runs here in numpy; the frontend only renders.
"""

from typing import Any
import numpy as np


def _entropy_bernoulli(p: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _kl_bernoulli(p: float, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))


def _ce_bernoulli(p: float, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return -(p * np.log2(q) + (1 - p) * np.log2(1 - q))


def compute(params: dict[str, Any]) -> dict[str, Any]:
    mode = params.get("mode", "bernoulli")
    n = 300

    if mode == "bernoulli":
        p = float(params.get("p", 0.7))
        q_curve = np.linspace(0.001, 0.999, n)
        h_curve = _entropy_bernoulli(q_curve)
        kl_curve = _kl_bernoulli(p, q_curve)
        ce_curve = _ce_bernoulli(p, q_curve)

        hp = float(_entropy_bernoulli(np.array([p]))[0])
        kl_at_p = 0.0
        ce_at_p = float(_ce_bernoulli(p, np.array([p]))[0])

        return {
            "mode": "bernoulli",
            "x": q_curve.round(6).tolist(),
            "entropy": h_curve.round(6).tolist(),
            "kl": kl_curve.round(6).tolist(),
            "crossEntropy": ce_curve.round(6).tolist(),
            "domain": {"x": [0.0, 1.0], "y": [0.0, 4.0]},
            "p": round(p, 4),
            "entropyP": round(hp, 4),
            "klAtP": round(kl_at_p, 4),
            "ceAtP": round(ce_at_p, 4),
            "formula": "KL(P||Q) = p log(p/q) + (1-p) log((1-p)/(1-q))",
        }

    # Categorical: two distributions over k categories
    k = int(params.get("k", 5))
    k = max(2, min(10, k))
    rng = np.random.default_rng(int(params.get("seed", 1)))

    # P is a fixed peaked distribution; Q is controlled by "temperature" t
    # applied to the same logits — so the KL/entropy change meaningfully.
    logits_p = rng.normal(0, 1.5, size=k)
    p_soft = np.exp(logits_p - logits_p.max())
    p_dist = p_soft / p_soft.sum()

    t = float(params.get("temperature", 1.0))
    t = max(0.1, t)
    logits_q = logits_p / t
    q_soft = np.exp(logits_q - logits_q.max())
    q_dist = q_soft / q_soft.sum()

    def entropy(d: np.ndarray) -> float:
        return float(-np.sum(d * np.log2(d + 1e-12)))

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2((a + 1e-12) / (b + 1e-12))))

    hp = entropy(p_dist)
    hq = entropy(q_dist)
    ce = float(-np.sum(p_dist * np.log2(q_dist + 1e-12)))
    kl_val = kl(p_dist, q_dist)

    return {
        "mode": "categorical",
        "categories": [f"c{i+1}" for i in range(k)],
        "p": p_dist.round(4).tolist(),
        "q": q_dist.round(4).tolist(),
        "domain": {"x": [0, 1], "y": [0, 1]},
        "entropyP": round(hp, 4),
        "entropyQ": round(hq, 4),
        "crossEntropy": round(ce, 4),
        "kl": round(kl_val, 4),
        "temperature": round(t, 3),
        "formula": "KL(P||Q) = sum P(i) log2(P(i)/Q(i))",
    }
