"""
Self-attention: scaled dot-product attention matrix.

Attention(Q,K,V) = softmax( Q K^T / sqrt(d) + mask ) V

We synthesize small token embeddings so the heatmap is deterministic per seed.
Returns Q, K, scores, weights and the weighted output so the UI can show the
full pipeline.
"""

from typing import Any

import numpy as np


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def compute(params: dict[str, Any]) -> dict[str, Any]:
    n = int(params.get("tokens", 6))
    temperature = float(params.get("temperature", 1.0))
    causal = bool(params.get("causal", True))
    seed = int(params.get("seed", 7))
    d = 8  # embedding / head dim

    rng = np.random.default_rng(seed)
    # Token embeddings with a little structure: each token is a direction
    # plus learnable-looking Q/K projections (fixed random matrices here).
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    X = np.hstack([X, rng.normal(0, 0.4, size=(n, d - 2))])

    Wq = rng.normal(0, 1 / np.sqrt(d), size=(d, d))
    Wk = rng.normal(0, 1 / np.sqrt(d), size=(d, d))
    Wv = rng.normal(0, 1 / np.sqrt(d), size=(d, d))

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Scale: user-supplied "temperature" multiplies 1/sqrt(d), i.e. lower
    # temperature => sharper attention.
    scale = temperature / np.sqrt(d)
    scores = (Q @ K.T) * scale

    mask = None
    if causal:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        masked_scores = np.where(mask, -1e9, scores)
    else:
        masked_scores = scores

    weights = _stable_softmax(masked_scores, axis=-1)
    output = weights @ V

    # Entropy per query row (bits) — how focused the attention is
    entropy = []
    for row in weights:
        p = row[row > 1e-9]
        entropy.append(float(-np.sum(p * np.log2(p))))

    return {
        "tokens": [f"t{i}" for i in range(n)],
        "n": n,
        "d": d,
        "temperature": temperature,
        "causal": causal,
        "Q": Q.round(4).tolist(),
        "K": K.round(4).tolist(),
        "scores": scores.round(4).tolist(),
        "weights": weights.round(4).tolist(),
        "output": output.round(4).tolist(),
        "entropy": [round(e, 3) for e in entropy],
        "maxEntropy": round(float(np.log2(n)), 3),
    }
