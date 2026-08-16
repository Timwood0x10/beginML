"""
Token Society — "who looks at whom" as an experiment (plan §13, P0).

Metaphor: a sentence is a small society. Each token is a member; each
attention head is a different kind of observer. We compute, for every head,
a behavioural fingerprint (does it watch neighbours? long distances? the
token one step back?) and give the head a deterministic name from rules —
not hardcoded labels, but names derived from the observed attention pattern.

SIMULATION MODE: embeddings and Q/K projections are synthetic (seeded). This
does NOT inspect a real model — it demonstrates the mechanism.

Simulation contract: realtime + seeded.

No-LLM principle: every number is computed; head names come from a
deterministic rule over the fingerprint.
"""

from typing import Any

import numpy as np

SENTENCES = [
    "the animal didn't cross the street because it was too tired",
    "the cat sat on the mat and watched the bird",
    "she sells seashells by the seashore",
    "when the bell rings the students run to class",
]

D = 12  # embedding dim


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _name_head(fingerprint: dict[str, float]) -> str:
    """Deterministic naming from the observed fingerprint (rules, not labels)."""
    if fingerprint["diag_ratio"] > 0.35:
        return "The Repeater"  # strong one-step-back look (induction-ish)
    if fingerprint["avg_dist"] > 3.0:
        return "The Long-Distance Scout"
    if fingerprint["local_ratio"] > 0.55:
        return "The Nearby One"
    return "The Connector"


def compute(params: dict[str, Any]) -> dict[str, Any]:
    sent_idx = int(params.get("sentence", 0)) % len(SENTENCES)
    sentence = SENTENCES[sent_idx]
    heads = int(params.get("heads", 6))
    seed = int(params.get("seed", 11))

    tokens = sentence.split()
    n = len(tokens)
    rng = np.random.default_rng(seed)

    # --- synthetic token embeddings: semantic groups share a direction ----
    # tokens that belong together (e.g. "it" -> "animal") get correlated
    # vectors; we encode this by hashing each token into one of a few groups.
    groups = [hash(t) % 4 for t in tokens]  # deterministic grouping
    E = np.zeros((n, D))
    for i, g in enumerate(groups):
        E[i] = rng.normal(0, 0.2, size=D)
        group_vec = np.zeros(D)
        group_vec[g] = 2.0  # group direction dominates (padded to D dims)
        E[i] += group_vec
    E = E / np.linalg.norm(E, axis=1, keepdims=True)

    head_results = []
    for h in range(heads):
        # per-head Q/K projections
        Wq = rng.normal(0, 1 / np.sqrt(D), size=(D, D))
        Wk = rng.normal(0, 1 / np.sqrt(D), size=(D, D))
        Q = E @ Wq
        K = E @ Wk

        # behavioural bias per head (deterministic by head index):
        # h0 sharp (low temp), h1 one-step-back, h2 far, h3 local, others mild
        temp = 0.8 + 0.25 * (h % 3)
        scores = (Q @ K.T) / temp

        if h % 4 == 1:  # the Repeater: bump the token one step back
            for i in range(1, n):
                scores[i, i - 1] += 6.0
        elif h % 4 == 2:  # the Long-Distance Scout: bump far tokens
            for i in range(n):
                for j in range(n):
                    if abs(i - j) >= 3:
                        scores[i, j] += 4.0
        elif h % 4 == 3:  # the Nearby One: bump adjacent tokens
            for i in range(n):
                for j in range(n):
                    if abs(i - j) <= 1:
                        scores[i, j] += 3.0
        # h % 4 == 0 stays the Connector (mild, data-driven)

        weights = _stable_softmax(scores, axis=-1)

        # --- fingerprint -------------------------------------------------
        dists = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        avg_dist = float((weights * dists).sum()) / n
        local_ratio = float(weights[np.abs(dists) <= 1].sum()) / n
        diag_ratio = float(np.mean([weights[i, i - 1] for i in range(1, n)]))

        fp = {
            "head": h,
            "name": _name_head(
                {
                    "avg_dist": avg_dist,
                    "local_ratio": local_ratio,
                    "diag_ratio": diag_ratio,
                }
            ),
            "avg_dist": round(avg_dist, 3),
            "local_ratio": round(local_ratio, 3),
            "diag_ratio": round(diag_ratio, 3),
            "weights": np.round(weights, 4).tolist(),
        }
        head_results.append(fp)

    return {
        "tokens": tokens,
        "n": n,
        "sentence": sentence,
        "heads": head_results,
        "n_heads": heads,
        "seed": seed,
        "simulation_mode": True,
        "provenance": f"seeded(synthetic society, seed {seed})",
    }
