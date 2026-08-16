"""
Feature Hunt — find the real features in a synthetic activation zoo
(plan §13, P1).

Metaphor: a feature zoo. We expose N candidate "neurons" and their
activations over a set of tokens. Most are noise; a few are REAL features —
they fire strongly and consistently on one semantic group of tokens (like
Detective's suspects, but for interpretability: sparse, concentrated
activation patterns).

The learner hunts: scan the activation heatmap, click a neuron, see what it
fires on, and bet which ones are real. The verdict is computed from the
activation statistics (max strength + concentration), never guessed.

SIMULATION MODE: synthetic sparse activations, not a real model.

Simulation contract: realtime + seeded.

No-LLM principle: every number is computed; "real feature" is a rule over
the activation matrix.
"""

from typing import Any

import numpy as np

N_TOKENS = 12
N_FEATURES = 16
N_TRUE = 4  # how many neurons are real features
SPARSE = [
    "bias",
    "position",
    "gender",
    "tense",
    "topic",
    "sentiment",
    "case",
    "negation",
]
TOKENS = [
    "the",
    "cat",
    "sat",
    "on",
    "the",
    "mat",
    "she",
    "ran",
    "fast",
    "through",
    "the",
    "park",
]


def compute(params: dict[str, Any]) -> dict[str, Any]:
    n_tokens = int(params.get("tokens", N_TOKENS)) % len(TOKENS) or N_TOKENS
    n_features = int(params.get("features", N_FEATURES))
    n_true = int(min(max(int(params.get("n_true", N_TRUE)), 1), n_features))
    seed = int(params.get("seed", 9))
    reveal = bool(params.get("reveal", False))

    rng = np.random.default_rng(seed)
    tokens = TOKENS[:n_tokens]

    # --- real features: each fires strongly on one semantic group --------
    # groups are deterministic slices of the token list (e.g. every 3rd token)
    true_idx = rng.choice(n_features, size=n_true, replace=False)
    true_idx = sorted(true_idx.tolist())
    group_of = {i: i % n_true for i in range(n_true)}  # true feature i -> group
    act = rng.uniform(0, 0.18, size=(n_tokens, n_features))  # noise floor

    for k, fi in enumerate(true_idx):
        g = group_of[k]
        members = [t for t in range(n_tokens) if t % n_true == g]
        for t in members:
            act[t, fi] = rng.uniform(0.75, 1.0)

    # --- per-feature statistics ------------------------------------------
    features = []
    for fi in range(n_features):
        col = act[:, fi]
        active = [t for t in range(n_tokens) if col[t] > 0.5]
        is_true = fi in true_idx
        features.append(
            {
                "idx": fi,
                "name": f"neuron_{fi}",
                "max": round(float(col.max()), 4),
                "mean": round(float(col.mean()), 4),
                "sparsity": round(float(np.mean(col > 0.05)), 4),
                "active_tokens": active,
                "is_true": is_true,
                "true_semantic": SPARSE[true_idx.index(fi) % len(SPARSE)]
                if is_true
                else None,
            }
        )

    return {
        "tokens": tokens,
        "n_tokens": n_tokens,
        "n_features": n_features,
        "n_true": n_true,
        "true_idx": true_idx,
        "activation": np.round(act, 4).tolist(),  # [token][feature]
        "features": features,
        "reveal": reveal,
        "seed": seed,
        "simulation_mode": True,
        "provenance": f"seeded(feature zoo, seed {seed})",
    }
