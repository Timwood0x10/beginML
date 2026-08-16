"""
Representation River — synthetic residual stream (plan §11.3, Mode A).

Metaphor: a river that runs through every layer of a Transformer. The
residual stream is the river; each layer's Attention and FFN are tributaries
that inject information into it. We track how a token's representation moves
through the layers — its "trajectory" in representation space.

SIMULATION MODE (frozen): this experiment MODELS representation dynamics; it
does NOT inspect a real model. There is no real Transformer activation here —
the evolution is a deterministic synthetic simulation (attention mixes token
representations, FFN applies a nonlinear transform, both injected additively
into the residual stream, exactly like x + Attn(x) + FFN(x)).

Simulation contract: realtime + seeded.

No-LLM principle: every number is computed; nothing is generated.
"""

from typing import Any

import numpy as np

N_LAYERS = 12
DIM = 8  # hidden dim of the synthetic stream


def _attention_pattern(n: int) -> np.ndarray:
    """Ring attention: nearby tokens attend to each other, decaying with
    distance. This is what makes related tokens drift together."""
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))  # ring distance
            sim[i, j] = np.exp(-d)
    return sim / sim.sum(axis=1, keepdims=True)


def compute(params: dict[str, Any]) -> dict[str, Any]:
    n_tokens = int(params.get("tokens", 6))
    max_layer = int(params.get("layer", N_LAYERS))
    show = str(params.get("show", "all"))  # attention | ffn | all
    seed = int(params.get("seed", 5))

    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_tokens, endpoint=False)
    # token embeddings start spread on a circle so the 2D projection is
    # meaningful from layer 0
    H = np.hstack(
        [
            np.stack([np.cos(angles), np.sin(angles)], axis=1),
            rng.normal(0, 0.25, size=(n_tokens, DIM - 2)),
        ]
    )

    attn = _attention_pattern(n_tokens)
    W_ffn = rng.normal(0, 1.0 / np.sqrt(DIM), size=(DIM, DIM))

    # per-token 2D trajectory across layers (layer 0 = input embedding)
    traj = np.empty((max_layer + 1, n_tokens, 2))
    traj[0] = H[:, :2]
    injections = []

    h = H.copy()
    for l in range(1, max_layer + 1):
        depth = l / N_LAYERS  # 0..1 through the stack

        # attention tributary: pull each token toward the ones it attends to
        a_strength = 0.25 + 0.65 * depth
        attn_out = attn @ h
        # ffn tributary: nonlinear individual transform
        ffn_out = np.tanh(h @ W_ffn)
        f_strength = 0.15 + 0.7 * depth

        h_prev = h.copy()
        if show in ("attention", "all"):
            h = h + a_strength * (attn_out - h)
        if show in ("ffn", "all"):
            h = h + f_strength * ffn_out

        traj[l] = h[:, :2]
        injections.append(
            {
                "layer": l,
                "attention": round(
                    float(np.mean(np.linalg.norm(attn_out - h_prev, axis=1))), 4
                ),
                "ffn": round(float(np.mean(np.linalg.norm(ffn_out, axis=1))), 4),
            }
        )

    return {
        "tokens": [f"t{i}" for i in range(n_tokens)],
        "n_tokens": n_tokens,
        "layers": list(range(max_layer + 1)),
        "trajectories": np.round(traj, 4).tolist(),  # [layer][token][x,y]
        "injections": injections,
        "show": show,
        "seed": seed,
        "simulation_mode": True,
        "provenance": f"seeded(synthetic residual stream, seed {seed})",
    }
