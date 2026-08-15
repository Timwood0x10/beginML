"""
MoE Expert Routing — "who picks up this token?" (plan §13, P1).

Metaphor: an expert triage room. Tokens arrive; a gate network routes each
one to the experts most suited to it. Loads build up per expert; top-k
routing keeps only the strongest connections so the routing is sparse and
the load distribution is visible.

SIMULATION MODE: token embeddings and the gate weights are synthetic
(seeded). This does NOT inspect a real MoE model — it demonstrates the
routing mechanism.

Simulation contract: realtime + seeded.

No-LLM principle: every number is computed; expert names come from a
deterministic rule over the routing load.
"""

from typing import Any

import numpy as np

D = 8  # embedding dim

SPECIALTIES = [
    "Syntax", "Semantics", "Math", "Common-Sense",
    "World-Knowledge", "Coding", "Translation", "Reasoning",
]


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def compute(params: dict[str, Any]) -> dict[str, Any]:
    n_experts = int(min(max(int(params.get("experts", 4)), 2), 8))
    top_k = int(min(max(int(params.get("top_k", 2)), 1), n_experts))
    n_tokens = int(params.get("tokens", 8))
    temperature = float(params.get("temperature", 1.0))
    seed = int(params.get("seed", 17))

    rng = np.random.default_rng(seed)

    # --- expert centres + deterministic specialty names -------------------
    centers = rng.normal(0, 1, size=(n_experts, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    specialties = [SPECIALTIES[i % len(SPECIALTIES)] for i in range(n_experts)]

    # --- token embeddings: each token leans toward one expert group ------
    tokens = [f"t{i}" for i in range(n_tokens)]
    E = rng.normal(0, 0.25, size=(n_tokens, D))
    for i in range(n_tokens):
        g = (i * 3 + seed) % n_experts  # deterministic affinity per token
        E[i] += centers[g] * 1.6

    # --- gate network: softmax(E @ W_gate) -------------------------------
    W_gate = rng.normal(0, 1.0 / np.sqrt(D), size=(D, n_experts))
    logits = (E @ W_gate) / max(temperature, 1e-3)
    routing = _stable_softmax(logits, axis=-1)  # [n_tokens, n_experts]

    # --- top-k routing: zero out everything below the k-th strongest -----
    mask = np.zeros_like(routing, dtype=bool)
    for i in range(n_tokens):
        idx = np.argsort(-routing[i])[:top_k]
        mask[i, idx] = True
    routed = np.where(mask, routing, 0.0)
    routed /= routed.sum(axis=1, keepdims=True)  # renormalize

    # --- expert load: total routing weight received ----------------------
    loads = routed.sum(axis=0)  # [n_experts]
    total = loads.sum() or 1.0

    experts = []
    for e in range(n_experts):
        experts.append({
            "id": e,
            "name": specialties[e],
            "load": round(float(loads[e]), 3),
            "load_pct": round(100.0 * float(loads[e]) / total, 1),
            "tokens": [i for i in range(n_tokens) if routed[i, e] > 0.05],
        })

    return {
        "tokens": tokens,
        "n_tokens": n_tokens,
        "n_experts": n_experts,
        "top_k": top_k,
        "temperature": round(temperature, 2),
        "routing": np.round(routed, 4).tolist(),  # [token][expert]
        "experts": experts,
        "loads": [round(float(l), 3) for l in loads],
        "load_total": round(float(total), 3),
        "seed": seed,
        "simulation_mode": True,
        "provenance": f"seeded(gate network, seed {seed})",
    }
