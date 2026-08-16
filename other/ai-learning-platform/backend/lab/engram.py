"""Engram — conditional memory: an O(1) hash-lookup axis for LLMs.

Deterministic synthetic simulation (numpy, seeded) of the core ideas from
the paper note zh/paper/ds_engram.md — no real model is trained or run:

  1. Retrieval cost:  O(1) hash lookup per token vs O(L) attention
  2. Multi-head hashing:  collision rate decays as heads grow
  3. Context-aware gating:  alpha_t = sigmoid(alignment), filters noise
  4. U-shaped scaling law:  Engram/MoE parameter split (rho) has a sweet spot
  5. Parameter/FLOPs decoupling:  more memory slots, same inference compute
"""

from typing import Any

import numpy as np

SEED = 7
OPT_RHO = 22.0  # paper: ~20-25% of sparse params go to Engram


def compute(params: dict[str, Any]) -> dict[str, Any]:
    n_gram = int(min(max(int(params.get("n_gram", 3)), 2), 5))
    heads = int(min(max(int(params.get("heads", 3)), 1), 8))
    L = int(min(max(int(params.get("tokens", 64)), 8), 512))
    slots = int(min(max(int(params.get("slots", 4096)), 256), 65536))
    rho = float(min(max(float(params.get("rho", 22.0)), 0.0), 100.0))
    rng = np.random.default_rng(SEED)

    # ---- 1) retrieval cost: Engram O(1) vs attention O(L) ----------------
    lengths = list(range(1, L + 1))
    cost_engram = [1.0 * l for l in lengths]  # each token: one hash lookup
    cost_attn = [l * l for l in lengths]  # each token attends to all previous
    # at the final length
    attn_over_engram = (L * L) / max(1.0 * L, 1.0)

    # ---- 2) multi-head hashing collision rate ----------------------------
    # Birthday-paradox collision probability for ONE head over L tokens in
    # `slots` buckets; K independent heads collide only if ALL heads do.
    p1 = 1.0 - float(np.exp(-(L * (L - 1)) / (2.0 * slots)))
    head_curve = [1, 2, 3, 4, 5, 6, 7, 8]
    p_all = [round(min(1.0, p1**k), 5) for k in head_curve]
    collision_now = p_all[heads - 1]

    # ---- 3) context-aware gating alpha_t ---------------------------------
    # Synthetic alignment scores: mostly aligned (matched n-gram), some noise.
    n_tok = 16
    aligned = rng.normal(0.8, 0.15, n_tok)
    noise = rng.normal(-0.4, 0.2, n_tok)
    scores = np.concatenate([aligned[: n_tok // 2], noise[n_tok // 2 :]])
    alpha = 1.0 / (1.0 + np.exp(-scores))
    # paper: alpha -> 0 filters retrieved knowledge that conflicts with context
    gate_curve = [round(float(a), 4) for a in alpha]
    gate_mean = round(float(alpha.mean()), 4)

    # ---- 4) U-shaped scaling law over rho --------------------------------
    rhos = list(range(0, 101, 5))
    # loss rises away from the optimum (squared penalty, plus base + noise)
    loss = [
        round(
            1.40 + 0.55 * ((r - OPT_RHO) / 100.0) ** 2 + float(rng.normal(0, 0.004)), 4
        )
        for r in rhos
    ]
    rho_now = int(round(rho / 5) * 5)
    loss_now = loss[rhos.index(min(rho_now, 100))] if rho_now <= 100 else loss[-1]

    # ---- 5) parameter / FLOPs decoupling ---------------------------------
    slots_axis = [256, 1024, 4096, 16384, 65536]
    params_growth = [round(1.0 * s / 256.0, 3) for s in slots_axis]  # linear in slots
    flops_flat = [1.0] * len(slots_axis)  # same FLOPs regardless of memory

    return {
        "n_gram": n_gram,
        "heads": heads,
        "tokens": L,
        "slots": slots,
        "rho": rho,
        "lengths": lengths,
        "cost_engram": [round(v, 2) for v in cost_engram],
        "cost_attn": [round(v, 2) for v in cost_attn],
        "attn_over_engram": round(attn_over_engram, 2),
        "collision_curve": p_all,
        "collision_now": collision_now,
        "gate_curve": gate_curve,
        "gate_mean": gate_mean,
        "rho_axis": rhos,
        "loss_curve": loss,
        "loss_now": loss_now,
        "slots_axis": slots_axis,
        "params_growth": params_growth,
        "flops_flat": flops_flat,
        "opt_rho": OPT_RHO,
        "provenance": "seeded(synthetic Engram model, no real LLM)",
        "simulation_mode": True,
    }
