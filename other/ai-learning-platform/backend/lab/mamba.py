"""
Mamba Memory Race — O(L) vs O(L²) as an experiment (plan §13, P1).

Metaphor: a memory race. Every token in a sequence must "remember and
revisit" the past. A Transformer attention layer does O(L²) work per layer —
every token looks back at every previous token. A Mamba-style SSM keeps a
fixed-size state, so each token only touches that state: O(L) per layer.
Grow the sequence and watch the quadratic curve pull ahead.

Numbers here are a complexity MODEL (deterministic formulas), not a runtime
benchmark. This is honest: we show flops and memory as proportional counts
per sequence length.

Simulation contract: realtime.

No-LLM principle: everything is computed; nothing is generated.
"""

from typing import Any

D = 16  # hidden dim used in the complexity model
LAYERS = 4


def _flops(L: int) -> tuple[float, float]:
    """(transformer, mamba) FLOPs for one forward pass."""
    # Transformer attention: L²·d per layer (QKᵀ + softmax + AV) + MLP L·d²
    trans = LAYERS * (2 * L * L * D + 2 * L * D * D)
    # Mamba SSM: per layer the state update and output projection are L·d
    mamba = LAYERS * (2 * L * D + 2 * L * D * D)
    return float(trans), float(mamba)


def _memory(L: int) -> tuple[float, float]:
    """(transformer, mamba) peak activation memory as proportional counts."""
    # Transformer must materialize the L×L attention matrix
    trans = float(L * L + L * D)
    # Mamba keeps only the fixed d×d state + L·d activations
    mamba = float(L * D + D * D)
    return trans, mamba


def compute(params: dict[str, Any]) -> dict[str, Any]:
    L = int(params.get("length", 64))
    L = max(4, min(256, L))

    lengths = list(range(4, 257, 4))
    trans_flops = []
    mamba_flops = []
    trans_mem = []
    mamba_mem = []
    for l in lengths:
        tf, mf = _flops(l)
        tm, mm = _memory(l)
        trans_flops.append(tf)
        mamba_flops.append(mf)
        trans_mem.append(tm)
        mamba_mem.append(mm)

    tf, mf = _flops(L)
    tm, mm = _memory(L)
    flops_ratio = tf / max(mf, 1e-9)
    mem_ratio = tm / max(mm, 1e-9)

    # "cross point": the smallest L where transformer is >2× heavier
    cross = None
    for l in lengths:
        t, m = _flops(l)
        if t > 2 * m:
            cross = l
            break

    return {
        "length": L,
        "d": D,
        "layers": LAYERS,
        "lengths": lengths,
        "transformer_flops": [round(v, 3) for v in trans_flops],
        "mamba_flops": [round(v, 3) for v in mamba_flops],
        "transformer_mem": [round(v, 3) for v in trans_mem],
        "mamba_mem": [round(v, 3) for v in mamba_mem],
        "transformer_flops_now": round(tf, 3),
        "mamba_flops_now": round(mf, 3),
        "flops_ratio": round(flops_ratio, 3),
        "transformer_mem_now": round(tm, 3),
        "mamba_mem_now": round(mm, 3),
        "mem_ratio": round(mem_ratio, 3),
        "cross_point": cross,
        "provenance": "realtime(complexity model)",
    }
