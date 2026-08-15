"""
Transformer Detective — "why did the model predict this?" (plan §13, P0).

Metaphor: a case file. The model made a prediction; the user investigates
suspects (heads, features, FFN, residual) and closes the case with the most
influential evidence.

Evidence = algorithmically generated from a curated case library (frozen
plan decision: no real model). For every case we store the observed
attention rows and feature activations; the influence ranking is computed
deterministically (evidence score = attention weight × feature strength).

SIMULATION MODE: synthetic attention/features, not a real model.

Simulation contract: cached (case library lookup).

No-LLM principle: predictions, evidence and rankings are all computed; the
explanation text is authored by a human in the case library.
"""

from typing import Any

import hashlib

import numpy as np

CASES: list[dict[str, Any]] = [
    {
        "id": "cat-mat",
        "sentence": "the cat sat on the mat",
        "target": "mat",
        "question": "为什么模型预测 mat？",
        "prediction": [
            {"token": "mat", "prob": 0.61},
            {"token": "floor", "prob": 0.19},
            {"token": "chair", "prob": 0.07},
            {"token": "table", "prob": 0.04},
        ],
        # who looks at the target: head -> {source: weight}
        "head_rows": {
            "Head 2 (H2)": {"sat": 0.42, "on": 0.31, "the": 0.09},
            "Head 4 (H4)": {"on": 0.48, "sat": 0.22, "mat": 0.12},
            "Head 7 (H7)": {"cat": 0.35, "mat": 0.28, "sat": 0.11},
        },
        "features": [
            {"name": "Feature 821", "activated": ["sat", "on"], "strength": 0.55,
             "hypothesis": "spatial-completion pattern"},
            {"name": "Feature 1842", "activated": ["mat"], "strength": 0.34,
             "hypothesis": "noun-slot pattern"},
        ],
        "explanation": "sat/on 的注意力把「mat」抬进候选池；Feature 821 的空间补全模式最终选中它。",
        "explanation_en": "Attention from sat/on lifts 'mat' into the candidate pool; the spatial-completion feature finalizes it.",
    },
    {
        "id": "france-paris",
        "sentence": "the capital of france is paris",
        "target": "paris",
        "question": "为什么模型预测 paris？",
        "prediction": [
            {"token": "paris", "prob": 0.74},
            {"token": "london", "prob": 0.08},
            {"token": "rome", "prob": 0.05},
            {"token": "berlin", "prob": 0.03},
        ],
        "head_rows": {
            "Head 2 (H2)": {"capital": 0.38, "france": 0.29, "of": 0.12},
            "Head 5 (H5)": {"france": 0.52, "capital": 0.21},
            "Head 1 (H1)": {"is": 0.31, "france": 0.27},
        },
        "features": [
            {"name": "Feature 1842", "activated": ["capital", "france"], "strength": 0.68,
             "hypothesis": "capital-city relation"},
            {"name": "Feature 77", "activated": ["paris"], "strength": 0.29,
             "hypothesis": "country-capital lookup"},
        ],
        "explanation": "capital/france 激活 capital-city 关系特征，把「paris」提到榜首。",
        "explanation_en": "capital/france fire the capital-city relation feature, pushing 'paris' to the top.",
    },
    {
        "id": "rain-because",
        "sentence": "she walked to the store because it was raining",
        "target": "because",
        "question": "为什么模型预测 because？",
        "prediction": [
            {"token": "because", "prob": 0.47},
            {"token": "and", "prob": 0.18},
            {"token": "although", "prob": 0.12},
            {"token": "but", "prob": 0.09},
        ],
        "head_rows": {
            "Head 3 (H3)": {"raining": 0.44, "store": 0.26, "walked": 0.18},
            "Head 6 (H6)": {"walked": 0.38, "raining": 0.33},
            "Head 9 (H9)": {"store": 0.29, "raining": 0.25},
        },
        "features": [
            {"name": "Feature 421", "activated": ["walked", "raining"], "strength": 0.6,
             "hypothesis": "cause-effect bridge"},
            {"name": "Feature 15", "activated": ["because"], "strength": 0.31,
             "hypothesis": "conjunction slot"},
        ],
        "explanation": "walked/raining 激活因果桥特征，衔接词槽落定「because」。",
        "explanation_en": "walked/raining fire the cause-effect bridge; the conjunction slot lands on 'because'.",
    },
]


def _influence_rank(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic ranking of suspects by their evidence strength."""
    suspects: list[dict[str, Any]] = []

    for name, rows in case["head_rows"].items():
        total = sum(rows.values())
        top_src = max(rows, key=rows.get)  # type: ignore[arg-type]
        suspects.append({
            "type": "head",
            "name": name,
            "score": round(total, 3),
            "detail": f"{top_src} → {case['target']} ({rows[top_src]:.2f})",
        })
    for feat in case["features"]:
        suspects.append({
            "type": "feature",
            "name": feat["name"],
            "score": round(feat["strength"], 3),
            "detail": f"activates on {', '.join(feat['activated'])}",
        })

    suspects.sort(key=lambda s: s["score"], reverse=True)
    return suspects


def _row_seed(case_id: str, head: str) -> int:
    """Deterministic per-head seed from the case id + head name."""
    return int(hashlib.sha256(f"{case_id}:{head}".encode()).hexdigest()[:8], 16)


def _full_attention(
    tokens: list[str], target: str, head_row: dict[str, float], seed: int
) -> np.ndarray:
    """Build a complete n×n attention matrix for one head (deterministic).

    - The target row is pinned to the curated `head_row` (renormalized), so
      the existing influence ranking and challenge stay consistent.
    - Every other row is generated: sources the head prefers (present in the
      curated row) score higher, plus a position-decay term (nearby tokens
      look at each other) and a seeded jitter. This lets the user trace a
      chain — click a source token to see who IT looks at.
    """
    n = len(tokens)
    rng = np.random.default_rng(seed)
    pref = np.array([2.0 if t in head_row else 0.0 for t in tokens], dtype=float)

    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            scores[i, j] = pref[j] + 0.8 * np.exp(-d) + rng.normal(0, 0.15)

    # softmax over rows
    scores -= scores.max(axis=1, keepdims=True)
    W = np.exp(scores)
    W /= W.sum(axis=1, keepdims=True)

    # pin the target row to the curated evidence
    ti = tokens.index(target)
    curated = np.array([head_row.get(t, 0.0) for t in tokens], dtype=float)
    total = curated.sum()
    if total > 0:
        curated /= total
    W[ti] = curated
    return W


def compute(params: dict[str, Any]) -> dict[str, Any]:
    idx = int(params.get("case", 0)) % len(CASES)
    case = CASES[idx]
    ranking = _influence_rank(case)
    tokens = case["sentence"].split()

    # full attention matrix per head, keyed by token INDEX (token names can
    # repeat in a sentence, e.g. "the", and would collide as dict keys):
    #   {head: {"<query_idx>": {"<source_idx>": weight}}}
    head_rows_full: dict[str, dict[str, dict[str, float]]] = {}
    for name, rows in case["head_rows"].items():
        W = _full_attention(tokens, case["target"], rows, _row_seed(case["id"], name))
        head_rows_full[name] = {
            str(i): {str(j): round(float(W[i, j]), 4) for j in range(len(tokens))}
            for i in range(len(tokens))
        }

    return {
        "case_id": case["id"],
        "sentence": tokens,
        "target": case["target"],
        "question": case["question"],
        "prediction": case["prediction"],
        "head_rows": case["head_rows"],
        "head_rows_full": head_rows_full,
        "features": case["features"],
        "ranking": ranking,
        "explanation": case["explanation"],
        "explanation_en": case["explanation_en"],
        "n_cases": len(CASES),
        "provenance": f"cached(case library, case {case['id']})",
    }
