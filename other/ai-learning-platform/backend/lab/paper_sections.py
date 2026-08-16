"""paper_sections.py — section ↔ source ↔ run mapping for transformer.pdf.

Each selected section of the paper gets:
  * `code`   — a self-contained, runnable numpy implementation of that
               section's core formula (shown in the middle column)
  * `run()`  — the same logic executed with a fixed seed, returning the
               numbers it actually computes (right column)

No-LLM principle: every number is computed, nothing is generated.
Sections map to pdf_paper.parse_paper() ids by order:
  s1  1.1 single query            -> attention of one query
  s2  1.2 Scaled Dot-Product Attn -> SDPA = softmax(QKᵀ/√dₖ)V
  s4  1.4 multiple queries        -> matrix form
  s7  3.1 K-V cache motivation    -> FLOPs saved by caching
  s8  3.2 Multi-head Latent Attn  -> MLA: compressed KV
  s9  3.3 RoPE                    -> rotary position embedding
  s10 3.4 Decoupled RoPE          -> Q full-rotate + K partial
  s12 4.1 online softmax          -> flash-attention style chunking
  s15 5.1 residual connection     -> x_L = x_{L-1} + F(x_{L-1})
"""

from typing import Any, Callable

import numpy as np

SEED = 7


def _softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


# ---- s1: 1.1 single query attention --------------------------------------
S1_CODE = '''
import numpy as np
rng = np.random.default_rng(7)
d = 4
q = rng.normal(0, 1, d)          # one query
K = rng.normal(0, 1, (5, d))     # five keys
V = rng.normal(0, 1, (5, d))     # five values

scores = K @ q                   # query-key dot products
w = np.exp(scores) / np.exp(scores).sum()
out = w @ V                      # weighted sum of values
'''


def run_s1() -> dict:
    rng = np.random.default_rng(SEED)
    d = 4
    q = rng.normal(0, 1, d)
    K = rng.normal(0, 1, (5, d))
    V = rng.normal(0, 1, (5, d))
    scores = K @ q
    w = np.exp(scores) / np.exp(scores).sum()
    out = w @ V
    return {"scores": [round(float(x), 4) for x in scores], "weights": [round(float(x), 4) for x in w],
            "out": [round(float(x), 4) for x in out], "weights_sum": round(float(w.sum()), 4)}


# ---- s2: 1.2 Scaled Dot-Product Attention ---------------------------------
S2_CODE = '''
import numpy as np
rng = np.random.default_rng(7)
d_k = 8
Q = rng.normal(0, 1, (3, d_k))
K = rng.normal(0, 1, (5, d_k))
V = rng.normal(0, 1, (5, d_k))

scores = Q @ K.T / np.sqrt(d_k)   # scaled by 1/sqrt(d_k)
w = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
out = w @ V
'''


def run_s2() -> dict:
    rng = np.random.default_rng(SEED)
    d_k = 8
    Q = rng.normal(0, 1, (3, d_k))
    K = rng.normal(0, 1, (5, d_k))
    V = rng.normal(0, 1, (5, d_k))
    scores = Q @ K.T / np.sqrt(d_k)
    w = _softmax(scores)
    out = w @ V
    return {"unscaled_max": round(float((Q @ K.T).max()), 4), "scaled_max": round(float(scores.max()), 4),
            "rows_sum_1": [round(float(r), 6) for r in w.sum(axis=-1)],
            "out_shape": list(out.shape)}


# ---- s4: 1.4 multiple queries (matrix form) -------------------------------
S4_CODE = '''
import numpy as np
rng = np.random.default_rng(7)
n, d = 4, 6
X = rng.normal(0, 1, (n, d))
Wq, Wk, Wv = (rng.normal(0, 1, (d, d)) for _ in range(3))

Q, K, V = X @ Wq, X @ Wk, X @ Wv
A = Q @ K.T / np.sqrt(d)
P = np.exp(A) / np.exp(A).sum(axis=-1, keepdims=True)
Y = P @ V                       # n x d output
'''


def run_s4() -> dict:
    rng = np.random.default_rng(SEED)
    n, d = 4, 6
    X = rng.normal(0, 1, (n, d))
    Wq, Wk, Wv = (rng.normal(0, 1, (d, d)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    A = Q @ K.T / np.sqrt(d)
    P = _softmax(A)
    Y = P @ V
    return {"attn_shape": list(A.shape), "row_entropy": round(float(-(P * np.log(P + 1e-12)).sum(axis=-1).mean()), 4),
            "y_norm": round(float(np.linalg.norm(Y)), 4)}


# ---- s7: 3.1 K-V cache motivation (FLOPs saved) ---------------------------
S7_CODE = '''
# decoding token t: without KV cache we recompute ALL past attention;
# with a cache we only touch the new token.
t = 10            # tokens already decoded
d = 64
no_cache = (t + 1) * (t + 1) * d   # attention over all pairs
with_cache = (t + 1) * d           # only the new row
print("saved", (1 - with_cache / no_cache) * 100, "%")
'''


def run_s7(params: dict) -> dict:
    t = int(min(max(int(params.get("tokens", 10)), 2), 512))
    d = int(min(max(int(params.get("d", 64)), 8), 1024))
    no_cache = (t + 1) * (t + 1) * d
    with_cache = (t + 1) * d
    return {"t": t, "d": d, "flops_no_cache": no_cache, "flops_with_cache": with_cache,
            "saved_pct": round((1 - with_cache / no_cache) * 100, 2)}


# ---- s8: 3.2 Multi-head Latent Attention (MLA) ----------------------------
S8_CODE = '''
# MLA compresses K/V into a small latent z before the cache,
# so the KV cache stores z (tiny) instead of the full K,V.
import numpy as np
rng = np.random.default_rng(7)
n, d, d_c = 8, 64, 16
X = rng.normal(0, 1, (n, d))
W_down = rng.normal(0, 1, (d, d_c)) / np.sqrt(d)

z = X @ W_down                # latent: n x d_c  (compressed)
kv_cache_full = n * 2 * d     # classic KV cache size
kv_cache_mla = n * d_c        # MLA cache size
'''


def run_s8(params: dict) -> dict:
    n = int(min(max(int(params.get("tokens", 8)), 2), 128))
    d = int(min(max(int(params.get("d", 64)), 8), 512))
    d_c = int(min(max(int(params.get("d_c", 16)), 4), 64))
    rng = np.random.default_rng(SEED)
    X = rng.normal(0, 1, (n, d))
    W_down = rng.normal(0, 1, (d, d_c)) / np.sqrt(d)
    z = X @ W_down
    full = n * 2 * d
    mla = n * d_c
    return {"z_shape": list(z.shape), "kv_full": full, "kv_mla": mla,
            "compression": round(full / mla, 2)}


# ---- s9: 3.3 RoPE ---------------------------------------------------------
S9_CODE = '''
import numpy as np
def rope(x, m, theta=10000.0):
    d = x.shape[-1]
    freqs = theta ** (-np.arange(0, d, 2) / d)
    ang = m * freqs
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).reshape(x.shape)

x = np.array([1.0, 0.0, 1.0, 0.0])
q4 = rope(x, 4)
q8 = rope(x, 8)
print("sim(4,8) =", q4 @ q8 / (np.linalg.norm(q4) * np.linalg.norm(q8)))
'''


def run_s9() -> dict:
    def rope(x, m, theta=10000.0):
        d = x.shape[-1]
        freqs = theta ** (-np.arange(0, d, 2) / d)
        ang = m * freqs
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return np.stack([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).reshape(x.shape)

    x = np.array([1.0, 0.0, 1.0, 0.0])
    q4 = rope(x, 4)
    q8 = rope(x, 8)
    sim = float(q4 @ q8 / (np.linalg.norm(q4) * np.linalg.norm(q8)))
    return {"sim_4_8": round(sim, 4), "norm": round(float(np.linalg.norm(q4)), 4)}


# ---- s10: 3.4 Decoupled RoPE ----------------------------------------------
S10_CODE = '''
# Decoupled RoPE (DeepSeek): rotate ONLY the query and part of the key,
# keep the value (and most of the key) unrotated for cache-friendliness.
import numpy as np
q = np.array([0.8, 0.6, 0.5, 0.4])
k = np.array([0.9, 0.2, 0.7, 0.3])
q_half, q_rope = q[:2], q[2:]      # half query gets rotation
k_half, k_rope = k[:2], k[2:]
theta = 0.5
rot = lambda v: np.array([v[0]*np.cos(theta) - v[1]*np.sin(theta),
                          v[0]*np.sin(theta) + v[1]*np.cos(theta)])
q_rope_r, k_rope_r = rot(q_rope), rot(k_rope)
score = (q_half @ k_half + q_rope_r @ k_rope_r) / np.sqrt(len(q))
'''


def run_s10() -> dict:
    q = np.array([0.8, 0.6, 0.5, 0.4])
    k = np.array([0.9, 0.2, 0.7, 0.3])
    q_half, q_rope = q[:2], q[2:]
    k_half, k_rope = k[:2], k[2:]
    theta = 0.5
    rot = lambda v: np.array([v[0] * np.cos(theta) - v[1] * np.sin(theta),
                              v[0] * np.sin(theta) + v[1] * np.cos(theta)])
    q_rope_r, k_rope_r = rot(q_rope), rot(k_rope)
    score = float((q_half @ k_half + q_rope_r @ k_rope_r) / np.sqrt(len(q)))
    return {"score": round(score, 4), "rotated_q": [round(float(x), 4) for x in q_rope_r]}


# ---- s12: 4.1 online softmax (flash-attention style) ----------------------
S12_CODE = '''
# Online softmax: run max & exp-sum incrementally over chunks, so we never
# materialize the full QK^T matrix (flash-attention's core trick).
import numpy as np
scores = np.array([2.0, 1.0, 3.0, 0.5, 2.5])
m, l = -np.inf, 0.0
for x in scores:                       # one pass, chunk = one element
    m_new = max(m, x)
    l = l * np.exp(m - m_new) + np.exp(x - m_new)
    m = m_new
w = np.exp(scores - m) / l
'''


def run_s12() -> dict:
    scores = np.array([2.0, 1.0, 3.0, 0.5, 2.5])
    m, l = -np.inf, 0.0
    for x in scores:
        m_new = max(m, x)
        l = l * np.exp(m - m_new) + np.exp(x - m_new)
        m = m_new
    w = np.exp(scores - m) / l
    ref = _softmax(scores.reshape(1, -1))[0]
    return {"weights": [round(float(x), 4) for x in w],
            "max_err_vs_full": round(float(np.abs(w - ref).max()), 8),
            "exp_sum": round(float(l), 4)}


# ---- s15: 5.1 residual connection -----------------------------------------
S15_CODE = '''
# Residual stream: x_L = x_{L-1} + F(x_{L-1}) — the block learns a
# correction F, so gradients flow through the identity path.
import numpy as np
rng = np.random.default_rng(7)
x = rng.normal(0, 1, 8)
W = rng.normal(0, 1, (8, 8)) / 4
F = np.tanh(x @ W)          # a tiny "transformer block"
x_next = x + F              # residual add
print("residual norm kept:", np.linalg.norm(x_next) / np.linalg.norm(x))
'''


def run_s15() -> dict:
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, 8)
    W = rng.normal(0, 1, (8, 8)) / 4
    F = np.tanh(x @ W)
    x_next = x + F
    return {"x_norm": round(float(np.linalg.norm(x)), 4),
            "x_next_norm": round(float(np.linalg.norm(x_next)), 4),
            "ratio": round(float(np.linalg.norm(x_next) / np.linalg.norm(x)), 4)}


# ---- section registry ------------------------------------------------------

_SECTIONS: dict[str, dict[str, Any]] = {
    "s1": {"title": "1.1 single query", "code": S1_CODE, "run": run_s1},
    "s2": {"title": "1.2 Scaled Dot-Product Attention", "code": S2_CODE, "run": run_s2},
    "s4": {"title": "1.4 multiple queries", "code": S4_CODE, "run": run_s4},
    "s7": {"title": "3.1 K-V cache motivation", "code": S7_CODE, "run": run_s7},
    "s8": {"title": "3.2 Multi-head Latent Attention", "code": S8_CODE, "run": run_s8},
    "s9": {"title": "3.3 RoPE", "code": S9_CODE, "run": run_s9},
    "s10": {"title": "3.4 Decoupled RoPE", "code": S10_CODE, "run": run_s10},
    "s12": {"title": "4.1 online softmax", "code": S12_CODE, "run": run_s12},
    "s15": {"title": "5.1 residual connection", "code": S15_CODE, "run": run_s15},
}


def section_ids() -> list[str]:
    return list(_SECTIONS.keys())


def section_source(section_id: str) -> dict[str, Any] | None:
    s = _SECTIONS.get(section_id)
    if s is None:
        return None
    return {"id": section_id, "title": s["title"], "code": s["code"]}


def run_section(section_id: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    s = _SECTIONS.get(section_id)
    if s is None:
        return None
    fn: Callable[..., dict] = s["run"]
    try:
        result = fn(params or {}) if section_id in ("s7", "s8") else fn()
    except TypeError:
        result = fn()
    return {"id": section_id, "title": s["title"], "result": result}
