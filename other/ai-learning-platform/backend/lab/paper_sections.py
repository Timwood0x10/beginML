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

from collections.abc import Callable
from typing import Any

import numpy as np

SEED = 7


def _softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


# ---- s1: 1.1 single query attention --------------------------------------
S1_CODE = """
import numpy as np
rng = np.random.default_rng(7)
d = 4
q = rng.normal(0, 1, d)          # one query
K = rng.normal(0, 1, (5, d))     # five keys
V = rng.normal(0, 1, (5, d))     # five values

scores = K @ q                   # query-key dot products
w = np.exp(scores) / np.exp(scores).sum()
out = w @ V                      # weighted sum of values
"""


def run_s1() -> dict:
    rng = np.random.default_rng(SEED)
    d = 4
    q = rng.normal(0, 1, d)
    K = rng.normal(0, 1, (5, d))
    V = rng.normal(0, 1, (5, d))
    scores = K @ q
    w = np.exp(scores) / np.exp(scores).sum()
    out = w @ V
    return {
        "scores": [round(float(x), 4) for x in scores],
        "weights": [round(float(x), 4) for x in w],
        "out": [round(float(x), 4) for x in out],
        "weights_sum": round(float(w.sum()), 4),
    }


# ---- s2: 1.2 Scaled Dot-Product Attention ---------------------------------
S2_CODE = """
import numpy as np
rng = np.random.default_rng(7)
d_k = 8
Q = rng.normal(0, 1, (3, d_k))
K = rng.normal(0, 1, (5, d_k))
V = rng.normal(0, 1, (5, d_k))

scores = Q @ K.T / np.sqrt(d_k)   # scaled by 1/sqrt(d_k)
w = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
out = w @ V
"""


def run_s2(params: dict | None = None) -> dict:
    rng = np.random.default_rng(SEED)
    d_k = int(min(max(int((params or {}).get("d_k", 8)), 2), 128))
    Q = rng.normal(0, 1, (3, d_k))
    K = rng.normal(0, 1, (5, d_k))
    V = rng.normal(0, 1, (5, d_k))
    raw = Q @ K.T
    scores = raw / np.sqrt(d_k)
    w = _softmax(scores)
    out = w @ V
    return {
        "d_k": d_k,
        "unscaled_max": round(float(raw.max()), 4),
        "scaled_max": round(float(scores.max()), 4),
        "unscaled_var": round(float(raw.var()), 4),
        "scaled_var": round(float(scores.var()), 4),
        "rows_sum_1": [round(float(r), 6) for r in w.sum(axis=-1)],
        "entropy": round(float(-(w * np.log(w + 1e-12)).sum(axis=-1).mean()), 4),
        "out_shape": list(out.shape),
        "visual": {
            "type": "attn",
            "matrix": [[round(float(v), 4) for v in row] for row in w.tolist()],
            "rows": 3,
            "cols": 5,
        },
    }


# ---- s4: 1.4 multiple queries (matrix form) -------------------------------
S4_CODE = """
import numpy as np
rng = np.random.default_rng(7)
n, d = 4, 6
X = rng.normal(0, 1, (n, d))
Wq, Wk, Wv = (rng.normal(0, 1, (d, d)) for _ in range(3))

Q, K, V = X @ Wq, X @ Wk, X @ Wv
A = Q @ K.T / np.sqrt(d)
P = np.exp(A) / np.exp(A).sum(axis=-1, keepdims=True)
Y = P @ V                       # n x d output
"""


def run_s4() -> dict:
    rng = np.random.default_rng(SEED)
    n, d = 4, 6
    X = rng.normal(0, 1, (n, d))
    Wq, Wk, Wv = (rng.normal(0, 1, (d, d)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    A = Q @ K.T / np.sqrt(d)
    P = _softmax(A)
    Y = P @ V
    return {
        "attn_shape": list(A.shape),
        "row_entropy": round(float(-(P * np.log(P + 1e-12)).sum(axis=-1).mean()), 4),
        "y_norm": round(float(np.linalg.norm(Y)), 4),
    }


# ---- s7: 3.1 K-V cache motivation (FLOPs saved) ---------------------------
S7_CODE = """
# decoding token t: without KV cache we recompute ALL past attention;
# with a cache we only touch the new token.
t = 10            # tokens already decoded
d = 64
no_cache = (t + 1) * (t + 1) * d   # attention over all pairs
with_cache = (t + 1) * d           # only the new row
print("saved", (1 - with_cache / no_cache) * 100, "%")
"""


def run_s7(params: dict) -> dict:
    t = int(min(max(int(params.get("tokens", 10)), 2), 512))
    d = int(min(max(int(params.get("d", 64)), 8), 1024))
    no_cache = (t + 1) * (t + 1) * d
    with_cache = (t + 1) * d
    return {
        "t": t,
        "d": d,
        "flops_no_cache": no_cache,
        "flops_with_cache": with_cache,
        "saved_pct": round((1 - with_cache / no_cache) * 100, 2),
    }


# ---- s8: 3.2 Multi-head Latent Attention (MLA) ----------------------------
S8_CODE = """
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
"""


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
    return {
        "z_shape": list(z.shape),
        "kv_full": full,
        "kv_mla": mla,
        "compression": round(full / mla, 2),
    }


# ---- s9: 3.3 RoPE ---------------------------------------------------------
S9_CODE = """
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
"""


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
    return {
        "sim_4_8": round(sim, 4),
        "norm": round(float(np.linalg.norm(q4)), 4),
        "visual": {
            "type": "flow",
            "steps": [
                {"label": "input x", "shape": "4", "value": [round(float(v), 3) for v in x]},
                {"label": "rope(x, 4)", "shape": "4", "value": [round(float(v), 3) for v in q4]},
                {"label": "rope(x, 8)", "shape": "4", "value": [round(float(v), 3) for v in q8]},
                {"label": "cos sim(4, 8)", "shape": "1", "value": round(sim, 4)},
            ],
        },
    }


# ---- s10: 3.4 Decoupled RoPE ----------------------------------------------
S10_CODE = """
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
"""


def run_s10() -> dict:
    q = np.array([0.8, 0.6, 0.5, 0.4])
    k = np.array([0.9, 0.2, 0.7, 0.3])
    q_half, q_rope = q[:2], q[2:]
    k_half, k_rope = k[:2], k[2:]
    theta = 0.5
    rot = lambda v: np.array(
        [
            v[0] * np.cos(theta) - v[1] * np.sin(theta),
            v[0] * np.sin(theta) + v[1] * np.cos(theta),
        ]
    )
    q_rope_r, k_rope_r = rot(q_rope), rot(k_rope)
    score = float((q_half @ k_half + q_rope_r @ k_rope_r) / np.sqrt(len(q)))
    return {
        "score": round(score, 4),
        "rotated_q": [round(float(x), 4) for x in q_rope_r],
    }


# ---- s12: 4.1 online softmax (flash-attention style) ----------------------
S12_CODE = """
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
"""


def run_s12() -> dict:
    scores = np.array([2.0, 1.0, 3.0, 0.5, 2.5])
    m, l = -np.inf, 0.0
    for x in scores:
        m_new = max(m, x)
        l = l * np.exp(m - m_new) + np.exp(x - m_new)
        m = m_new
    w = np.exp(scores - m) / l
    ref = _softmax(scores.reshape(1, -1))[0]
    return {
        "weights": [round(float(x), 4) for x in w],
        "max_err_vs_full": round(float(np.abs(w - ref).max()), 8),
        "exp_sum": round(float(l), 4),
    }


# ---- s15: 5.1 residual connection -----------------------------------------
S15_CODE = """
# Residual stream: x_L = x_{L-1} + F(x_{L-1}) — the block learns a
# correction F, so gradients flow through the identity path.
import numpy as np
rng = np.random.default_rng(7)
x = rng.normal(0, 1, 8)
W = rng.normal(0, 1, (8, 8)) / 4
F = np.tanh(x @ W)          # a tiny "transformer block"
x_next = x + F              # residual add
print("residual norm kept:", np.linalg.norm(x_next) / np.linalg.norm(x))
"""


def run_s15() -> dict:
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, 8)
    W = rng.normal(0, 1, (8, 8)) / 4
    F = np.tanh(x @ W)
    x_next = x + F
    return {
        "x_norm": round(float(np.linalg.norm(x)), 4),
        "x_next_norm": round(float(np.linalg.norm(x_next)), 4),
        "ratio": round(float(np.linalg.norm(x_next) / np.linalg.norm(x)), 4),
        "visual": {
            "type": "flow",
            "steps": [
                {"label": "x", "shape": "8", "value": [round(float(v), 3) for v in x]},
                {"label": "F(x) = tanh(x·W)", "shape": "8", "value": [round(float(v), 3) for v in F]},
                {"label": "x + F(x)", "shape": "8", "value": [round(float(v), 3) for v in x_next]},
            ],
        },
    }


# ---- s3: 1.3 seq2seq attention ---------------------------------------------
S3_CODE = """
import numpy as np
rng = np.random.default_rng(7)
d = 6
q = rng.normal(0, 1, d)            # decoder hidden state (query)
K = rng.normal(0, 1, (8, d))       # encoder keys (source tokens)
V = rng.normal(0, 1, (8, d))       # encoder values
scores = K @ q
w = np.exp(scores) / np.exp(scores).sum()
context = w @ V                    # context vector fed to the decoder
"""


def run_s3() -> dict:
    rng = np.random.default_rng(SEED)
    d = 6
    q = rng.normal(0, 1, d)
    K = rng.normal(0, 1, (8, d))
    V = rng.normal(0, 1, (8, d))
    scores = K @ q
    w = np.exp(scores) / np.exp(scores).sum()
    context = w @ V
    return {
        "top_key": int(np.argmax(w)),
        "top_weight": round(float(w.max()), 4),
        "weights_sum": round(float(w.sum()), 4),
        "context_norm": round(float(np.linalg.norm(context)), 4),
    }


# ---- s5: 2.1 causal self-attention ------------------------------------------
S5_CODE = """
import numpy as np
rng = np.random.default_rng(7)
n, d = 6, 8
X = rng.normal(0, 1, (n, d))
Wq, Wk, Wv = (rng.normal(0, 1, (d, d)) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = Q @ K.T / np.sqrt(d)
mask = np.triu(np.full((n, n), -1e9), k=1)   # hide future tokens
w = np.exp(scores + mask) / np.exp(scores + mask).sum(axis=-1, keepdims=True)
out = w @ V
"""


def run_s5() -> dict:
    rng = np.random.default_rng(SEED)
    n, d = 6, 8
    X = rng.normal(0, 1, (n, d))
    Wq, Wk, Wv = (rng.normal(0, 1, (d, d)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(d)
    mask = np.triu(np.full((n, n), -1e9), k=1)
    w = np.exp(scores + mask) / np.exp(scores + mask).sum(axis=-1, keepdims=True)
    return {
        "row0_supports": int((w[0] > 1e-6).sum()),  # only itself
        "rowN_supports": int((w[-1] > 1e-6).sum()),  # all past
        "row0_top": int(np.argmax(w[0])),
        "rowN_top": int(np.argmax(w[-1])),
        "visual": {
            "type": "attn",
            "matrix": [[round(float(v), 4) for v in row] for row in w.tolist()],
            "rows": n,
            "cols": n,
        },
    }


# ---- s6: 2.2 cross-attention ------------------------------------------------
S6_CODE = """
# Cross-attention: Q comes from the target sequence, K/V from the source.
import numpy as np
rng = np.random.default_rng(7)
d = 6
Q = rng.normal(0, 1, (4, d))    # target (decoder)
K = rng.normal(0, 1, (7, d))    # source (encoder)
V = rng.normal(0, 1, (7, d))
A = Q @ K.T / np.sqrt(d)
P = np.exp(A) / np.exp(A).sum(axis=-1, keepdims=True)
Y = P @ V
"""


def run_s6() -> dict:
    rng = np.random.default_rng(SEED)
    d = 6
    Q = rng.normal(0, 1, (4, d))
    K = rng.normal(0, 1, (7, d))
    V = rng.normal(0, 1, (7, d))
    A = Q @ K.T / np.sqrt(d)
    P = np.exp(A) / np.exp(A).sum(axis=-1, keepdims=True)
    Y = P @ V
    return {
        "attn_shape": list(A.shape),
        "rows_sum_1": [round(float(r), 6) for r in P.sum(axis=-1)],
        "y_norm": round(float(np.linalg.norm(Y)), 4),
    }


# ---- s11: 3.5 KV cache extension --------------------------------------------
S11_CODE = """
# KV cache extension: decode k new tokens in ONE batch.
t = 10   # tokens already cached
k = 4    # new tokens decoded together
d = 64
no_cache = (t + k) * (t + k) * d      # recompute all pairs
with_cache = (t + k) * d              # only the new rows
"""


def run_s11(params: dict | None = None) -> dict:
    t = int(min(max(int((params or {}).get("tokens", 10)), 2), 512))
    k = int(min(max(int((params or {}).get("batch", 4)), 1), 64))
    d = int(min(max(int((params or {}).get("d", 64)), 8), 1024))
    no_cache = (t + k) * (t + k) * d
    with_cache = (t + k) * d
    return {
        "t": t,
        "k": k,
        "flops_no_cache": no_cache,
        "flops_with_cache": with_cache,
        "saved_pct": round((1 - with_cache / no_cache) * 100, 2),
    }


# ---- s13: 4.2 online softmax loop -------------------------------------------
S13_CODE = """
# Online softmax, LOOP form: update (m, l) chunk by chunk.
import numpy as np
chunks = [np.array([2.0, 1.0]), np.array([3.0, 0.5]), np.array([2.5])]
m, l = -np.inf, 0.0
for x in chunks:
    m_new = max(m, x.max())
    l = l * np.exp(m - m_new) + np.exp(x - m_new).sum()
    m = m_new
w = np.exp(np.concatenate(chunks) - m) / l
"""


def run_s13() -> dict:
    chunks = [np.array([2.0, 1.0]), np.array([3.0, 0.5]), np.array([2.5])]
    m, l = -np.inf, 0.0
    for x in chunks:
        m_new = max(m, x.max())
        l = l * np.exp(m - m_new) + np.exp(x - m_new).sum()
        m = m_new
    all_scores = np.concatenate(chunks)
    w = np.exp(all_scores - m) / l
    ref = _softmax(all_scores.reshape(1, -1))[0]
    return {
        "weights": [round(float(x), 4) for x in w],
        "max_err_vs_full": round(float(np.abs(w - ref).max()), 8),
        "exp_sum": round(float(l), 4),
    }


# ---- s14: 4.3 single-query row form -----------------------------------------
S14_CODE = """
# Single-query row form: one query q scores against the whole key matrix.
import numpy as np
rng = np.random.default_rng(7)
d = 8
q = rng.normal(0, 1, d)
K = rng.normal(0, 1, (5, d))
V = rng.normal(0, 1, (5, d))
row = (q @ K.T) / np.sqrt(d)    # 1 x m score row
w = np.exp(row) / np.exp(row).sum()
out = w @ V
"""


def run_s14() -> dict:
    rng = np.random.default_rng(SEED)
    d = 8
    q = rng.normal(0, 1, d)
    K = rng.normal(0, 1, (5, d))
    V = rng.normal(0, 1, (5, d))
    row = (q @ K.T) / np.sqrt(d)
    w = np.exp(row) / np.exp(row).sum()
    out = w @ V
    return {
        "row_shape": list(row.shape),
        "weights": [round(float(x), 4) for x in w],
        "weights_sum": round(float(w.sum()), 4),
        "out_norm": round(float(np.linalg.norm(out)), 4),
        "visual": {
            "type": "attn",
            "matrix": [[round(float(v), 4) for v in w.tolist()]],
            "rows": 1,
            "cols": 5,
        },
    }


# ---- s16: 5.2 residual gradient ---------------------------------------------
S16_CODE = """
# Residual gradient: dy/dx = I + dF/dx. The identity path keeps gradients
# O(1) across depth, where a plain chain would vanish.
import numpy as np
dFdx = 0.05            # a "small" block Jacobian
L = 50
plain = dFdx ** L      # without identity: vanishes
resid = (1.0 + dFdx) ** L
"""


def run_s16() -> dict:
    dFdx = 0.05
    L = 50
    plain = dFdx**L
    resid = (1.0 + dFdx) ** L
    return {
        "depth": L,
        "plain_chain_grad": f"{plain:.2e}",
        "residual_grad": round(float(resid), 4),
        "plain_vs_residual": f"{resid / max(plain, 1e-300):.2e}x",
    }


# ---- section registry ------------------------------------------------------

_SECTIONS: dict[str, dict[str, Any]] = {
    "s1": {"title": "1.1 single query", "code": S1_CODE, "run": run_s1},
    "s2": {"title": "1.2 Scaled Dot-Product Attention", "code": S2_CODE, "run": run_s2},
    "s3": {"title": "1.3 seq2seq attention", "code": S3_CODE, "run": run_s3},
    "s4": {"title": "1.4 multiple queries", "code": S4_CODE, "run": run_s4},
    "s5": {"title": "2.1 causal self-attention", "code": S5_CODE, "run": run_s5},
    "s6": {"title": "2.2 cross-attention", "code": S6_CODE, "run": run_s6},
    "s7": {"title": "3.1 K-V cache motivation", "code": S7_CODE, "run": run_s7},
    "s8": {"title": "3.2 Multi-head Latent Attention", "code": S8_CODE, "run": run_s8},
    "s9": {"title": "3.3 RoPE", "code": S9_CODE, "run": run_s9},
    "s10": {"title": "3.4 Decoupled RoPE", "code": S10_CODE, "run": run_s10},
    "s11": {"title": "3.5 KV cache extension", "code": S11_CODE, "run": run_s11},
    "s12": {"title": "4.1 online softmax", "code": S12_CODE, "run": run_s12},
    "s13": {"title": "4.2 online softmax loop", "code": S13_CODE, "run": run_s13},
    "s14": {"title": "4.3 single-query row form", "code": S14_CODE, "run": run_s14},
    "s15": {"title": "5.1 residual connection", "code": S15_CODE, "run": run_s15},
    "s16": {"title": "5.2 residual gradient", "code": S16_CODE, "run": run_s16},
}


def section_ids() -> list[str]:
    return list(_SECTIONS.keys())


def section_source(section_id: str) -> dict[str, Any] | None:
    s = _SECTIONS.get(section_id)
    if s is None:
        return None
    return {"id": section_id, "title": s["title"], "code": s["code"]}


def run_section(
    section_id: str, params: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    s = _SECTIONS.get(section_id)
    if s is None:
        return None
    fn: Callable[..., dict] = s["run"]
    try:
        result = fn(params or {}) if section_id in ("s2", "s7", "s8") else fn()
    except TypeError:
        result = fn()
    return {"id": section_id, "title": s["title"], "result": result}
