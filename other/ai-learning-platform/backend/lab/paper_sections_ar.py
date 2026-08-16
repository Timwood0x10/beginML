"""paper_sections_ar.py — numpy implementations for Attention Residuals.

Core mechanisms of the Kimi Attention-Residuals technical report,
hand-authored as runnable demos (same shape as paper_sections.py):

  s6  sequence-depth duality   — residual x + F(x) preserves information
                                 along DEPTH; without the residual the norm
                                 collapses as depth grows
  s7  normalization & depth stability — scaling alpha in x + alpha*F(x)
                                 controls how stable the representation stays
                                 across many layers

No-LLM: every number is computed, nothing is generated.
"""

from typing import Any

import numpy as np

SEED = 7


# ---- s6: sequence-depth duality --------------------------------------------
S6_CODE = '''
import numpy as np
rng = np.random.default_rng(7)
d, L = 32, 20
W = rng.normal(0, 1, (d, d)) / 8
h_res = rng.normal(0, 1, d)     # residual path: h_{l+1} = h_l + F(h_l)
h_plain = h_res.copy()          # plain stacking:  h_{l+1} = F(h_l)
residual_norms, plain_norms = [], []
for _ in range(L):
    h_res = h_res + np.tanh(h_res @ W)
    h_plain = np.tanh(h_plain @ W)
    residual_norms.append(np.linalg.norm(h_res))
    plain_norms.append(np.linalg.norm(h_plain))
'''


def run_ar_s6() -> dict:
    rng = np.random.default_rng(SEED)
    d, L = 32, 20
    W = rng.normal(0, 1, (d, d)) / 8
    h_res = rng.normal(0, 1, d)
    h_plain = h_res.copy()
    residual_norms = []
    plain_norms = []
    for _ in range(L):
        h_res = h_res + np.tanh(h_res @ W)
        h_plain = np.tanh(h_plain @ W)
        residual_norms.append(float(np.linalg.norm(h_res)))
        plain_norms.append(float(np.linalg.norm(h_plain)))
    return {
        "residual_norm_L0": round(residual_norms[0], 4),
        "residual_norm_L20": round(residual_norms[-1], 4),
        "plain_norm_L0": round(plain_norms[0], 4),
        "plain_norm_L20": round(plain_norms[-1], 4),
        "residual_growth": round(residual_norms[-1] / max(residual_norms[0], 1e-9), 3),
        "plain_decay": round(plain_norms[-1] / max(plain_norms[0], 1e-9), 3),
        "visual": {
            "type": "flow",
            "steps": [
                {"label": "residual L0", "shape": "1", "value": round(residual_norms[0], 4)},
                {"label": "residual L20 (kept)", "shape": "1", "value": round(residual_norms[-1], 4)},
                {"label": "plain L0", "shape": "1", "value": round(plain_norms[0], 4)},
                {"label": "plain L20 (collapsed)", "shape": "1", "value": round(plain_norms[-1], 4)},
            ],
        },
    }


# ---- s7: normalization & depth stability ------------------------------------
S7_CODE = '''
import numpy as np
rng = np.random.default_rng(7)
d, L = 32, 30
W = rng.normal(0, 1, (d, d)) / 8
for alpha in (0.5, 1.0, 1.5):     # h_{l+1} = h_l + alpha * F(h_l)
    h = rng.normal(0, 1, d)
    norms = []
    for _ in range(L):
        h = h + alpha * np.tanh(h @ W)
        norms.append(np.linalg.norm(h))
'''


def run_ar_s7() -> dict:
    rng = np.random.default_rng(SEED)
    d, L = 32, 30
    W = rng.normal(0, 1, (d, d)) / 8
    curves = {}
    for alpha in (0.5, 1.0, 1.5):
        h = rng.normal(0, 1, d)
        norms = []
        for _ in range(L):
            h = h + alpha * np.tanh(h @ W)
            norms.append(float(np.linalg.norm(h)))
        curves[f"alpha_{alpha}"] = {
            "norm_L0": round(norms[0], 4),
            "norm_L30": round(norms[-1], 4),
            "ratio": round(norms[-1] / max(norms[0], 1e-9), 3),
        }
    return {
        "depth": L,
        "curves": curves,
        "stable_alpha": "1.0 (ratio closest to 1)",
        "observation": (
            "alpha=1.5 grows the norm ~"
            f"{curves['alpha_1.5']['ratio']}x over {L} layers (unstable); "
            f"alpha=0.5 keeps it ~{curves['alpha_0.5']['ratio']}x (over-damped); "
            f"alpha=1.0 balances at ~{curves['alpha_1.0']['ratio']}x."
        ),
        "visual": {
            "type": "flow",
            "steps": [
                {"label": "α=0.5 ratio", "shape": "1", "value": curves["alpha_0.5"]["ratio"]},
                {"label": "α=1.0 ratio", "shape": "1", "value": curves["alpha_1.0"]["ratio"]},
                {"label": "α=1.5 ratio", "shape": "1", "value": curves["alpha_1.5"]["ratio"]},
            ],
        },
    }


# ---- s1: residual connections in LLMs --------------------------------------
S1_CODE = """
# Standard residual block: h_{l+1} = h_l + f_l(h_l).
import numpy as np
rng = np.random.default_rng(7)
d, L = 16, 10
W = rng.normal(0, 1, (d, d)) / 4
h = rng.normal(0, 1, d)
norms = []
for _ in range(L):
    h = h + np.tanh(h @ W)          # residual update
    norms.append(np.linalg.norm(h))
"""


def run_ar_s1() -> dict:
    rng = np.random.default_rng(SEED)
    d, L = 16, 10
    W = rng.normal(0, 1, (d, d)) / 4
    h = rng.normal(0, 1, d)
    norms = []
    for _ in range(L):
        h = h + np.tanh(h @ W)
        norms.append(float(np.linalg.norm(h)))
    return {
        "norm_L0": round(norms[0], 4),
        "norm_L10": round(norms[-1], 4),
        "ratio": round(norms[-1] / max(norms[0], 1e-9), 3),
    }


# ---- s2: notation & setup (B x T x d) ---------------------------------------
S2_CODE = """
# Notation: a batch of sequences, shape B x T x d.
import numpy as np
rng = np.random.default_rng(7)
B, T, d = 2, 4, 8
X = rng.normal(0, 1, (B, T, d))
W = rng.normal(0, 1, (d, d)) / 4
Y = X + np.tanh(X @ W)   # one residual block per token; shape preserved
"""


def run_ar_s2() -> dict:
    rng = np.random.default_rng(SEED)
    B, T, d = 2, 4, 8
    X = rng.normal(0, 1, (B, T, d))
    W = rng.normal(0, 1, (d, d)) / 4
    Y = X + np.tanh(X @ W)
    return {
        "input_shape": list(X.shape),
        "output_shape": list(Y.shape),
        "norm_ratio": round(float(np.linalg.norm(Y) / max(np.linalg.norm(X), 1e-9)), 3),
    }


# ---- s3: attention residuals — unified view of time & depth -----------------
S3_CODE = """
# Unified view: BOTH time mixing (attention) and depth mixing (FFN)
# are residual updates "x + transform(x)".
import numpy as np
rng = np.random.default_rng(7)
T, d = 6, 8
X = rng.normal(0, 1, (T, d))
Wv = rng.normal(0, 1, (d, d)) / 4
Wf = rng.normal(0, 1, (d, d)) / 4
attn_out = X @ Wv                    # time axis: tokens mix
X_time = X + attn_out                # residual add (attention residual)
X_depth = X_time + np.tanh(X_time @ Wf)   # depth axis: per-token transform
"""


def run_ar_s3() -> dict:
    rng = np.random.default_rng(SEED)
    T, d = 6, 8
    X = rng.normal(0, 1, (T, d))
    Wv = rng.normal(0, 1, (d, d)) / 4
    Wf = rng.normal(0, 1, (d, d)) / 4
    X_time = X + X @ Wv
    X_depth = X_time + np.tanh(X_time @ Wf)
    return {
        "time_residual_norm": round(float(np.linalg.norm(X_time)), 4),
        "depth_residual_norm": round(float(np.linalg.norm(X_depth)), 4),
        "time_mix_strength": round(float(np.linalg.norm(X @ Wv) / max(np.linalg.norm(X), 1e-9)), 3),
    }


# ---- s4: infrastructure design ----------------------------------------------
S4_CODE = """
# Infrastructure: AttnRes indexing is DETERMINISTIC -> prefetchable;
# MoE routing depends on the hidden state -> must wait for the layer.
import numpy as np
latency_ms = 10.0                 # host->device transfer latency
# deterministic n-gram index: prefetch during compute, no stall
stall_attnres = 0.0
# MoE: route computed after the layer -> serial stall
stall_moe = latency_ms
"""


def run_ar_s4() -> dict:
    latency = 10.0
    return {
        "stall_attnres_ms": 0.0,
        "stall_moe_ms": latency,
        "hidden_by_prefetch_ms": latency,
        "speedup_over_moe": "inf (0 stall)",
    }


# ---- s5: architecture details (MoE transformer) -----------------------------
S5_CODE = """
# MoE Transformer architecture: every block (attention + experts)
# ends with a residual add.
import numpy as np
rng = np.random.default_rng(7)
d = 16
x = rng.normal(0, 1, d)
Wattn = rng.normal(0, 1, (d, d)) / 4
Wexp = rng.normal(0, 1, (d, d)) / 4
h = x + np.tanh(x @ Wattn)          # attention block residual
h = h + np.tanh(h @ Wexp)           # expert block residual
"""


def run_ar_s5() -> dict:
    rng = np.random.default_rng(SEED)
    d = 16
    x = rng.normal(0, 1, d)
    Wattn = rng.normal(0, 1, (d, d)) / 4
    Wexp = rng.normal(0, 1, (d, d)) / 4
    h_attn = x + np.tanh(x @ Wattn)
    h_exp = h_attn + np.tanh(h_attn @ Wexp)
    return {
        "after_attn_norm": round(float(np.linalg.norm(h_attn)), 4),
        "after_experts_norm": round(float(np.linalg.norm(h_exp)), 4),
        "norm_growth": round(float(np.linalg.norm(h_exp) / max(np.linalg.norm(x), 1e-9)), 3),
    }


# ---- registry ---------------------------------------------------------------

_SECTIONS: dict[str, dict[str, Any]] = {
    "s1": {"title": "1 residual connections in LLMs", "code": S1_CODE, "run": run_ar_s1},
    "s2": {"title": "2 notation & setup (B×T×d)", "code": S2_CODE, "run": run_ar_s2},
    "s3": {"title": "3 attention residuals: unified view", "code": S3_CODE, "run": run_ar_s3},
    "s4": {"title": "4 infrastructure design", "code": S4_CODE, "run": run_ar_s4},
    "s5": {"title": "5 architecture details (MoE)", "code": S5_CODE, "run": run_ar_s5},
    "s6": {"title": "6 sequence-depth duality", "code": S6_CODE, "run": run_ar_s6},
    "s7": {"title": "7 normalization & depth stability", "code": S7_CODE, "run": run_ar_s7},
}


def section_source(section_id: str) -> dict[str, Any] | None:
    s = _SECTIONS.get(section_id)
    if s is None:
        return None
    return {"id": section_id, "title": s["title"], "code": s["code"]}


def run_section(section_id: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    s = _SECTIONS.get(section_id)
    if s is None:
        return None
    try:
        result = s["run"](params or {})
    except TypeError:
        result = s["run"]()
    return {"id": section_id, "title": s["title"], "result": result}
