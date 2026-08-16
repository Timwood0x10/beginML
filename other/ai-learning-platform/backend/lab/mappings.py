"""mappings.py — MANUALLY authored formula → implementation → experiment.

The paper parser only LOCATES formulas (paper_formulas.py) and names their
concept (anchors.py). It never guesses what the author meant. Everything
below is the human-authored mapping layer: for each section that has a
numpy implementation in paper_sections.py we declare

  implementation : which file / symbols / lines implement the math
  experiment     : what you can change (inputs) and what to observe

The frontend renders these directly: ImplementationPanel shows the
formula ↔ code correspondence, ExperimentPanel shows the input controls +
observation text next to the run output.
"""

from typing import Any

_SOURCE_FILE = "paper_sections.py"

# section id → where the math is implemented (human-confirmed)
IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "s1": {"file": _SOURCE_FILE, "symbols": ["run_s1"], "lines": [20, 46]},
    "s2": {"file": _SOURCE_FILE, "symbols": ["run_s2", "_softmax"], "lines": [49, 73]},
    "s4": {"file": _SOURCE_FILE, "symbols": ["run_s4", "_softmax"], "lines": [82, 108]},
    "s7": {"file": _SOURCE_FILE, "symbols": ["run_s7"], "lines": [116, 138]},
    "s8": {"file": _SOURCE_FILE, "symbols": ["run_s8"], "lines": [142, 166]},
    "s9": {"file": _SOURCE_FILE, "symbols": ["run_s9", "rope"], "lines": [177, 211]},
    "s10": {"file": _SOURCE_FILE, "symbols": ["run_s10"], "lines": [214, 241]},
    "s12": {"file": _SOURCE_FILE, "symbols": ["run_s12", "_softmax"], "lines": [248, 274]},
    "s15": {"file": _SOURCE_FILE, "symbols": ["run_s15"], "lines": [281, 304]},
}

# section id → runnable experiment: adjustable inputs + what to observe
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "s1": {
        "runner": "run_s1",
        "inputs": {},
        "observation_zh": "单个 query 与 5 个 key 的点积打分 → softmax 权重 → 加权求和。权重之和恒为 1。",
        "observation_en": "One query scores against 5 keys, softmax weights weight-sum the values. Weights always sum to 1.",
    },
    "s2": {
        "runner": "run_s2",
        "inputs": {"d_k": {"label": "d_k", "min": 2, "max": 64, "step": 2, "default": 8}},
        "observation_zh": "调大 d_k：不除以 √dₖ 的 QKᵀ 分数方差变大，softmax 趋于饱和（分数更极端）。除以 √dₖ 保持方差稳定。",
        "observation_en": "Increase d_k: unscaled QKᵀ scores spread out and softmax saturates; dividing by √dₖ keeps variance stable.",
    },
    "s4": {
        "runner": "run_s4",
        "inputs": {},
        "observation_zh": "矩阵形式：每行是某个 query 对所有 key 的注意力分布，行和 = 1；输出的范数反映混合强度。",
        "observation_en": "Matrix form: each row is one query's attention over all keys, rows sum to 1.",
    },
    "s7": {
        "runner": "run_s7",
        "inputs": {"tokens": {"label": "decoded tokens t", "min": 2, "max": 512, "step": 2, "default": 10}},
        "observation_zh": "解码第 t 步：无 KV 缓存要重算全部历史注意力 O((t+1)²·d)，有缓存只算新的一行 O((t+1)·d)——t 越大节省越夸张。",
        "observation_en": "Decoding step t: without a KV cache all past attention is recomputed O((t+1)²d); with a cache only the new row O((t+1)d).",
    },
    "s8": {
        "runner": "run_s8",
        "inputs": {"d_c": {"label": "latent dim d_c", "min": 4, "max": 64, "step": 4, "default": 16}},
        "observation_zh": "MLA 把 K/V 压缩进潜在向量 z：KV 缓存从 n·2d 降到 n·d_c——压缩比 = 2d / d_c。",
        "observation_en": "MLA compresses K/V into a latent z: the KV cache drops from n·2d to n·d_c — compression = 2d / d_c.",
    },
    "s9": {
        "runner": "run_s9",
        "inputs": {},
        "observation_zh": "RoPE 按位置 m 旋转：不同频率维度旋转角度不同，同一向量在位置 4 与 8 的余弦相似度随距离衰减。",
        "observation_en": "RoPE rotates by position m with per-frequency angles; the cosine similarity of the same vector at positions 4 and 8 decays with distance.",
    },
    "s10": {
        "runner": "run_s10",
        "inputs": {},
        "observation_zh": "解耦 RoPE：只旋转 query 与部分 key，value 与剩余 key 不旋转——缓存更友好，分数是旋转部分与非旋转部分的点积和。",
        "observation_en": "Decoupled RoPE rotates only the query and part of the key; the score is the sum of rotated and unrotated dot products.",
    },
    "s12": {
        "runner": "run_s12",
        "inputs": {},
        "observation_zh": "online softmax 用单遍流式更新 max 与 exp-sum，不物化完整 QKᵀ 矩阵；与完整 softmax 结果逐元素一致（误差 ≈ 0）。",
        "observation_en": "Online softmax streams max & exp-sum in one pass without materializing QKᵀ; element-wise identical to full softmax (error ≈ 0).",
    },
    "s15": {
        "runner": "run_s15",
        "inputs": {},
        "observation_zh": "残差连接 x + F(x)：输出范数与输入接近，梯度经恒等路径直达——深层网络可训的关键。",
        "observation_en": "Residual x + F(x): output norm stays close to input, gradients flow through the identity path — key to training deep nets.",
    },
}


# section id → paper symbol ↔ code line correspondence (human-confirmed).
# This is what makes the ImplementationPanel meaningful: "the paper says
# softmax(QKᵀ/√dₖ)V — in code that is these lines."
CODE_MAP: dict[str, list[dict[str, str]]] = {
    "s1": [
        {"paper": "scores = K·q", "code": "scores = K @ q"},
        {"paper": "softmax(scores)", "code": "w = np.exp(scores) / np.exp(scores).sum()"},
        {"paper": "out = Σᵢ wᵢ·vᵢ", "code": "out = w @ V"},
    ],
    "s2": [
        {"paper": "QKᵀ", "code": "raw = Q @ K.T"},
        {"paper": "1 / √dₖ", "code": "scores = raw / np.sqrt(d_k)"},
        {"paper": "softmax(·)", "code": "w = _softmax(scores)"},
        {"paper": "softmax(·)V", "code": "out = w @ V"},
    ],
    "s4": [
        {"paper": "Q = XWq, K = XWk, V = XWv", "code": "Q, K, V = X @ Wq, X @ Wk, X @ Wv"},
        {"paper": "P = softmax(QKᵀ/√d)", "code": "P = _softmax(Q @ K.T / np.sqrt(d))"},
        {"paper": "Y = PV", "code": "Y = P @ V"},
    ],
    "s7": [
        {"paper": "无缓存: O((t+1)²·d)", "code": "no_cache = (t+1) * (t+1) * d"},
        {"paper": "有缓存: O((t+1)·d)", "code": "with_cache = (t+1) * d"},
    ],
    "s8": [
        {"paper": "z = X·W_down（潜在压缩）", "code": "z = X @ W_down"},
        {"paper": "缓存 n·d_c vs n·2d", "code": "kv_cache_mla = n * d_c"},
    ],
    "s9": [
        {"paper": "θᵢ = θ^(−2i/d)", "code": "freqs = theta ** (-np.arange(0, d, 2) / d)"},
        {"paper": "R(m)·x 旋转", "code": "x1*c - x2*s, x1*s + x2*c"},
    ],
    "s10": [
        {"paper": "q/k 仅部分旋转", "code": "rot(q_rope), rot(k_rope)"},
        {"paper": "score = (q·k)/√d", "code": "score / np.sqrt(len(q))"},
    ],
    "s12": [
        {"paper": "m ← max(m, x)", "code": "m_new = max(m, x)"},
        {"paper": "l ← l·e^(m−m_new) + e^(x−m_new)", "code": "l = l*np.exp(m-m_new) + np.exp(x-m_new)"},
    ],
    "s15": [
        {"paper": "x_L = x_{L−1} + F(x_{L−1})", "code": "x_next = x + F"},
        {"paper": "恒等捷径（梯度直通）", "code": "# residual add keeps the identity path"},
    ],
}


def code_map_for(section_id: str | None) -> list[dict[str, str]] | None:
    if not section_id:
        return None
    return CODE_MAP.get(section_id)


def implementation_for(section_id: str | None) -> dict[str, Any] | None:
    if not section_id:
        return None
    return IMPLEMENTATIONS.get(section_id)


def experiment_for(section_id: str | None) -> dict[str, Any] | None:
    if not section_id:
        return None
    return EXPERIMENTS.get(section_id)
