"""mappings.py — MANUALLY authored formula → implementation → experiment.

The paper parser only LOCATES formulas (paper_formulas.py) and names their
concept (anchors.py). It never guesses what the author meant. Everything
below is the human-authored mapping layer, organised PER PAPER:

  MAPPINGS[paper_id] = {
    "implementations": section_id -> {file, symbols, lines}
    "experiments":     section_id -> {runner, inputs, observation_zh/en}
    "code_map":        section_id -> [{paper, code}, ...]
  }

The frontend renders these directly: ImplementationPanel shows the
formula ↔ code correspondence, ExperimentPanel shows the input controls +
observation text next to the run output.
"""

from typing import Any

_SOURCE_FILE = "paper_sections.py"
_SOURCE_FILE_AR = "paper_sections_ar.py"

MAPPINGS: dict[str, dict[str, dict[str, Any]]] = {
    # ---- transformer.pdf ------------------------------------------------
    "transformer": {
        "implementations": {
            "s1": {"file": _SOURCE_FILE, "symbols": ["run_s1"], "lines": [20, 46]},
            "s2": {"file": _SOURCE_FILE, "symbols": ["run_s2", "_softmax"], "lines": [49, 73]},
            "s4": {"file": _SOURCE_FILE, "symbols": ["run_s4", "_softmax"], "lines": [82, 108]},
            "s7": {"file": _SOURCE_FILE, "symbols": ["run_s7"], "lines": [116, 138]},
            "s8": {"file": _SOURCE_FILE, "symbols": ["run_s8"], "lines": [142, 166]},
            "s9": {"file": _SOURCE_FILE, "symbols": ["run_s9", "rope"], "lines": [177, 211]},
            "s10": {"file": _SOURCE_FILE, "symbols": ["run_s10"], "lines": [214, 241]},
            "s12": {"file": _SOURCE_FILE, "symbols": ["run_s12", "_softmax"], "lines": [248, 274]},
            "s15": {"file": _SOURCE_FILE, "symbols": ["run_s15"], "lines": [281, 304]},
            "s3": {"file": _SOURCE_FILE, "symbols": ["run_s3"], "lines": [324, 355]},
            "s5": {"file": _SOURCE_FILE, "symbols": ["run_s5"], "lines": [360, 395]},
            "s6": {"file": _SOURCE_FILE, "symbols": ["run_s6"], "lines": [400, 430]},
            "s11": {"file": _SOURCE_FILE, "symbols": ["run_s11"], "lines": [435, 465]},
            "s13": {"file": _SOURCE_FILE, "symbols": ["run_s13"], "lines": [470, 500]},
            "s14": {"file": _SOURCE_FILE, "symbols": ["run_s14"], "lines": [505, 535]},
            "s16": {"file": _SOURCE_FILE, "symbols": ["run_s16"], "lines": [540, 570]},
        },
        "experiments": {
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
            "s3": {
                "runner": "run_s3",
                "inputs": {},
                "observation_zh": "seq2seq：decoder 的 query 对 encoder 的 key 打分，softmax 加权求和得到上下文向量——注意力把源序列信息带进解码。",
                "observation_en": "seq2seq: the decoder query scores encoder keys; softmax-weighted sum yields the context vector that carries source info into decoding.",
            },
            "s5": {
                "runner": "run_s5",
                "inputs": {},
                "observation_zh": "因果掩码：每个 token 只能看到自己及之前的 token——第 0 行只支持 1 个位置，最后一行支持全部过去位置。",
                "observation_en": "Causal masking: each token only sees itself and earlier tokens — row 0 attends to 1 position, the last row to all past positions.",
            },
            "s6": {
                "runner": "run_s6",
                "inputs": {},
                "observation_zh": "交叉注意力：Q 来自目标序列（decoder），K/V 来自源序列（encoder）——目标每个 token 关注源的全部 token。",
                "observation_en": "Cross-attention: Q from the target sequence, K/V from the source; each target token attends to all source tokens.",
            },
            "s11": {
                "runner": "run_s11",
                "inputs": {
                    "tokens": {"label": "cached tokens t", "min": 2, "max": 512, "step": 2, "default": 10},
                    "batch": {"label": "new tokens k", "min": 1, "max": 64, "step": 1, "default": 4},
                },
                "observation_zh": "批解码 k 个新 token：无缓存要重算全部 (t+k)² 对，有缓存只算新行——t 越大节省越夸张。",
                "observation_en": "Batched decoding of k new tokens: without a cache all (t+k)² pairs are recomputed; with a cache only the new rows.",
            },
            "s13": {
                "runner": "run_s13",
                "inputs": {},
                "observation_zh": "online softmax 循环版：逐块更新 (m, l)，与完整 softmax 逐元素一致（误差 ≈ 0）——Flash Attention 的核心循环。",
                "observation_en": "Online softmax loop: chunk-by-chunk (m, l) updates, element-wise identical to full softmax — the Flash Attention loop.",
            },
            "s14": {
                "runner": "run_s14",
                "inputs": {},
                "observation_zh": "单查询行式：一个 query 与整张 key 矩阵点积得到一行分数，softmax 后加权求和 value。",
                "observation_en": "Single-query row form: one query scores against the whole key matrix; softmax weights weight-sum the values.",
            },
            "s16": {
                "runner": "run_s16",
                "inputs": {},
                "observation_zh": "残差梯度：dy/dx = I + dF/dx，恒等路径让梯度跨 50 层保持 O(1)；纯链式链则衰减到 ≈0。",
                "observation_en": "Residual gradient: dy/dx = I + dF/dx keeps gradients O(1) across 50 layers; a plain chain decays to ≈0.",
            },
        },
        "code_map": {
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
            "s3": [
                {"paper": "context = Σᵢ wᵢ·vᵢ (decoder←encoder)", "code": "context = w @ V"},
            ],
            "s5": [
                {"paper": "掩码隐藏未来 token", "code": "np.triu(np.full((n,n), -1e9), k=1)"},
                {"paper": "softmax(QKᵀ/√d + mask)", "code": "w = softmax(scores + mask)"},
            ],
            "s6": [
                {"paper": "Q 目标 / K,V 源", "code": "Q @ K.T / np.sqrt(d)"},
                {"paper": "Y = PV", "code": "Y = P @ V"},
            ],
            "s11": [
                {"paper": "无缓存: O((t+k)²·d)", "code": "no_cache = (t+k) * (t+k) * d"},
                {"paper": "有缓存: O((t+k)·d)", "code": "with_cache = (t+k) * d"},
            ],
            "s13": [
                {"paper": "m ← max(m, chunk.max)", "code": "m_new = max(m, x.max())"},
                {"paper": "l ← l·e^(m−m_new) + Σe^(x−m_new)", "code": "l = l*np.exp(m-m_new) + np.exp(x-m_new).sum()"},
            ],
            "s14": [
                {"paper": "row = qKᵀ/√d", "code": "row = (q @ K.T) / np.sqrt(d)"},
                {"paper": "out = softmax(row)·V", "code": "out = w @ V"},
            ],
            "s16": [
                {"paper": "dy/dx = I + dF/dx", "code": "resid = (1.0 + dFdx) ** L"},
                {"paper": "无恒等路径: 衰减", "code": "plain = dFdx ** L"},
            ],
        },
    },
    # ---- attention_residuals.pdf -----------------------------------------
    "attention-residuals": {
        "implementations": {
            "s1": {"file": _SOURCE_FILE_AR, "symbols": ["run_ar_s1"], "lines": [30, 55]},
            "s2": {"file": _SOURCE_FILE_AR, "symbols": ["run_ar_s2"], "lines": [60, 85]},
            "s3": {"file": _SOURCE_FILE_AR, "symbols": ["run_ar_s3"], "lines": [90, 115]},
            "s4": {"file": _SOURCE_FILE_AR, "symbols": ["run_ar_s4"], "lines": [120, 140]},
            "s5": {"file": _SOURCE_FILE_AR, "symbols": ["run_ar_s5"], "lines": [145, 170]},
            "s6": {"file": _SOURCE_FILE_AR, "symbols": ["run_ar_s6"], "lines": [15, 40]},
            "s7": {"file": _SOURCE_FILE_AR, "symbols": ["run_ar_s7"], "lines": [45, 75]},
        },
        "experiments": {
            "s6": {
                "runner": "run_ar_s6",
                "inputs": {},
                "observation_zh": "残差 h_{l+1} = h_l + F(h_l) 沿深度传播：范数基本保持（信息逐层累积）；无残差的纯堆叠 h_{l+1} = F(h_l) 范数塌缩——深度无法扩展。",
                "observation_en": "Residual h_{l+1} = h_l + F(h_l) keeps the norm over depth (information accumulates); plain stacking h_{l+1} = F(h_l) collapses — depth cannot scale.",
            },
            "s7": {
                "runner": "run_ar_s7",
                "inputs": {},
                "observation_zh": "残差缩放 α（h_{l+1} = h_l + αF(h_l)）：α=1.5 深层范数爆炸（不稳定），α=0.5 过度阻尼，α=1.0 最稳——残差缩放是深度稳定性的旋钮。",
                "observation_en": "Residual scaling α (h_{l+1} = h_l + αF(h_l)): α=1.5 blows up over depth (unstable), α=0.5 over-damps, α=1.0 is the stable sweet spot.",
            },
            "s1": {
                "runner": "run_ar_s1",
                "inputs": {},
                "observation_zh": "标准残差块 h_{l+1} = h_l + f_l(h_l)：10 层后范数仍保持（信息累积），这是现代 LLM 的基础构件。",
                "observation_en": "Standard residual block h_{l+1} = h_l + f_l(h_l): the norm holds over 10 layers — the de facto LLM building block.",
            },
            "s2": {
                "runner": "run_ar_s2",
                "inputs": {},
                "observation_zh": "记号 B×T×d：残差块作用于批次中每个 token，形状保持不变。",
                "observation_en": "Notation B×T×d: the residual block applies per token and preserves the shape.",
            },
            "s3": {
                "runner": "run_ar_s3",
                "inputs": {},
                "observation_zh": "统一视图：时间混合（attention）与深度变换（FFN）都是“x + transform(x)”的残差更新——注意力残差把两个轴统一起来。",
                "observation_en": "Unified view: time mixing (attention) and depth transform (FFN) are both residual updates 'x + transform(x)'.",
            },
            "s4": {
                "runner": "run_ar_s4",
                "inputs": {},
                "observation_zh": "基础设施：AttnRes 索引是确定性的 → 可预取（0 停顿）；MoE 路由依赖隐藏状态 → 每层串行停顿。",
                "observation_en": "Infrastructure: AttnRes indexing is deterministic → prefetchable (0 stall); MoE routing depends on the hidden state → serial stall.",
            },
            "s5": {
                "runner": "run_ar_s5",
                "inputs": {},
                "observation_zh": "架构细节：MoE Transformer 中每个块（attention + 专家）都以残差相加结尾，范数逐块缓慢增长。",
                "observation_en": "Architecture: every MoE block (attention + experts) ends with a residual add; the norm grows slowly block by block.",
            },
        },
        "code_map": {
            "s6": [
                {"paper": "h_{l+1} = h_l + F(h_l)（残差）", "code": "h_res = h_res + np.tanh(h_res @ W)"},
                {"paper": "h_{l+1} = F(h_l)（纯堆叠）", "code": "h_plain = np.tanh(h_plain @ W)"},
            ],
            "s7": [
                {"paper": "h_{l+1} = h_l + α·F(h_l)", "code": "h = h + alpha * np.tanh(h @ W)"},
                {"paper": "深度稳定性 = 范数保持", "code": "norms.append(np.linalg.norm(h))"},
            ],
            "s1": [
                {"paper": "h_{l+1} = h_l + f_l(h_l)", "code": "h = h + np.tanh(h @ W)"},
            ],
            "s2": [
                {"paper": "B×T×d 批次", "code": "X @ W (per token)"},
                {"paper": "形状保持", "code": "Y = X + np.tanh(X @ W)"},
            ],
            "s3": [
                {"paper": "时间混合（attention）", "code": "X_time = X + X @ Wv"},
                {"paper": "深度变换（FFN）", "code": "X_depth = X_time + np.tanh(X_time @ Wf)"},
            ],
            "s4": [
                {"paper": "确定性索引 → 预取", "code": "stall_attnres = 0.0"},
                {"paper": "MoE 动态路由 → 停顿", "code": "stall_moe = latency_ms"},
            ],
            "s5": [
                {"paper": "attention 块残差", "code": "h = x + np.tanh(x @ Wattn)"},
                {"paper": "专家块残差", "code": "h = h + np.tanh(h @ Wexp)"},
            ],
        },
    },
}


def implementation_for(paper_id: str, section_id: str | None) -> dict[str, Any] | None:
    if not section_id:
        return None
    return MAPPINGS.get(paper_id, {}).get("implementations", {}).get(section_id)


def experiment_for(paper_id: str, section_id: str | None) -> dict[str, Any] | None:
    if not section_id:
        return None
    return MAPPINGS.get(paper_id, {}).get("experiments", {}).get(section_id)


def code_map_for(paper_id: str, section_id: str | None) -> list[dict[str, str]] | None:
    if not section_id:
        return None
    return MAPPINGS.get(paper_id, {}).get("code_map", {}).get(section_id)
