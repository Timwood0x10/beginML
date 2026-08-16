"""anchors.py — semantic anchors for paper formulas (per-paper concepts).

A formula is not just a bbox: it carries a semantic anchor that names WHAT
it is (type, section, label, concept). Concept strings are MANUALLY
authored (paper → section → concept), in English and Chinese — the parser
only locates formulas, it never guesses the author's meaning.

The concept key ties formula → implementation → experiment per paper.
"""

from typing import Any

# ---- per-paper concept tables (manually confirmed) -------------------------

CONCEPTS: dict[str, dict[str, str]] = {
    "transformer": {
        "s1": "dot-product attention (single query)",
        "s2": "scaled dot-product attention",
        "s3": "attention in seq2seq",
        "s4": "attention in matrix form",
        "s5": "causal self-attention",
        "s6": "cross-attention",
        "s7": "KV cache",
        "s8": "multi-head latent attention (MLA)",
        "s9": "rotary position embedding (RoPE)",
        "s10": "decoupled RoPE",
        "s11": "KV cache extension",
        "s12": "online softmax (flash attention)",
        "s13": "online softmax loop",
        "s14": "single-query row form",
        "s15": "residual connection",
        "s16": "residual gradient",
    },
    "attention-residuals": {
        "s1": "residual connections in LLMs",
        "s2": "notation & setup (B×T×d)",
        "s3": "attention residuals: unified view of time & depth",
        "s4": "infrastructure design for AttnRes",
        "s5": "architecture details (MoE transformer)",
        "s6": "sequence-depth duality",
        "s7": "normalization & depth stability",
    },
}

CONCEPTS_ZH: dict[str, dict[str, str]] = {
    "transformer": {
        "s1": "单查询点积注意力",
        "s2": "缩放点积注意力",
        "s3": "seq2seq 中的注意力",
        "s4": "矩阵形式的注意力",
        "s5": "因果自注意力",
        "s6": "交叉注意力",
        "s7": "KV 缓存",
        "s8": "多头潜在注意力 (MLA)",
        "s9": "旋转位置编码 (RoPE)",
        "s10": "解耦 RoPE",
        "s11": "KV 缓存扩展",
        "s12": "在线 softmax（Flash Attention）",
        "s13": "在线 softmax 循环",
        "s14": "单查询行形式",
        "s15": "残差连接",
        "s16": "残差梯度",
    },
    "attention-residuals": {
        "s1": "大模型中的残差连接",
        "s2": "记号与设定（B×T×d）",
        "s3": "注意力残差：时间与深度的统一视角",
        "s4": "AttnRes 的基础设施设计",
        "s5": "架构细节（MoE Transformer）",
        "s6": "序列-深度对偶性",
        "s7": "标准化与深度稳定性",
    },
}

UNKNOWN_CONCEPT = "unknown concept"
UNKNOWN_CONCEPT_ZH = "未知概念"


# virtual section for formulas before the first heading (title/abstract)
_COVER_CONCEPT = "front matter (title / abstract)"
_COVER_CONCEPT_ZH = "封面与摘要"


def concept_for(paper_id: str, section_id: str | None) -> str:
    if not section_id:
        return UNKNOWN_CONCEPT
    if section_id == "cover":
        return _COVER_CONCEPT
    return CONCEPTS.get(paper_id, {}).get(section_id, UNKNOWN_CONCEPT)


def concept_zh_for(paper_id: str, section_id: str | None) -> str:
    if not section_id:
        return UNKNOWN_CONCEPT_ZH
    if section_id == "cover":
        return _COVER_CONCEPT_ZH
    return CONCEPTS_ZH.get(paper_id, {}).get(section_id, UNKNOWN_CONCEPT_ZH)


def build_anchor(paper_id: str, formula: dict[str, Any], label: str) -> dict[str, Any]:
    """Attach the semantic anchor to one extracted formula.

    label is the per-section equation number (e.g. "Equation 2") — computed
    by the caller from the formula's position inside its section.
    """
    sid = formula.get("section_id")
    return {
        "type": "equation",
        "section": formula.get("section_title") or "",
        "label": label,
        "concept": concept_for(paper_id, sid),
        "concept_zh": concept_zh_for(paper_id, sid),
    }
