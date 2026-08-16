"""anchors.py — semantic anchors for paper formulas.

A formula is not just a bbox: it carries a semantic anchor that names WHAT
it is (type, section, label, concept). The concept strings are MANUALLY
authored mapping (section → concept) — the parser only locates formulas,
it never guesses the author's meaning. This lets the frontend say:

  "This is the scaled dot-product attention equation"
  instead of "this is formula #17".

The same concept key then ties formula → implementation → experiment.
"""

from typing import Any

# Manually confirmed: PDF section id → the mathematical concept it expresses.
CONCEPT_BY_SECTION: dict[str, str] = {
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
}

UNKNOWN_CONCEPT = "unknown concept"


def concept_for(section_id: str | None) -> str:
    if not section_id:
        return UNKNOWN_CONCEPT
    return CONCEPT_BY_SECTION.get(section_id, UNKNOWN_CONCEPT)


def build_anchor(formula: dict[str, Any], label: str) -> dict[str, Any]:
    """Attach the semantic anchor to one extracted formula.

    label is the per-section equation number (e.g. "Equation 2") — computed
    by the caller from the formula's position inside its section.
    """
    sid = formula.get("section_id")
    return {
        "type": "equation",
        "section": formula.get("section_title") or "",
        "label": label,
        "concept": concept_for(sid),
    }
