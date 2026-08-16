"""paper_formulas.py — extract the paper's formulas and link each to a section.

The paper PDF renders LaTeX; pymupdf sees formula fragments as small text
spans. This module works at BLOCK level (a block is a layout unit, so a
formula block is usually the whole formula), decides "is math" by symbol
density, records page + bbox, and assigns each formula to the section whose
heading sits directly above it (same page).

The frontend then just renders: highlight each formula's bbox on the page
image and, on click, fetch that formula's section source + run output.

No-LLM: nothing is generated, only located and linked.
"""

from pathlib import Path

import pymupdf

from .pdf_paper import PAPER_PATH, parse_paper

# characters that strongly suggest math content
_MATH_CHARS = set("∈≤≥∂√∑πθαβλμσφ∇·×≈≠⊥⊤∞±=")

# a block is "math" if at least this many math characters appear
MIN_SYMBOLS = 2
# ... and math chars make up at least this fraction of the block (filters
# prose paragraphs that merely mention a symbol, e.g. "∂z are zero!").
# 0.08 keeps letter-heavy formulas like "softmax(QKᵀ/√dₖ)V" (ratio ≈ 0.1)
# while dropping prose.
MIN_SYMBOL_RATIO = 0.08
# formula blocks are compact; anything this tall is a paragraph, not a formula
MAX_BLOCK_HEIGHT = 160.0  # PDF points


def _is_math_block(text: str) -> bool:
    if not text:
        return False
    count = sum(1 for ch in text if ch in _MATH_CHARS)
    return count >= MIN_SYMBOLS and (count / max(len(text), 1)) >= MIN_SYMBOL_RATIO


def _y_overlap(a: list[float], b: list[float]) -> bool:
    """True if two bboxes share any vertical space (same formula line)."""
    return a[1] < b[3] and b[1] < a[3]


def extract_formulas() -> list[dict]:
    """Return formula blocks with page, bbox (PDF points) and section link.

    Formula fragments on the same line are merged; each formula is assigned
    to the section whose heading is the closest one ABOVE it in document
    order (headings live on section-start pages; formulas on any page in
    between belong to the section that opened before them).

    Each formula: {id, page, bbox: [x0,y0,x1,y1], text, section_id, section_title}
    """
    paper = parse_paper()

    # document-wide heading sequence: (page, y0, id, title)
    headings = sorted(
        ((s["page"], s["bbox"][1], s["id"], s["title"]) for s in paper["sections"]),
        key=lambda h: (h[0], h[1]),
    )

    doc = pymupdf.open(PAPER_PATH)

    # ---- collect raw math blocks -----------------------------------------
    raw: list[dict] = []
    for pno, page in enumerate(doc):
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            lines = block.get("lines") or []
            if not lines:
                continue
            text = "".join(
                span.get("text", "")
                for line in lines
                for span in line.get("spans", [])
            ).strip()
            if not text:
                continue
            bbox = [float(v) for v in block.get("bbox", [0, 0, 0, 0])]
            if bbox[3] - bbox[1] > MAX_BLOCK_HEIGHT:
                continue  # paragraph block, not a formula
            if not _is_math_block(text):
                continue
            raw.append({"page": pno + 1, "bbox": bbox, "text": text})

    # ---- merge fragments sharing a horizontal line (same formula) ---------
    raw.sort(key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]))
    merged: list[dict] = []
    for b in raw:
        if merged and b["page"] == merged[-1]["page"] and _y_overlap(b["bbox"], merged[-1]["bbox"]):
            last = merged[-1]
            last["bbox"] = [
                min(last["bbox"][0], b["bbox"][0]),
                min(last["bbox"][1], b["bbox"][1]),
                max(last["bbox"][2], b["bbox"][2]),
                max(last["bbox"][3], b["bbox"][3]),
            ]
            last["text"] = (last["text"] + " " + b["text"]).strip()
        else:
            merged.append(dict(b))

    # ---- assign each formula to the closest heading above it --------------
    formulas: list[dict] = []
    for fid, m in enumerate(merged, 1):
        yc = (m["bbox"][1] + m["bbox"][3]) / 2.0
        sec_id = None
        sec_title = ""
        for hpage, hy, sid, stitle in headings:
            if (hpage, hy) <= (m["page"], yc):
                sec_id, sec_title = sid, stitle
            else:
                break
        formulas.append(
            {
                "id": f"f{fid}",
                "page": m["page"],
                "bbox": [round(float(v), 2) for v in m["bbox"]],
                "text": m["text"][:200],
                "section_id": sec_id,
                "section_title": sec_title,
            }
        )

    return formulas


def get_formulas() -> dict:
    """Public contract for the frontend: formulas + their section links."""
    return {"count": None, "formulas": extract_formulas()}
