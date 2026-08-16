"""paper_formulas.py — extract the paper's formulas by FONT, not by guessing.

The paper is LaTeX-rendered: body text uses NimbusRom, while every math
glyph uses the Computer Modern family (CMR / CMMI / CMSY / CMEX / CMBX…).
That makes formula detection exact — no symbol-density heuristics, no OCR,
no ML:

  span.font.startswith("CM")  ⟹  this span is part of a formula

Spans sharing a horizontal line are merged; vertically adjacent lines (a
multi-line formula like a fraction) are merged into one formula block with
page + bbox. Each formula is assigned to the section whose heading is the
closest one ABOVE it in document order.

No-LLM: the parser locates and extracts; it never guesses meaning.
"""

from pathlib import Path

import pymupdf

from .pdf_paper import parse_paper, paper_path

# Computer Modern fonts — LaTeX's math typefaces (CMR roman, CMMI italic,
# CMSY symbols, CMEX extension, CMBX bold). Body text never uses these.
_MATH_FONT_PREFIXES = ("CMR", "CMMI", "CMSY", "CMEX", "CMBX")

# max vertical gap (PDF points) between two lines to still merge them into
# one formula block (fractions/stacked math)
MAX_LINE_GAP = 8.0


def _is_math_font(font: str) -> bool:
    return any(font.startswith(p) for p in _MATH_FONT_PREFIXES)


def _collect_math_chars(page) -> list[dict]:
    """CM-font characters with their geometry: {x0,y0,x1,y1,c}.

    Working at CHAR level (with per-char origin/bbox) lets us re-assemble
    the formula in reading order — span-level text comes out scrambled for
    complex math layouts.
    """
    chars: list[dict] = []
    d = page.get_text("rawdict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not _is_math_font(span.get("font", "")):
                    continue
                for ch in span.get("chars", []):
                    bb = ch.get("bbox")
                    if bb:
                        x0, y0, x1, y1 = (float(v) for v in bb)
                    else:
                        x0, y0 = ch["origin"]
                        x1, y1 = x0 + 2.0, y0 - 2.0  # tiny fallback box
                    c = ch.get("c", "")
                    if c.strip():
                        chars.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "c": c})
    return chars


def _formula_blocks(
    chars: list[dict],
    line_tol: float = 4.0,
    max_line_gap: float = MAX_LINE_GAP,
) -> list[dict]:
    """Cluster chars into formula blocks.

    1) group chars into lines (baseline tolerance), sort each line by x →
       reassembled reading-order text;
    2) merge vertically-adjacent lines (multi-line formulas) into blocks.
    """
    if not chars:
        return []
    chars.sort(key=lambda c: (c["y0"], c["x0"]))

    # 1) lines
    rows: list[list[dict]] = []
    cur = [chars[0]]
    for c in chars[1:]:
        if abs(c["y0"] - cur[-1]["y0"]) <= line_tol:
            cur.append(c)
        else:
            rows.append(cur)
            cur = [c]
    rows.append(cur)

    line_blocks: list[dict] = []
    for row in rows:
        row.sort(key=lambda c: c["x0"])
        text = "".join(c["c"] for c in row).strip()
        if not text:
            continue
        line_blocks.append(
            {
                "bbox": [
                    min(c["x0"] for c in row),
                    min(c["y0"] for c in row),
                    max(c["x1"] for c in row),
                    max(c["y1"] for c in row),
                ],
                "text": text,
            }
        )

    # 2) vertical merge
    blocks: list[dict] = []
    for lb in line_blocks:
        if blocks and (lb["bbox"][1] - blocks[-1]["bbox"][3]) <= max_line_gap:
            last = blocks[-1]
            last["bbox"] = [
                min(last["bbox"][0], lb["bbox"][0]),
                min(last["bbox"][1], lb["bbox"][1]),
                max(last["bbox"][2], lb["bbox"][2]),
                max(last["bbox"][3], lb["bbox"][3]),
            ]
            last["text"] = (last["text"] + " " + lb["text"]).strip()
        else:
            blocks.append(dict(lb))
    return blocks


def extract_formulas(paper_id: str = "transformer") -> list[dict]:
    """Return formula blocks with page, bbox (PDF points) and section link.

    Each formula: {id, page, bbox: [x0,y0,x1,y1], text, section_id, section_title}
    """
    paper = parse_paper(paper_path(paper_id))

    # document-wide heading sequence: (page, y0, id, title)
    headings = sorted(
        ((s["page"], s["bbox"][1], s["id"], s["title"]) for s in paper["sections"]),
        key=lambda h: (h[0], h[1]),
    )

    doc = pymupdf.open(paper_path(paper_id))
    formulas: list[dict] = []
    fid = 0

    for pno, page in enumerate(doc):
        blocks = _formula_blocks(_collect_math_chars(page))
        for b in blocks:
            yc = (b["bbox"][1] + b["bbox"][3]) / 2.0
            sec_id = None
            sec_title = ""
            for hpage, hy, sid, stitle in headings:
                if (hpage, hy) <= (pno + 1, yc):
                    sec_id, sec_title = sid, stitle
                else:
                    break
            fid += 1
            formulas.append(
                {
                    "id": f"f{fid}",
                    "page": pno + 1,
                    "bbox": [round(float(v), 2) for v in b["bbox"]],
                    "text": b["text"][:200],
                    "section_id": sec_id,
                    "section_title": sec_title,
                }
            )

    return formulas


def get_formulas() -> dict:
    """Public contract for the frontend: formulas + their section links."""
    return {"count": None, "formulas": extract_formulas()}
