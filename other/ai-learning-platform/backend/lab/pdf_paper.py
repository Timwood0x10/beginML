"""pdf_paper.py — parse a paper PDF into clickable sections.

Uses pymupdf's layout dict to reconstruct lines with their font size, then
splits the paper into sections by heading size:
  14.3  -> level-1 heading (e.g. "1 Dot-Product Attention (DPA)")
  12.0  -> level-2 heading (e.g. "1.2 Scaled Dot-Product Attention (SDPA)")
  <=11  -> body text, accumulated into the current section

Each section carries a `math` flag (contains LaTeX/unicode math) so the
frontend can show a "formula" hint. The section list feeds the paper <->
source <- > run page: clicking a section shows that section's numpy
implementation and its computed output.
"""

from pathlib import Path
import base64

import pymupdf

PAPER_PATH = Path(__file__).parent / "papers" / "transformer.pdf"

H1_SIZE = 13.5  # >= this is a level-1 heading
H2_SIZE = 11.8  # >= this (and numbered) is a level-2 heading
VIEW_ZOOM = 2.0  # pixmap zoom for the on-page clickable view

# characters that strongly suggest math content in a line
_MATH_HINTS = ("$", "\\", "∑", "∂", "√", "π", "θ", "α", "β", "λ", "μ", "σ", "φ", "∇", "∈", "≤", "≥", "·", "×")


def _line_is_math(line: str) -> bool:
    return any(h in line for h in _MATH_HINTS)


def _split_heading(text: str) -> tuple[int | None, str]:
    """Return (level, title) if `text` looks like a numbered heading."""
    t = text.strip()
    if not t:
        return None, ""
    # "1 Title" / "1.2 Subtitle" — level from number of dotted parts
    parts = t.split(None, 1)
    if not parts:
        return None, ""
    num, rest = parts[0], parts[1] if len(parts) > 1 else ""
    if not num.replace(".", "").isdigit() or "." not in num:
        return None, ""
    level = num.count(".")
    return (1 if level == 0 else 2 if level == 1 else 3), rest.strip()


def parse_paper(path: Path | None = None) -> dict:
    """Extract sections from the paper PDF.

    Returns {title, author, pages, sections: [{id, level, title, text, math}]}.
    """
    p = path or PAPER_PATH
    doc = pymupdf.open(p)

    title = ""
    author = ""
    sections: list[dict] = []
    current: dict | None = None

    def push():
        nonlocal current
        if current is not None and current["text"].strip():
            current["math"] = _line_is_math(current["text"])
            sections.append(current)
        current = None

    for pno, page in enumerate(doc):
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line.get("spans", [])
                if not spans:
                    continue
                size = max(s.get("size", 0) for s in spans)
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue

                if size >= H1_SIZE:
                    # level-1 heading — first one on page 1 might be the doc title
                    level, rest = _split_heading(text)
                    if level == 1:
                        push()
                        current = {
                            "id": f"s{len(sections) + 1}",
                            "level": 1,
                            "title": rest or text,
                            "text": "",
                            "page": pno + 1,
                            "bbox": [round(float(v), 2) for v in line.get("bbox", [0, 0, 0, 0])],
                        }
                        continue
                    if pno == 0 and not title:
                        title = text
                        continue
                    continue

                if size >= H2_SIZE:
                    level, rest = _split_heading(text)
                    if level == 2:
                        push()
                        current = {
                            "id": f"s{len(sections) + 1}",
                            "level": 2,
                            "title": rest or text,
                            "text": "",
                            "page": pno + 1,
                            "bbox": [round(float(v), 2) for v in line.get("bbox", [0, 0, 0, 0])],
                        }
                        continue
                    if pno == 0 and text and not author:
                        author = text
                        continue
                    # other 12.0 spans on page 1 (date, etc.) — skip on cover
                    if pno == 0:
                        continue
                    continue

                # body text — accumulate into the current section
                if current is not None:
                    current["text"] += text + "\n"

    push()

    return {
        "title": title,
        "author": author,
        "pages": len(doc),
        "sections": sections,
    }


def get_paper() -> dict:
    return parse_paper()


def render_pages(zoom: float = VIEW_ZOOM) -> list[dict]:
    """Render every PDF page to a PNG (base64) plus its pixel size.

    The frontend shows these images directly — the paper appears as itself
    — and overlays clickable regions using the sections' page/bbox (PDF
    points scaled by `zoom` to pixels).
    """
    doc = pymupdf.open(PAPER_PATH)
    out: list[dict] = []
    for pno, page in enumerate(doc):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        out.append(
            {
                "page": pno + 1,
                "image": base64.b64encode(pix.tobytes("png")).decode(),
                "width": pix.width,
                "height": pix.height,
            }
        )
    return out


def get_paper_view() -> dict:
    """Full clickable view: rendered pages + sections with page/bbox.

    section.bbox is in PDF points (x0,y0,x1,y1); multiply by VIEW_ZOOM to get
    pixel coordinates on the rendered page image.
    """
    paper = parse_paper()
    return {
        "title": paper["title"],
        "author": paper["author"],
        "pages": paper["pages"],
        "zoom": VIEW_ZOOM,
        "images": render_pages(VIEW_ZOOM),
        "sections": paper["sections"],
    }
