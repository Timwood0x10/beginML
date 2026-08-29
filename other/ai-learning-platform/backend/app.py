"""
AI Learning Platform — Backend Service

Serves Markdown notes with:
  * TF-IDF semantic search (scikit-learn)
  * Per-note "related notes" via cosine similarity
  * A 2D knowledge map computed with classical MDS (scikit-learn)
  * Rendered HTML (code highlighting, tables, TOC)
  * Static image serving so relative image links in notes keep working
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
import os
from typing import Any

import latex2mathml.converter as latex_converter  # type: ignore
import numpy as np
import pymdownx.superfences  # type: ignore
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from markdown import markdown as render_markdown
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# backend/ resides under <notes>/ai-learning-platform/backend; notes root is two levels up
NOTES_ROOT = Path(
    os.getenv("NOTES_ROOT", Path(__file__).resolve().parents[2])
).resolve()

# Directories containing notes (relative to NOTES_ROOT). Notes are split by
# language: Original Chinese notes under zh/, English translations under en/
# Do NOT scan the frontend/backend/templates directories.
SCAN_DIRS = [
    NOTES_ROOT / "zh" / "math",
    NOTES_ROOT / "zh" / "Self-Attention",
    NOTES_ROOT / "zh" / "Hybrid-models",
    NOTES_ROOT / "zh" / "paper",
    NOTES_ROOT / "zh" / "agent",
    NOTES_ROOT / "en" / "math",
    NOTES_ROOT / "en" / "Self-Attention",
    NOTES_ROOT / "en" / "Hybrid-models",
    NOTES_ROOT / "en" / "paper",
    NOTES_ROOT / "en" / "agent",
]

# Image / asset roots — served under /assets/<mount>/...
ASSET_ROOTS: dict[str, Path] = {
    "images": NOTES_ROOT / "images",
    "math-images": NOTES_ROOT / "zh" / "math" / "images",
    "self-attention-images": NOTES_ROOT / "zh" / "Self-Attention" / "images",
    "hybrid-images": NOTES_ROOT / "zh" / "Hybrid-models" / "image",
    "paper-images": NOTES_ROOT / "zh" / "paper" / "images",
}

MARKDOWN_EXTENSIONS = [
    "markdown.extensions.fenced_code",
    "markdown.extensions.tables",
    "markdown.extensions.toc",
    "markdown.extensions.nl2br",
    "markdown.extensions.sane_lists",
    "pymdownx.arithmatex",
    "pymdownx.highlight",
    "pymdownx.superfences",
    "pymdownx.inlinehilite",
    "pymdownx.tilde",
]

MARKDOWN_EXTENSION_CONFIGS = {
    "pymdownx.arithmatex": {"generic": True},
    "pymdownx.highlight": {"linenums": False, "anchor_linenums": False},
    # Render fenced ```mermaid blocks as <div class="mermaid">…</div> so the
    # frontend can hand them to the mermaid.js renderer (graphs, flowcharts).
    "pymdownx.superfences": {
        "custom_fences": [
            {
                "name": "mermaid",
                "class": "mermaid",
                "format": pymdownx.superfences.fence_div_format,
            }
        ]
    },
}

# Category metadata, ordered for display.
CATEGORY_DEFS: dict[str, dict[str, str]] = {
    "math": {"id": "math", "en": "Mathematics", "icon": "functions"},
    "attention": {"id": "attention", "en": "Self-Attention", "icon": "psychology"},
    "hybrid": {"id": "hybrid", "en": "Hybrid Models", "icon": "bolt"},
    "paper": {"id": "paper", "en": "Research Papers", "icon": "description"},
    "agent": {"id": "agent", "en": "Agent Engineering", "icon": "smart_toy"},
    "general": {"id": "general", "en": "General", "icon": "article"},
}

# Map a lowercased path segment → category id.
_PATH_SEGMENT_TO_CATEGORY = {
    "math": "math",
    "self-attention": "attention",
    "hybrid-models": "hybrid",
    "paper": "paper",
    "agent": "agent",
}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

from lab.router import router as lab_router

app = FastAPI(title="AI Learning Platform", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "").split(",")
    if os.getenv("ALLOWED_ORIGINS")
    else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lab_router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_category(relative_path: str) -> dict[str, str]:
    for part in Path(relative_path).parts:
        key = part.lower().replace("_", "-")
        if key in _PATH_SEGMENT_TO_CATEGORY:
            return CATEGORY_DEFS[_PATH_SEGMENT_TO_CATEGORY[key]]
    return CATEGORY_DEFS["general"]


def _iter_non_code_lines(content: str):
    """Yield (line, stripped) for every line, skipping fenced code blocks.

    Notes embed Python/other fenced blocks whose comment lines (`# ...`) must
    not be mistaken for Markdown headings when extracting titles or headings.
    """
    in_code = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_code = not in_code
            continue
        if in_code:
            continue
        yield line, stripped


def extract_title(content: str) -> str:
    for _, stripped in _iter_non_code_lines(content):
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def extract_description(content: str) -> str:
    """First non-heading, non-empty paragraph — used for card subtitles.

    Inline math ($...$, \\(...\\)) and display math ($$...$$, \\[...\\]) are
    stripped so card previews never leak raw LaTeX.
    """
    lines = content.split("\n")
    buffer: list[str] = []
    in_paragraph = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_paragraph and buffer:
                break
            in_paragraph = False
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(("```", "---", "|", "- ", "* ", "> ")):
            continue
        in_paragraph = True
        buffer.append(stripped)
    text = " ".join(buffer)
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)  # $$...$$ display math
    text = re.sub(r"\\\([\s\S]*?\\\)", "", text)  # \(...\) inline math
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)  # \[...\] display math
    text = re.sub(r"\$[^$\n]*?\$", "", text)  # $...$ inline math
    text = re.sub(r"[#*`_>\[\]]", "", text).strip()
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) > 220:
        text = text[:217].rstrip() + "…"
    return text


def count_words(content: str) -> int:
    # CJK-safe approximate word count: split on whitespace + count CJK chars.
    cjk = len(re.findall(r"[\u4e00-\u9fff]", content))
    words = len(re.findall(r"[A-Za-z0-9_]+", content))
    return words + cjk


def estimate_reading_time(word_count: int) -> int:
    return max(1, word_count // 200)


def humanize_filename(stem: str) -> str:
    text = stem.replace("_", " ").replace("-", " ")
    text = re.sub(r"^\d+(\.\d+)*[\.\s]*", "", text)  # strip leading numbering
    return text.strip().title()


def normalize_math(content: str) -> str:
    """Normalize display math so pymdownx-arithmatex always recognizes it.

    Notes use ``$$...$$`` for block math, often without surrounding blank
    lines. The ``nl2br`` markdown extension then inserts ``<br>`` tags that
    break arithmatex's block detection, leaving the dollar signs as literal
    text. We rewrite both multi-line and single-line ``$$`` forms to the
    bracket delimiters ``\\[...\\]`` with blank lines around them, which
    arithmatex matches reliably regardless of surrounding newlines.
    """
    # Multi-line:  $$\n ... \n$$  (own block)
    content = re.sub(
        r"^[ \t]*\$\$\s*\n(.*?)\n\s*\$\$\s*$",
        r"\n\n\\[\n\1\n\\]\n\n",
        content,
        flags=re.DOTALL | re.MULTILINE,
    )
    # Single-line:  $$ ... $$
    content = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", content, flags=re.DOTALL)
    # Ensure any bracket-delimited block \[ ... \] sits in its own paragraph
    # (blank lines around it) so arithmatex emits a display <div> rather than
    # an inline <span>. This covers notes that already used \[...\] directly.
    content = re.sub(
        r"[ \t]*(\\\[.*?\\\])[ \t]*",
        lambda m: f"\n\n{m.group(1)}\n\n",
        content,
        flags=re.DOTALL,
    )
    # Collapse 3+ newlines created above back to a clean pair.
    content = re.sub(r"\n{3,}", "\n\n", content)
    # Inline math written with stray whitespace next to the delimiters
    # (`$ x $` / `$x $` / `$ x$`) is not recognized by arithmatex, which
    # requires the content to hug the dollar signs. Rewrite such pairs to
    # the `\(x\)` form (content trimmed). Done pair-by-pair on every line
    # outside fenced code blocks so well-formed `$...$` (like a pair of
    # `$(\lambda_i>0)$` and `$(\lambda_i<0)$` on one line) is never touched.
    lines = content.split("\n")
    in_code = False
    for li, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith(("```", "~~~")):
            in_code = not in_code
            continue
        if in_code:
            continue
        out: list[str] = []
        i = 0
        while True:
            j = line.find("$", i)
            if j < 0:
                out.append(line[i:])
                break
            k = line.find("$", j + 1)
            if k < 0:
                out.append(line[i:])
                break
            body = line[j + 1 : k]
            leading_space = bool(body) and body[0] in " \t"
            trailing_space = bool(body) and body[-1] in " \t"
            if leading_space or trailing_space:
                out.append(line[i:j] + "\\(" + body.strip() + "\\)")
            else:
                out.append(line[i : k + 1])
            i = k + 1
        lines[li] = "".join(out)
    return "\n".join(lines)


# Matches pymdownx-arithmatex output: <span class="arithmatex">\(...\)</span>
# and <div class="arithmatex">\[...\]</div>.
_ARITHMATEX_RE = re.compile(
    r'<(span|div) class="arithmatex">\\([\(\[])(.*?)\\([\)\]])</\1>',
    flags=re.DOTALL,
)


def mathml_from_tex(tex: str, display: bool) -> str:
    """Render LaTeX to MathML server-side (native browser rendering)."""
    try:
        mode = "block" if display else "inline"
        return latex_converter.convert(tex, display=mode)
    except Exception:  # noqa: BLE001 - fallback so content never vanishes
        # Fall back to a safe inline wrapper so content never vanishes.
        tag = "div" if display else "span"
        return f'<{tag} class="ailearn-math-error">\\[{tex}\\]</{tag}>'


def render_math_to_mathml(html: str) -> str:
    """Replace arithmatex wrappers with server-rendered MathML."""

    def repl(match: re.Match) -> str:
        tag, left, tex, _right = (
            match.group(1),
            match.group(2),
            match.group(3),
            match.group(4),
        )
        display = left == "["
        mml = mathml_from_tex(tex.strip(), display)
        if mml.startswith("<math") and not mml.startswith("<" + tag):
            return mml
        return f'<{tag} class="ailearn-math">{mml}</{tag}>'

    return _ARITHMATEX_RE.sub(repl, html)


def rewrite_image_paths(html: str, relative_note_path: str) -> str:
    """
    Rewrite relative image src in rendered HTML so they resolve against
    the note's directory through the /assets endpoint.

    e.g. `images/foo.png` in math/3.md -> /assets/math-images/foo.png
    """
    note_dir = (NOTES_ROOT / relative_note_path).parent

    def repl(match: re.Match) -> str:
        prefix, src, suffix = match.group(1), match.group(2), match.group(3)
        if src.startswith(("http://", "https://", "data:", "/")):
            return match.group(0)
        target = (note_dir / src).resolve()
        # English notes live under en/, but their images are co-located with
        # the Chinese originals under zh/ — fall back to the zh/ sibling.
        if not target.exists() and relative_note_path.startswith("en/"):
            zh_path = NOTES_ROOT / "zh" / relative_note_path[len("en/") :]
            target = (zh_path.parent / src).resolve()
        try:
            rel = target.relative_to(NOTES_ROOT.resolve())
        except ValueError:
            return match.group(0)
        parts = rel.parts
        # Find the best asset mount by matching the first path segment.
        lower_parts = [p.lower() for p in parts]
        for mount, root in ASSET_ROOTS.items():
            root_parts = [p.lower() for p in root.relative_to(NOTES_ROOT).parts]
            if lower_parts[: len(root_parts)] == root_parts:
                rest = "/".join(parts[len(root_parts) :])
                return f"{prefix}/assets/{mount}/{rest}{suffix}"
        # Fallback: serve via generic raw file route
        return f"{prefix}/raw/{rel.as_posix()}{suffix}"

    return re.sub(r'(<img[^>]*\ssrc=")([^"]+)(")', repl, html)


# ---------------------------------------------------------------------------
# Note index
# ---------------------------------------------------------------------------


class NoteIndex:
    """Scans notes, builds a TF-IDF index and a 2D semantic map."""

    def __init__(self) -> None:
        self.notes: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.similarity = None
        self.map_points: list[dict[str, Any]] = []
        self.topic_keywords: list[dict[str, Any]] = []

    # -- building -----------------------------------------------------------

    def build(self) -> None:
        self.notes = []
        md_files: list[Path] = []

        for scan_dir in SCAN_DIRS:
            if scan_dir.is_dir():
                md_files.extend(sorted(scan_dir.rglob("*.md")))

        # Loose notes at the language roots (not inside a category dir)
        for lang in ("zh", "en"):
            md_files.extend(sorted((NOTES_ROOT / lang).glob("*.md")))

        seen: set[Path] = set()
        for file_path in md_files:
            if file_path in seen:
                continue
            seen.add(file_path)
            # Skip files inside code/ directories and platform/template dirs
            if any(
                part
                in {
                    "code",
                    "node_modules",
                    ".venv",
                    "ai-learning-platform",
                    "templates",
                    ".codescope",
                }
                for part in file_path.parts
            ):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            relative_path = str(file_path.relative_to(NOTES_ROOT))
            category = infer_category(relative_path)
            title = extract_title(content) or humanize_filename(file_path.stem)
            word_count = count_words(content)

            note = {
                "id": hashlib.md5(relative_path.encode()).hexdigest()[:12],
                "path": relative_path,
                "filename": file_path.name,
                "title": title,
                "description": extract_description(content),
                "category": category,
                "wordCount": word_count,
                "readingTime": estimate_reading_time(word_count),
                "headings": self._extract_headings(content),
            }
            self.notes.append(note)

        self.by_id = {n["id"]: n for n in self.notes}
        self._build_tfidf()
        self._build_similarity()
        self._build_map()
        self._build_topics()

    @staticmethod
    def _extract_headings(content: str, max_depth: int = 3) -> list[dict[str, Any]]:
        headings: list[dict[str, Any]] = []
        slug_counts: dict[str, int] = {}
        for _, stripped in _iter_non_code_lines(content):
            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if not m:
                continue
            level = len(m.group(1))
            if level > max_depth:
                continue
            text = m.group(2).strip()
            slug = re.sub(r"[^\w\- ]", "", text.lower()).strip().replace(" ", "-")
            slug_counts[slug] = slug_counts.get(slug, 0) + 1
            if slug_counts[slug] > 1:
                slug = f"{slug}-{slug_counts[slug]}"
            headings.append({"text": text, "level": level, "slug": slug})
        return headings

    def _corpus(self) -> list[str]:
        return [
            f"{n['title']} {n['description']} {' '.join(h['text'] for h in n['headings'])}"
            for n in self.notes
        ]

    def _build_tfidf(self) -> None:
        corpus = self._corpus()
        if not corpus:
            return
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9+\-_]*\b",
        )
        # Build the word matrix, then augment with CJK char-bigrams so Chinese
        # notes (which have no whitespace word boundaries) still get meaningful
        # features for similarity / search.
        word_matrix = self.vectorizer.fit_transform(corpus)  # type: ignore
        cjk = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 3),
            min_df=1,
            sublinear_tf=True,
            max_features=4000,
        )
        # Restrict char n-grams to CJK content to avoid drowning out words.
        cjk_corpus = [" ".join(re.findall(r"[一-鿿]+", t)) for t in corpus]
        cjk_matrix = cjk.fit_transform(cjk_corpus)

        from scipy.sparse import hstack  # type: ignore

        self.tfidf_matrix = hstack([word_matrix, cjk_matrix]).tocsr()
        # Keep both vectorizers only for debugging; feature names for topics
        # come from the word vectorizer.
        self._cjk_vectorizer = cjk

    def _build_similarity(self) -> None:
        if self.tfidf_matrix is None:
            return
        self.similarity = cosine_similarity(self.tfidf_matrix)

    def _build_map(self) -> None:
        """Classical (metric) MDS on the TF-IDF distance = 1 - cosine."""
        if self.tfidf_matrix is None or len(self.notes) < 3:
            self.map_points = []
            return

        # Reduce sparsity first for a stable MDS — SVD to <=50 dims, then
        # cosine distance in that dense space.
        n_components = min(
            50, self.tfidf_matrix.shape[0] - 1, self.tfidf_matrix.shape[1]
        )
        if n_components >= 2:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense = svd.fit_transform(self.tfidf_matrix)
        else:
            dense = self.tfidf_matrix.toarray()

        sim = cosine_similarity(dense)
        np.clip(sim, -1.0, 1.0, out=sim)
        distance = 1.0 - sim
        # Symmetrize & zero diagonal numerically
        distance = (distance + distance.T) / 2.0
        np.fill_diagonal(distance, 0.0)

        try:
            # scikit-learn >= 1.9 renamed `dissimilarity='precomputed'` to
            # `metric='precomputed'` alongside `metric_mds=True`.
            mds = MDS(
                n_components=2,
                metric_mds=True,
                metric="precomputed",
                random_state=42,
                normalized_stress="auto",
                n_init=4,
                max_iter=300,
                init="random",
            )
        except TypeError:
            mds = MDS(
                n_components=2,
                metric=True,
                dissimilarity="precomputed",
                random_state=42,
                normalized_stress="auto",
                n_init=4,
                max_iter=300,
            )
        coords = mds.fit_transform(distance)

        # Normalize to a [-1, 1] canvas with a little padding.
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = maxs - mins
        span[span == 0] = 1.0
        normalized = (coords - mins) / span * 2.0 - 1.0

        points = []
        for i, note in enumerate(self.notes):
            try:
                x = float(normalized[i, 0])
                y = float(normalized[i, 1])
            except Exception:
                x = y = 0.0
            points.append(
                {
                    "id": note["id"],
                    "title": note["title"],
                    "category": note["category"]["id"],
                    "x": x,
                    "y": y,
                    "readingTime": note["readingTime"],
                }
            )
        self.map_points = points

    def _build_topics(self) -> None:
        """Top keywords per category — used as chips / topic clouds."""
        if not self.vectorizer or self.tfidf_matrix is None:
            self.topic_keywords = []
            return
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        n_word_features = len(feature_names)
        topics: list[dict[str, Any]] = []
        for cat_id in {n["category"]["id"] for n in self.notes}:
            indices = [
                i for i, n in enumerate(self.notes) if n["category"]["id"] == cat_id
            ]
            if not indices:
                continue
            # Only score over the word-feature columns (feature_names maps 1:1);
            # CJK char-n-grams occupy the columns after that.
            scores = np.asarray(
                self.tfidf_matrix[indices, :n_word_features].mean(axis=0)
            ).ravel()
            top_idx = scores.argsort()[::-1][:12]
            keywords = []
            for j in top_idx:
                if scores[j] > 0:
                    try:
                        weight = float(np.round(scores[j], 4))
                    except Exception:
                        weight = 0.0
                    keywords.append({"word": str(feature_names[j]), "weight": weight})
            topics.append({"category": cat_id, "keywords": keywords})
        self.topic_keywords = topics

    # -- queries ------------------------------------------------------------

    def _transform_query(self, query: str):
        from scipy.sparse import hstack  # type: ignore

        wv = self.vectorizer.transform([query])  # type: ignore
        cjk_text = " ".join(re.findall(r"[一-鿿]+", query))
        cv = (
            self._cjk_vectorizer.transform([cjk_text])
            if hasattr(self, "_cjk_vectorizer")
            else None
        )
        return hstack([wv, cv]).tocsr() if cv is not None else wv

    def search(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        if not self.vectorizer or self.tfidf_matrix is None:
            return self.notes[:top_k]
        qv = self._transform_query(query)
        scores = cosine_similarity(qv, self.tfidf_matrix).flatten()
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        out: list[dict[str, Any]] = []
        for idx, score in ranked[:top_k]:
            if score <= 0.01:
                continue
            note = dict(self.notes[idx])
            try:
                note["score"] = round(float(score), 4)
            except Exception:
                note["score"] = 0.0
            out.append(note)
        return out

    def related(self, note_id: str, top_k: int = 6) -> list[dict[str, Any]]:
        if self.similarity is None or note_id not in self.by_id:
            return []
        idx = next(i for i, n in enumerate(self.notes) if n["id"] == note_id)
        sims = self.similarity[idx]
        order = sorted(enumerate(sims), key=lambda x: -x[1])
        out: list[dict[str, Any]] = []
        for other_idx, score in order:
            other = self.notes[other_idx]
            if other["id"] == note_id:
                continue
            if score <= 0.05:
                continue
            item = dict(other)
            try:
                item["score"] = round(float(score), 4)
            except Exception:
                item["score"] = 0.0
            out.append(item)
            if len(out) >= top_k:
                break
        return out


note_index = NoteIndex()


@app.on_event("startup")
def _startup_build_index() -> None:
    note_index.build()


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "notes": len(note_index.notes)}


@app.get("/api/stats")
def stats(lang: str = Query("zh")) -> dict[str, Any]:
    prefix = f"{lang}/"
    notes = [n for n in note_index.notes if n["path"].startswith(prefix)]
    categories: dict[str, int] = {}
    for note in notes:
        cid = note["category"]["id"]
        categories[cid] = categories.get(cid, 0) + 1
    total_words = sum(n["wordCount"] for n in notes)
    cat_list = []
    for cid, count in categories.items():
        cat_list.append(
            {**CATEGORY_DEFS.get(cid, CATEGORY_DEFS["general"]), "count": count}
        )
    cat_list.sort(key=lambda c: -c["count"])
    return {
        "totalNotes": len(notes),
        "totalWords": total_words,
        "totalReadingMinutes": estimate_reading_time(total_words),
        "categories": cat_list,
        "topics": note_index.topic_keywords,
    }


@app.get("/api/notes")
def list_notes(
    category: str | None = Query(None),
    search: str | None = Query(None),
    lang: str = Query("zh"),
    limit: int = Query(200, ge=1, le=500),
) -> dict[str, Any]:
    prefix = f"{lang}/"
    if search:
        results = note_index.search(search, top_k=limit)
    else:
        results = list(note_index.notes)

    results = [n for n in results if n["path"].startswith(prefix)]

    if category:
        results = [n for n in results if n["category"]["id"] == category]

    cat_counts: dict[str, dict[str, Any]] = {}
    for note in note_index.notes:
        if not note["path"].startswith(prefix):
            continue
        cid = note["category"]["id"]
        if cid not in cat_counts:
            cat_counts[cid] = {
                **CATEGORY_DEFS.get(cid, CATEGORY_DEFS["general"]),
                "count": 0,
            }
        cat_counts[cid]["count"] += 1

    return {
        "notes": results[:limit],
        "total": len(results),
        "categories": sorted(cat_counts.values(), key=lambda c: -c["count"]),
    }


def localized_note(note: dict[str, Any], lang: str) -> dict[str, Any]:
    """Return a note dict with title/description translated when an .en.md
    sibling exists and lang == 'en'."""
    if lang != "en":
        return note
    # English counterpart lives under en/ mirroring the zh/ relative path.
    note_rel = Path(note["path"])
    if not (note_rel.parts and note_rel.parts[0] == "zh"):
        return note  # already an en/ (or unpaired) note
    en_path = NOTES_ROOT / "en" / Path(*note_rel.parts[1:])
    if not en_path.exists():
        return note
    try:
        content = en_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return note
    out = dict(note)
    out["title"] = extract_title(content) or note["title"]
    out["description"] = extract_description(content) or note["description"]
    return out


@app.get("/api/notes/{note_id}")
def get_note(note_id: str, lang: str = Query("zh")) -> dict[str, Any]:
    note = note_index.by_id.get(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    file_path = NOTES_ROOT / note["path"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Note file missing on disk")

    # English translations live under en/ mirroring the zh/ relative path.
    use_en = False
    source_path = file_path
    note_rel = Path(note["path"])
    if lang == "en" and note_rel.parts and note_rel.parts[0] == "zh":
        en_path = NOTES_ROOT / "en" / Path(*note_rel.parts[1:])
        if en_path.exists():
            use_en = True
            source_path = en_path

    content = source_path.read_text(encoding="utf-8", errors="replace")
    content = normalize_math(content)
    html = render_markdown(
        content,
        extensions=MARKDOWN_EXTENSIONS,
        extension_configs=MARKDOWN_EXTENSION_CONFIGS,
        output_format="html5",  # type: ignore
    )
    html = render_math_to_mathml(html)
    # The frontend renders the note title in its own header; drop the first
    # <h1> from the body so titles never appear twice.
    html = re.sub(r"<h1[^>]*>.*?</h1>", "", html, count=1, flags=re.DOTALL)
    html = rewrite_image_paths(html, note["path"])

    # Prefer the translated file's own title/description when present.
    title = extract_title(content) or note["title"]
    description = extract_description(content) or note["description"]
    headings = note_index._extract_headings(content) if use_en else note["headings"]

    return {
        **note,
        "title": title,
        "description": description,
        "headings": headings,
        "content": content,
        "html": html,
        "translated": use_en,
        "related": [
            localized_note(r, lang)
            for r in note_index.related(note_id)
            if r["path"].startswith(f"{lang}/")
        ],
    }


@app.get("/api/map")
def knowledge_map(
    category: str | None = Query(None), lang: str = Query("zh")
) -> dict[str, Any]:
    prefix = f"{lang}/"
    # map_points carry ids but not paths; resolve each point's owning note.
    points = [
        p
        for p in note_index.map_points
        if (note := note_index.by_id.get(p["id"])) and note["path"].startswith(prefix)
    ]
    if category:
        points = [p for p in points if p["category"] == category]
    categories = [
        {**CATEGORY_DEFS[cid], "count": sum(1 for p in points if p["category"] == cid)}
        for cid in {p["category"] for p in points}
    ]
    return {"points": points, "categories": categories}


# ---------------------------------------------------------------------------
# Routes — raw assets / images
# ---------------------------------------------------------------------------


@app.get("/raw/{file_path:path}")
def raw_file(file_path: str) -> FileResponse:
    target = (NOTES_ROOT / file_path).resolve()
    # Prevent path traversal outside NOTES_ROOT
    try:
        target.relative_to(NOTES_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


# Mount each known asset directory under /assets/<mount>
for mount_name, asset_dir in ASSET_ROOTS.items():
    if asset_dir.is_dir():
        app.mount(
            f"/assets/{mount_name}",
            StaticFiles(directory=str(asset_dir)),
            name=f"assets-{mount_name}",
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
