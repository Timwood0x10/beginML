"""
FastAPI router exposing the Math Lab modules and their compute endpoints.

  GET  /api/lab/modules             -> module metadata (drives the sidebar)
  GET  /api/lab/modules/{module_id} -> one module's metadata
  POST /api/lab/compute/{module_id} -> run the module's numerical backend

The compute body is a free-form dict of control values (sliders/toggles/selects)
defined in lab/modules.py.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import (
    activations,
    anchors,
    attention,
    bias_variance,
    convolution,
    detective,
    distributions,
    double_descent,
    engram,
    entropy,
    feature_hunt,
    forge,
    losses,
    mamba,
    mappings,
    matrix_transform,
    moe,
    neural_net,
    optimizers,
    paper_formulas,
    paper_sections,
    pca,
    pdf_paper,
    quantization,
    regularization,
    residual_river,
    rope,
    sampling,
    svm,
    token_society,
    transformer,
    transformer_mri,
)
from .modules import MODULES, get_module

router = APIRouter(prefix="/api/lab", tags=["lab"])

# module_id -> compute(module_params) -> dict
_COMPUTERS = {
    "gradient-descent": optimizers.compute,
    "activations": activations.compute,
    "losses": losses.compute,
    "convolution": convolution.compute,
    "matrix-transform": matrix_transform.compute,
    "distributions": distributions.compute,
    "entropy": entropy.compute,
    "neural-net": neural_net.compute,
    "attention": attention.compute,
    "transformer-training": transformer.compute,
    "agent-forge": forge.compute,
    "pca": pca.compute,
    "regularization": regularization.compute,
    "svm": svm.compute,
    "sampling-machine": sampling.compute,
    "rotary-observatory": rope.compute,
    "dangerous-mountain": double_descent.compute,
    "shooting-range": bias_variance.compute,
    "weight-freezer": quantization.compute,
    "representation-river": residual_river.compute,
    "token-society": token_society.compute,
    "transformer-detective": detective.compute,
    "moe-expert-routing": moe.compute,
    "mamba-memory-race": mamba.compute,
    "transformer-mri": transformer_mri.compute,
    "feature-hunt": feature_hunt.compute,
    "paper-engram": engram.compute,
}

# module_id -> python source file (for the paper ↔ source ↔ run page).
# Whitelist only — the UI may never read arbitrary paths.
_SOURCE_FILES: dict[str, str] = {
    "sampling-machine": "sampling.py",
    "rotary-observatory": "rope.py",
    "dangerous-mountain": "double_descent.py",
    "shooting-range": "bias_variance.py",
    "weight-freezer": "quantization.py",
    "representation-river": "residual_river.py",
    "token-society": "token_society.py",
    "transformer-detective": "detective.py",
    "moe-expert-routing": "moe.py",
    "mamba-memory-race": "mamba.py",
    "transformer-mri": "transformer_mri.py",
    "feature-hunt": "feature_hunt.py",
    "paper-engram": "engram.py",
}


class ComputeRequest(BaseModel):
    params: dict[str, Any] = {}


@router.get("/modules")
def list_modules() -> dict[str, Any]:
    return {"modules": MODULES}


@router.get("/modules/{module_id}")
def get_one_module(module_id: str) -> dict[str, Any]:
    m = get_module(module_id)
    if not m:
        raise HTTPException(status_code=404, detail="Unknown lab module")
    return m


@router.post("/compute/{module_id}")
def compute(module_id: str, req: ComputeRequest) -> dict[str, Any]:
    fn = _COMPUTERS.get(module_id)
    if fn is None:
        raise HTTPException(status_code=404, detail="Unknown lab module")
    try:
        return fn(req.params or {})
    except Exception as exc:  # surface numerical errors cleanly to the UI
        raise HTTPException(status_code=400, detail=f"compute failed: {exc}") from exc


@router.get("/source/{module_id}")
def get_source(module_id: str) -> dict[str, Any]:
    """Return the python source of a lab module — feeds the middle column of
    the paper ↔ source ↔ run page."""
    fname = _SOURCE_FILES.get(module_id)
    if not fname:
        raise HTTPException(status_code=404, detail="No source for this module")
    path = Path(__file__).parent / fname
    try:
        code = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Source file missing") from exc
    return {"module_id": module_id, "filename": fname, "code": code}


# ---- paper ↔ source ↔ run (clickable sections) ----------------------------


@router.get("/paper")
def get_paper() -> dict[str, Any]:
    """Parse the paper PDF into clickable sections."""
    return pdf_paper.get_paper()


@router.get("/paper/view")
def get_paper_view() -> dict[str, Any]:
    """Rendered PDF pages (PNG) + sections with page/bbox for on-page
    clicking — the paper appears as itself and sections are clickable."""
    return pdf_paper.get_paper_view()


@router.get("/paper/formulas")
def get_paper_formulas() -> dict[str, Any]:
    """Formula blocks (page + bbox) linked to their sections.

    The backend does all the work: formula detection, fragment merging,
    section assignment, PLUS the semantic anchor (concept/label) and the
    manually-authored implementation + experiment mappings. The frontend
    only renders the highlight regions and fetches source + run output."""
    formulas = paper_formulas.extract_formulas()
    counters: dict[str, int] = {}
    for f in formulas:
        sid = f["section_id"]
        counters[sid] = counters.get(sid, 0) + 1
        f["anchor"] = anchors.build_anchor(f, f"Equation {counters[sid]}")
        f["implementation"] = mappings.implementation_for(sid)
        f["experiment"] = mappings.experiment_for(sid)
        f["code_map"] = mappings.code_map_for(sid)
    return {"count": len(formulas), "formulas": formulas}


@router.get("/paper/source/{section_id}")
def get_paper_source(section_id: str) -> dict[str, Any]:
    """Numpy implementation for one paper section (middle column)."""
    src = paper_sections.section_source(section_id)
    if src is None:
        raise HTTPException(status_code=404, detail="Unknown paper section")
    return src


@router.post("/paper/run/{section_id}")
def run_paper_section(section_id: str, req: ComputeRequest) -> dict[str, Any]:
    """Execute one paper section's implementation (right column)."""
    out = paper_sections.run_section(section_id, req.params or {})
    if out is None:
        raise HTTPException(status_code=404, detail="Unknown paper section")
    return out
