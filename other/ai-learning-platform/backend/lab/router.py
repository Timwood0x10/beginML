"""
FastAPI router exposing the Math Lab modules and their compute endpoints.

  GET  /api/lab/modules             -> module metadata (drives the sidebar)
  GET  /api/lab/modules/{module_id} -> one module's metadata
  POST /api/lab/compute/{module_id} -> run the module's numerical backend

The compute body is a free-form dict of control values (sliders/toggles/selects)
defined in lab/modules.py.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import (
    activations,
    attention,
    bias_variance,
    convolution,
    detective,
    distributions,
    double_descent,
    entropy,
    forge,
    losses,
    matrix_transform,
    moe,
    neural_net,
    optimizers,
    pca,
    quantization,
    regularization,
    residual_river,
    rope,
    sampling,
    svm,
    token_society,
    transformer,
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
