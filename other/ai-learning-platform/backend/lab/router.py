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

from . import attention, optimizers, pca, regularization, svm
from .modules import MODULES, get_module

router = APIRouter(prefix="/api/lab", tags=["lab"])

# module_id -> compute(module_params) -> dict
_COMPUTERS = {
    "gradient-descent": optimizers.compute,
    "attention": attention.compute,
    "pca": pca.compute,
    "regularization": regularization.compute,
    "svm": svm.compute,
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
