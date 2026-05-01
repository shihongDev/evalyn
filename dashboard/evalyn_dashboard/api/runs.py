"""Stub /api/runs router (Phase 1 A1.5; replaced by Phase 2 lane B1)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED = JSONResponse(
    {"error": "not implemented", "lane": "B1"}, status_code=501
)


@router.get("")
async def list_runs() -> JSONResponse:
    return _NOT_IMPLEMENTED


@router.get("/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    del run_id
    return _NOT_IMPLEMENTED
