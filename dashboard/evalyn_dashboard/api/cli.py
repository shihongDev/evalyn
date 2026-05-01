"""Stub /api/cli router (Phase 1 A1.5; replaced by Phase 2 lane B1)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED = JSONResponse(
    {"error": "not implemented", "lane": "B1"}, status_code=501
)


@router.get("")
async def get_catalog() -> JSONResponse:
    return _NOT_IMPLEMENTED


@router.post("/run")
async def run_cli() -> JSONResponse:
    return _NOT_IMPLEMENTED
