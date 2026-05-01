"""Stub /api/files router (Phase 1 A1.5; replaced by Phase 2 lane B1)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED = JSONResponse(
    {"error": "not implemented", "lane": "B1"}, status_code=501
)


@router.get("/tree")
async def file_tree() -> JSONResponse:
    return _NOT_IMPLEMENTED


@router.get("/read")
async def read_file() -> JSONResponse:
    return _NOT_IMPLEMENTED
