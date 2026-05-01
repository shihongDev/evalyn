"""Stub /api/agent router (Phase 1 A1.5; replaced by Phase 3 lane C1)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED = JSONResponse(
    {"error": "not implemented", "lane": "C1"}, status_code=501
)


@router.post("/chat")
async def chat() -> JSONResponse:
    return _NOT_IMPLEMENTED


@router.post("/chat/{thread_id}/confirm")
async def confirm(thread_id: str) -> JSONResponse:
    del thread_id
    return _NOT_IMPLEMENTED
