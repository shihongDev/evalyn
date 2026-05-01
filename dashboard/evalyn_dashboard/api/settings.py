"""Stub /api/settings router (Phase 1 A1.5; replaced by Phase 3 lane C1)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED = JSONResponse(
    {"error": "not implemented", "lane": "C1"}, status_code=501
)


@router.get("")
async def get_settings() -> JSONResponse:
    return _NOT_IMPLEMENTED


@router.post("/active")
async def set_active() -> JSONResponse:
    return _NOT_IMPLEMENTED


@router.get("/models/{provider}")
async def list_models(provider: str) -> JSONResponse:
    del provider
    return _NOT_IMPLEMENTED


@router.post("/test/{provider}")
async def test_provider(provider: str) -> JSONResponse:
    del provider
    return _NOT_IMPLEMENTED


@router.get("/{provider}")
async def get_provider(provider: str) -> JSONResponse:
    del provider
    return _NOT_IMPLEMENTED


@router.post("/{provider}")
async def set_provider(provider: str) -> JSONResponse:
    del provider
    return _NOT_IMPLEMENTED
