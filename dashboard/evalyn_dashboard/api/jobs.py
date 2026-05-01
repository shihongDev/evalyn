"""Stub /api/jobs router (Phase 1 A1.5; replaced by Phase 2 lane B1)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_NOT_IMPLEMENTED = JSONResponse(
    {"error": "not implemented", "lane": "B1"}, status_code=501
)


@router.get("/recent")
async def recent_jobs() -> JSONResponse:
    return _NOT_IMPLEMENTED


@router.get("/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    del job_id
    return _NOT_IMPLEMENTED


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    del job_id
    return _NOT_IMPLEMENTED
