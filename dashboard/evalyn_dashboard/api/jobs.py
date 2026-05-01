"""``/api/jobs`` router: list / inspect / cancel (Lane B1.3, B1.4).

GET ``/api/jobs/recent?limit=N`` returns the most recent jobs from the
shared :class:`JobManager`, projected to a JSON-friendly shape.

GET ``/api/jobs/{job_id}`` returns a single job's metadata.

POST ``/api/jobs/{job_id}/cancel`` issues SIGTERM (with grace period
SIGKILL escalation handled by the manager). Returns 404 if the job is
unknown.

The WebSocket route ``/ws/jobs/{job_id}`` lives in ``api/jobs_ws.py`` and
is mounted directly on the FastAPI app rather than via this router so it
sits on the ``/ws/`` prefix instead of ``/api/``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..jobs import Job

router = APIRouter()


def _job_to_dict(job: Job) -> dict:
    """Project a :class:`Job` to its public JSON shape.

    Excludes private bookkeeping fields (``_process``, ``_subscribers``,
    raw event log). The event log is reachable separately via the
    WebSocket; pushing the entire log down ``/api/jobs/{id}`` would
    bloat the payload for completed multi-megabyte runs.
    """
    return {
        "id": job.id,
        "cmd": job.cmd,
        "state": job.state,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "exit_code": job.exit_code,
        "pid": job.pid,
        "duration": job.duration,
    }


@router.get("/recent")
async def recent_jobs(request: Request, limit: int = 100) -> JSONResponse:
    """Return up to ``limit`` recent jobs in reverse chronological order."""
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be <= 1000")
    jm = request.app.state.job_manager
    return JSONResponse([_job_to_dict(j) for j in jm.recent(n=limit)])


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str) -> JSONResponse:
    jm = request.app.state.job_manager
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return JSONResponse(_job_to_dict(job))


@router.post("/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> JSONResponse:
    jm = request.app.state.job_manager
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    await jm.cancel(job_id)
    # The manager runs the SIGTERM/grace/SIGKILL dance synchronously; by the
    # time we reach this line, ``state`` is "cancelled" or already terminal.
    return JSONResponse(_job_to_dict(jm.get(job_id) or job))


__all__ = ["router"]
