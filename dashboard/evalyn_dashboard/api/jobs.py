"""``/api/jobs`` router: list / inspect / cancel (Lane B1.3, B1.4).

GET ``/api/jobs/recent?limit=N`` returns the most recent jobs from the
shared :class:`JobManager`, projected to a JSON-friendly shape. When a
sqlite mirror is configured on the manager, persisted rows are merged
in (in-memory wins on collisions) so the Recent Jobs drawer shows
history across restarts.

GET ``/api/jobs/{job_id}`` returns a single job's metadata. On
in-memory miss the route falls through to the sqlite mirror so evicted
or post-restart jobs still resolve. Both miss -> 404.

POST ``/api/jobs/{job_id}/cancel`` issues SIGTERM (with grace period
SIGKILL escalation handled by the manager). Returns 404 if the job is
unknown.

The WebSocket route ``/ws/jobs/{job_id}`` lives in ``api/jobs_ws.py`` and
is mounted directly on the FastAPI app rather than via this router so it
sits on the ``/ws/`` prefix instead of ``/api/``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
        # cli_id (catalog id passed at spawn time) included so /api/jobs/recent
        # responses are uniform between in-memory and persisted sources, and
        # so clients can filter by cli_id without parsing cmd[1].
        "cli_id": job.cli_id,
        "state": job.state,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "exit_code": job.exit_code,
        "pid": job.pid,
        "duration": job.duration,
        # Monotonic count of stderr lines emitted by the subprocess.
        # Survives the events ring trim so this reflects the true total
        # even on long jobs that overflow max_log. Frontend's Recent
        # Jobs drawer surfaces it as "5 stderr" inline.
        "stderr_count": job.stderr_count,
    }


def _iso_to_epoch(value: Any) -> float | None:
    """Parse an ISO-8601 string back to a unix timestamp, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _persisted_to_dict(row: dict) -> dict:
    """Project a sqlite row to the same shape as :func:`_job_to_dict`."""
    cmd_str = row.get("cmd") or ""
    cmd_list = cmd_str.split(" ") if cmd_str else []
    return {
        "id": row.get("job_id"),
        "cmd": cmd_list,
        "cli_id": row.get("cli_id") or "",
        "state": row.get("status") or "unknown",
        "started_at": _iso_to_epoch(row.get("started_at_iso")),
        "ended_at": _iso_to_epoch(row.get("ended_at_iso")),
        "exit_code": row.get("exit_code"),
        "pid": None,
        "duration": row.get("duration_s"),
        # As of the previous backend tick the sqlite mirror persists
        # stderr_count via a forward migration. Older rows back-fill 0
        # via the column DEFAULT. Either way the response shape is
        # uniform with in-memory rows.
        "stderr_count": row.get("stderr_count") or 0,
    }


def _persistence_for(jm) -> Any:
    """Return the JobPersistence attached to the manager, or None."""
    return getattr(jm, "_persistence", None)


@router.get("/recent")
async def recent_jobs(
    request: Request,
    limit: int = 100,
    cli_id: str | None = None,
    status: str | None = None,
) -> JSONResponse:
    """Return up to ``limit`` recent jobs in reverse chronological order.

    Merges in-memory jobs with persisted rows (in-memory wins on id
    collision so the freshest state surfaces). Sort key is ``started_at``
    descending across both sources.

    Optional filters:

    - ``cli_id`` matches the catalog id passed at spawn time. Lets a
      caller fetch "my last 10 run-eval invocations" without paging.
    - ``status`` matches one of ``queued`` / ``running`` / ``complete`` /
      ``failed`` / ``cancelled``. Lets a caller list "only failed jobs"
      for a regression scan.

    Both filters are pushed down to SQL on the persisted side and
    applied in-memory for the live set.
    """
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be <= 1000")
    jm = request.app.state.job_manager
    in_memory_jobs = jm.recent(n=limit)
    if cli_id is not None:
        in_memory_jobs = [j for j in in_memory_jobs if j.cli_id == cli_id]
    if status is not None:
        in_memory_jobs = [j for j in in_memory_jobs if j.state == status]
    in_memory = [_job_to_dict(j) for j in in_memory_jobs]
    seen = {entry["id"] for entry in in_memory}
    merged = list(in_memory)
    persistence = _persistence_for(jm)
    if persistence is not None:
        for row in persistence.list_recent(
            limit=limit, cli_id=cli_id, status=status
        ):
            entry = _persisted_to_dict(row)
            if entry["id"] in seen:
                continue
            merged.append(entry)
            seen.add(entry["id"])
    merged.sort(key=lambda e: (e.get("started_at") or 0.0), reverse=True)
    return JSONResponse(merged[:limit])


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str) -> JSONResponse:
    jm = request.app.state.job_manager
    job = jm.get(job_id)
    if job is not None:
        return JSONResponse(_job_to_dict(job))
    persistence = _persistence_for(jm)
    if persistence is not None:
        row = persistence.get(job_id)
        if row is not None:
            return JSONResponse(_persisted_to_dict(row))
    raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")


@router.get("/{job_id}/output")
async def get_job_output(request: Request, job_id: str) -> JSONResponse:
    """Return the captured stdout/stderr tails for a finished job.

    Read sources, in priority order:

      1. In-memory job: assemble tails from ``job.events`` (capped at
         ``MAX_PERSISTED_OUTPUT`` chars per stream so the payload stays
         predictable for chatty runs).
      2. Persisted sqlite row: return ``stdout_tail`` and ``stderr_tail``
         as already capped by ``set_output_tails``.
      3. Both miss -> 404.

    Useful when a client wants the final output without setting up a
    WebSocket - e.g. an agent fetching the result of a tool call, a
    deep link to "show this completed run's logs", or a CLI tool
    scripting around the dashboard. Live-tail consumers should still
    use the ``/ws/jobs/{id}`` stream.

    Response shape:
      {
        "id": "<job_id>",
        "state": "<state>",
        "stdout_tail": str,
        "stderr_tail": str,
        "stderr_count": int,
        "total_chars": int,  # len(stdout_tail) + len(stderr_tail)
      }
    """
    from ..jobs_persistence import MAX_PERSISTED_OUTPUT

    jm = request.app.state.job_manager
    job = jm.get(job_id)
    if job is not None:
        # Build tails from the in-memory event log. Each line is
        # joined by '\n' to match the persisted shape, then capped.
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        for event in job.events:
            t = event.get("type")
            if t == "stdout":
                stdout_buf.append(event.get("line", ""))
            elif t == "stderr":
                stderr_buf.append(event.get("line", ""))
        stdout_tail = "\n".join(stdout_buf)
        stderr_tail = "\n".join(stderr_buf)
        if len(stdout_tail) > MAX_PERSISTED_OUTPUT:
            stdout_tail = stdout_tail[-MAX_PERSISTED_OUTPUT:]
        if len(stderr_tail) > MAX_PERSISTED_OUTPUT:
            stderr_tail = stderr_tail[-MAX_PERSISTED_OUTPUT:]
        return JSONResponse(
            {
                "id": job.id,
                "state": job.state,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stderr_count": job.stderr_count,
                "total_chars": len(stdout_tail) + len(stderr_tail),
            }
        )

    persistence = _persistence_for(jm)
    if persistence is not None:
        row = persistence.get(job_id)
        if row is not None:
            stdout_tail = row.get("stdout_tail") or ""
            stderr_tail = row.get("stderr_tail") or ""
            return JSONResponse(
                {
                    "id": row.get("job_id"),
                    "state": row.get("status") or "unknown",
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                    "stderr_count": row.get("stderr_count") or 0,
                    "total_chars": len(stdout_tail) + len(stderr_tail),
                }
            )
    raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")


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
