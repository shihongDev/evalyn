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
from fastapi.responses import JSONResponse, PlainTextResponse

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


def _validate_recent_args(
    limit: int, since: float | None, before: float | None = None
) -> None:
    """Shared validation for /recent, /recent.csv, /recent.ndjson."""
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be <= 1000")
    if since is not None and since < 0:
        raise HTTPException(status_code=400, detail="since must be >= 0")
    if before is not None and before < 0:
        raise HTTPException(status_code=400, detail="before must be >= 0")
    if since is not None and before is not None and before <= since:
        raise HTTPException(
            status_code=400,
            detail="before must be > since (windowed queries are open intervals)",
        )


def _parse_cli_ids(cli_ids_param: str | None) -> list[str] | None:
    """Parse a comma-separated cli_ids query param into a list of
    non-empty trimmed strings. Returns None when not provided so the
    SQL builder skips the filter entirely. Empty/whitespace-only
    inputs and trailing commas are tolerated."""
    if cli_ids_param is None:
        return None
    parts = [p.strip() for p in cli_ids_param.split(",")]
    parts = [p for p in parts if p]
    return parts if parts else None


def _collect_recent_rows(
    jm,
    limit: int,
    cli_id: str | None,
    status: str | None,
    since: float | None,
    before: float | None = None,
    cli_ids: list[str] | None = None,
) -> list[dict]:
    """Filter + merge in-memory and persisted jobs into a single list,
    sorted started_at-descending and capped at ``limit``. Shared by
    the JSON, CSV, and NDJSON variants of /api/jobs/recent.
    """
    in_memory_jobs = jm.recent(n=limit)
    # cli_ids takes precedence over cli_id when both are set so a
    # caller migrating between the two variants does not get an empty
    # AND of "id == X AND id IN [Y, Z]".
    if cli_ids is not None and len(cli_ids) > 0:
        cli_ids_set = set(cli_ids)
        in_memory_jobs = [j for j in in_memory_jobs if j.cli_id in cli_ids_set]
    elif cli_id is not None:
        in_memory_jobs = [j for j in in_memory_jobs if j.cli_id == cli_id]
    if status is not None:
        in_memory_jobs = [j for j in in_memory_jobs if j.state == status]
    if since is not None:
        in_memory_jobs = [j for j in in_memory_jobs if j.started_at > since]
    if before is not None:
        in_memory_jobs = [j for j in in_memory_jobs if j.started_at < before]
    in_memory = [_job_to_dict(j) for j in in_memory_jobs]
    seen = {entry["id"] for entry in in_memory}
    merged = list(in_memory)
    persistence = _persistence_for(jm)
    if persistence is not None:
        # ISO-8601 with +00:00 has the property that lexicographic
        # ordering matches chronological ordering, so the SQL >? / <?
        # comparisons on the started_at_iso column are correct.
        since_iso: str | None = None
        before_iso: str | None = None
        if since is not None or before is not None:
            from datetime import datetime, timezone

            if since is not None:
                since_iso = datetime.fromtimestamp(
                    since, tz=timezone.utc
                ).isoformat()
            if before is not None:
                before_iso = datetime.fromtimestamp(
                    before, tz=timezone.utc
                ).isoformat()
        for row in persistence.list_recent(
            limit=limit,
            cli_id=cli_id,
            cli_ids=cli_ids,
            status=status,
            since_iso=since_iso,
            before_iso=before_iso,
        ):
            entry = _persisted_to_dict(row)
            if entry["id"] in seen:
                continue
            merged.append(entry)
            seen.add(entry["id"])
    merged.sort(key=lambda e: (e.get("started_at") or 0.0), reverse=True)
    return merged[:limit]


@router.get("/recent")
async def recent_jobs(
    request: Request,
    limit: int = 100,
    cli_id: str | None = None,
    cli_ids: str | None = None,
    status: str | None = None,
    since: float | None = None,
    before: float | None = None,
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
    - ``since`` is a unix epoch (float) keeping only jobs whose
      ``started_at > since``. Useful for polling - hand back the
      previous response's max ``started_at`` to fetch only the delta.

    All filters are pushed down to SQL on the persisted side and
    applied in-memory for the live set.
    """
    _validate_recent_args(limit, since, before)
    parsed_cli_ids = _parse_cli_ids(cli_ids)
    rows = _collect_recent_rows(
        request.app.state.job_manager,
        limit,
        cli_id,
        status,
        since,
        before,
        cli_ids=parsed_cli_ids,
    )
    return JSONResponse(rows)


@router.get("/recent.csv")
async def recent_jobs_csv(
    request: Request,
    limit: int = 100,
    cli_id: str | None = None,
    cli_ids: str | None = None,
    status: str | None = None,
    since: float | None = None,
    before: float | None = None,
) -> PlainTextResponse:
    """CSV variant of /api/jobs/recent with the same query semantics.

    Useful for spreadsheet workflows ("download my last 30 failed
    runs as a CSV"), archive tooling, and quick pipes to ``column -ts,``
    on the command line. Columns: id, cli_id, state, started_at,
    ended_at, exit_code, duration, stderr_count.

    Body bypasses _job_to_dict to keep the row layout small + flat:
    no nested cmd array, no pid (always None for persisted rows
    anyway). Caller wanting the full shape should use the JSON
    endpoint and convert client-side.
    """
    _validate_recent_args(limit, since, before)
    parsed_cli_ids = _parse_cli_ids(cli_ids)
    rows = _collect_recent_rows(
        request.app.state.job_manager,
        limit,
        cli_id,
        status,
        since,
        before,
        cli_ids=parsed_cli_ids,
    )

    import csv
    import io

    columns = [
        "id",
        "cli_id",
        "state",
        "started_at",
        "ended_at",
        "exit_code",
        "duration",
        "stderr_count",
    ]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for r in rows:
        writer.writerow(
            [
                r.get(c) if r.get(c) is not None else ""
                for c in columns
            ]
        )
    return PlainTextResponse(
        buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="evalyn-jobs-recent.csv"',
        },
    )


@router.get("/recent.ndjson")
async def recent_jobs_ndjson(
    request: Request,
    limit: int = 100,
    cli_id: str | None = None,
    cli_ids: str | None = None,
    status: str | None = None,
    since: float | None = None,
    before: float | None = None,
) -> PlainTextResponse:
    """NDJSON variant of /api/jobs/recent: one full job dict per line.

    Same query semantics + filters as the JSON endpoint. Useful for
    streaming consumers that read one record at a time without
    parsing a top-level array - jq pipes (``curl ... | jq -c .``),
    log aggregators (Vector / Filebeat input plugins), or any tool
    that ingests JSON Lines.

    Returns ``application/x-ndjson`` (the de-facto media type) with
    Unix-style ``\\n`` line endings. No trailing newline so concatenating
    multiple responses doesn't produce blank lines.
    """
    import json

    _validate_recent_args(limit, since, before)
    parsed_cli_ids = _parse_cli_ids(cli_ids)
    rows = _collect_recent_rows(
        request.app.state.job_manager,
        limit,
        cli_id,
        status,
        since,
        before,
        cli_ids=parsed_cli_ids,
    )
    body = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    return PlainTextResponse(
        body,
        media_type="application/x-ndjson",
    )


@router.get("/stats")
async def jobs_stats(
    request: Request,
    recent_window_s: int = 86400,
) -> JSONResponse:
    """Aggregate counts over the job history.

    Returns ``{total, by_status, total_stderr, recent_failures}``.
    ``recent_window_s`` controls the age threshold for the
    ``recent_failures`` count (default 24h). The persisted side does
    the heavy lifting via ``JobPersistence.stats``; in-memory queued
    or running jobs that haven't yet hit ``_persist_job_terminal``
    are merged in by adjusting the ``by_status`` counts so a fresh
    "running" job is not undercounted.

    Useful for an at-a-glance dashboard health badge or for an admin
    answering "how many failed jobs in the last day?" without paging.
    """
    if recent_window_s < 0:
        raise HTTPException(
            status_code=400, detail="recent_window_s must be >= 0"
        )
    jm = request.app.state.job_manager
    # Capacity surface is independent of persistence - always include
    # so a frontend can render a "X / Y running" chip without an extra
    # round-trip. ``max_concurrent=0`` means the cap is disabled.
    capacity = {
        "running": jm.running_count(),
        "max_concurrent": jm.max_concurrent,
    }
    persistence = _persistence_for(jm)
    if persistence is None:
        # Test setups without persistence: build the same shape from
        # in-memory only.
        in_mem = list(jm.recent(n=10_000))
        by_status: dict[str, int] = {}
        for j in in_mem:
            by_status[j.state] = by_status.get(j.state, 0) + 1
        return JSONResponse(
            {
                "total": len(in_mem),
                "by_status": by_status,
                "total_stderr": sum(j.stderr_count for j in in_mem),
                "recent_failures": sum(
                    1 for j in in_mem if j.state == "failed"
                ),
                **capacity,
            }
        )
    body = persistence.stats(recent_window_s)
    body.update(capacity)
    return JSONResponse(body)


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


@router.get("/{job_id}/output.txt")
async def get_job_output_txt(
    request: Request,
    job_id: str,
    stream: str | None = None,
    tail: int | None = None,
    download: bool = False,
) -> PlainTextResponse:
    """Plain-text variant of ``/output`` with stdout and stderr
    interleaved in chronological emit order (the order they hit the
    subprocess pipes). Suitable for ``curl ... | grep error`` or
    redirecting straight into a log file.

    Optional ``?stream=stdout`` or ``?stream=stderr`` filters to one
    stream. Default (omitted) keeps the existing interleaved behavior.
    Useful for ``curl ... | grep`` only on stdout, or pulling the
    error log alone for further analysis.

    Optional ``?download=1`` adds a ``Content-Disposition: attachment``
    header so a browser visit downloads the file rather than
    rendering it inline. Filename derives from the job_id prefix and
    the stream filter (if any) so multiple downloads to the same
    Downloads folder don't clobber each other.

    Sources:
      1. In-memory: walks ``job.events`` in event_id order so stdout
         and stderr stay correctly interleaved (when no stream filter).
      2. Persisted only: returns ``stdout_tail`` and/or ``stderr_tail``
         depending on the filter; with no filter, concatenates both
         with a divider since the persisted tails lost their original
         chronology.
      3. Both miss -> 404.

    Caps: in-memory builds a string at most ``MAX_PERSISTED_OUTPUT * 2``
    chars long (one per stream). For very chatty jobs the live tail
    via ``/ws/jobs/{id}`` remains the right tool.
    """
    from ..jobs_persistence import MAX_PERSISTED_OUTPUT

    if stream is not None and stream not in ("stdout", "stderr"):
        raise HTTPException(
            status_code=400,
            detail="stream must be 'stdout' or 'stderr' if provided",
        )
    if tail is not None and tail < 1:
        raise HTTPException(
            status_code=400, detail="tail must be >= 1 if provided"
        )

    def _download_headers() -> dict[str, str]:
        """Header dict for ?download=1 responses, empty otherwise.
        Filename derives from the job_id prefix + stream filter so a
        user downloading both streams of the same job to the same
        folder doesn't clobber the prior file. Stream-suffix only when
        a filter is active. ``-tail`` segment when truncated. The
        prefix is the first 8 chars of the job_id - enough to
        disambiguate within a normal Downloads folder, short enough
        to be readable in the file save dialog.
        """
        if not download:
            return {}
        suffix_parts: list[str] = []
        if stream:
            suffix_parts.append(stream)
        if tail is not None:
            suffix_parts.append(f"tail{tail}")
        suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""
        filename = f"evalyn-job-{job_id[:8]}{suffix}.log"
        return {
            "Content-Disposition": f'attachment; filename="{filename}"',
        }

    def _maybe_tail(text: str) -> str:
        """Trim ``text`` to the last ``tail`` lines if ``tail`` is set.
        Stable on input that does/doesn't end with a newline. Strips
        trailing empty splits so we count actual lines, not separators.
        """
        if tail is None:
            return text
        if not text:
            return text
        # Split lines and re-join. splitlines() handles "\n" and "\r\n"
        # uniformly. We re-add the trailing newline so the response
        # remains a well-formed text file regardless of input.
        lines = text.splitlines()
        if len(lines) <= tail:
            return text
        return "\n".join(lines[-tail:]) + "\n"

    jm = request.app.state.job_manager
    job = jm.get(job_id)
    if job is not None:
        # Walk events in event_id order. When stream filter is active,
        # only emit lines of that kind; the chronological order within
        # the chosen stream is preserved.
        keep = ("stdout", "stderr") if stream is None else (stream,)
        ordered_lines: list[str] = []
        for event in sorted(job.events, key=lambda e: e.get("event_id", 0)):
            if event.get("type") in keep:
                ordered_lines.append(event.get("line", ""))
        text = "\n".join(ordered_lines)
        # Symmetric cap with the JSON endpoint - 2 * MAX_PERSISTED_OUTPUT
        # so the total budget matches stdout + stderr separately.
        cap = MAX_PERSISTED_OUTPUT * 2
        if len(text) > cap:
            text = text[-cap:]
        if text and not text.endswith("\n"):
            text = text + "\n"
        return PlainTextResponse(
            _maybe_tail(text),
            media_type="text/plain; charset=utf-8",
            headers=_download_headers(),
        )

    persistence = _persistence_for(jm)
    if persistence is not None:
        row = persistence.get(job_id)
        if row is not None:
            stdout_tail = row.get("stdout_tail") or ""
            stderr_tail = row.get("stderr_tail") or ""
            if stream == "stdout":
                body = stdout_tail
                if body and not body.endswith("\n"):
                    body = body + "\n"
                return PlainTextResponse(
                    _maybe_tail(body),
                    media_type="text/plain; charset=utf-8",
                    headers=_download_headers(),
                )
            if stream == "stderr":
                body = stderr_tail
                if body and not body.endswith("\n"):
                    body = body + "\n"
                return PlainTextResponse(
                    _maybe_tail(body),
                    media_type="text/plain; charset=utf-8",
                    headers=_download_headers(),
                )
            parts: list[str] = []
            if stdout_tail:
                parts.append(stdout_tail)
                if not stdout_tail.endswith("\n"):
                    parts.append("\n")
            if stderr_tail:
                # Divider so tooling reading text/plain doesn't conflate
                # stdout and stderr. The persisted tails lost their
                # original interleaving; this is the best we can do.
                parts.append("--- stderr ---\n")
                parts.append(stderr_tail)
                if not stderr_tail.endswith("\n"):
                    parts.append("\n")
            return PlainTextResponse(
                _maybe_tail("".join(parts)),
                media_type="text/plain; charset=utf-8",
                headers=_download_headers(),
            )
    raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")


@router.delete("/{job_id}")
async def delete_job(request: Request, job_id: str) -> JSONResponse:
    """Remove a finished job from in-memory + persistence.

    Status codes:
      - 204: deleted (or was already absent and the call was idempotent)
      - 404: not found in either store
      - 409: still queued/running; client must POST /cancel first

    The 409 path is the load-bearing safety: we never purge a job whose
    subprocess is alive because that would orphan the reaper task and
    the captured streams. Cancel synchronously, then DELETE.
    """
    jm = request.app.state.job_manager
    try:
        removed = jm.purge(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if not removed:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return JSONResponse({"ok": True, "id": job_id}, status_code=200)


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


@router.post("/{job_id}/restart")
async def restart_job(request: Request, job_id: str) -> JSONResponse:
    """Re-spawn a finished job using the same ``cli_id`` and ``args``
    as the source. Returns ``{job_id: <new_id>}`` for the fresh run.

    Source lookup: in-memory first, then persistence. Both miss -> 404.

    The source must be terminal (complete / failed / cancelled). 409 if
    the source is still queued/running - the user should wait for it
    to finish or cancel first; restarting in parallel would land two
    concurrent runs of the same command, which is rarely the intent.

    The argv is rebuilt from the stored ``args`` dict via
    ``args_to_argv`` rather than reusing the source's persisted ``cmd``
    string (which is space-joined and lossy for args with whitespace).
    The source's ``cli_id`` must still be in the catalog for the
    rebuild; 404 with a clear message otherwise.
    """
    from .cli import args_to_argv

    jm = request.app.state.job_manager

    cli_id: str | None = None
    args: dict | None = None
    state: str | None = None

    job = jm.get(job_id)
    if job is not None:
        cli_id = job.cli_id
        args = dict(job.args)
        state = job.state
    else:
        persistence = _persistence_for(jm)
        if persistence is not None:
            row = persistence.get(job_id)
            if row is not None:
                cli_id = row.get("cli_id") or ""
                args = row.get("args") or {}
                state = row.get("status")

    if cli_id is None or args is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    if not cli_id:
        raise HTTPException(
            status_code=409,
            detail=f"source job {job_id} has no cli_id; cannot restart",
        )
    if state in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"source job {job_id} is {state}; wait for it to finish "
                f"or cancel first"
            ),
        )

    catalog = request.app.state.cli_catalog
    schema = next((s for s in catalog if s.id == cli_id), None)
    if schema is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"source job's cli_id {cli_id!r} is no longer in the catalog; "
                f"restart cannot rebuild argv"
            ),
        )

    argv_tail = args_to_argv(schema, args)
    cmd = ["evalyn", cli_id, *argv_tail]
    from ..jobs import ConcurrencyLimitExceeded

    try:
        new_job_id = await jm.spawn(cmd, cli_id=cli_id, args=args)
    except ConcurrencyLimitExceeded as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "running": jm.running_count(),
                "max_concurrent": jm.max_concurrent,
            },
            status_code=503,
            headers={"Retry-After": "5"},
        )
    return JSONResponse({"job_id": new_job_id})


__all__ = ["router"]
