"""Subprocess job manager for the evalyn dashboard backend.

Spawns CLI subprocesses, captures stdout/stderr line-by-line into a per-job
append-only event log, and fans the events out to multiple WebSocket
subscribers. Cancellation is SIGTERM with a configurable grace period,
escalating to SIGKILL.

No persistence in v1: jobs and event logs live entirely in memory and are
lost when the dashboard process restarts.

Event shapes (each event has ``event_id`` and ``ts``):
    {"type": "stdout",    "line": str, "ts": float, "event_id": int}
    {"type": "stderr",    "line": str, "ts": float, "event_id": int}
    {"type": "exit",      "code": int, "duration": float, "ts": float,
                          "event_id": int}
    {"type": "truncated", "count": int, "ts": float, "event_id": int}

Job states:
    pending   - spawn() returned but transport tasks not yet running
    running   - subprocess alive, capture tasks active
    complete  - subprocess exited with code 0
    failed    - subprocess exited with non-zero code
    cancelled - cancel() was called
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Literal

from .jobs_persistence import MAX_PERSISTED_OUTPUT, JobPersistence

logger = logging.getLogger(__name__)

JobState = Literal["pending", "running", "complete", "failed", "cancelled"]
EventType = Literal["stdout", "stderr", "exit", "truncated"]

# Defaults (per spec section 7 jobs.py + plan A3).
DEFAULT_MAX_LOG = 10_000
DEFAULT_GRACE_SECONDS = 3.0
DEFAULT_HISTORY_SIZE = 100
# Persistence GC: keep at most this many rows in the sqlite mirror,
# pruned every DEFAULT_PERSISTENCE_GC_INTERVAL terminal events. Without
# the GC call the .evalyn/data/jobs.sqlite table grew unboundedly.
DEFAULT_PERSISTENCE_KEEP = 200
DEFAULT_PERSISTENCE_GC_INTERVAL = 50
# Concurrency cap. Without one, an accidental rapid-fire spawn (a script
# that loops POST /api/cli/run, an over-eager UI test, a stuck retry
# loop) can spawn dozens of subprocesses and OOM the host. The default
# is intentionally generous - a normal interactive session never hits
# it - but the cap exists so pathological cases get a clean 503 from
# the API layer rather than a wedged box.
DEFAULT_MAX_CONCURRENT = 16


class ConcurrencyLimitExceeded(RuntimeError):
    """Raised by :meth:`JobManager.spawn` when ``running_count`` is at
    ``max_concurrent``. The API layer translates this to HTTP 503 with
    a Retry-After hint; downstream code should NOT catch and silently
    retry (that's how thundering herds form). Callers that legitimately
    want to wait can poll ``running_count`` until a slot frees."""


def _iso_utc(ts: float) -> str:
    """Format a unix timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class _EventStream:
    """Async-iterable wrapper around a per-subscriber ``asyncio.Queue``.

    Iteration yields events as they arrive. The stream auto-terminates
    after delivering an ``exit`` event so consumers can write the
    idiomatic ``async for evt in stream`` loop without bookkeeping. Use
    ``put_nowait`` for fanout from JobManager.

    Replay phase: ``JobManager.subscribe`` flips the stream into replay
    mode before registering it for live fanout. While replaying, live
    events from ``_emit`` divert into ``_live_buffer`` instead of the
    queue; ``_end_replay`` flushes the buffer in emit order, then
    switches the stream to direct fanout. This preserves correct
    ordering even if a future change makes queue puts yield mid-replay.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._replaying = False
        self._live_buffer: list[dict] = []

    def put_nowait(self, event: dict) -> None:
        if self._replaying:
            self._live_buffer.append(event)
        else:
            self._queue.put_nowait(event)

    async def put(self, event: dict) -> None:
        await self._queue.put(event)

    def _begin_replay(self) -> None:
        self._replaying = True

    def _end_replay(self) -> None:
        for event in self._live_buffer:
            self._queue.put_nowait(event)
        self._live_buffer.clear()
        self._replaying = False

    def __aiter__(self) -> _EventStream:
        return self

    async def __anext__(self) -> dict:
        if self._closed:
            raise StopAsyncIteration
        event = await self._queue.get()
        if event["type"] == "exit":
            self._closed = True
        return event


@dataclass
class Job:
    """In-memory job record. Mutated in place as the subprocess progresses."""

    id: str
    cmd: list[str]
    state: JobState = "pending"
    started_at: float = 0.0
    ended_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    # Optional metadata for persistence + UI reconstruction. cli_id matches
    # the catalog id passed by /api/cli/run; args is the original kwargs.
    cli_id: str = ""
    args: dict = field(default_factory=dict)
    # Append-only ring buffer of events. Each entry is a full event dict.
    # Capped at max_log via _enforce_log_cap() which prepends a truncated
    # marker when oldest entries are dropped.
    events: list[dict] = field(default_factory=list)
    # Cumulative count of stderr lines observed for this job. Incremented
    # by _emit() on every type=='stderr' event. Survives the events ring
    # trim (the events list is bounded by max_log; the count is monotonic
    # and unbounded - useful for "did this run produce errors?" questions
    # without scanning the full event log). Exposed via /api/jobs/{id}
    # so the Recent Jobs drawer can surface "5 stderr" inline even after
    # a server restart.
    stderr_count: int = 0
    # Set of _EventStream instances, one per active subscriber. Each stream
    # receives every new event after subscribe time. Replay of older events
    # is performed inside subscribe() before the stream joins fanout.
    _subscribers: set[_EventStream] = field(default_factory=set)
    # Internal coordination.
    _process: asyncio.subprocess.Process | None = None
    _capture_tasks: list[asyncio.Task] = field(default_factory=list)
    _exit_event: asyncio.Event = field(default_factory=asyncio.Event)
    _next_event_id: int = 1
    # Lowest event_id still present in self.events. Anything below this id
    # has been dropped due to backpressure. A subscriber asking for events
    # since N where N+1 < truncated_horizon needs a truncated marker.
    _truncated_horizon: int = 1
    _truncated_dropped: int = 0  # cumulative count of dropped events
    duration: float | None = None

    @property
    def is_done(self) -> bool:
        return self.state in ("complete", "failed", "cancelled")


class JobManager:
    """Spawn, observe, and cancel subprocess jobs.

    Args:
        grace_seconds: Time to wait between SIGTERM and SIGKILL on cancel.
        max_log: Maximum number of events kept per job's event log. When
            exceeded, oldest events are dropped and a truncated marker
            replaces them at the front of the log.
        history_size: Maximum number of jobs kept in ``recent()``. Older
            completed jobs are pruned. Running jobs are never pruned.
        persistence: Optional sqlite mirror for jobs.
        persistence_keep: When persistence is set, retain at most this
            many rows in the sqlite mirror; older rows are pruned by the
            periodic GC fired from ``_persist_job_terminal``.
        persistence_gc_interval: Run the persistence GC every Nth
            terminal event. Tuning lever: lower = tighter cap but more
            DELETE traffic; higher = laxer cap with cheaper terminals.
    """

    def __init__(
        self,
        *,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
        max_log: int = DEFAULT_MAX_LOG,
        history_size: int = DEFAULT_HISTORY_SIZE,
        persistence: JobPersistence | None = None,
        persistence_keep: int = DEFAULT_PERSISTENCE_KEEP,
        persistence_gc_interval: int = DEFAULT_PERSISTENCE_GC_INTERVAL,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        self.grace_seconds = grace_seconds
        self.max_log = max_log
        self.history_size = history_size
        # 0 or negative disables the cap (used in tests that intentionally
        # spawn many cheap jobs). Positive values are the hard limit on
        # concurrent ``running`` subprocesses.
        self.max_concurrent = max(0, int(max_concurrent))
        self._jobs: dict[str, Job] = {}
        # Insertion order tracked by dict; Python 3.7+ preserves it.
        # Optional sqlite mirror. None disables persistence (test default).
        self._persistence = persistence
        self._persistence_keep = max(1, persistence_keep)
        self._persistence_gc_interval = max(1, persistence_gc_interval)
        # Counts terminal events seen so far; used to trigger periodic GC.
        # Resets implicitly via modulo so we never run GC twice for the
        # same terminal.
        self._persistence_terminal_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def persistence(self) -> JobPersistence | None:
        """Public accessor for the optional sqlite mirror.

        External callers (api/health, api/jobs, api/cli) used to reach
        into ``_persistence`` via ``getattr(jm, "_persistence", None)``,
        which silently returned None if the attribute were ever
        renamed - a refactor footgun. Going through this property
        makes the contract explicit AND keeps the underscore field
        as a hint that the storage SHAPE is private (callers
        shouldn't construct a JobPersistence themselves; they read
        what the JM was wired with).

        Returns ``None`` when persistence is disabled (test default
        or unwired test rigs).
        """
        return self._persistence

    async def spawn(
        self,
        cmd: list[str],
        *,
        cli_id: str = "",
        args: dict | None = None,
    ) -> str:
        """Launch a subprocess and return a fresh job id.

        The subprocess is started immediately. stdout/stderr capture tasks
        are scheduled and will append events to the job's event log as
        lines arrive.

        ``cli_id`` and ``args`` are optional metadata used by the sqlite
        mirror (when configured) so the dashboard's Recent Jobs drawer can
        reconstruct state after a restart. They have no effect on subprocess
        execution.

        Raises :class:`ConcurrencyLimitExceeded` if ``max_concurrent`` is
        positive and ``running_count`` is already at the cap. The check
        happens BEFORE the subprocess is created so a rejected spawn
        never starts a process. The API layer maps this to HTTP 503.
        """
        _validate_cmd(cmd)
        if self.max_concurrent and self.running_count() >= self.max_concurrent:
            raise ConcurrencyLimitExceeded(
                f"job manager at capacity: {self.running_count()} running "
                f"(max_concurrent={self.max_concurrent})"
            )

        job_id = uuid.uuid4().hex
        job = Job(
            id=job_id,
            cmd=list(cmd),
            started_at=time.time(),
            cli_id=cli_id,
            args=dict(args) if args else {},
        )
        self._jobs[job_id] = job
        self._prune_history()
        self._persist_job_create(job)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        job._process = process
        job.pid = process.pid
        job.state = "running"

        stdout_task = asyncio.create_task(
            self._pump_stream(job, process.stdout, "stdout"),
            name=f"job-{job_id}-stdout",
        )
        stderr_task = asyncio.create_task(
            self._pump_stream(job, process.stderr, "stderr"),
            name=f"job-{job_id}-stderr",
        )
        job._capture_tasks = [stdout_task, stderr_task]

        # Reaper: wait for both capture tasks AND process exit, then emit
        # exit event. Spawned as a fire-and-forget task so spawn() returns
        # immediately.
        asyncio.create_task(
            self._reap(job),
            name=f"job-{job_id}-reap",
        )

        return job_id

    async def cancel(self, job_id: str) -> None:
        """Cancel a running job: SIGTERM, await grace, then SIGKILL.

        No-op if the job is unknown or already finished.
        """
        job = self._jobs.get(job_id)
        if job is None or job.is_done:
            return
        process = job._process
        if process is None or process.returncode is not None:
            return

        # Mark state early so the reaper sees this as a cancel.
        job.state = "cancelled"

        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGTERM)

        try:
            await asyncio.wait_for(process.wait(), timeout=self.grace_seconds)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                # Bound the SIGKILL wait so a buggy kernel can't hang us.
                await asyncio.wait_for(process.wait(), timeout=self.grace_seconds)

    async def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Block until the job finishes (or timeout). Returns the Job."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.is_done:
            return job
        await asyncio.wait_for(job._exit_event.wait(), timeout=timeout)
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the Job for ``job_id`` or None if unknown."""
        return self._jobs.get(job_id)

    def running_count(self) -> int:
        """Number of jobs currently in the ``running`` state.

        Used by ``spawn``'s capacity check and by ``/api/jobs/stats`` to
        surface saturation in the dashboard. Does NOT include ``pending``
        (no such intermediate state in the current pipeline) or terminal
        states.
        """
        return sum(1 for j in self._jobs.values() if j.state == "running")

    def purge(self, job_id: str) -> bool:
        """Drop a finished job from in-memory + persistence by id.

        Returns True if anything was removed (matched in either store),
        False on miss in both. Raises ``ValueError`` if the job is
        still queued/running - the caller must cancel first so we
        never purge a job whose subprocess is alive (would orphan
        the reaper task and the captured streams).

        Idempotent: a second purge for the same id is a no-op + False.
        """
        job = self._jobs.get(job_id)
        if job is not None and not job.is_done:
            raise ValueError(
                f"job {job_id} is {job.state}; cancel before purge"
            )
        removed = False
        if job is not None:
            del self._jobs[job_id]
            removed = True
        if self._persistence is not None:
            try:
                if self._persistence.delete(job_id):
                    removed = True
            except Exception as exc:  # noqa: BLE001 - never crash the caller
                logger.warning(
                    "JobPersistence delete(%s) raised: %s", job_id, exc
                )
        return removed

    def recent(self, n: int = 100) -> list[Job]:
        """Return the most recent jobs in reverse chronological order."""
        jobs = list(self._jobs.values())
        # Sort by started_at descending. Stable sort preserves insertion
        # order for ties.
        jobs.sort(key=lambda j: j.started_at, reverse=True)
        return jobs[:n]

    def history(self, job_id: str) -> list[tuple[str, str, float]]:
        """Return ``(kind, line, ts)`` triples for stdout/stderr only.

        Convenience accessor used by tests and the agent runtime when it
        wants the raw text of a finished job. Excludes exit and truncated
        markers.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return []
        return [
            (e["type"], e["line"], e["ts"])
            for e in job.events
            if e["type"] in ("stdout", "stderr")
        ]

    @asynccontextmanager
    async def subscribe(
        self, job_id: str, since: int | None = None
    ) -> AsyncIterator[_EventStream]:
        """Subscribe to a job's event stream.

        Yields an async-iterable stream that receives every event with
        ``event_id`` > ``since``. If ``since`` is ``None`` (default),
        replays the entire event log first.

        If ``since`` is below the truncation horizon (data has been
        dropped), the first event delivered is a ``truncated`` marker so
        the consumer knows the stream was lossy.

        On context exit, the stream is removed from the fanout set.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)

        stream = _EventStream()
        cursor = since if since is not None else 0

        # Atomic registration: flip into replay mode, register for live
        # fanout, snapshot the replay set. asyncio is cooperative so the
        # following sync block is not interleaved with _emit. After this,
        # any live event arriving via _emit goes to stream._live_buffer
        # (replaying=True) and will be flushed in order after the replay
        # loop, preventing both event loss and out-of-order delivery.
        stream._begin_replay()
        job._subscribers.add(stream)
        snapshot_id = job._next_event_id - 1
        needs_truncated_marker = (
            job._truncated_dropped > 0 and cursor < job._truncated_horizon - 1
        )
        replay_events = [
            evt for evt in job.events
            if cursor < evt["event_id"] <= snapshot_id
        ]

        try:
            if needs_truncated_marker:
                # Marker uses event_id 0 so it sorts before any real event.
                await stream.put(
                    {
                        "type": "truncated",
                        "count": job._truncated_dropped,
                        "ts": time.time(),
                        "event_id": 0,
                    }
                )
            for event in replay_events:
                await stream.put(event)
            stream._end_replay()
            yield stream
        finally:
            job._subscribers.discard(stream)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _pump_stream(
        self,
        job: Job,
        stream: asyncio.StreamReader | None,
        kind: str,
    ) -> None:
        if stream is None:
            return
        while True:
            try:
                raw = await stream.readline()
            except (asyncio.CancelledError, ValueError):
                # ValueError can be raised when a single line exceeds the
                # default StreamReader buffer; treat as end-of-stream
                # rather than crashing the capture task.
                break
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            self._emit(job, {"type": kind, "line": line, "ts": time.time()})

    async def _reap(self, job: Job) -> None:
        """Wait for capture tasks + process exit, then emit exit event."""
        process = job._process
        assert process is not None
        # Wait for both stdout and stderr capture tasks to finish so all
        # output is in the log before the exit marker.
        if job._capture_tasks:
            await asyncio.gather(*job._capture_tasks, return_exceptions=True)
        return_code = await process.wait()

        # Update terminal state. cancel() may have already set state to
        # "cancelled"; preserve that.
        job.exit_code = return_code
        job.ended_at = time.time()
        job.duration = job.ended_at - job.started_at
        if job.state != "cancelled":
            job.state = "complete" if return_code == 0 else "failed"

        self._emit(
            job,
            {
                "type": "exit",
                "code": return_code,
                "duration": job.duration,
                "ts": job.ended_at,
            },
        )
        # Persist final status + last-N output (terminal-only strategy:
        # one batch write at the end avoids per-line write storms).
        self._persist_job_terminal(job)
        job._exit_event.set()

    def _emit(self, job: Job, event: dict) -> None:
        """Stamp an event with an id, append to the log, fan out to subs."""
        event_id = job._next_event_id
        job._next_event_id += 1
        event["event_id"] = event_id
        job.events.append(event)
        # Maintain a monotonic stderr counter independent of the bounded
        # events ring. A subscriber asking "did this run produce errors?"
        # via /api/jobs/{id} should get a true total even if the early
        # stderr lines have already been evicted past max_log.
        if event.get("type") == "stderr":
            job.stderr_count += 1
        self._enforce_log_cap(job)
        # Fanout to all live subscribers. put_nowait is safe because the
        # underlying queues are unbounded.
        for stream in job._subscribers:
            stream.put_nowait(event)

    def _enforce_log_cap(self, job: Job) -> None:
        """Drop oldest events when the log exceeds ``max_log``.

        A single ``truncated`` marker is kept at index 0 (event_id 0) so
        any subscriber whose cursor falls below the truncation horizon
        learns that data was lost. The marker's ``count`` field is updated
        in place across successive drops.
        """
        if len(job.events) <= self.max_log:
            return

        has_marker = job.events[0]["type"] == "truncated"
        # Index of the first real event (after any existing marker).
        real_start = 1 if has_marker else 0
        # Number of real events to drop. When inserting a fresh marker we
        # also reserve one slot so the post-trim length equals max_log.
        overflow = len(job.events) - self.max_log
        if not has_marker:
            overflow += 1

        dropped = job.events[real_start : real_start + overflow]
        del job.events[real_start : real_start + overflow]
        job._truncated_dropped += len(dropped)

        if not has_marker:
            job.events.insert(
                0,
                {
                    "type": "truncated",
                    "count": job._truncated_dropped,
                    "ts": time.time(),
                    "event_id": 0,
                },
            )
        else:
            job.events[0]["count"] = job._truncated_dropped

        # Update horizon to the lowest surviving real event id.
        if len(job.events) > 1:
            job._truncated_horizon = job.events[1]["event_id"]
        else:
            job._truncated_horizon = job._next_event_id

    # ------------------------------------------------------------------
    # Persistence hooks (best-effort; never raise out of the hot path)
    # ------------------------------------------------------------------

    def _persist_job_create(self, job: Job) -> None:
        if self._persistence is None:
            return
        try:
            self._persistence.upsert_job(
                job_id=job.id,
                cli_id=job.cli_id,
                args=job.args,
                cmd=" ".join(job.cmd),
                status="running",
                started_at_iso=_iso_utc(job.started_at),
            )
        except Exception as exc:  # noqa: BLE001 - never crash the spawn
            logger.warning("persist_job_create(%s) failed: %s", job.id, exc)

    def _persist_job_terminal(self, job: Job) -> None:
        if self._persistence is None:
            return
        try:
            stdout_buf: list[str] = []
            stderr_buf: list[str] = []
            for event in job.events:
                if event["type"] == "stdout":
                    stdout_buf.append(event["line"])
                elif event["type"] == "stderr":
                    stderr_buf.append(event["line"])
            stdout_tail = "\n".join(stdout_buf)
            stderr_tail = "\n".join(stderr_buf)
            if len(stdout_tail) > MAX_PERSISTED_OUTPUT:
                stdout_tail = stdout_tail[-MAX_PERSISTED_OUTPUT:]
            if len(stderr_tail) > MAX_PERSISTED_OUTPUT:
                stderr_tail = stderr_tail[-MAX_PERSISTED_OUTPUT:]
            self._persistence.set_output_tails(job.id, stdout_tail, stderr_tail)
            self._persistence.patch_status(
                job_id=job.id,
                status=job.state,
                exit_code=job.exit_code,
                ended_at_iso=_iso_utc(job.ended_at) if job.ended_at else None,
                duration_s=job.duration,
                stderr_count=job.stderr_count,
            )
        except Exception as exc:  # noqa: BLE001 - never crash the reaper
            logger.warning("persist_job_terminal(%s) failed: %s", job.id, exc)
        # GC runs in its own try/except so a prune failure cannot poison
        # the per-job persist path. Without this hook the sqlite mirror
        # grew unboundedly.
        self._persist_gc_maybe()

    def _persist_gc_maybe(self) -> None:
        """Periodically prune the sqlite mirror to bound DB growth.

        Fires every ``persistence_gc_interval`` terminal events and asks
        the mirror to keep at most ``persistence_keep`` rows. Cheap when
        no pruning is needed (a single COUNT-style query). Never raises.
        """
        if self._persistence is None:
            return
        self._persistence_terminal_count += 1
        if self._persistence_terminal_count % self._persistence_gc_interval != 0:
            return
        try:
            deleted = self._persistence.delete_old(keep=self._persistence_keep)
            if deleted > 0:
                logger.info(
                    "JobPersistence GC pruned %d old rows (keeping %d)",
                    deleted,
                    self._persistence_keep,
                )
        except Exception as exc:  # noqa: BLE001 - never crash the reaper
            logger.warning("JobPersistence GC failed: %s", exc)

    def _prune_history(self) -> None:
        """Trim completed jobs beyond ``history_size``. Running jobs stay."""
        if len(self._jobs) <= self.history_size:
            return
        # Walk from oldest insertion forward; drop completed jobs first.
        for jid in list(self._jobs):
            if len(self._jobs) <= self.history_size:
                break
            if self._jobs[jid].is_done:
                del self._jobs[jid]


def _validate_cmd(cmd) -> None:
    if not isinstance(cmd, list):
        raise TypeError(f"cmd must be a list, got {type(cmd).__name__}")
    if not cmd:
        raise ValueError("cmd must be a non-empty list")
    for i, part in enumerate(cmd):
        if not isinstance(part, str):
            raise TypeError(
                f"cmd[{i}] must be str, got {type(part).__name__}"
            )


__all__ = ["Job", "JobManager", "JobState", "EventType"]
