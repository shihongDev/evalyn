"""Tests for ``evalyn_dashboard.jobs.JobManager`` (Lane A3).

Covers tasks A3.1 - A3.5:
- A3.1: spawn() with stdout/stderr capture, history, exit codes.
- A3.2: cancel() via SIGTERM + grace + SIGKILL.
- A3.3: subscribe() WS-fanout to multiple subscribers.
- A3.4: backpressure with truncation marker.
- A3.5: recent() history retention.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from evalyn_dashboard.jobs import JobManager


# ---------------------------------------------------------------------------
# A3.1: spawn() and stdout/stderr capture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_echo_captures_output():
    jm = JobManager()
    job_id = await jm.spawn([sys.executable, "-c", "print('hello')"])
    job = await jm.wait(job_id, timeout=5)
    assert job.exit_code == 0
    assert job.state == "complete"
    output = "".join(line for kind, line, _ts in jm.history(job_id) if kind == "stdout")
    assert "hello" in output


@pytest.mark.asyncio
async def test_spawn_returns_unique_ids():
    jm = JobManager()
    a = await jm.spawn([sys.executable, "-c", "pass"])
    b = await jm.spawn([sys.executable, "-c", "pass"])
    assert a != b
    await jm.wait(a, timeout=5)
    await jm.wait(b, timeout=5)


@pytest.mark.asyncio
async def test_spawn_captures_stderr():
    jm = JobManager()
    job_id = await jm.spawn(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n')"]
    )
    await jm.wait(job_id, timeout=5)
    err = "".join(line for kind, line, _ts in jm.history(job_id) if kind == "stderr")
    assert "boom" in err


@pytest.mark.asyncio
async def test_spawn_failed_exit_marks_failed_state():
    jm = JobManager()
    job_id = await jm.spawn([sys.executable, "-c", "import sys; sys.exit(2)"])
    job = await jm.wait(job_id, timeout=5)
    assert job.exit_code == 2
    assert job.state == "failed"


@pytest.mark.asyncio
async def test_spawn_validates_cmd_is_list():
    jm = JobManager()
    with pytest.raises(TypeError):
        await jm.spawn("echo hi")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_spawn_validates_cmd_elements_are_str():
    jm = JobManager()
    with pytest.raises(TypeError):
        await jm.spawn([sys.executable, 123])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_spawn_validates_cmd_not_empty():
    jm = JobManager()
    with pytest.raises(ValueError):
        await jm.spawn([])


@pytest.mark.asyncio
async def test_get_returns_job():
    jm = JobManager()
    job_id = await jm.spawn([sys.executable, "-c", "pass"])
    job = jm.get(job_id)
    assert job is not None
    assert job.id == job_id
    assert job.cmd == [sys.executable, "-c", "pass"]
    await jm.wait(job_id, timeout=5)


@pytest.mark.asyncio
async def test_get_unknown_returns_none():
    jm = JobManager()
    assert jm.get("nope") is None


# ---------------------------------------------------------------------------
# A3.2: cancel() with SIGTERM + grace + SIGKILL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_sends_sigterm():
    jm = JobManager(grace_seconds=0.5)
    job_id = await jm.spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    await asyncio.sleep(0.2)
    await jm.cancel(job_id)
    job = await jm.wait(job_id, timeout=5)
    assert job.state == "cancelled"
    assert job.exit_code != 0


@pytest.mark.asyncio
async def test_cancel_sigkill_when_sigterm_ignored():
    """A subprocess that ignores SIGTERM should still be killed via SIGKILL after grace."""
    # Busy loop with SIGTERM ignored. signal.signal(SIGTERM, SIG_IGN) makes
    # SIGTERM do nothing on POSIX; only SIGKILL terminates the process.
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )
    jm = JobManager(grace_seconds=0.3)
    job_id = await jm.spawn([sys.executable, "-c", code])
    await asyncio.sleep(0.2)
    await jm.cancel(job_id)
    job = await jm.wait(job_id, timeout=5)
    assert job.state == "cancelled"
    # On POSIX the process exits via SIGKILL; exit code is negative signal
    # number for asyncio subprocesses (-9). Just assert non-zero.
    assert job.exit_code != 0


@pytest.mark.asyncio
async def test_cancel_unknown_is_noop():
    jm = JobManager()
    # Should not raise.
    await jm.cancel("does-not-exist")


@pytest.mark.asyncio
async def test_cancel_completed_is_noop():
    jm = JobManager()
    job_id = await jm.spawn([sys.executable, "-c", "pass"])
    await jm.wait(job_id, timeout=5)
    await jm.cancel(job_id)  # already done; should be a no-op
    job = jm.get(job_id)
    assert job.state == "complete"


# ---------------------------------------------------------------------------
# A3.3: subscribe() fanout + cursor replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_same_output():
    jm = JobManager()
    received1: list = []
    received2: list = []

    async def consume(q, sink):
        async for evt in q:
            sink.append(evt)
            if evt["type"] == "exit":
                break

    job_id = await jm.spawn([sys.executable, "-c", "print('a'); print('b')"])

    async with jm.subscribe(job_id) as q1, jm.subscribe(job_id) as q2:
        await asyncio.wait_for(
            asyncio.gather(consume(q1, received1), consume(q2, received2)),
            timeout=5,
        )

    assert any("a" in str(e.get("line", "")) for e in received1)
    assert any("a" in str(e.get("line", "")) for e in received2)
    assert any("b" in str(e.get("line", "")) for e in received1)
    assert any("b" in str(e.get("line", "")) for e in received2)
    assert received1[-1]["type"] == "exit"
    assert received2[-1]["type"] == "exit"


@pytest.mark.asyncio
async def test_subscribe_after_completion_replays_history():
    jm = JobManager()
    job_id = await jm.spawn([sys.executable, "-c", "print('replay')"])
    await jm.wait(job_id, timeout=5)

    received = []
    async with jm.subscribe(job_id) as q:
        async for evt in q:
            received.append(evt)
            if evt["type"] == "exit":
                break

    assert any("replay" in str(e.get("line", "")) for e in received)
    assert received[-1]["type"] == "exit"


@pytest.mark.asyncio
async def test_subscribe_since_replays_only_new_events():
    jm = JobManager()
    job_id = await jm.spawn([sys.executable, "-c", "print('one'); print('two')"])
    await jm.wait(job_id, timeout=5)

    # First subscription: collect everything, find event_id of 'one' line.
    full = []
    async with jm.subscribe(job_id) as q:
        async for evt in q:
            full.append(evt)
            if evt["type"] == "exit":
                break

    one_evt = next(
        e for e in full if e["type"] == "stdout" and "one" in e.get("line", "")
    )
    cursor = one_evt["event_id"]

    replay = []
    async with jm.subscribe(job_id, since=cursor) as q:
        async for evt in q:
            replay.append(evt)
            if evt["type"] == "exit":
                break

    assert all(e["event_id"] > cursor for e in replay)
    assert any("two" in str(e.get("line", "")) for e in replay)
    assert not any("one" in str(e.get("line", "")) for e in replay)


@pytest.mark.asyncio
async def test_subscribe_unknown_job_raises():
    jm = JobManager()
    with pytest.raises(KeyError):
        async with jm.subscribe("nope") as _q:
            pass


@pytest.mark.asyncio
async def test_event_ids_are_monotonic():
    jm = JobManager()
    job_id = await jm.spawn(
        [sys.executable, "-c", "for i in range(5): print(i)"]
    )
    await jm.wait(job_id, timeout=5)
    received = []
    async with jm.subscribe(job_id) as q:
        async for evt in q:
            received.append(evt)
            if evt["type"] == "exit":
                break
    ids = [e["event_id"] for e in received]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# A3.4: Backpressure with truncation marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_capped_with_truncation_marker():
    """When the per-job event log exceeds ``max_log``, oldest entries drop and
    a single ``truncated`` marker is inserted at the front of the log."""
    jm = JobManager(max_log=50)
    # Print 200 lines; combined with the exit event, log will be truncated.
    job_id = await jm.spawn(
        [sys.executable, "-c", "import sys\nfor i in range(200):\n    print(i)\n"]
    )
    await jm.wait(job_id, timeout=10)

    history = jm.history(job_id)
    assert len(history) <= 50  # capped

    received = []
    async with jm.subscribe(job_id) as q:
        async for evt in q:
            received.append(evt)
            if evt["type"] == "exit":
                break

    # First event in replay must be a truncated marker so the subscriber
    # knows data was lost.
    assert received[0]["type"] == "truncated"
    assert received[0]["count"] > 0
    # Final event still the exit.
    assert received[-1]["type"] == "exit"


@pytest.mark.asyncio
async def test_subscribe_since_below_horizon_gets_truncation_marker():
    jm = JobManager(max_log=20)
    job_id = await jm.spawn(
        [sys.executable, "-c", "for i in range(100): print(i)"]
    )
    await jm.wait(job_id, timeout=10)

    # Cursor 0 is well below the truncation horizon; replay must start with
    # a truncated marker.
    received = []
    async with jm.subscribe(job_id, since=0) as q:
        async for evt in q:
            received.append(evt)
            if evt["type"] == "exit":
                break

    assert received[0]["type"] == "truncated"


# ---------------------------------------------------------------------------
# A3.5: recent() history retention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_returns_reverse_chronological():
    jm = JobManager()
    ids = []
    for _ in range(3):
        ids.append(await jm.spawn([sys.executable, "-c", "pass"]))
        # ensure distinct started_at timestamps
        await asyncio.sleep(0.01)
    for jid in ids:
        await jm.wait(jid, timeout=5)

    recent = jm.recent(n=10)
    recent_ids = [j.id for j in recent]
    # Most recent first.
    assert recent_ids == list(reversed(ids))


@pytest.mark.asyncio
async def test_recent_includes_running_and_completed():
    jm = JobManager()
    finished = await jm.spawn([sys.executable, "-c", "pass"])
    await jm.wait(finished, timeout=5)
    running = await jm.spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        recent = jm.recent(n=10)
        ids = {j.id for j in recent}
        assert finished in ids
        assert running in ids
        states = {j.id: j.state for j in recent}
        assert states[finished] in ("complete", "failed")
        assert states[running] in ("pending", "running")
    finally:
        await jm.cancel(running)
        await jm.wait(running, timeout=5)


@pytest.mark.asyncio
async def test_recent_caps_at_n():
    jm = JobManager()
    ids = []
    for _ in range(5):
        ids.append(await jm.spawn([sys.executable, "-c", "pass"]))
    for jid in ids:
        await jm.wait(jid, timeout=5)
    recent = jm.recent(n=2)
    assert len(recent) == 2
