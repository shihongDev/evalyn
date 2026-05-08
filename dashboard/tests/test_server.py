"""Tests for ``evalyn_dashboard.server`` FastAPI app skeleton (A1.1)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.server import build_app


def test_healthcheck() -> None:
    client = TestClient(build_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    # Backwards-compatible: ok field still present.
    assert body["ok"] is True
    # Newer fields useful for monitors and build-tagging.
    assert isinstance(body["version"], str) and len(body["version"]) > 0
    assert isinstance(body["started_at"], (int, float))
    assert body["started_at"] > 0
    assert isinstance(body["uptime_seconds"], int)
    assert body["uptime_seconds"] >= 0
    # Capacity snapshot for SRE dashboards.
    assert isinstance(body["running"], int)
    assert body["running"] >= 0
    assert isinstance(body["max_concurrent"], int)
    assert body["max_concurrent"] >= 0
    # Agent thread snapshot.
    assert isinstance(body["agent_threads"], int)
    assert body["agent_threads"] >= 0
    assert isinstance(body["agent_open_threads"], int)
    assert body["agent_open_threads"] >= 0
    # Open threads cannot exceed total.
    assert body["agent_open_threads"] <= body["agent_threads"]
    # Persistence visibility.
    assert isinstance(body["jobs_persisted"], int)
    assert body["jobs_persisted"] >= 0
    assert isinstance(body["jobs_db_bytes"], int)
    assert body["jobs_db_bytes"] >= 0
    # last_vacuum_at: None until a vacuum has run in-process.
    # On a freshly-built test app no vacuum has fired so it's None.
    assert body["last_vacuum_at"] is None or isinstance(
        body["last_vacuum_at"], (int, float)
    )
    # recent_failures_24h: from the cached stats() call. >= 0
    # always; on a fresh app with no jobs it's exactly 0.
    assert isinstance(body["recent_failures_24h"], int)
    assert body["recent_failures_24h"] >= 0


def test_healthcheck_capacity_zero_when_idle() -> None:
    """A freshly-built app with no jobs yet reports running=0 and
    a positive max_concurrent (the default JobManager cap)."""
    client = TestClient(build_app())
    body = client.get("/api/health").json()
    assert body["running"] == 0
    # Default cap is 16; we don't assert the exact value (it could
    # change) but it must be > 0 since the cap is enabled by default.
    assert body["max_concurrent"] > 0


def test_healthcheck_survives_missing_job_manager(monkeypatch) -> None:
    """If the job_manager is somehow missing (early-startup race or
    a test rig that doesn't attach one), /api/health must NOT 500 -
    it falls back to running=0 / max_concurrent=0 so external
    monitors keep getting 200s."""
    app = build_app()
    # Simulate the degraded state by clearing the attribute.
    monkeypatch.delattr(app.state, "job_manager", raising=False)
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["running"] == 0
    assert body["max_concurrent"] == 0
    # Agent fields are also defensive.
    assert body["agent_threads"] == 0
    assert body["agent_open_threads"] == 0
    # Persistence fields fall back to zero when JM is missing.
    assert body["jobs_persisted"] == 0
    assert body["jobs_db_bytes"] == 0
    # last_vacuum_at falls back to None in the degraded path.
    assert body["last_vacuum_at"] is None
    # recent_failures_24h falls through to 0 in the degraded path.
    assert body["recent_failures_24h"] == 0


def test_healthcheck_sends_no_store_cache_control() -> None:
    """Polled health endpoint must not be cached. A corporate
    proxy or CDN serving a stale response would freeze the
    SystemStatusCard's uptime/running/recent_failures fields.
    The header is checked explicitly because FastAPI/Starlette
    does NOT add Cache-Control by default - it's the route's
    responsibility."""
    client = TestClient(build_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc.lower(), (
        f"expected 'no-store' in Cache-Control, got: {cc!r}"
    )


def test_index_served() -> None:
    client = TestClient(build_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_static_assets_served() -> None:
    client = TestClient(build_app())
    r = client.get("/static/index.html")
    assert r.status_code == 200


def test_spa_fallback_returns_index_for_unknown_paths() -> None:
    client = TestClient(build_app())
    r = client.get("/some/unknown/route")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_unknown_api_route_does_not_fall_back_to_index() -> None:
    client = TestClient(build_app())
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404


def test_unknown_api_route_returns_detail_shape() -> None:
    """The SPA fallback's API 404 response uses the `{"detail": ...}`
    shape, matching FastAPI's HTTPException convention. Previously
    used `{"error": ...}` which broke the FE error parser
    consistency (settings.ts handled both via `j.error ?? j.detail`
    but other modules just `res.text()` and surfaced the body
    verbatim - keying off the JSON shape made debug logs noisier).
    Pin so this can't drift back to the `error` key.
    """
    client = TestClient(build_app())
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert "detail" in body
    assert body["detail"] == "not found"
    # Defensive: the old key must NOT also be present.
    assert "error" not in body


def test_csrf_rejection_uses_detail_shape() -> None:
    """The CSRF middleware's 403 response uses the `{"detail": ...}`
    shape, matching FastAPI's HTTPException convention. Same
    rationale as the SPA fallback test above.
    """
    client = TestClient(build_app())
    # POST without the X-Workbench-Token header trips the CSRF
    # middleware before any route handler runs.
    r = client.post("/api/jobs/admin/vacuum")
    assert r.status_code == 403
    body = r.json()
    assert "detail" in body
    assert "workbench token" in body["detail"].lower()
    assert "error" not in body


# ---- A1.4: browser auto-open scheduling -----------------------------------


class _FakeServer:
    def __init__(self, started: bool = True) -> None:
        self.started = started


def test_schedule_browser_open_calls_opener_when_started() -> None:
    from evalyn_dashboard.server import _schedule_browser_open

    seen: list[str] = []
    t = _schedule_browser_open(
        _FakeServer(started=True),
        "http://127.0.0.1:7401/",
        open_delay=0.0,
        opener=lambda url: seen.append(url) or True,
    )
    t.join(timeout=2.0)
    assert seen == ["http://127.0.0.1:7401/"]


def test_schedule_browser_open_swallows_opener_exceptions() -> None:
    from evalyn_dashboard.server import _schedule_browser_open

    def boom(_url: str) -> bool:
        raise RuntimeError("no browser available")

    t = _schedule_browser_open(
        _FakeServer(started=True),
        "http://127.0.0.1:7401/",
        open_delay=0.0,
        opener=boom,
    )
    # Thread must exit cleanly even when opener raises.
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_schedule_browser_open_times_out_when_server_never_starts() -> None:
    from evalyn_dashboard.server import _schedule_browser_open

    seen: list[str] = []
    t = _schedule_browser_open(
        _FakeServer(started=False),
        "http://127.0.0.1:7401/",
        open_delay=0.0,
        timeout=0.1,
        opener=lambda url: seen.append(url) or True,
    )
    t.join(timeout=2.0)
    # Even if the server never reports started, the opener still fires
    # after the timeout window so the operator at least gets a tab.
    assert seen == ["http://127.0.0.1:7401/"]


# ---- /api/files and /api/runs return empty arrays when .evalyn/ is absent --


@pytest.mark.parametrize(
    "path",
    [
        "/api/files/tree",
        "/api/runs",
    ],
)
def test_files_and_runs_return_empty_when_no_evalyn(tmp_path, monkeypatch, path: str) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(build_app())
    r = client.get(path)
    assert r.status_code == 200
    assert r.json() == []


def test_stub_post_routes_require_csrf() -> None:
    """Mutating stub routes still 403 without the workbench token."""

    client = TestClient(build_app())
    r = client.post("/api/cli/run", json={})
    assert r.status_code == 403


def test_settings_get_returns_redacted_view() -> None:
    """``/api/settings`` is no longer a stub; it returns the public view."""
    client = TestClient(build_app())
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "providers" in body and "active" in body


def test_resolve_positive_float_env_falls_back_safely(monkeypatch) -> None:
    """``_resolve_positive_float_env`` must accept positive numerics
    (int or float), fall back to ``default`` on unset/empty/garbage/
    zero/negative/NaN. Used by the agent-thread auto-purge knobs
    (EVALYN_AGENT_PURGE_INTERVAL_S, EVALYN_AGENT_THREAD_TTL_S) so
    a typo in operator config can't silently convert "purge every
    hour" into "purge every second" or "never purge."
    """
    from evalyn_dashboard.server import _resolve_positive_float_env

    cases: list[tuple[str | None, float, float]] = [
        # (env value, default, expected)
        (None, 60.0, 60.0),  # unset -> default
        ("", 60.0, 60.0),  # empty -> default
        ("  ", 60.0, 60.0),  # whitespace -> default
        ("3600", 60.0, 3600.0),  # int string
        ("3600.5", 60.0, 3600.5),  # float string
        ("  3600  ", 60.0, 3600.0),  # whitespace tolerant
        ("garbage", 60.0, 60.0),  # unparseable -> default
        ("0", 60.0, 60.0),  # zero rejected -> default
        ("-1", 60.0, 60.0),  # negative rejected -> default
        ("nan", 60.0, 60.0),  # NaN rejected -> default
        ("inf", 60.0, float("inf")),  # +inf is positive, accepted
        # Note: -inf is negative, rejected.
        ("-inf", 60.0, 60.0),
    ]
    for raw, default, expected in cases:
        if raw is None:
            monkeypatch.delenv("EVALYN_TEST_FLOAT_VAR", raising=False)
        else:
            monkeypatch.setenv("EVALYN_TEST_FLOAT_VAR", raw)
        actual = _resolve_positive_float_env("EVALYN_TEST_FLOAT_VAR", default)
        assert actual == expected, f"raw={raw!r} expected={expected} got={actual}"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", 20),  # INFO default when unset
        ("DEBUG", 10),
        ("debug", 10),  # case-insensitive
        ("INFO", 20),
        ("WARNING", 30),
        ("warning", 30),
        ("ERROR", 40),
        ("CRITICAL", 50),
        ("  WARNING  ", 30),  # whitespace tolerant
        ("BOGUS", 20),  # unknown -> INFO (no crash)
        ("123", 20),  # numeric strings not honored (would be misleading)
        ("WARN", 30),  # stdlib alias for WARNING
        ("FATAL", 50),  # stdlib alias for CRITICAL
    ],
)
def test_resolve_app_log_level(monkeypatch, raw: str, expected: int) -> None:
    """``EVALYN_LOG_LEVEL`` resolves to logging level ints. Unknown
    or unset values fall back to INFO so the audit-log trail stays
    visible by default.

    Why this matters for operators:
      - Production daemons can flip to ``WARNING`` to suppress the
        per-mutation audit-log INFO chatter without losing real
        warnings/errors.
      - A typo in the env var must NOT silently switch the logger
        to a useless level (e.g. CRITICAL-only) - the resolver
        defends with a hardcoded INFO fallback.
    """
    import logging
    from evalyn_dashboard.server import _resolve_app_log_level

    if raw:
        monkeypatch.setenv("EVALYN_LOG_LEVEL", raw)
    else:
        monkeypatch.delenv("EVALYN_LOG_LEVEL", raising=False)
    assert _resolve_app_log_level() == expected
    # Sanity: the resolver never returns a string (the bug we
    # defended against - logging.getLevelName returns a string for
    # unknown names instead of an int).
    assert isinstance(_resolve_app_log_level(), int)
    # And the value must match a real level for the logger.
    logging.getLogger("test").setLevel(_resolve_app_log_level())


def test_agent_thread_auto_purge_task_starts_and_cancels() -> None:
    """Periodic agent-thread purge background task is started on
    app startup and cancelled cleanly on shutdown. Without this
    sweep, an unattended long-running dashboard accumulates closed
    threads in memory (~10KB each) because the
    /api/agent/threads/purge-old endpoint has to be triggered
    manually.

    Pin: app.state carries a non-None task on startup, and after
    the lifespan shutdown completes the task is done (cancelled
    cleanly). The test does NOT advance time enough to actually
    trigger a purge tick - that's covered by
    test_purge_old_threads_drops_closed_old_threads.
    """
    import asyncio

    app = build_app()
    with TestClient(app):
        task = getattr(app.state, "agent_thread_purge_task", None)
        # The task is created if an agent runtime is attached.
        # In the build_app default an agent runtime IS attached so
        # the task should exist.
        assert task is not None, (
            "agent_thread_purge_task should be created on startup"
        )
        assert isinstance(task, asyncio.Task)
        # Mid-lifespan: task is running its sleep loop.
        assert not task.done()

    # After lifespan shutdown: task should be cancelled.
    assert task.done(), "purge task should be cancelled on shutdown"


def test_shutdown_hook_calls_persistence_vacuum() -> None:
    """The shutdown lifespan hook fires JobPersistence.vacuum() so the
    on-disk sqlite mirror gets compacted on clean exit. Without this,
    long-running dashboards accumulate fragmentation that delete_old's
    page-mark-only behavior never reclaims.

    Uses TestClient as a context manager so the lifespan startup +
    shutdown both run. We swap in a counter on the persistence
    object's vacuum method to observe the call.
    """
    app = build_app()
    jm = app.state.job_manager
    persistence = jm._persistence
    assert persistence is not None
    calls = {"n": 0}
    real_vacuum = persistence.vacuum

    def counting_vacuum() -> bool:
        calls["n"] += 1
        return real_vacuum()

    persistence.vacuum = counting_vacuum  # type: ignore[method-assign]

    with TestClient(app):
        # No-op block: just enter+exit the lifespan to trigger startup
        # + shutdown.
        pass

    assert calls["n"] == 1, "shutdown hook should call vacuum exactly once"
