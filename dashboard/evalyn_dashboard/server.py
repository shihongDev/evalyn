"""FastAPI app + uvicorn launcher for the evalyn dashboard.

Public surface:
- :func:`build_app` factory: returns a configured FastAPI instance. Accepts an
  optional ``token`` override (used by tests); otherwise a fresh CSRF token
  is generated per call via :func:`secrets.token_urlsafe`.

Routes:
- ``GET /api/health`` -> ``{"ok": True}``.
- ``GET /static/*`` mounts the bundled React build.
- ``GET /`` and SPA fallback for unknown non-API routes serve the bundled
  ``index.html`` with the CSRF token injected as
  ``<meta name="workbench-token" content="...">``.
- Strict 404 for unknown ``/api/*`` paths.

CSRF (A1.2):
- Random per-startup token, exposed via the meta tag in served HTML.
- ``X-Workbench-Token`` header required on all mutating verbs
  (``POST/PUT/DELETE/PATCH``) to ``/api/*``. GETs are exempt.
- Wrong or missing token -> 403.

Localhost binding guard / browser auto-open (A1.3, A1.4) and stub API
routers (A1.5) layer on top.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
CSRF_HEADER = "X-Workbench-Token"
CSRF_META_NAME = "workbench-token"
MUTATING_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


class CSRFMiddleware(BaseHTTPMiddleware):
    """Reject mutating ``/api/*`` requests missing the workbench token.

    GETs and HEADs are exempt. Non-API mutating requests are also exempt -
    the dashboard does not own arbitrary write paths outside ``/api``.
    Comparison uses :func:`secrets.compare_digest` to avoid timing leaks.
    """

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        if method in MUTATING_METHODS and path.startswith("/api/"):
            sent = request.headers.get(CSRF_HEADER)
            if not sent or not secrets.compare_digest(sent, self.token):
                return JSONResponse(
                    {"error": "missing or invalid workbench token"},
                    status_code=403,
                )
        return await call_next(request)


def _build_timing_logger() -> logging.Logger:
    """Configure a stdout-attached logger for the timing middleware.

    Uvicorn's default access logger uses a positional-arg formatter we
    can't reuse, and the root logger has no handler in the bare uvicorn
    config. So we attach our own ``StreamHandler`` once and disable
    propagation - cheap, no third-party deps, works in tests.
    """
    log = logging.getLogger("evalyn.timing")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("TIMING: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


_timing_logger = _build_timing_logger()


class TimingMiddleware(BaseHTTPMiddleware):
    """Log ``method path -> status (Xms)`` for every ``/api/*`` request.

    Cheap perf telemetry - one ``time.perf_counter`` pair per request.
    Scoped to API routes so static asset noise stays out of the log.
    Emits one line per request under the ``evalyn.timing`` logger.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _timing_logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
        )
        return response


def _read_index_html(token: str) -> str:
    """Return the bundled ``index.html`` with the CSRF meta tag injected.

    Falls back to a minimal HTML shell if the static bundle is missing
    (e.g. a dev checkout without ``npm run build``).
    """

    if INDEX_FILE.exists():
        raw = INDEX_FILE.read_text(encoding="utf-8")
    else:
        raw = (
            "<!doctype html><html><head>"
            "<meta charset=\"utf-8\">"
            "<title>Evalyn Dashboard</title>"
            "</head><body>"
            "<div id=\"root\">Frontend bundle not built. "
            "Run `npm run build` in dashboard/frontend.</div>"
            "</body></html>"
        )

    meta = f'<meta name="{CSRF_META_NAME}" content="{token}">'
    if "<head>" in raw:
        return raw.replace("<head>", "<head>\n    " + meta, 1)
    if "<html" in raw:
        idx = raw.find(">", raw.find("<html"))
        if idx != -1:
            return raw[: idx + 1] + "<head>" + meta + "</head>" + raw[idx + 1 :]
    return meta + raw


def _register_api_routers(app: FastAPI) -> None:
    """Register API routers.

    All Phase 1-3 routers wire here. ``files`` and ``runs`` still return
    501 (Phase 2 lane B1 territory) but the agent and settings routers
    are real (Phase 3 Lane C1).
    The import is wrapped in try/except so this function stays usable in
    early checkouts that have not yet vendored the ``api`` package.
    """

    try:
        from .api import (
            agent as agent_api,
            cli as cli_api,
            compare as compare_api,
            demo as demo_api,
            files as files_api,
            jobs as jobs_api,
            promote as promote_api,
            runs as runs_api,
            settings as settings_api,
            threads as threads_api,
        )
        from .api.agent_ws import register_agent_ws_routes
        from .api.jobs_ws import register_ws_routes
    except ImportError:
        return

    app.include_router(cli_api.router, prefix="/api/cli", tags=["cli"])
    app.include_router(jobs_api.router, prefix="/api/jobs", tags=["jobs"])
    app.include_router(files_api.router, prefix="/api/files", tags=["files"])
    app.include_router(runs_api.router, prefix="/api/runs", tags=["runs"])
    app.include_router(compare_api.router, prefix="/api/compare", tags=["compare"])
    app.include_router(agent_api.router, prefix="/api/agent", tags=["agent"])
    app.include_router(
        settings_api.router, prefix="/api/settings", tags=["settings"]
    )
    app.include_router(promote_api.router, prefix="/api/promote", tags=["promote"])
    app.include_router(demo_api.router, prefix="/api/demo", tags=["demo"])
    app.include_router(threads_api.router, prefix="/api/threads", tags=["threads"])
    register_ws_routes(app)
    register_agent_ws_routes(app)
    _register_v2_routers(app)


def _register_v2_routers(app: FastAPI) -> None:
    """Register ``/api/v2/*`` routers (the new dashboard surface).

    Wrapped so a missing module fails for that prefix only - the rest of
    the API stays online. Each module is independently importable.
    Also mounts ``/ws/v2/events`` (the cache-bust event stream) and
    schedules its dataset-roots watcher on app startup so the FE gets
    push notifications when ``run-eval`` writes new runs to disk.
    """
    try:
        from .api.v2 import home as v2_home
    except ImportError as exc:
        logger.warning("v2 home router import failed; /api/v2/home will 404: %s", exc)
        return

    app.include_router(v2_home.router, prefix="/api/v2/home", tags=["v2"])

    # Optional v2 modules. We log import failures (including transitive
    # ones) so a dropped router shows up in startup logs instead of
    # manifesting as mysterious 404s in the browser.
    for module_name, prefix in (
        ("experiments", "/api/v2/experiments"),
        ("datasets", "/api/v2/datasets"),
        ("rubrics", "/api/v2/rubrics"),
        ("review", "/api/v2/review"),
        ("reports", "/api/v2/reports"),
        ("annotation", "/api/v2/annotation"),
    ):
        try:
            mod = __import__(
                f"evalyn_dashboard.api.v2.{module_name}",
                fromlist=["router"],
            )
            app.include_router(mod.router, prefix=prefix, tags=["v2"])
        except ImportError as exc:
            logger.warning(
                "v2 %s router import failed; %s will 404: %s",
                module_name,
                prefix,
                exc,
            )
            continue

    # Mount the v2 event WS + start the cache-watcher loop. Both are
    # independent of any individual /api/v2/* router (the event stream
    # is useful even if some routers failed to import).
    try:
        from .api.v2.v2_ws import register_v2_ws_routes
        from .api.v2._shared import prewarm, start_watcher, stop_watcher
    except ImportError as exc:
        logger.warning("v2 ws wiring failed; /ws/v2/events will 404: %s", exc)
        return

    register_v2_ws_routes(app)

    @app.on_event("startup")
    async def _v2_start_watcher() -> None:
        # Pre-warm the run cache in a background thread so the first
        # request after launch hits memory instead of the cold FS walk +
        # JSON parse. Fire-and-forget: a slow walk on a fresh workspace
        # must not delay the first HTTP/WS handshake.
        prewarm()
        start_watcher()

    @app.on_event("shutdown")
    async def _v2_stop_watcher() -> None:
        await stop_watcher()

    @app.on_event("shutdown")
    async def _vacuum_jobs_persistence() -> None:
        """Compact the jobs sqlite mirror on clean shutdown.

        ``DELETE``-based GC (``delete_old`` in JobManager) only marks
        pages as free; the ``.evalyn/data/jobs.sqlite`` file itself
        never shrinks. Over hours/days of running, the file grows
        beyond the working-set size implied by ``persistence_keep``.
        Running ``VACUUM`` here rewrites the file as a tightly-packed
        copy and triggers an implicit WAL checkpoint.

        Best-effort: VACUUM is heavy (rewrites the whole file) but
        fast for the dashboard's small jobs table; an unhealthy
        process should still exit promptly. Errors are logged inside
        ``JobPersistence.vacuum()`` and never bubble.
        """
        jm = getattr(app.state, "job_manager", None)
        persistence = getattr(jm, "_persistence", None) if jm else None
        if persistence is not None and hasattr(persistence, "vacuum"):
            persistence.vacuum()


def build_app(
    token: Optional[str] = None,
    *,
    credential_store: Optional[object] = None,
    agent_runtime: Optional[object] = None,
) -> FastAPI:
    """Construct the FastAPI app.

    ``token`` overrides the auto-generated CSRF token (test seam).
    ``credential_store`` and ``agent_runtime`` allow tests to inject
    fakes; production callers leave them as ``None`` to use the
    defaults (file-backed :class:`CredentialStore` and a real
    :class:`AgentRuntime`).

    Wires four singletons onto ``app.state`` for the API routers to share:
    - ``cli_catalog``: cached output of :func:`build_catalog` (computed
      once at app construction; ~10ms).
    - ``job_manager``: a single :class:`JobManager` shared across all
      requests so subscribe/cancel can find the same job.
    - ``credential_store``: file-backed credential store powering the
      settings API and the agent provider factory.
    - ``agent_runtime``: an :class:`AgentRuntime` whose provider factory
      reads the active provider out of ``credential_store`` on each
      turn (so settings changes take effect immediately).
    """

    csrf_token = token or secrets.token_urlsafe(32)
    app = FastAPI(title="evalyn-dashboard", version="0.1.0")
    app.state.workbench_token = csrf_token
    # Wall-clock at startup, used by /api/health to expose uptime so a
    # monitor can verify the process started recently. Stored as a unix
    # epoch float (UTC) for cheap subtraction.
    import time as _time

    app.state.started_at = _time.time()

    # Lazily imported to avoid making ``server.build_app`` pay the cost
    # of catalog walking on every test that only touches healthcheck.
    # Catalog and job manager are app-scoped singletons.
    from .agent import AgentRuntime, make_provider_factory
    from .credentials import CredentialStore
    from .introspect import build_catalog
    from .jobs import JobManager

    app.state.cli_catalog = build_catalog()
    # Attach the sqlite mirror so jobs survive restarts. Construction is
    # cheap (no filesystem IO) -- the .evalyn/data/jobs.sqlite file is only
    # created on the first persisted write, so tests that build_app() in a
    # fresh tmp_path don't get a stray .evalyn/ directory.
    from .jobs_persistence import JobPersistence

    app.state.job_manager = JobManager(persistence=JobPersistence())
    app.state.credential_store = credential_store or CredentialStore()
    if agent_runtime is None:
        agent_runtime = AgentRuntime(
            provider_factory=make_provider_factory(app.state.credential_store),
            catalog=app.state.cli_catalog,
            job_manager=app.state.job_manager,
        )
    app.state.agent_runtime = agent_runtime

    app.add_middleware(CSRFMiddleware, token=csrf_token)
    app.add_middleware(TimingMiddleware)

    @app.get("/api/health")
    async def healthcheck() -> dict:
        """Return basic process health + capacity snapshot.

        Shape:
          {
            ok: True,
            version: "<semver>",
            started_at: <unix epoch float>,
            uptime_seconds: <int>,
            running: <int>,        # currently-running jobs
            max_concurrent: <int>, # cap; 0 = disabled
            jobs_persisted: <int>, # rows in the sqlite mirror
            jobs_db_bytes: <int>,  # main + WAL + SHM file size
            last_vacuum_at: <float|None>,  # epoch of last VACUUM, None if
                                           # not yet vacuumed in this process
            recent_failures_24h: <int>,  # COUNT(jobs WHERE status='failed'
                                         # AND started_at > now - 24h)
          }

        Useful for:
          - external monitors (curl-based liveness checks, process
            supervisors, dev-server "is it up yet?" loops)
          - tagging UI responses to a specific build at debug time
          - SRE dashboards that want to alert on saturation
            (running >= max_concurrent for sustained periods is a
            signal the cap should be raised or the workload spread)

        ``running`` and ``max_concurrent`` are best-effort: if the
        job_manager is somehow not yet attached (early in startup),
        we report zeros rather than failing the healthcheck. The
        endpoint MUST stay reachable even when the job system is in
        a degraded state; that's its whole point.
        """
        import time as _time

        started_at = float(getattr(app.state, "started_at", _time.time()))
        jm = getattr(app.state, "job_manager", None)
        running = 0
        max_concurrent = 0
        if jm is not None:
            try:
                running = jm.running_count()
                max_concurrent = jm.max_concurrent
            except Exception:  # noqa: BLE001 - never fail healthcheck on JM
                pass
        agent = getattr(app.state, "agent_runtime", None)
        agent_threads = 0
        agent_open_threads = 0
        if agent is not None and hasattr(agent, "thread_counts"):
            try:
                tc = agent.thread_counts()
                agent_threads = int(tc.get("total", 0))
                agent_open_threads = int(tc.get("open", 0))
            except Exception:  # noqa: BLE001 - never fail healthcheck on agent
                pass
        # Persistence visibility: row count + on-disk footprint
        # (main + WAL + SHM). SREs watching long-running dashboards
        # can spot bloat trending toward a vacuum before it pages.
        # Both are best-effort; degraded persistence reports zeros
        # rather than failing the healthcheck.
        jobs_persisted = 0
        jobs_db_bytes = 0
        last_vacuum_at: float | None = None
        recent_failures_24h = 0
        persistence = getattr(jm, "_persistence", None) if jm is not None else None
        if persistence is not None:
            # Pull both totals from a single stats() call. The TTL
            # cache means a second stats() call within the window
            # is free; calling once and reading two fields is
            # still cleaner and removes a dependency on the cache
            # for correctness.
            try:
                stats = persistence.stats()
                jobs_persisted = int(stats.get("total", 0))
                recent_failures_24h = int(stats.get("recent_failures", 0))
            except Exception:  # noqa: BLE001
                pass
            try:
                jobs_db_bytes = int(persistence.db_size_bytes())
            except Exception:  # noqa: BLE001
                pass
            try:
                last_vacuum_at = persistence.last_vacuum_at()
            except Exception:  # noqa: BLE001
                pass
        return {
            "ok": True,
            "version": app.version,
            "started_at": started_at,
            "uptime_seconds": int(_time.time() - started_at),
            "running": running,
            "max_concurrent": max_concurrent,
            "agent_threads": agent_threads,
            "agent_open_threads": agent_open_threads,
            "jobs_persisted": jobs_persisted,
            "jobs_db_bytes": jobs_db_bytes,
            "last_vacuum_at": last_vacuum_at,
            "recent_failures_24h": recent_failures_24h,
        }

    _register_api_routers(app)

    if STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="static",
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Response:
        return HTMLResponse(_read_index_html(csrf_token))

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def spa_fallback(full_path: str) -> Response:
        if full_path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        return HTMLResponse(_read_index_html(csrf_token))

    return app


def _schedule_browser_open(
    server: "uvicorn.Server",  # type: ignore[name-defined]
    url: str,
    *,
    open_delay: float = 0.6,
    timeout: float = 10.0,
    opener: Callable[[str], bool] | None = None,
) -> threading.Thread:
    """Spawn a daemon thread that opens ``url`` once uvicorn signals ready.

    Parameters are exposed for tests; production callers should use the
    defaults.
    """

    open_fn = opener or webbrowser.open

    def _wait_then_open() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                break
            time.sleep(0.05)
        if open_delay > 0:
            time.sleep(open_delay)
        try:
            open_fn(url)
        except Exception:
            # Browser open is best-effort; never fatal.
            pass

    t = threading.Thread(target=_wait_then_open, daemon=True)
    t.start()
    return t


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 7401,
    no_browser: bool = False,
    dev: bool = False,
) -> int:
    """Block running uvicorn. Opens a browser tab unless suppressed.

    Skips the browser-open in two cases:
    - ``no_browser=True`` (explicit operator opt-out).
    - ``dev=True`` (frontend served by Vite on :5173).

    Returns 0 on clean shutdown (uvicorn exits via SIGINT/SIGTERM).
    """

    import uvicorn  # imported lazily so unit tests can build_app cheaply.

    app = build_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    if not no_browser and not dev:
        _schedule_browser_open(server, f"http://{host}:{port}/")

    server.run()
    return 0


def main(
    host: str = "127.0.0.1",
    port: int = 7401,
    no_browser: bool = False,
    dev: bool = False,
    agent_budget: float | None = None,
) -> int:
    """Spec-shaped entry point. ``agent_budget`` is reserved for Phase 3."""

    del agent_budget  # unused in Phase 1
    return run_server(host=host, port=port, no_browser=no_browser, dev=dev)


__all__ = [
    "build_app",
    "run_server",
    "main",
    "CSRFMiddleware",
    "CSRF_HEADER",
    "CSRF_META_NAME",
    "STATIC_DIR",
    "INDEX_FILE",
]
