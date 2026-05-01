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

import secrets
from pathlib import Path
from typing import Optional

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


def build_app(token: Optional[str] = None) -> FastAPI:
    """Construct the FastAPI app.

    ``token`` overrides the auto-generated CSRF token (test seam).
    """

    csrf_token = token or secrets.token_urlsafe(32)
    app = FastAPI(title="evalyn-dashboard", version="0.1.0")
    app.state.workbench_token = csrf_token

    app.add_middleware(CSRFMiddleware, token=csrf_token)

    @app.get("/api/health")
    async def healthcheck() -> dict:
        return {"ok": True}

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


__all__ = [
    "build_app",
    "CSRFMiddleware",
    "CSRF_HEADER",
    "CSRF_META_NAME",
    "STATIC_DIR",
    "INDEX_FILE",
]
