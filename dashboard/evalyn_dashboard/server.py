"""FastAPI app + uvicorn launcher for the evalyn dashboard.

A1.1 surface (this commit):
- :func:`build_app` factory: returns a configured FastAPI instance.
- ``GET /api/health`` -> ``{"ok": True}``.
- ``GET /`` and SPA fallback for unknown non-API routes serve the bundled
  ``static/index.html`` (with a minimal placeholder when the bundle is
  missing).
- ``/static/`` mounts the React build directory.
- Strict 404 for unknown ``/api/*`` paths (no SPA bleed-through).

CSRF middleware (A1.2), localhost binding guard / browser auto-open (A1.3,
A1.4), and the stub API routers (A1.5) layer onto this skeleton.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def _read_index_html() -> str:
    """Return the bundled ``index.html`` text, or a fallback shell."""

    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8")
    return (
        "<!doctype html><html><head>"
        "<meta charset=\"utf-8\">"
        "<title>Evalyn Dashboard</title>"
        "</head><body>"
        "<div id=\"root\">Frontend bundle not built. "
        "Run `npm run build` in dashboard/frontend.</div>"
        "</body></html>"
    )


def build_app(token: Optional[str] = None) -> FastAPI:
    """Construct the FastAPI app.

    The ``token`` argument is reserved for the CSRF middleware added in
    A1.2; it is accepted now so tests and downstream code don't churn.
    """

    del token  # unused until A1.2

    app = FastAPI(title="evalyn-dashboard", version="0.1.0")

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
        return HTMLResponse(_read_index_html())

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def spa_fallback(full_path: str) -> Response:
        # Unknown ``/api/*`` routes must 404 explicitly: the SPA must not
        # mask backend errors.
        if full_path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        return HTMLResponse(_read_index_html())

    return app


__all__ = ["build_app", "STATIC_DIR", "INDEX_FILE"]
