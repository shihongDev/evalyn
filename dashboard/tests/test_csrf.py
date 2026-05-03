"""CSRF token middleware tests (A1.2).

Spec ref: design §7 ``server.py`` and §10 (Network exposure / API keys).
- Random token generated at startup.
- Injected into served ``index.html`` as ``<meta name="workbench-token">``.
- Required on POST/PUT/DELETE/PATCH for ``/api/*``.
- GET routes exempt.
- Wrong token rejected.
- Token is stable across requests for the same app.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.server import CSRF_HEADER, CSRF_META_NAME, build_app

META_RE = re.compile(
    rf'<meta\s+name="{CSRF_META_NAME}"\s+content="([^"]+)"', re.IGNORECASE
)


def _extract_token(html: str) -> str:
    m = META_RE.search(html)
    assert m, f"missing meta tag in: {html[:300]}"
    return m.group(1)


def test_csrf_token_in_index() -> None:
    client = TestClient(build_app())
    r = client.get("/")
    assert r.status_code == 200
    assert f'name="{CSRF_META_NAME}"' in r.text


def test_csrf_token_is_non_empty_random() -> None:
    client = TestClient(build_app())
    token = _extract_token(client.get("/").text)
    assert len(token) >= 20  # ``secrets.token_urlsafe(32)`` is ~43 chars


def test_csrf_token_stable_within_app() -> None:
    """Two requests against the same app instance see the same token."""
    app = build_app()
    client = TestClient(app)
    t1 = _extract_token(client.get("/").text)
    t2 = _extract_token(client.get("/").text)
    assert t1 == t2


def test_csrf_token_differs_across_apps() -> None:
    """Each ``build_app()`` call generates a fresh token."""
    t1 = _extract_token(TestClient(build_app()).get("/").text)
    t2 = _extract_token(TestClient(build_app()).get("/").text)
    assert t1 != t2


def test_post_without_token_rejected() -> None:
    client = TestClient(build_app())
    r = client.post("/api/cli/run", json={})
    assert r.status_code == 403


def test_post_with_wrong_token_rejected() -> None:
    client = TestClient(build_app())
    r = client.post(
        "/api/cli/run",
        json={},
        headers={CSRF_HEADER: "not-the-real-token"},
    )
    assert r.status_code == 403


def test_post_with_correct_token_passes_middleware() -> None:
    """With the right token, the request is NOT 403'd by the middleware.

    The route may 404/405/501 depending on whether stub routers are
    registered, but the contract here is purely middleware behaviour.
    """

    client = TestClient(build_app())
    token = _extract_token(client.get("/").text)
    r = client.post("/api/cli/run", json={}, headers={CSRF_HEADER: token})
    assert r.status_code != 403


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_other_mutating_methods_require_token(method: str) -> None:
    client = TestClient(build_app())
    r = client.request(method, "/api/cli/run")
    assert r.status_code == 403


def test_get_does_not_require_token() -> None:
    client = TestClient(build_app())
    r = client.get("/api/health")
    assert r.status_code == 200


def test_non_api_post_unaffected_by_csrf() -> None:
    """The middleware only guards /api/*; other POSTs (e.g. unknown SPA
    routes) are not 403'd, they just hit the SPA fallback or 404."""
    client = TestClient(build_app())
    r = client.post("/some/unknown/page")
    # Either 404/405 (no POST route) - the key contract is "not 403".
    assert r.status_code != 403
