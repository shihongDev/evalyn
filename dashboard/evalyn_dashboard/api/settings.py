"""``/api/settings`` router (Lane C1.8, C1.9).

Backed by :class:`evalyn_dashboard.credentials.CredentialStore` (kept on
``app.state.credential_store`` by ``server.build_app``).

Routes:

- ``GET    /api/settings``             - safe public view (never plaintext keys).
- ``POST   /api/settings/{provider}``  - upsert ``{api_key?, model?, base_url?}``.
- ``POST   /api/settings/test/{provider}`` - 1-token connection probe.
- ``GET    /api/settings/models/{provider}`` - hard-coded list for cloud
  providers; live ``GET {base_url}/api/tags`` for ``ollama``.
- ``POST   /api/settings/active``      - set active provider.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()


# Hard-coded model catalogues for cloud providers. We deliberately ship a
# curated short-list rather than calling the provider's ``/models`` API
# during a settings render: the call would require a valid key (chicken
# and egg) and produce hundreds of irrelevant entries. Updating this list
# is a normal version bump.
HARDCODED_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-5.1",
        "gpt-5.0",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-mini",
    ],
    "anthropic": [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ],
}


def _get_store(request: Request) -> Any:
    store = getattr(request.app.state, "credential_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="credential store not configured")
    return store


@router.get("")
async def get_settings(request: Request) -> JSONResponse:
    store = _get_store(request)
    return JSONResponse(store.public_view())


@router.post("/active")
async def set_active(request: Request) -> JSONResponse:
    store = _get_store(request)
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")
    provider = body.get("provider")
    if not isinstance(provider, str) or not provider:
        raise HTTPException(status_code=400, detail="provider must be a non-empty string")
    try:
        store.set_active(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "active": provider})


@router.get("/models/{provider}")
async def list_models(request: Request, provider: str) -> JSONResponse:
    if provider in HARDCODED_MODELS:
        return JSONResponse({"models": list(HARDCODED_MODELS[provider])})

    if provider == "ollama":
        # Ollama has no cloud directory; query the local server.
        store = _get_store(request)
        record = store.get_provider("ollama") or {}
        from ..agent import DEFAULT_OLLAMA_BASE_URL

        base_url = record.get("base_url") or DEFAULT_OLLAMA_BASE_URL
        url = base_url.rstrip("/") + "/api/tags"
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"ollama tags fetch failed: {exc}") from exc
        models = []
        for entry in data.get("models", []) or []:
            name = entry.get("name") or entry.get("model")
            if name:
                models.append(name)
        return JSONResponse({"models": models})

    raise HTTPException(status_code=404, detail=f"unknown provider: {provider}")


@router.post("/test/{provider}")
async def test_provider(request: Request, provider: str) -> JSONResponse:
    store = _get_store(request)
    result = store.test_provider(provider)
    status = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status)


@router.post("/{provider}")
async def set_provider(request: Request, provider: str) -> JSONResponse:
    store = _get_store(request)
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")

    api_key = body.get("api_key")
    model = body.get("model")
    base_url = body.get("base_url")
    for label, value in (("api_key", api_key), ("model", model), ("base_url", base_url)):
        if value is not None and not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"{label} must be a string")

    if api_key is None and model is None and base_url is None:
        raise HTTPException(
            status_code=400,
            detail="at least one of api_key, model, base_url is required",
        )

    store.set_provider(provider, api_key=api_key, model=model, base_url=base_url)
    return JSONResponse({"ok": True, "provider": provider})


__all__ = ["router"]
