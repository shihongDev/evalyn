"""Tests for ``/api/settings/*`` (Lane C1.8, C1.9)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.credentials import CredentialStore
from evalyn_dashboard.server import CSRF_HEADER, build_app


def _token_from(client: TestClient) -> str:
    html = client.get("/").text
    m = re.search(r'content="([^"]+)"', html)
    assert m
    return m.group(1)


def _make_client(tmp_path: Path) -> tuple[TestClient, Any, str]:
    store = CredentialStore(path=tmp_path / "cred.json")
    app = build_app(credential_store=store)
    client = TestClient(app)
    token = _token_from(client)
    return client, store, token


def test_get_settings_returns_empty_initially(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["providers"] == {}
    assert body["active"] is None


def test_post_provider_persists_and_redacts(tmp_path: Path) -> None:
    client, store, token = _make_client(tmp_path)
    r = client.post(
        "/api/settings/openai",
        json={"api_key": "sk-secret", "model": "gpt-5.1"},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "provider": "openai"}

    # Public view never leaks the key.
    public = client.get("/api/settings").json()
    assert public["providers"]["openai"]["is_set"] is True
    assert public["providers"]["openai"]["model"] == "gpt-5.1"
    assert "sk-secret" not in r.text
    # But it persisted on disk via the real store.
    rec = store.get_provider("openai")
    assert rec is not None
    assert rec["api_key"] == "sk-secret"


def test_post_provider_requires_csrf_token(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    r = client.post(
        "/api/settings/openai",
        json={"api_key": "sk-x", "model": "gpt-5.1"},
    )
    assert r.status_code == 403


def test_post_provider_rejects_empty_body(tmp_path: Path) -> None:
    client, _, token = _make_client(tmp_path)
    r = client.post(
        "/api/settings/openai", json={}, headers={CSRF_HEADER: token}
    )
    assert r.status_code == 400


def test_post_active_sets_default_provider(tmp_path: Path) -> None:
    client, store, token = _make_client(tmp_path)
    store.set_provider("openai", api_key="sk", model="gpt-5.1")
    r = client.post(
        "/api/settings/active",
        json={"provider": "openai"},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 200
    assert r.json()["active"] == "openai"
    assert client.get("/api/settings").json()["active"] == "openai"


def test_post_active_unknown_provider_400(tmp_path: Path) -> None:
    client, _, token = _make_client(tmp_path)
    r = client.post(
        "/api/settings/active",
        json={"provider": "ghost"},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 400


def test_models_openai_hardcoded(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    r = client.get("/api/settings/models/openai")
    assert r.status_code == 200
    models = r.json()["models"]
    assert isinstance(models, list)
    assert any("gpt" in m for m in models)


def test_models_anthropic_hardcoded(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    r = client.get("/api/settings/models/anthropic")
    assert r.status_code == 200
    models = r.json()["models"]
    assert any("claude" in m for m in models)


def test_models_unknown_provider_404(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    r = client.get("/api/settings/models/unicorn")
    assert r.status_code == 404


def test_models_ollama_calls_local_tags(tmp_path: Path) -> None:
    client, store, _ = _make_client(tmp_path)
    store.set_provider("ollama", model="llama3", base_url="http://localhost:11434")

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "models": [
            {"name": "llama3:70b"},
            {"name": "mistral:7b"},
        ]
    }
    with patch("httpx.get", return_value=fake_response) as mock_get:
        r = client.get("/api/settings/models/ollama")
    assert r.status_code == 200
    assert r.json()["models"] == ["llama3:70b", "mistral:7b"]
    mock_get.assert_called_once()
    args, _ = mock_get.call_args
    assert args[0] == "http://localhost:11434/api/tags"


def test_models_ollama_failure_502(tmp_path: Path) -> None:
    client, _, _ = _make_client(tmp_path)
    with patch("httpx.get", side_effect=RuntimeError("connection refused")):
        r = client.get("/api/settings/models/ollama")
    assert r.status_code == 502


def test_test_provider_ok_returns_200(tmp_path: Path) -> None:
    client, store, token = _make_client(tmp_path)
    store.set_provider("openai", api_key="sk", model="gpt-5.1")

    fake_module = MagicMock()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock()
    fake_module.OpenAI.return_value = fake_client

    with patch.dict(sys.modules, {"openai": fake_module}):
        r = client.post(
            "/api/settings/test/openai", headers={CSRF_HEADER: token}
        )
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_test_provider_failure_returns_400(tmp_path: Path) -> None:
    client, store, token = _make_client(tmp_path)
    store.set_provider("openai", api_key="sk", model="gpt-5.1")

    fake_module = MagicMock()
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("rate limit")
    fake_module.OpenAI.return_value = fake_client

    with patch.dict(sys.modules, {"openai": fake_module}):
        r = client.post(
            "/api/settings/test/openai", headers={CSRF_HEADER: token}
        )
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert "rate limit" in body["error"]
