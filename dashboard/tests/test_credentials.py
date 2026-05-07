"""Tests for ``evalyn_dashboard.credentials.CredentialStore``.

Covers A4.1 (atomic write + chmod 600 + public_view safety) and
A4.2 (provider connection test with mocked HTTP clients).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evalyn_dashboard.credentials import CredentialStore


# ---------------------------------------------------------------------------
# A4.1: atomic write + chmod 600 + safe public view
# ---------------------------------------------------------------------------


def test_fresh_path_public_view_surfaces_ollama_only(tmp_path: Path) -> None:
    """Fresh store still synthesizes an ollama entry as is_set=True.

    Ollama is a local provider with a default base_url; we want the FE
    Test connection button to be enabled even before the user clicks
    Save, so public_view always surfaces ollama. Cloud providers stay
    absent until set_provider records them.
    """
    cs = CredentialStore(path=tmp_path / "cred.json")
    view = cs.public_view()
    assert view["active"] is None
    assert view["providers"] == {
        "ollama": {"is_set": True, "model": None, "added_at": None, "updated_at": None},
    }


def test_fresh_path_get_provider_returns_none(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    assert cs.get_provider("openai") is None


def test_set_provider_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    cs = CredentialStore(path=target)
    cs.set_provider("openai", api_key="sk-test", model="gpt-5.1")
    assert target.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod semantics")
def test_write_chmods_to_600(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    cs = CredentialStore(path=target)
    cs.set_provider("openai", api_key="sk-test", model="gpt-5.1")
    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o600


def test_round_trip_preserves_api_key(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    cs = CredentialStore(path=target)
    cs.set_provider("openai", api_key="sk-test", model="gpt-5.1")
    cs2 = CredentialStore(path=target)
    record = cs2.get_provider("openai")
    assert record is not None
    assert record["api_key"] == "sk-test"
    assert record["model"] == "gpt-5.1"
    assert "added_at" in record


def test_public_view_never_contains_plaintext_key(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-secret-XXX", model="gpt-5.1")
    public = cs.public_view()
    serialized = json.dumps(public)
    assert "sk-secret-XXX" not in serialized
    assert public["providers"]["openai"]["is_set"] is True
    assert public["providers"]["openai"]["model"] == "gpt-5.1"
    assert "added_at" in public["providers"]["openai"]
    assert "api_key" not in public["providers"]["openai"]


def test_public_view_includes_active(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-1", model="gpt-5.1")
    cs.set_active("openai")
    assert cs.public_view()["active"] == "openai"


def test_test_provider_ollama_with_no_record_does_not_short_circuit(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: Test connection used to fail with 'not configured'
    for a fresh ollama install. The fix lets ``test_provider('ollama')``
    fall through to ``_test_ollama({})`` so the localhost probe runs.
    """
    cs = CredentialStore(path=tmp_path / "cred.json")
    called_with: dict[str, Any] = {}

    def fake_test_ollama(rec: dict[str, Any]) -> dict[str, Any]:
        called_with["rec"] = rec
        return {"ok": True}

    monkeypatch.setattr(cs, "_test_ollama", staticmethod(fake_test_ollama))
    result = cs.test_provider("ollama")
    assert result == {"ok": True}
    assert called_with["rec"] == {}


def test_test_provider_unknown_cloud_still_short_circuits(tmp_path: Path) -> None:
    """OpenAI/Anthropic without a saved key still return the
    'not configured' error - the ollama exception is local-only.
    """
    cs = CredentialStore(path=tmp_path / "cred.json")
    result = cs.test_provider("openai")
    assert result == {"ok": False, "error": "provider 'openai' not configured"}


def test_set_active_persists_across_instances(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    cs = CredentialStore(path=target)
    cs.set_provider("anthropic", api_key="sk-ant", model="claude-sonnet-4-6")
    cs.set_active("anthropic")
    cs2 = CredentialStore(path=target)
    assert cs2.public_view()["active"] == "anthropic"


def test_set_active_unknown_provider_raises(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    with pytest.raises(ValueError):
        cs.set_active("nonexistent")


def test_set_provider_partial_update_keeps_old_fields(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-1", model="gpt-5.1")
    # Update only model
    cs.set_provider("openai", model="gpt-5.2")
    rec = cs.get_provider("openai")
    assert rec is not None
    assert rec["api_key"] == "sk-1"
    assert rec["model"] == "gpt-5.2"


def test_ollama_record_with_base_url(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("ollama", model="llama3:70b", base_url="http://localhost:11434")
    rec = cs.get_provider("ollama")
    assert rec is not None
    assert rec["base_url"] == "http://localhost:11434"
    assert rec["model"] == "llama3:70b"


def test_malformed_json_treated_as_empty(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    target.write_text("{not json}")
    cs = CredentialStore(path=target)
    # Malformed file falls back to an empty store; public_view still
    # surfaces ollama as the always-testable local provider.
    view = cs.public_view()
    assert view["active"] is None
    assert view["providers"] == {
        "ollama": {"is_set": True, "model": None, "added_at": None, "updated_at": None},
    }
    # Should be able to recover by writing
    cs.set_provider("openai", api_key="sk-1", model="gpt-5.1")
    assert cs.get_provider("openai") is not None


def test_atomic_write_no_temp_file_left_behind(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-1", model="gpt-5.1")
    leftover = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftover == []


def test_default_path_under_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    # Also patch USERPROFILE for Windows-style expanduser behavior.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cs = CredentialStore()
    assert cs.path == tmp_path / ".evalyn" / "credentials.json"


# ---------------------------------------------------------------------------
# A4.2: test_provider connection check (mocked)
# ---------------------------------------------------------------------------


def test_test_provider_unknown_returns_error(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    result = cs.test_provider("openai")
    assert result["ok"] is False
    assert "error" in result


def test_test_provider_openai_ok(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-test", model="gpt-5.1")

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    fake_module = MagicMock()
    fake_module.OpenAI.return_value = fake_client

    with patch.dict(sys.modules, {"openai": fake_module}):
        result = cs.test_provider("openai")

    assert result["ok"] is True
    fake_module.OpenAI.assert_called_once_with(api_key="sk-test")
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.1"
    assert kwargs["max_tokens"] == 1
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_test_provider_openai_failure(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-test", model="gpt-5.1")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("rate limited")
    fake_module = MagicMock()
    fake_module.OpenAI.return_value = fake_client

    with patch.dict(sys.modules, {"openai": fake_module}):
        result = cs.test_provider("openai")

    assert result["ok"] is False
    assert "rate limited" in result["error"]


def test_test_provider_anthropic_ok(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("anthropic", api_key="sk-ant", model="claude-sonnet-4-6")

    fake_client = MagicMock()
    fake_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="ok")]
    )
    fake_module = MagicMock()
    fake_module.Anthropic.return_value = fake_client

    with patch.dict(sys.modules, {"anthropic": fake_module}):
        result = cs.test_provider("anthropic")

    assert result["ok"] is True
    fake_module.Anthropic.assert_called_once_with(api_key="sk-ant")
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 1


def test_test_provider_anthropic_failure(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("anthropic", api_key="sk-ant", model="claude-sonnet-4-6")

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("invalid key")
    fake_module = MagicMock()
    fake_module.Anthropic.return_value = fake_client

    with patch.dict(sys.modules, {"anthropic": fake_module}):
        result = cs.test_provider("anthropic")

    assert result["ok"] is False
    assert "invalid key" in result["error"]


def test_test_provider_ollama_ok(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("ollama", model="llama3", base_url="http://localhost:11434")

    fake_get = MagicMock()
    fake_get.status_code = 200
    fake_get.raise_for_status.return_value = None
    # Tool-probe response: 200 with tool_calls -> tool-capable.
    fake_post = MagicMock()
    fake_post.status_code = 200
    fake_post.json.return_value = {
        "message": {"tool_calls": [{"function": {"name": "echo"}}]}
    }

    with patch("httpx.get", return_value=fake_get) as mock_get, patch(
        "httpx.post", return_value=fake_post
    ) as mock_post:
        result = cs.test_provider("ollama")

    assert result["ok"] is True
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "http://localhost:11434/api/tags"
    # Tool-probe should hit /api/chat exactly once.
    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert post_args[0] == "http://localhost:11434/api/chat"


def test_test_provider_ollama_failure(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("ollama", model="llama3", base_url="http://localhost:11434")

    with patch("httpx.get", side_effect=RuntimeError("connection refused")):
        result = cs.test_provider("ollama")

    assert result["ok"] is False
    assert "connection refused" in result["error"]


def test_test_provider_ollama_default_base_url(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    # Provision ollama record without explicit base_url
    cs.set_provider("ollama", model="llama3")

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_post = MagicMock()
    fake_post.status_code = 200
    fake_post.json.return_value = {
        "message": {"tool_calls": [{"function": {"name": "echo"}}]}
    }

    with patch("httpx.get", return_value=fake_response) as mock_get, patch(
        "httpx.post", return_value=fake_post
    ):
        result = cs.test_provider("ollama")

    assert result["ok"] is True
    args, _ = mock_get.call_args
    assert args[0] == "http://localhost:11434/api/tags"


def test_test_provider_unknown_provider_name(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk", model="gpt-5.1")
    result = cs.test_provider("mystery")
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# O3: Ollama tool-support detection in _test_ollama
# ---------------------------------------------------------------------------


def test_ollama_probe_rejects_text_only_model(tmp_path: Path) -> None:
    """O3: when the configured Ollama model returns text without
    `tool_calls` on the probe, _test_ollama must report the model is not
    tool-capable so the user is not silently misled."""
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("ollama", model="phi3", base_url="http://localhost:11434")

    fake_get = MagicMock()
    fake_get.raise_for_status.return_value = None
    # Probe response: 200 with text only, no tool_calls -> NOT tool-capable.
    fake_post = MagicMock()
    fake_post.status_code = 200
    fake_post.json.return_value = {"message": {"content": "hello"}}

    with patch("httpx.get", return_value=fake_get), patch(
        "httpx.post", return_value=fake_post
    ) as mock_post:
        result = cs.test_provider("ollama")

    assert result["ok"] is False
    assert "tool calling" in result["error"].lower()
    assert "phi3" in result["error"]
    mock_post.assert_called_once()
    post_args, _ = mock_post.call_args
    assert post_args[0] == "http://localhost:11434/api/chat"


def test_ollama_probe_rejects_model_that_4xx_on_tools(tmp_path: Path) -> None:
    """O3 (refined after live testing): when Ollama returns HTTP 4xx
    because the model does not accept the `tools` payload (the actual
    failure mode for llama3 and other pre-3.1 models), the probe must
    surface this as ok=False rather than swallowing it via best-effort."""
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("ollama", model="llama3", base_url="http://localhost:11434")

    fake_get = MagicMock()
    fake_get.raise_for_status.return_value = None
    fake_post = MagicMock()
    fake_post.status_code = 400

    with patch("httpx.get", return_value=fake_get), patch(
        "httpx.post", return_value=fake_post
    ):
        result = cs.test_provider("ollama")

    assert result["ok"] is False
    assert "HTTP 400" in result["error"]
    assert "llama3" in result["error"]


def test_ollama_probe_network_error_degrades_to_ok(tmp_path: Path) -> None:
    """When the probe POST hits a transport-level error (timeout,
    connection drop), don't mask the (already successful) reachability
    check - the user should still be able to save credentials and the
    tool-support check is best-effort."""
    import httpx

    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("ollama", model="llama3", base_url="http://localhost:11434")

    fake_get = MagicMock()
    fake_get.raise_for_status.return_value = None

    with patch("httpx.get", return_value=fake_get), patch(
        "httpx.post", side_effect=httpx.ConnectError("network down")
    ):
        result = cs.test_provider("ollama")

    assert result["ok"] is True


def test_ollama_probe_skipped_when_no_model_configured(tmp_path: Path) -> None:
    """O3: when no model is configured, the probe is skipped and the
    reachability check alone determines ok=True. Setup mid-flow must not
    fail just because the model picker has not been used yet."""
    cs = CredentialStore(path=tmp_path / "cred.json")
    # Provision ollama with no model.
    cs.set_provider("ollama", model="", base_url="http://localhost:11434")

    fake_get = MagicMock()
    fake_get.raise_for_status.return_value = None

    with patch("httpx.get", return_value=fake_get), patch(
        "httpx.post"
    ) as mock_post:
        result = cs.test_provider("ollama")

    assert result["ok"] is True
    mock_post.assert_not_called()
