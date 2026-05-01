"""Tests for ``evalyn_dashboard.credentials.CredentialStore`` (A4.1)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from evalyn_dashboard.credentials import CredentialStore


def test_fresh_path_public_view_is_empty(tmp_path: Path) -> None:
    cs = CredentialStore(path=tmp_path / "cred.json")
    view = cs.public_view()
    assert view == {"providers": {}, "active": None}


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
    assert cs.public_view() == {"providers": {}, "active": None}
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
