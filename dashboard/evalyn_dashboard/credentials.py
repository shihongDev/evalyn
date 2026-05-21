"""Credential storage for evalyn-dashboard agent providers.

Stores per-provider API keys (and optional ``base_url`` for self-hosted
providers like Ollama) in ``~/.evalyn/credentials.json`` with mode 0600.

Public API:
- ``CredentialStore.set_provider`` - atomic write of a provider record.
- ``CredentialStore.get_provider`` - INTERNAL: returns full record incl key.
- ``CredentialStore.public_view`` - safe shape for HTTP responses.
- ``CredentialStore.set_active`` - choose default provider.
- ``CredentialStore.test_provider`` - 1-token probe to verify credentials.

The on-disk schema:

    {
      "providers": {
        "openai":    {"api_key": "...", "model": "...", "added_at": "..."},
        "anthropic": {"api_key": "...", "model": "...", "added_at": "..."},
        "ollama":    {"base_url": "...", "model": "...", "added_at": "..."}
      },
      "active": "openai" | null
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _default_path() -> Path:
    return Path.home() / ".evalyn" / "credentials.json"


def _utcnow_iso() -> str:
    # Z-suffixed ISO-8601 for portability.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class CredentialStore:
    """File-backed credential store with atomic writes and 0600 mode."""

    path: Path | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path) if self.path else _default_path()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Load the on-disk file, returning an empty store on any error."""
        assert self.path is not None  # for type checker
        if not self.path.exists():
            return {"providers": {}, "active": None}
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("credentials file unreadable, treating as empty: %s", exc)
            return {"providers": {}, "active": None}
        # Tolerate older / malformed shapes by coercing.
        if not isinstance(data, dict):
            return {"providers": {}, "active": None}
        providers = data.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        active = data.get("active")
        if active is not None and not isinstance(active, str):
            active = None
        return {"providers": providers, "active": active}

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, data: dict[str, Any]) -> None:
        """Atomically write JSON to ``self.path`` with mode 0600.

        Strategy: temp file in same directory, fsync, ``os.rename`` (atomic on
        POSIX), then ``os.chmod(target, 0o600)``. Order matters: chmod after
        rename so the final inode carries the correct mode regardless of any
        umask interaction during the temp-file creation.
        """
        assert self.path is not None
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")

        # Open with restrictive mode up-front; we still chmod after rename
        # to be defensive across platforms.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(tmp_path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # fsync can fail on some filesystems; the rename is still
                    # atomic on POSIX.
                    pass
        except Exception:
            # Best-effort cleanup; ignore secondary errors.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        os.replace(tmp_path, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            # Non-POSIX filesystems may not support chmod; tolerate.
            logger.debug("chmod 600 failed on %s", target, exc_info=True)

    # ------------------------------------------------------------------
    # Provider records
    # ------------------------------------------------------------------

    def set_provider(
        self,
        name: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create or update a provider record. Partial updates preserve fields."""
        data = self._load()
        existing = data["providers"].get(name, {})
        record: dict[str, Any] = dict(existing)

        if api_key is not None:
            record["api_key"] = api_key
        if model is not None:
            record["model"] = model
        if base_url is not None:
            record["base_url"] = base_url
        # Capture the timestamp once so a fresh record gets the SAME
        # added_at and updated_at (two _utcnow_iso() calls would
        # differ by microseconds, which is harmless but surprises
        # anyone asserting equality).
        now_iso = _utcnow_iso()
        record["added_at"] = existing.get("added_at") or now_iso
        # Bump on every write so a future "rotated N days ago" UI can
        # distinguish "first set" from "last touched". added_at is the
        # creation moment; updated_at advances on each set_provider.
        record["updated_at"] = now_iso

        data["providers"][name] = record
        self._atomic_write(data)

    def get_provider(self, name: str) -> dict[str, Any] | None:
        """Internal: return the full provider record (may include api_key)."""
        data = self._load()
        rec = data["providers"].get(name)
        if rec is None:
            return None
        return dict(rec)

    def remove_provider(self, name: str) -> bool:
        """Drop a provider's stored credentials.

        Returns True if a record was removed, False on miss. If the
        removed provider was the active one, ``active`` is cleared
        too so the dashboard does not point at a now-missing record
        (which would surface as a 4xx on the next agent turn).
        Idempotent: a second remove for the same name is a no-op +
        False.
        """
        data = self._load()
        if name not in data["providers"]:
            return False
        del data["providers"][name]
        if data.get("active") == name:
            # Clear active rather than picking a random replacement;
            # the user will pick the next provider explicitly.
            data["active"] = ""
        self._atomic_write(data)
        return True

    def set_active(self, name: str) -> None:
        """Mark ``name`` as the active provider. Must already be configured."""
        data = self._load()
        if name not in data["providers"]:
            raise ValueError(f"provider {name!r} not configured")
        data["active"] = name
        self._atomic_write(data)

    def public_view(self) -> dict[str, Any]:
        """Return a redacted view safe to expose over HTTP.

        Never includes plaintext ``api_key``. Each provider entry exposes:
        ``is_set`` (bool), ``model`` (str | None), ``added_at`` (str | None),
        ``updated_at`` (str | None - present when the record has been
        written at least once via set_provider; useful for "rotated N
        days ago" displays), and ``base_url`` (str | None) when set.

        Ollama is a local provider with a sensible default base_url
        (``DEFAULT_OLLAMA_BASE_URL``); we surface it as ``is_set=True``
        even when no record has been saved so the FE Test connection
        button can probe localhost without forcing the user to Save an
        empty form first.
        """
        data = self._load()
        providers = dict(data["providers"])
        if "ollama" not in providers:
            providers["ollama"] = {}
        out_providers: dict[str, dict[str, Any]] = {}
        for name, rec in providers.items():
            is_set = bool(rec.get("api_key")) or bool(rec.get("base_url")) or name == "ollama"
            entry: dict[str, Any] = {
                "is_set": is_set,
                "model": rec.get("model"),
                "added_at": rec.get("added_at"),
                "updated_at": rec.get("updated_at"),
            }
            if "base_url" in rec:
                entry["base_url"] = rec["base_url"]
            out_providers[name] = entry
        return {"providers": out_providers, "active": data.get("active")}

    # ------------------------------------------------------------------
    # Connection probe
    # ------------------------------------------------------------------

    def test_provider(self, name: str) -> dict[str, Any]:
        """Attempt a minimal 1-token call against ``name``.

        Returns ``{"ok": True}`` on success or ``{"ok": False, "error": str}``
        on any failure. Imports provider SDKs lazily so the dashboard works
        even if optional dependencies are missing.

        Ollama with no saved record is allowed; ``_test_ollama`` falls
        back to ``DEFAULT_OLLAMA_BASE_URL`` so a fresh install can probe
        localhost without forcing a Save first. Cloud providers still
        require a saved api_key.
        """
        rec = self.get_provider(name)
        if rec is None:
            if name == "ollama":
                rec = {}
            else:
                return {"ok": False, "error": f"provider {name!r} not configured"}

        try:
            if name == "openai":
                return self._test_openai(rec)
            if name == "anthropic":
                return self._test_anthropic(rec)
            if name == "ollama":
                return self._test_ollama(rec)
            return {"ok": False, "error": f"unknown provider {name!r}"}
        except Exception as exc:  # noqa: BLE001 - surface any error verbatim
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _test_openai(rec: dict[str, Any]) -> dict[str, Any]:
        import openai  # type: ignore[import-not-found]

        api_key = rec.get("api_key")
        model = rec.get("model")
        if not api_key or not model:
            return {"ok": False, "error": "missing api_key or model"}
        client = openai.OpenAI(api_key=api_key)
        client.chat.completions.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"ok": True}

    @staticmethod
    def _test_anthropic(rec: dict[str, Any]) -> dict[str, Any]:
        import anthropic  # type: ignore[import-not-found]

        api_key = rec.get("api_key")
        model = rec.get("model")
        if not api_key or not model:
            return {"ok": False, "error": "missing api_key or model"}
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"ok": True}

    @staticmethod
    def _test_ollama(rec: dict[str, Any]) -> dict[str, Any]:
        import httpx

        base_url = (rec.get("base_url") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        # Step 1: server reachability via /api/tags.
        response = httpx.get(base_url + "/api/tags", timeout=5.0)
        response.raise_for_status()

        # O3 fix (part 1): if a model is configured, probe whether it
        # actually supports tool calling. Many Ollama models silently
        # ignore the `tools` payload and reply with free text; the user
        # would otherwise believe the agent is wired up while no tool
        # ever runs. We send a short /api/chat call with a clear,
        # tool-motivated prompt and check whether the response surfaces
        # a `tool_calls` field. If no model is configured yet, skip this
        # step: the user is mid-setup and "no model" is not a failure.
        #
        # Probe design notes (refined after live testing against
        # llama3 / gpt-oss:20b):
        #   - prompt must clearly motivate tool use, otherwise even
        #     tool-capable models reply with text only.
        #   - num_predict must be large enough to emit a tool_call shape;
        #     1 token is not enough.
        #   - parameters: required arg + clear description so the model
        #     has a reason to invoke the tool.
        model = rec.get("model")
        if not model:
            return {"ok": True}

        probe_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": ("Call the echo tool with text=hello to verify tool calling works."),
                }
            ],
            "stream": False,
            "options": {"num_predict": 64},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Echo the provided text back to the caller.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The text to echo.",
                                }
                            },
                            "required": ["text"],
                        },
                    },
                }
            ],
        }
        # Distinguish between network-level failures (treat as best-effort
        # ok=True so users can still save credentials when the daemon is
        # flaky) and protocol-level rejections (4xx/5xx, treat as evidence
        # the model does not support tools - this is THE failure mode O3
        # is meant to catch). Live testing showed Ollama returns HTTP 400
        # for models that do not accept the `tools` field, which the old
        # blanket-catch was masking as ok=True.
        try:
            probe = httpx.post(base_url + "/api/chat", json=probe_payload, timeout=15.0)
        except httpx.RequestError as exc:
            logger.debug("ollama tool-probe network failure: %s", exc)
            return {"ok": True}

        if probe.status_code >= 400:
            return {
                "ok": False,
                "error": (
                    f"Model {model} does not appear to support tool calling "
                    f"(Ollama returned HTTP {probe.status_code}). "
                    "Try qwen2.5, llama3.1:8b, or mistral."
                ),
            }

        try:
            body = probe.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ollama tool-probe response not JSON: %s", exc)
            return {"ok": True}

        message = (body.get("message") or {}) if isinstance(body, dict) else {}
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not tool_calls:
            return {
                "ok": False,
                "error": (
                    f"Model {model} does not appear to support tool calling. "
                    "Try qwen2.5, llama3.1:8b, or mistral."
                ),
            }
        return {"ok": True}
