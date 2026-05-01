"""Credential storage for evalyn-dashboard agent providers.

Stores per-provider API keys (and optional ``base_url`` for self-hosted
providers like Ollama) in ``~/.evalyn/credentials.json`` with mode 0600.

Public API:
- ``CredentialStore.set_provider`` - atomic write of a provider record.
- ``CredentialStore.get_provider`` - INTERNAL: returns full record incl key.
- ``CredentialStore.public_view`` - safe shape for HTTP responses.
- ``CredentialStore.set_active`` - choose default provider.

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
        record["added_at"] = existing.get("added_at") or _utcnow_iso()

        data["providers"][name] = record
        self._atomic_write(data)

    def get_provider(self, name: str) -> dict[str, Any] | None:
        """Internal: return the full provider record (may include api_key)."""
        data = self._load()
        rec = data["providers"].get(name)
        if rec is None:
            return None
        return dict(rec)

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
        and ``base_url`` (str | None) when set.
        """
        data = self._load()
        out_providers: dict[str, dict[str, Any]] = {}
        for name, rec in data["providers"].items():
            entry: dict[str, Any] = {
                "is_set": bool(rec.get("api_key")) or bool(rec.get("base_url")),
                "model": rec.get("model"),
                "added_at": rec.get("added_at"),
            }
            if "base_url" in rec:
                entry["base_url"] = rec["base_url"]
            out_providers[name] = entry
        return {"providers": out_providers, "active": data.get("active")}
