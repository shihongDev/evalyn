"""Storage read-only mode: prevent accidental writes during analysis.

Provides global read-only state with toggle functions, a context manager
for scoped read-only sections, and history tracking of state changes.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

_READ_ONLY: bool = False
_READ_ONLY_HISTORY: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ReadOnlyStatus:
    """Current read-only mode status."""

    enabled: bool
    since: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "since": self.since,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def enable_read_only(reason: str = "") -> ReadOnlyStatus:
    """Enable read-only mode."""
    global _READ_ONLY
    _READ_ONLY = True
    ts = _now_iso()
    _READ_ONLY_HISTORY.append(
        {"action": "enable", "timestamp": ts, "reason": reason}
    )
    return ReadOnlyStatus(enabled=True, since=ts, reason=reason)


def disable_read_only() -> ReadOnlyStatus:
    """Disable read-only mode."""
    global _READ_ONLY
    _READ_ONLY = False
    ts = _now_iso()
    _READ_ONLY_HISTORY.append({"action": "disable", "timestamp": ts})
    return ReadOnlyStatus(enabled=False)


def is_read_only() -> bool:
    """Check whether read-only mode is currently active."""
    return _READ_ONLY


def get_status() -> ReadOnlyStatus:
    """Get detailed read-only status including last toggle info."""
    if not _READ_ONLY:
        return ReadOnlyStatus(enabled=False)
    # Find the most recent enable entry for since/reason
    for entry in reversed(_READ_ONLY_HISTORY):
        if entry["action"] == "enable":
            return ReadOnlyStatus(
                enabled=True,
                since=entry["timestamp"],
                reason=entry.get("reason", ""),
            )
    return ReadOnlyStatus(enabled=True)


def check_write_allowed() -> tuple[bool, str]:
    """Return (allowed, reason) for use before write operations."""
    if _READ_ONLY:
        status = get_status()
        return (False, f"read-only since {status.since}: {status.reason}")
    return (True, "")


def get_history() -> list[dict[str, Any]]:
    """Return the full toggle history."""
    return list(_READ_ONLY_HISTORY)


def reset_read_only() -> None:
    """Reset to default (writable) state and clear history."""
    global _READ_ONLY
    _READ_ONLY = False
    _READ_ONLY_HISTORY.clear()


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


@contextmanager
def read_only_context(
    reason: str = "",
) -> Generator[None, None, None]:
    """Context manager that enables read-only on enter, restores previous state on exit."""
    previous = _READ_ONLY
    enable_read_only(reason=reason)
    try:
        yield
    finally:
        if previous:
            enable_read_only(reason="restored")
        else:
            disable_read_only()
