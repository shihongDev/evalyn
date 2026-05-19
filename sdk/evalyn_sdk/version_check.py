"""Version management and update checking for evalyn.

Pure Python, no external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class VersionInfo:
    """Version check result."""

    current_version: str
    latest_version: str | None
    update_available: bool
    checked_at: str  # ISO 8601 timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "update_available": self.update_available,
            "checked_at": self.checked_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionInfo:
        return cls(
            current_version=data["current_version"],
            latest_version=data.get("latest_version"),
            update_available=data.get("update_available", False),
            checked_at=data.get("checked_at", ""),
        )


def get_current_version() -> str:
    """Read __version__ from evalyn_sdk.__init__."""
    from evalyn_sdk import __version__

    return __version__


def parse_version(version_str: str) -> tuple:
    """Split '1.2.3' into comparable tuple.

    Each segment becomes (0, int_val, "") for numeric parts or
    (1, 0, str_val) for non-numeric parts. This ensures numeric
    segments sort before string segments and avoids TypeError
    when comparing mixed int/str tuples in Python 3.
    """
    parts: list = []
    for segment in version_str.split("."):
        try:
            parts.append((0, int(segment), ""))
        except ValueError:
            parts.append((1, 0, segment))
    return tuple(parts)


def compare_versions(a: str, b: str) -> int:
    """Compare two version strings.

    Returns -1 if a < b, 0 if a == b, 1 if a > b.
    """
    pa = parse_version(a)
    pb = parse_version(b)
    if pa < pb:
        return -1
    if pa > pb:
        return 1
    return 0


def check_for_updates(timeout: float = 3.0) -> VersionInfo:
    """Check PyPI for the latest evalyn version.

    On any network error returns VersionInfo with latest_version=None
    and update_available=False.
    """
    current = get_current_version()
    checked_at = datetime.now(timezone.utc).isoformat()

    try:
        req = urllib.request.Request(
            "https://pypi.org/pypi/evalyn/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest = data.get("info", {}).get("version")
        if latest is None:
            return VersionInfo(
                current_version=current,
                latest_version=None,
                update_available=False,
                checked_at=checked_at,
            )
        update_available = compare_versions(current, latest) < 0
        return VersionInfo(
            current_version=current,
            latest_version=latest,
            update_available=update_available,
            checked_at=checked_at,
        )
    except Exception:
        return VersionInfo(
            current_version=current,
            latest_version=None,
            update_available=False,
            checked_at=checked_at,
        )


def format_update_message(info: VersionInfo) -> str:
    """Return a human-readable update message, or empty string if up-to-date."""
    if not info.update_available or info.latest_version is None:
        return ""
    return (
        f"Update available: {info.current_version} -> {info.latest_version}. "
        f"Run 'pip install --upgrade evalyn' to update."
    )


def should_check_updates(config_path: str, interval_hours: float = 24.0) -> bool:
    """Check if enough time has passed since the last update check.

    Reads a JSON cache file at config_path. Returns True if the file
    does not exist, is unreadable, or the last check was more than
    interval_hours ago.
    """
    try:
        with open(config_path) as f:
            data = json.load(f)
        last_check = data.get("last_check_timestamp")
        if last_check is None:
            return True
        elapsed = time.time() - float(last_check)
        return elapsed >= interval_hours * 3600
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return True


def save_check_timestamp(config_path: str) -> None:
    """Save the current time to the cache file at config_path."""
    data: dict[str, Any] = {}
    try:
        with open(config_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass

    data["last_check_timestamp"] = time.time()

    # Ensure parent directory exists
    parent = os.path.dirname(config_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
