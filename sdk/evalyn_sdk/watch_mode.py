"""File-change watch mode for auto-rerunning evaluations.

Uses os.stat polling for change detection - no external dependencies.
Provides a callback-driven watch loop with debouncing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WatchConfig:
    """Configuration for the watch loop."""

    paths: List[str] = field(default_factory=list)
    debounce_seconds: float = 2.0
    poll_interval: float = 0.5

    def as_dict(self) -> Dict[str, Any]:
        return {
            "paths": list(self.paths),
            "debounce_seconds": self.debounce_seconds,
            "poll_interval": self.poll_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WatchConfig:
        return cls(
            paths=list(data.get("paths", [])),
            debounce_seconds=float(data.get("debounce_seconds", 2.0)),
            poll_interval=float(data.get("poll_interval", 0.5)),
        )


@dataclass
class FileState:
    """Snapshot of a single file's metadata."""

    path: str
    mtime: float
    size: int
    exists: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mtime": self.mtime,
            "size": self.size,
            "exists": self.exists,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileState:
        return cls(
            path=data["path"],
            mtime=float(data.get("mtime", 0.0)),
            size=int(data.get("size", 0)),
            exists=bool(data.get("exists", False)),
        )


@dataclass
class WatchEvent:
    """A detected file change event."""

    path: str
    event_type: str  # "modified", "created", or "deleted"
    timestamp: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WatchEvent:
        return cls(
            path=data["path"],
            event_type=data["event_type"],
            timestamp=data["timestamp"],
        )


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


class ChangeDetector:
    """Polls file metadata to detect modifications, creations, and deletions."""

    def __init__(self, paths: List[str]) -> None:
        self.paths = list(paths)

    def snapshot(self) -> Dict[str, FileState]:
        """Return the current FileState for every watched path."""
        result: Dict[str, FileState] = {}
        for p in self.paths:
            try:
                st = os.stat(p)
                result[p] = FileState(
                    path=p,
                    mtime=st.st_mtime,
                    size=st.st_size,
                    exists=True,
                )
            except OSError:
                result[p] = FileState(path=p, mtime=0.0, size=0, exists=False)
        return result

    @staticmethod
    def detect_changes(
        old: Dict[str, FileState],
        new: Dict[str, FileState],
    ) -> List[WatchEvent]:
        """Compare two snapshots and return a list of WatchEvents."""
        now_iso = datetime.now(timezone.utc).isoformat()
        events: List[WatchEvent] = []

        all_paths = sorted(set(old) | set(new))
        for p in all_paths:
            old_fs = old.get(p)
            new_fs = new.get(p)

            if old_fs is None and new_fs is not None and new_fs.exists:
                events.append(WatchEvent(path=p, event_type="created", timestamp=now_iso))
            elif new_fs is None and old_fs is not None and old_fs.exists:
                events.append(WatchEvent(path=p, event_type="deleted", timestamp=now_iso))
            elif old_fs is not None and new_fs is not None:
                if old_fs.exists and not new_fs.exists:
                    events.append(WatchEvent(path=p, event_type="deleted", timestamp=now_iso))
                elif not old_fs.exists and new_fs.exists:
                    events.append(WatchEvent(path=p, event_type="created", timestamp=now_iso))
                elif old_fs.exists and new_fs.exists:
                    if old_fs.mtime != new_fs.mtime or old_fs.size != new_fs.size:
                        events.append(WatchEvent(path=p, event_type="modified", timestamp=now_iso))

        return events

    @staticmethod
    def has_changes(
        old: Dict[str, FileState],
        new: Dict[str, FileState],
    ) -> bool:
        """Quick boolean check - are there any differences between snapshots?"""
        all_paths = set(old) | set(new)
        for p in all_paths:
            old_fs = old.get(p)
            new_fs = new.get(p)
            if old_fs is None or new_fs is None:
                return True
            if old_fs.exists != new_fs.exists:
                return True
            if old_fs.exists and new_fs.exists:
                if old_fs.mtime != new_fs.mtime or old_fs.size != new_fs.size:
                    return True
        return False


# ---------------------------------------------------------------------------
# Debouncing
# ---------------------------------------------------------------------------


def should_debounce(last_run_time: float, debounce: float) -> bool:
    """Return True if not enough time has passed since last_run_time."""
    return (time.time() - last_run_time) < debounce


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------


def run_watch_loop(
    paths: List[str],
    callback: Callable[[List[WatchEvent]], None],
    config: WatchConfig,
    max_iterations: int = 0,
) -> int:
    """Poll for file changes and invoke callback when changes are detected.

    Parameters
    ----------
    paths:
        File paths to watch.
    callback:
        Called with a list of WatchEvents whenever changes are detected.
    config:
        Watch configuration (debounce, poll interval).
    max_iterations:
        Number of poll cycles to run. 0 means run once (useful for testing).
        Use a negative value or a very large number for continuous watching.

    Returns
    -------
    int
        Total number of times *callback* was invoked.
    """
    detector = ChangeDetector(paths)
    old_snapshot = detector.snapshot()
    last_run_time = 0.0
    callbacks_invoked = 0
    iterations = 0

    if max_iterations == 0:
        max_iterations = 1

    while iterations < max_iterations:
        iterations += 1
        time.sleep(config.poll_interval)

        new_snapshot = detector.snapshot()
        events = detector.detect_changes(old_snapshot, new_snapshot)

        if events and not should_debounce(last_run_time, config.debounce_seconds):
            callback(events)
            callbacks_invoked += 1
            last_run_time = time.time()

        old_snapshot = new_snapshot

    return callbacks_invoked


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_watch_event(event: WatchEvent) -> str:
    """Return a human-readable description of a watch event."""
    labels = {
        "modified": "Modified",
        "created": "Created",
        "deleted": "Deleted",
    }
    label = labels.get(event.event_type, event.event_type)
    return f"[{event.timestamp}] {label}: {event.path}"
