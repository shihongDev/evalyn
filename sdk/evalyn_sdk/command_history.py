"""CLI command history: record and replay command sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class HistoryEntry:
    """A single recorded command invocation."""

    command: str
    args: list[str]
    timestamp: str
    exit_code: int = 0
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "args": self.args,
            "timestamp": self.timestamp,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HistoryEntry:
        return cls(
            command=data["command"],
            args=data.get("args", []),
            timestamp=data["timestamp"],
            exit_code=data.get("exit_code", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


@dataclass
class HistoryLog:
    """A collection of history entries bound to a file path."""

    entries: list[HistoryEntry] = field(default_factory=list)
    log_path: str = ""

    def as_dict(self) -> dict:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "log_path": self.log_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HistoryLog:
        return cls(
            entries=[HistoryEntry.from_dict(e) for e in data.get("entries", [])],
            log_path=data.get("log_path", ""),
        )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def append_history(log_path: str, entry: HistoryEntry) -> None:
    """Append a history entry as a JSON line to the log file.

    Creates parent directories if they do not exist.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.as_dict()) + "\n")


def read_history(log_path: str, limit: int = 50) -> HistoryLog:
    """Read the last *limit* entries from the history file.

    Returns an empty log when the file does not exist.
    """
    path = Path(log_path)
    if not path.exists():
        return HistoryLog(entries=[], log_path=log_path)

    entries: list[HistoryEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(HistoryEntry.from_dict(json.loads(line)))

    # Keep only the last N entries
    if len(entries) > limit:
        entries = entries[-limit:]

    return HistoryLog(entries=entries, log_path=log_path)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_history(
    log: HistoryLog,
    command: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> HistoryLog:
    """Filter history entries by command name and/or date range.

    Args:
        log: The history log to filter.
        command: If set, keep only entries whose command matches.
        after: ISO timestamp - keep entries at or after this time.
        before: ISO timestamp - keep entries at or before this time.

    Returns:
        A new HistoryLog with the filtered entries.
    """
    filtered = log.entries

    if command is not None:
        filtered = [e for e in filtered if e.command == command]

    if after is not None:
        filtered = [e for e in filtered if e.timestamp >= after]

    if before is not None:
        filtered = [e for e in filtered if e.timestamp <= before]

    return HistoryLog(entries=filtered, log_path=log.log_path)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def get_replay_sequence(log: HistoryLog, from_timestamp: str) -> list[HistoryEntry]:
    """Return entries from the given timestamp onward (inclusive)."""
    return [e for e in log.entries if e.timestamp >= from_timestamp]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_history(log: HistoryLog) -> str:
    """Format history entries as a human-readable table.

    Returns an empty string when there are no entries.
    """
    if not log.entries:
        return ""

    lines: list[str] = []
    header = f"{'TIMESTAMP':<26}  {'COMMAND':<20}  {'EXIT':>4}  {'DURATION':>8}  ARGS"
    lines.append(header)
    lines.append("-" * len(header))
    for e in log.entries:
        args_str = " ".join(e.args) if e.args else ""
        dur = f"{e.duration_seconds:.1f}s"
        lines.append(
            f"{e.timestamp:<26}  {e.command:<20}  {e.exit_code:>4}  {dur:>8}  {args_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_history_entry(
    command: str,
    args: list[str],
    exit_code: int = 0,
    duration: float = 0.0,
) -> HistoryEntry:
    """Create a HistoryEntry with the current UTC timestamp."""
    ts = datetime.now(timezone.utc).isoformat()
    return HistoryEntry(
        command=command,
        args=args,
        timestamp=ts,
        exit_code=exit_code,
        duration_seconds=duration,
    )
