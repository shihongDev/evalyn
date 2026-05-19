"""Immutable append-only audit log for evaluation actions.

Pure Python, no external dependencies. Provides deterministic config
hashing and JSONL-based persistent storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _get_evalyn_version() -> str:
    """Return the installed evalyn version string."""
    try:
        from evalyn_sdk import __version__

        return __version__
    except Exception:
        return "unknown"


def _get_current_user() -> str:
    """Auto-detect current user from environment."""
    return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of a config dict.

    Serializes with sorted keys for determinism.
    """
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    """A single audit log entry."""

    entry_id: str
    user: str
    timestamp: str
    command: str
    args: Dict[str, Any]
    config_hash: str
    result_summary: str
    evalyn_version: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "user": self.user,
            "timestamp": self.timestamp,
            "command": self.command,
            "args": dict(self.args),
            "config_hash": self.config_hash,
            "result_summary": self.result_summary,
            "evalyn_version": self.evalyn_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditEntry:
        return cls(
            entry_id=data["entry_id"],
            user=data["user"],
            timestamp=data["timestamp"],
            command=data["command"],
            args=dict(data.get("args", {})),
            config_hash=data.get("config_hash", ""),
            result_summary=data.get("result_summary", ""),
            evalyn_version=data.get("evalyn_version", "unknown"),
        )


@dataclass
class AuditLog:
    """Collection of audit entries with associated log path."""

    entries: List[AuditEntry] = field(default_factory=list)
    log_path: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "log_path": self.log_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditLog:
        return cls(
            entries=[
                AuditEntry.from_dict(e) for e in data.get("entries", [])
            ],
            log_path=data.get("log_path", ""),
        )


def create_audit_entry(
    command: str,
    args: Dict[str, Any],
    config: Dict[str, Any],
    result_summary: str = "",
    user: str = "",
) -> AuditEntry:
    """Create an audit entry with auto-generated fields.

    Auto-generates UUID, timestamp, config hash, user detection,
    and evalyn version.
    """
    return AuditEntry(
        entry_id=str(uuid.uuid4()),
        user=user or _get_current_user(),
        timestamp=_now_iso(),
        command=command,
        args=dict(args),
        config_hash=compute_config_hash(config),
        result_summary=result_summary,
        evalyn_version=_get_evalyn_version(),
    )


def append_entry(log_path: str, entry: AuditEntry) -> None:
    """Append a single audit entry as a JSON line to the log file.

    Creates parent directories if they do not exist.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.as_dict(), separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def read_audit_log(log_path: str) -> AuditLog:
    """Read all entries from a JSONL audit log file.

    Returns an empty AuditLog if the file is missing or empty.
    """
    path = Path(log_path)
    entries: List[AuditEntry] = []
    if not path.exists():
        return AuditLog(entries=entries, log_path=log_path)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(AuditEntry.from_dict(data))

    return AuditLog(entries=entries, log_path=log_path)


def filter_entries(
    log: AuditLog,
    command: str | None = None,
    user: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> AuditLog:
    """Filter audit log entries by command, user, and/or date range.

    Date range parameters (after, before) are ISO format strings.
    """
    filtered: List[AuditEntry] = []
    after_dt = datetime.fromisoformat(after) if after else None
    before_dt = datetime.fromisoformat(before) if before else None
    for entry in log.entries:
        if command is not None and entry.command != command:
            continue
        if user is not None and entry.user != user:
            continue
        if after_dt is not None and datetime.fromisoformat(entry.timestamp) < after_dt:
            continue
        if before_dt is not None and datetime.fromisoformat(entry.timestamp) > before_dt:
            continue
        filtered.append(entry)
    return AuditLog(entries=filtered, log_path=log.log_path)


def format_audit_log(log: AuditLog, limit: int = 50) -> str:
    """Format audit log entries as a human-readable table.

    Shows the most recent entries up to the limit.
    """
    entries = log.entries[-limit:]
    if not entries:
        return "No audit entries found."

    # Column widths
    id_w = 8
    user_w = 12
    cmd_w = 20
    time_w = 25
    summary_w = 30

    header = (
        f"{'ID':<{id_w}} "
        f"{'User':<{user_w}} "
        f"{'Command':<{cmd_w}} "
        f"{'Timestamp':<{time_w}} "
        f"{'Summary':<{summary_w}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for entry in entries:
        short_id = entry.entry_id[:8]
        user_col = entry.user[:user_w]
        cmd_col = entry.command[:cmd_w]
        time_col = entry.timestamp[:time_w]
        summary_col = entry.result_summary[:summary_w]
        lines.append(
            f"{short_id:<{id_w}} "
            f"{user_col:<{user_w}} "
            f"{cmd_col:<{cmd_w}} "
            f"{time_col:<{time_w}} "
            f"{summary_col:<{summary_w}}"
        )

    return "\n".join(lines)


def get_audit_stats(log: AuditLog) -> Dict[str, Any]:
    """Compute summary statistics for an audit log.

    Returns dict with: total_entries, unique_users, unique_commands,
    date_range, most_used_command.
    """
    if not log.entries:
        return {
            "total_entries": 0,
            "unique_users": 0,
            "unique_commands": 0,
            "date_range": {"earliest": None, "latest": None},
            "most_used_command": None,
        }

    users = set()
    commands: Dict[str, int] = {}
    timestamps: List[str] = []

    for entry in log.entries:
        users.add(entry.user)
        commands[entry.command] = commands.get(entry.command, 0) + 1
        timestamps.append(entry.timestamp)

    timestamps.sort()
    most_used = max(commands, key=lambda k: commands[k])

    return {
        "total_entries": len(log.entries),
        "unique_users": len(users),
        "unique_commands": len(commands),
        "date_range": {
            "earliest": timestamps[0],
            "latest": timestamps[-1],
        },
        "most_used_command": most_used,
    }
