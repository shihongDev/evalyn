"""
CLI execution audit log.

Log every CLI command invocation with full arguments, timing,
and exit status. Pure Python, no external dependencies.
"""

from __future__ import annotations

import getpass
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AuditConfig:
    """Configuration for the audit log."""

    log_path: str = ".evalyn/audit.jsonl"
    enabled: bool = True
    include_output: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_path": self.log_path,
            "enabled": self.enabled,
            "include_output": self.include_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditConfig:
        return cls(
            log_path=data.get("log_path", ".evalyn/audit.jsonl"),
            enabled=data.get("enabled", True),
            include_output=data.get("include_output", False),
        )


@dataclass
class AuditRecord:
    """A single audit log entry for one CLI invocation."""

    command: str
    args: list[str]
    timestamp: str
    exit_code: int
    duration_seconds: float
    user: str
    cwd: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": list(self.args),
            "timestamp": self.timestamp,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "user": self.user,
            "cwd": self.cwd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditRecord:
        return cls(
            command=data["command"],
            args=list(data.get("args", [])),
            timestamp=data["timestamp"],
            exit_code=data.get("exit_code", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            user=data.get("user", ""),
            cwd=data.get("cwd", ""),
        )


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def create_audit_record(
    command: str,
    args: list[str] | None = None,
    exit_code: int = 0,
    duration: float = 0.0,
) -> AuditRecord:
    """Factory that auto-fills timestamp, user, and cwd."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or getpass.getuser()
    except Exception:
        user = "unknown"
    cwd = os.getcwd()
    return AuditRecord(
        command=command,
        args=list(args or []),
        timestamp=ts,
        exit_code=exit_code,
        duration_seconds=duration,
        user=user,
        cwd=cwd,
    )


def append_audit(config: AuditConfig, record: AuditRecord) -> None:
    """Append a JSONL line to the audit log file.

    Creates parent directories if they do not exist.
    Does nothing when auditing is disabled.
    """
    if not config.enabled:
        return
    path = Path(config.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.as_dict()) + "\n")


def read_audit_log(log_path: str, limit: int = 100) -> list[AuditRecord]:
    """Read the last N records from a JSONL audit log.

    Returns an empty list if the file does not exist.
    Skips malformed lines silently.
    """
    path = Path(log_path)
    if not path.exists():
        return []
    records: list[AuditRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(AuditRecord.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
    # Return last N records
    if limit > 0 and len(records) > limit:
        records = records[-limit:]
    return records


def filter_audit(
    records: list[AuditRecord],
    command: str | None = None,
    after: str | None = None,
    exit_code: int | None = None,
) -> list[AuditRecord]:
    """Filter audit records by command name, timestamp, or exit code."""
    result = records
    if command is not None:
        result = [r for r in result if r.command == command]
    if after is not None:
        result = [r for r in result if r.timestamp > after]
    if exit_code is not None:
        result = [r for r in result if r.exit_code == exit_code]
    return result


def compute_audit_stats(records: list[AuditRecord]) -> dict[str, Any]:
    """Compute summary statistics from audit records.

    Returns dict with: total_commands, unique_commands, error_rate,
    most_used, avg_duration.
    """
    if not records:
        return {
            "total_commands": 0,
            "unique_commands": 0,
            "error_rate": 0.0,
            "most_used": "",
            "avg_duration": 0.0,
        }
    total = len(records)
    command_counts: dict[str, int] = {}
    error_count = 0
    total_duration = 0.0
    for r in records:
        command_counts[r.command] = command_counts.get(r.command, 0) + 1
        if r.exit_code != 0:
            error_count += 1
        total_duration += r.duration_seconds
    most_used = max(command_counts, key=command_counts.get)  # type: ignore[arg-type]
    return {
        "total_commands": total,
        "unique_commands": len(command_counts),
        "error_rate": error_count / total,
        "most_used": most_used,
        "avg_duration": total_duration / total,
    }


def format_audit_log(records: list[AuditRecord]) -> str:
    """Format audit records as a human-readable table."""
    if not records:
        return "No audit records."
    header = f"{'Timestamp':<28} {'Command':<16} {'Exit':<6} {'Duration':<10} {'Args'}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in records:
        args_str = " ".join(r.args) if r.args else ""
        lines.append(
            f"{r.timestamp:<28} {r.command:<16} {r.exit_code:<6} "
            f"{r.duration_seconds:<10.3f} {args_str}"
        )
    return "\n".join(lines)


def verify_audit_integrity(records: list[AuditRecord]) -> bool:
    """Check that timestamps are monotonically non-decreasing."""
    if len(records) <= 1:
        return True
    for i in range(1, len(records)):
        if records[i].timestamp < records[i - 1].timestamp:
            return False
    return True
