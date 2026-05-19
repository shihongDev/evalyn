"""
Incremental backup: periodic automatic backup of database to a secondary location.

Provides pure functions to generate backup IDs, plan backups, list and clean up
old backups, and build summary reports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BackupConfig:
    """Configuration for incremental backups."""

    source_path: str = ""
    backup_dir: str = ".evalyn_backups"
    max_backups: int = 5
    interval_minutes: float = 60.0
    compress: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "backup_dir": self.backup_dir,
            "max_backups": self.max_backups,
            "interval_minutes": self.interval_minutes,
            "compress": self.compress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupConfig:
        return cls(
            source_path=data.get("source_path", ""),
            backup_dir=data.get("backup_dir", ".evalyn_backups"),
            max_backups=data.get("max_backups", 5),
            interval_minutes=data.get("interval_minutes", 60.0),
            compress=data.get("compress", True),
        )


@dataclass
class BackupRecord:
    """A single backup entry."""

    backup_id: str = ""
    source_path: str = ""
    backup_path: str = ""
    timestamp: str = ""
    size_bytes: int = 0
    compressed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupRecord:
        return cls(
            backup_id=data.get("backup_id", ""),
            source_path=data.get("source_path", ""),
            backup_path=data.get("backup_path", ""),
            timestamp=data.get("timestamp", ""),
            size_bytes=data.get("size_bytes", 0),
            compressed=data.get("compressed", False),
        )


@dataclass
class BackupReport:
    """Summary report of all backups in a directory."""

    backups: list[BackupRecord] = field(default_factory=list)
    total_backups: int = 0
    total_size_bytes: int = 0
    oldest_backup: str = ""
    newest_backup: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "backups": [b.as_dict() for b in self.backups],
            "total_backups": self.total_backups,
            "total_size_bytes": self.total_size_bytes,
            "oldest_backup": self.oldest_backup,
            "newest_backup": self.newest_backup,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupReport:
        return cls(
            backups=[BackupRecord.from_dict(b) for b in data.get("backups", [])],
            total_backups=data.get("total_backups", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            oldest_backup=data.get("oldest_backup", ""),
            newest_backup=data.get("newest_backup", ""),
        )

    def format_text(self) -> str:
        lines = [
            "Backup Report",
            f"  total_backups: {self.total_backups}",
            f"  total_size_bytes: {self.total_size_bytes}",
            f"  oldest_backup: {self.oldest_backup}",
            f"  newest_backup: {self.newest_backup}",
        ]
        for rec in self.backups:
            lines.append(f"  - {rec.backup_id} ({rec.size_bytes} bytes)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def generate_backup_id() -> str:
    """Generate a timestamp-based backup ID like 'backup_20260328_143000'."""
    now = datetime.now(timezone.utc)
    return now.strftime("backup_%Y%m%d_%H%M%S")


def create_backup_path(config: BackupConfig, backup_id: str) -> str:
    """Generate full backup file path from config and backup ID."""
    ext = ".db.gz" if config.compress else ".db"
    filename = backup_id + ext
    return os.path.join(config.backup_dir, filename)


def should_backup(
    config: BackupConfig, last_backup_time: str | None = None
) -> bool:
    """Return True if enough time has passed since last backup.

    If last_backup_time is None, a backup is always needed.
    """
    if last_backup_time is None:
        return True
    try:
        last_dt = datetime.fromisoformat(last_backup_time)
    except (ValueError, TypeError):
        return True
    now = datetime.now(timezone.utc)
    # Ensure last_dt is timezone-aware for comparison
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    elapsed_minutes = (now - last_dt).total_seconds() / 60.0
    return elapsed_minutes >= config.interval_minutes


def plan_backup(config: BackupConfig) -> BackupRecord:
    """Plan a backup (create a record without actually copying files)."""
    backup_id = generate_backup_id()
    backup_path = create_backup_path(config, backup_id)
    now = datetime.now(timezone.utc).isoformat()
    return BackupRecord(
        backup_id=backup_id,
        source_path=config.source_path,
        backup_path=backup_path,
        timestamp=now,
        size_bytes=0,
        compressed=config.compress,
    )


def list_backups(backup_dir: str) -> list[BackupRecord]:
    """List existing backup records in a directory.

    Scans the directory for files matching the backup naming pattern
    and returns BackupRecord objects sorted by filename (oldest first).
    """
    if not os.path.isdir(backup_dir):
        return []
    records: list[BackupRecord] = []
    for name in sorted(os.listdir(backup_dir)):
        if not name.startswith("backup_"):
            continue
        path = os.path.join(backup_dir, name)
        if not os.path.isfile(path):
            continue
        compressed = name.endswith(".gz")
        # Extract backup_id by stripping extensions
        backup_id = name
        for suffix in (".db.gz", ".db"):
            if backup_id.endswith(suffix):
                backup_id = backup_id[: -len(suffix)]
                break
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        records.append(
            BackupRecord(
                backup_id=backup_id,
                source_path="",
                backup_path=path,
                timestamp="",
                size_bytes=size,
                compressed=compressed,
            )
        )
    return records


def cleanup_old_backups(
    records: list[BackupRecord], max_backups: int = 5
) -> list[BackupRecord]:
    """Return records to keep (newest max_backups).

    Assumes records are ordered oldest-first. Keeps the last max_backups items.
    """
    if max_backups <= 0:
        return []
    return records[-max_backups:]


def build_backup_report(records: list[BackupRecord]) -> BackupReport:
    """Build an aggregate report from a list of backup records."""
    if not records:
        return BackupReport()
    total_size = sum(r.size_bytes for r in records)
    timestamps = [r.timestamp for r in records if r.timestamp]
    oldest = min(timestamps) if timestamps else ""
    newest = max(timestamps) if timestamps else ""
    return BackupReport(
        backups=list(records),
        total_backups=len(records),
        total_size_bytes=total_size,
        oldest_backup=oldest,
        newest_backup=newest,
    )
