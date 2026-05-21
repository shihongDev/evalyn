"""Storage snapshot/restore: point-in-time snapshots for safe experimentation.

Provides functions to plan, list, find, and clean up snapshots, plus
metadata recording and hash computation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Snapshot:
    """Metadata for a single snapshot."""

    snapshot_id: str
    source_path: str = ""
    created_at: str = ""
    description: str = ""
    data_hash: str = ""
    size_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_path": self.source_path,
            "created_at": self.created_at,
            "description": self.description,
            "data_hash": self.data_hash,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Snapshot:
        return cls(
            snapshot_id=d.get("snapshot_id", ""),
            source_path=d.get("source_path", ""),
            created_at=d.get("created_at", ""),
            description=d.get("description", ""),
            data_hash=d.get("data_hash", ""),
            size_bytes=d.get("size_bytes", 0),
        )


@dataclass
class SnapshotConfig:
    """Configuration for snapshot management."""

    snapshot_dir: str = ".evalyn_snapshots"
    max_snapshots: int = 10
    auto_snapshot_before_calibration: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_dir": self.snapshot_dir,
            "max_snapshots": self.max_snapshots,
            "auto_snapshot_before_calibration": self.auto_snapshot_before_calibration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SnapshotConfig:
        return cls(
            snapshot_dir=d.get("snapshot_dir", ".evalyn_snapshots"),
            max_snapshots=d.get("max_snapshots", 10),
            auto_snapshot_before_calibration=d.get("auto_snapshot_before_calibration", True),
        )


@dataclass
class SnapshotReport:
    """Aggregated report of available snapshots."""

    snapshots: list[Snapshot] = field(default_factory=list)
    total_snapshots: int = 0
    total_size_bytes: int = 0
    oldest: str = ""
    newest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshots": [s.as_dict() for s in self.snapshots],
            "total_snapshots": self.total_snapshots,
            "total_size_bytes": self.total_size_bytes,
            "oldest": self.oldest,
            "newest": self.newest,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SnapshotReport:
        snapshots = [Snapshot.from_dict(s) for s in d.get("snapshots", [])]
        return cls(
            snapshots=snapshots,
            total_snapshots=d.get("total_snapshots", 0),
            total_size_bytes=d.get("total_size_bytes", 0),
            oldest=d.get("oldest", ""),
            newest=d.get("newest", ""),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Snapshot Report")
        lines.append("-" * 40)
        lines.append(f"Total snapshots: {self.total_snapshots}")
        lines.append(f"Total size: {self.total_size_bytes} bytes")
        if self.oldest:
            lines.append(f"Oldest: {self.oldest}")
        if self.newest:
            lines.append(f"Newest: {self.newest}")
        if self.snapshots:
            lines.append("")
            for snap in self.snapshots:
                desc = f" - {snap.description}" if snap.description else ""
                lines.append(f"  {snap.snapshot_id}{desc}")
        return "\n".join(lines)


def generate_snapshot_id(description: str = "") -> str:
    """Generate a timestamp-based snapshot ID with optional description suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if description:
        # Sanitize: lowercase, replace spaces with underscores, keep alphanumeric and underscores
        suffix = "".join(
            c if c.isalnum() or c == "_" else "_" for c in description.lower().replace(" ", "_")
        )
        # Trim to reasonable length
        suffix = suffix[:30].rstrip("_")
        return f"{ts}_{suffix}"
    return ts


def compute_snapshot_hash(data: str) -> str:
    """Compute SHA-256 hash of data, returning first 16 hex characters."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def create_snapshot(
    source_path: str,
    config: SnapshotConfig,
    description: str = "",
) -> Snapshot:
    """Plan a snapshot by recording metadata without actual file copy."""
    snapshot_id = generate_snapshot_id(description)
    created_at = datetime.now(timezone.utc).isoformat()

    return Snapshot(
        snapshot_id=snapshot_id,
        source_path=source_path,
        created_at=created_at,
        description=description,
    )


def list_snapshots(config: SnapshotConfig) -> list[Snapshot]:
    """List available snapshots. Returns empty list (no file I/O)."""
    return []


def find_snapshot(snapshots: list[Snapshot], snapshot_id: str) -> Snapshot | None:
    """Find a snapshot by ID. Returns None if not found."""
    for snap in snapshots:
        if snap.snapshot_id == snapshot_id:
            return snap
    return None


def cleanup_snapshots(
    snapshots: list[Snapshot], max_keep: int = 10
) -> tuple[list[Snapshot], list[Snapshot]]:
    """Split snapshots into (keep, remove). Keep the newest max_keep."""
    # Sort by created_at descending (newest first)
    sorted_snaps = sorted(snapshots, key=lambda s: s.created_at, reverse=True)
    keep = sorted_snaps[:max_keep]
    remove = sorted_snaps[max_keep:]
    return keep, remove


def build_snapshot_report(snapshots: list[Snapshot]) -> SnapshotReport:
    """Build an aggregate report from a list of snapshots."""
    if not snapshots:
        return SnapshotReport()

    total_size = sum(s.size_bytes for s in snapshots)
    timestamps = [s.created_at for s in snapshots if s.created_at]
    oldest = min(timestamps) if timestamps else ""
    newest = max(timestamps) if timestamps else ""

    return SnapshotReport(
        snapshots=list(snapshots),
        total_snapshots=len(snapshots),
        total_size_bytes=total_size,
        oldest=oldest,
        newest=newest,
    )
