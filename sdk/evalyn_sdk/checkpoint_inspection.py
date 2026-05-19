"""View and manage evaluation checkpoints.

Pure Python, no external dependencies. Scans checkpoint JSON files,
reports progress, and supports cleanup of stale checkpoints.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint file."""

    path: str
    run_id: str
    processed_count: int
    total_count: int
    created_at: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "run_id": self.run_id,
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointInfo:
        return cls(
            path=data["path"],
            run_id=data.get("run_id", ""),
            processed_count=data.get("processed_count", 0),
            total_count=data.get("total_count", 0),
            created_at=data.get("created_at", ""),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class CheckpointSummary:
    """Aggregated summary of all discovered checkpoints."""

    checkpoints: list[CheckpointInfo] = field(default_factory=list)
    total_count: int = 0
    total_size_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoints": [c.as_dict() for c in self.checkpoints],
            "total_count": self.total_count,
            "total_size_bytes": self.total_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointSummary:
        checkpoints = [
            CheckpointInfo.from_dict(c) for c in data.get("checkpoints", [])
        ]
        return cls(
            checkpoints=checkpoints,
            total_count=data.get("total_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
        )


def scan_checkpoints(data_dir: str) -> CheckpointSummary:
    """Find all checkpoint*.json files under data_dir and parse basic info.

    Walks the directory tree, reading each matching file to extract
    run_id, processed_count, and total_count from the JSON content.
    """
    checkpoints: list[CheckpointInfo] = []
    total_size = 0

    if not os.path.isdir(data_dir):
        return CheckpointSummary()

    for dirpath, _dirnames, filenames in os.walk(data_dir):
        for fname in sorted(filenames):
            if fname.startswith("checkpoint") and fname.endswith(".json"):
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                    with open(full_path) as f:
                        content = json.load(f)
                    info = CheckpointInfo(
                        path=full_path,
                        run_id=content.get("run_id", ""),
                        processed_count=content.get("processed_count", 0),
                        total_count=content.get("total_count", 0),
                        created_at=content.get("created_at", ""),
                        size_bytes=size,
                    )
                    checkpoints.append(info)
                    total_size += size
                except (json.JSONDecodeError, OSError):
                    continue

    return CheckpointSummary(
        checkpoints=checkpoints,
        total_count=len(checkpoints),
        total_size_bytes=total_size,
    )


def read_checkpoint(path: str) -> dict:
    """Read and parse a checkpoint JSON file."""
    with open(path) as f:
        return json.load(f)


def get_checkpoint_progress(info: CheckpointInfo) -> float:
    """Return progress as a 0-1 fraction (processed / total).

    Returns 0.0 if total_count is zero.
    """
    if info.total_count <= 0:
        return 0.0
    return info.processed_count / info.total_count


def delete_checkpoint(path: str) -> bool:
    """Delete a checkpoint file. Returns True if successful."""
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def delete_stale_checkpoints(
    summary: CheckpointSummary, max_age_days: float = 7.0
) -> int:
    """Delete checkpoints older than max_age_days. Returns count deleted."""
    now = _now_utc()
    deleted = 0
    for info in summary.checkpoints:
        if not info.created_at:
            continue
        try:
            created = datetime.fromisoformat(info.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).total_seconds() / 86400.0
            if age_days > max_age_days:
                if delete_checkpoint(info.path):
                    deleted += 1
        except (ValueError, TypeError):
            continue
    return deleted


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_age(created_at: str) -> str:
    """Format age relative to now."""
    if not created_at:
        return "unknown"
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_seconds = (_now_utc() - created).total_seconds()
        if age_seconds < 60:
            return f"{int(age_seconds)}s ago"
        if age_seconds < 3600:
            return f"{int(age_seconds / 60)}m ago"
        if age_seconds < 86400:
            return f"{age_seconds / 3600:.1f}h ago"
        return f"{age_seconds / 86400:.1f}d ago"
    except (ValueError, TypeError):
        return "unknown"


def format_checkpoint_list(summary: CheckpointSummary) -> str:
    """Human-readable table of checkpoints with path, progress, age, size."""
    if not summary.checkpoints:
        return "No checkpoints found."

    lines: list[str] = []
    lines.append(
        f"Checkpoints: {summary.total_count} "
        f"(total size: {_format_size(summary.total_size_bytes)})"
    )
    lines.append("")
    header = f"{'Path':<40} {'Progress':>10} {'Age':>12} {'Size':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for info in summary.checkpoints:
        progress = get_checkpoint_progress(info)
        pct = f"{progress * 100:.0f}%"
        progress_str = f"{info.processed_count}/{info.total_count} ({pct})"
        age_str = _format_age(info.created_at)
        size_str = _format_size(info.size_bytes)
        path_display = info.path
        if len(path_display) > 40:
            path_display = "..." + path_display[-37:]
        lines.append(
            f"{path_display:<40} {progress_str:>10} {age_str:>12} {size_str:>10}"
        )

    return "\n".join(lines)


def format_checkpoint_detail(info: CheckpointInfo) -> str:
    """Detailed view of a single checkpoint."""
    progress = get_checkpoint_progress(info)
    lines = [
        f"Checkpoint: {info.path}",
        f"  Run ID:    {info.run_id}",
        f"  Progress:  {info.processed_count}/{info.total_count} ({progress * 100:.0f}%)",
        f"  Created:   {info.created_at or 'unknown'}",
        f"  Age:       {_format_age(info.created_at)}",
        f"  Size:      {_format_size(info.size_bytes)}",
    ]
    return "\n".join(lines)
