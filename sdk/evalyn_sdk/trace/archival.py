from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        return datetime.fromisoformat(raw)
    return datetime.now(timezone.utc)


@dataclass
class ArchivalConfig:
    """Configuration for trace archival."""

    max_age_days: int = 90
    archive_path: str = "archive"
    dry_run: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_age_days": self.max_age_days,
            "archive_path": self.archive_path,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArchivalConfig:
        return cls(
            max_age_days=data.get("max_age_days", 90),
            archive_path=data.get("archive_path", "archive"),
            dry_run=data.get("dry_run", False),
        )


@dataclass
class ArchivalResult:
    """Result summary from an archival operation."""

    archived_count: int
    skipped_count: int
    total_size_bytes: int = 0
    archive_path: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "archived_count": self.archived_count,
            "skipped_count": self.skipped_count,
            "total_size_bytes": self.total_size_bytes,
            "archive_path": self.archive_path,
        }

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Archival Result")
        lines.append("-" * 30)
        lines.append(f"  Archived: {self.archived_count}")
        lines.append(f"  Skipped:  {self.skipped_count}")
        lines.append(f"  Size:     {self.total_size_bytes} bytes")
        if self.archive_path:
            lines.append(f"  Path:     {self.archive_path}")
        return "\n".join(lines)


@dataclass
class ArchivedTrace:
    """Record of a single archived trace."""

    trace_id: str
    archived_at: datetime
    original_path: str = ""
    span_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "archived_at": _iso(self.archived_at),
            "original_path": self.original_path,
            "span_count": self.span_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArchivedTrace:
        return cls(
            trace_id=data["trace_id"],
            archived_at=_parse_datetime(data.get("archived_at")),
            original_path=data.get("original_path", ""),
            span_count=data.get("span_count", 0),
        )


def should_archive(trace_timestamp: datetime, max_age_days: int = 90) -> bool:
    """Return True if the trace is older than max_age_days from now."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    # Ensure timestamp is timezone-aware for comparison
    ts = trace_timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts < cutoff


def plan_archival(
    traces: List[Dict[str, Any]],
    config: ArchivalConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split traces into (to_archive, to_keep).

    Each trace dict has "id", "created_at" (ISO string or datetime), "span_count".
    """
    to_archive: List[Dict[str, Any]] = []
    to_keep: List[Dict[str, Any]] = []
    for t in traces:
        ts = _parse_datetime(t.get("created_at"))
        if should_archive(ts, config.max_age_days):
            to_archive.append(t)
        else:
            to_keep.append(t)
    return to_archive, to_keep


def create_archive_manifest(
    archived_traces: List[Dict[str, Any]],
) -> List[ArchivedTrace]:
    """Create a manifest of archived traces."""
    now = datetime.now(timezone.utc)
    manifest: List[ArchivedTrace] = []
    for t in archived_traces:
        manifest.append(
            ArchivedTrace(
                trace_id=t.get("id", ""),
                archived_at=now,
                original_path=t.get("original_path", ""),
                span_count=t.get("span_count", 0),
            )
        )
    return manifest


def estimate_archive_size(
    traces: List[Dict[str, Any]],
    avg_bytes_per_span: int = 500,
) -> int:
    """Estimate archive size in bytes based on span counts."""
    total = 0
    for t in traces:
        total += t.get("span_count", 0) * avg_bytes_per_span
    return total
