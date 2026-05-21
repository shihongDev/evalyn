"""
Storage statistics: database size, row counts, and growth rate over time.

Provides pure functions to create snapshots, compute growth, build reports,
and render ASCII charts for storage usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class TableStats:
    """Row count and estimated size for a single table."""

    table_name: str
    row_count: int = 0
    estimated_size_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "estimated_size_bytes": self.estimated_size_bytes,
        }


@dataclass
class StorageSnapshot:
    """Point-in-time snapshot of storage usage."""

    timestamp: str
    total_size_bytes: int = 0
    total_rows: int = 0
    tables: list[TableStats] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_size_bytes": self.total_size_bytes,
            "total_rows": self.total_rows,
            "tables": [t.as_dict() for t in self.tables],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageSnapshot:
        tables = [
            TableStats(
                table_name=t["table_name"],
                row_count=t.get("row_count", 0),
                estimated_size_bytes=t.get("estimated_size_bytes", 0),
            )
            for t in data.get("tables", [])
        ]
        return cls(
            timestamp=data["timestamp"],
            total_size_bytes=data.get("total_size_bytes", 0),
            total_rows=data.get("total_rows", 0),
            tables=tables,
        )


@dataclass
class GrowthRate:
    """Growth rate between two snapshots."""

    period: str = "daily"
    size_growth_bytes: int = 0
    row_growth: int = 0
    growth_pct: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "size_growth_bytes": self.size_growth_bytes,
            "row_growth": self.row_growth,
            "growth_pct": self.growth_pct,
        }


@dataclass
class StorageReport:
    """Full storage report with history, growth, and recommendations."""

    current: StorageSnapshot
    history: list[StorageSnapshot] = field(default_factory=list)
    growth: GrowthRate | None = None
    largest_table: str = ""
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.as_dict(),
            "history": [s.as_dict() for s in self.history],
            "growth": self.growth.as_dict() if self.growth else None,
            "largest_table": self.largest_table,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageReport:
        current = StorageSnapshot.from_dict(data["current"])
        history = [StorageSnapshot.from_dict(s) for s in data.get("history", [])]
        growth = None
        if data.get("growth"):
            g = data["growth"]
            growth = GrowthRate(
                period=g.get("period", "daily"),
                size_growth_bytes=g.get("size_growth_bytes", 0),
                row_growth=g.get("row_growth", 0),
                growth_pct=g.get("growth_pct", 0.0),
            )
        return cls(
            current=current,
            history=history,
            growth=growth,
            largest_table=data.get("largest_table", ""),
            recommendations=data.get("recommendations", []),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Storage Report")
        lines.append("=" * 40)
        lines.append(f"Total size: {format_size(self.current.total_size_bytes)}")
        lines.append(f"Total rows: {self.current.total_rows}")
        if self.largest_table:
            lines.append(f"Largest table: {self.largest_table}")
        if self.growth:
            lines.append("")
            lines.append(f"Growth ({self.growth.period}):")
            lines.append(f"  Size: +{format_size(self.growth.size_growth_bytes)}")
            lines.append(f"  Rows: +{self.growth.row_growth}")
            lines.append(f"  Rate: {self.growth.growth_pct:.1f}%")
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_snapshot(tables: list[dict[str, Any]], total_size_bytes: int = 0) -> StorageSnapshot:
    """Create a snapshot from table info dicts.

    Each dict should have "name", "row_count", "size_bytes".
    """
    table_stats: list[TableStats] = []
    total_rows = 0
    computed_size = 0
    for t in tables:
        ts = TableStats(
            table_name=t.get("name", ""),
            row_count=t.get("row_count", 0),
            estimated_size_bytes=t.get("size_bytes", 0),
        )
        table_stats.append(ts)
        total_rows += ts.row_count
        computed_size += ts.estimated_size_bytes

    if total_size_bytes == 0:
        total_size_bytes = computed_size

    timestamp = datetime.now(timezone.utc).isoformat()
    return StorageSnapshot(
        timestamp=timestamp,
        total_size_bytes=total_size_bytes,
        total_rows=total_rows,
        tables=table_stats,
    )


def compute_growth_rate(
    snapshots: list[StorageSnapshot],
) -> GrowthRate | None:
    """Compare latest to earliest snapshot and compute growth."""
    if len(snapshots) < 2:
        return None

    earliest = snapshots[0]
    latest = snapshots[-1]

    size_growth = latest.total_size_bytes - earliest.total_size_bytes
    row_growth = latest.total_rows - earliest.total_rows

    if earliest.total_size_bytes > 0:
        growth_pct = (size_growth / earliest.total_size_bytes) * 100.0
    else:
        growth_pct = 0.0

    return GrowthRate(
        period="daily",
        size_growth_bytes=size_growth,
        row_growth=row_growth,
        growth_pct=round(growth_pct, 2),
    )


def build_storage_report(
    snapshots: list[StorageSnapshot],
) -> StorageReport:
    """Build full report with growth analysis and recommendations."""
    if not snapshots:
        empty = StorageSnapshot(timestamp=datetime.now(timezone.utc).isoformat())
        return StorageReport(current=empty)

    current = snapshots[-1]
    growth = compute_growth_rate(snapshots)

    # Find largest table
    largest_table = ""
    max_size = 0
    for ts in current.tables:
        if ts.estimated_size_bytes > max_size:
            max_size = ts.estimated_size_bytes
            largest_table = ts.table_name

    report = StorageReport(
        current=current,
        history=snapshots,
        growth=growth,
        largest_table=largest_table,
    )
    report.recommendations = suggest_recommendations(report)
    return report


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string (B, KB, MB, GB)."""
    if bytes_val < 0:
        return f"-{format_size(-bytes_val)}"
    if bytes_val < 1024:
        return f"{bytes_val} B"
    if bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    if bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    return f"{bytes_val / (1024 * 1024 * 1024):.1f} GB"


def suggest_recommendations(report: StorageReport) -> list[str]:
    """Generate storage recommendations based on report data."""
    recs: list[str] = []
    size_mb = report.current.total_size_bytes / (1024 * 1024)

    if size_mb > 500:
        recs.append("Consider compaction - database exceeds 500 MB")
    if size_mb > 1000:
        recs.append("Enable archival - database exceeds 1 GB")

    if report.growth and report.growth.growth_pct > 50:
        recs.append("High growth rate detected - review retention policies")

    if report.current.total_rows > 100_000:
        recs.append("Large row count - consider indexing or partitioning")

    if report.largest_table:
        for ts in report.current.tables:
            if ts.table_name == report.largest_table:
                table_pct = 0.0
                if report.current.total_size_bytes > 0:
                    table_pct = ts.estimated_size_bytes / report.current.total_size_bytes * 100
                if table_pct > 80:
                    recs.append(
                        f"Table '{report.largest_table}' dominates storage"
                        f" ({table_pct:.0f}%) - consider cleanup"
                    )
                break

    return recs


def render_storage_chart(report: StorageReport, width: int = 60) -> str:
    """Render an ASCII bar chart of table sizes."""
    tables = report.current.tables
    if not tables:
        return "(no tables)"

    max_size = max(t.estimated_size_bytes for t in tables)
    if max_size == 0:
        max_size = 1  # avoid division by zero

    # Find longest table name for alignment
    max_name_len = max(len(t.table_name) for t in tables)
    bar_width = width - max_name_len - 2  # 2 for padding
    if bar_width < 10:
        bar_width = 10

    lines: list[str] = []
    for t in sorted(tables, key=lambda x: x.estimated_size_bytes, reverse=True):
        bar_len = int((t.estimated_size_bytes / max_size) * bar_width)
        bar_len = max(bar_len, 0)
        bar = "#" * bar_len
        label = t.table_name.ljust(max_name_len)
        size_str = format_size(t.estimated_size_bytes)
        lines.append(f"{label}  {bar} {size_str}")

    return "\n".join(lines)
