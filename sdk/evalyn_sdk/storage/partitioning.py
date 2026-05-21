"""
Storage partitioning: partition databases by time period for better performance.

Provides pure functions to compute partition keys, plan partitions, generate
file paths, and merge partition reports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PartitionConfig:
    """Configuration for time-based partitioning."""

    period: str = "monthly"  # "daily", "weekly", "monthly", "yearly"
    base_dir: str = "."
    naming_pattern: str = "evalyn_{period}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "base_dir": self.base_dir,
            "naming_pattern": self.naming_pattern,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartitionConfig:
        return cls(
            period=data.get("period", "monthly"),
            base_dir=data.get("base_dir", "."),
            naming_pattern=data.get("naming_pattern", "evalyn_{period}"),
        )


@dataclass
class Partition:
    """A single partition with its metadata."""

    partition_id: str
    period_label: str
    start_date: str
    end_date: str
    file_path: str = ""
    row_count: int = 0
    size_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "period_label": self.period_label,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "file_path": self.file_path,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Partition:
        return cls(
            partition_id=data.get("partition_id", ""),
            period_label=data.get("period_label", ""),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            file_path=data.get("file_path", ""),
            row_count=data.get("row_count", 0),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class PartitionReport:
    """Report summarizing all partitions."""

    partitions: list[Partition] = field(default_factory=list)
    total_partitions: int = 0
    total_rows: int = 0
    total_size_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "partitions": [p.as_dict() for p in self.partitions],
            "total_partitions": self.total_partitions,
            "total_rows": self.total_rows,
            "total_size_bytes": self.total_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartitionReport:
        partitions = [Partition.from_dict(p) for p in data.get("partitions", [])]
        return cls(
            partitions=partitions,
            total_partitions=data.get("total_partitions", 0),
            total_rows=data.get("total_rows", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Partition Report")
        lines.append("=" * 40)
        lines.append(f"Total partitions: {self.total_partitions}")
        lines.append(f"Total rows: {self.total_rows}")
        lines.append(f"Total size: {self.total_size_bytes} bytes")

        if self.partitions:
            lines.append("")
            lines.append("Partitions:")
            for p in self.partitions:
                lines.append(f"  {p.partition_id}: {p.period_label}")
                lines.append(f"    Range: {p.start_date} - {p.end_date}")
                lines.append(f"    Rows: {p.row_count}, Size: {p.size_bytes} bytes")
                if p.file_path:
                    lines.append(f"    Path: {p.file_path}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_partition_key(timestamp: str, period: str = "monthly") -> str:
    """Extract partition key from an ISO timestamp.

    Returns:
        "daily"   -> "2026-03-28"
        "weekly"  -> "2026-W13"
        "monthly" -> "2026-03"
        "yearly"  -> "2026"
    """
    dt = datetime.fromisoformat(timestamp)

    if period == "daily":
        return dt.strftime("%Y-%m-%d")
    elif period == "weekly":
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    elif period == "monthly":
        return dt.strftime("%Y-%m")
    elif period == "yearly":
        return dt.strftime("%Y")
    else:
        return dt.strftime("%Y-%m")


def generate_partition_path(config: PartitionConfig, partition_key: str) -> str:
    """Generate file path for a partition.

    Substitutes {period} in the naming pattern with the partition key.
    """
    name = config.naming_pattern.replace("{period}", partition_key)
    return os.path.join(config.base_dir, f"{name}.db")


def assign_to_partition(
    record: dict[str, Any],
    config: PartitionConfig,
    timestamp_field: str = "created_at",
) -> str:
    """Return the partition key for a single record."""
    ts = record.get(timestamp_field, "")
    if not ts:
        return ""
    return compute_partition_key(ts, config.period)


def plan_partitions(
    records: list[dict[str, Any]],
    config: PartitionConfig,
    timestamp_field: str = "created_at",
) -> PartitionReport:
    """Group records by partition key and create a partition plan."""
    if not records:
        return PartitionReport()

    # Group records by partition key
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = assign_to_partition(record, config, timestamp_field)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append(record)

    if not groups:
        return PartitionReport()

    # Build partitions sorted by key
    partitions: list[Partition] = []
    total_rows = 0

    for key in sorted(groups.keys()):
        group = groups[key]
        row_count = len(group)
        total_rows += row_count

        # Extract timestamp range within this partition
        timestamps: list[str] = []
        for r in group:
            ts = r.get(timestamp_field, "")
            if ts:
                timestamps.append(ts)
        timestamps.sort()

        start_date = timestamps[0] if timestamps else ""
        end_date = timestamps[-1] if timestamps else ""

        file_path = generate_partition_path(config, key)

        partitions.append(
            Partition(
                partition_id=key,
                period_label=key,
                start_date=start_date,
                end_date=end_date,
                file_path=file_path,
                row_count=row_count,
            )
        )

    return PartitionReport(
        partitions=partitions,
        total_partitions=len(partitions),
        total_rows=total_rows,
    )


def merge_partition_reports(reports: list[PartitionReport]) -> PartitionReport:
    """Combine partition reports from multiple sources.

    Merges partitions with the same partition_id by summing row counts
    and size bytes.
    """
    if not reports:
        return PartitionReport()

    merged: dict[str, Partition] = {}

    for report in reports:
        for p in report.partitions:
            if p.partition_id in merged:
                existing = merged[p.partition_id]
                existing.row_count += p.row_count
                existing.size_bytes += p.size_bytes
                # Extend the date range
                if p.start_date and (not existing.start_date or p.start_date < existing.start_date):
                    existing.start_date = p.start_date
                if p.end_date and (not existing.end_date or p.end_date > existing.end_date):
                    existing.end_date = p.end_date
            else:
                merged[p.partition_id] = Partition(
                    partition_id=p.partition_id,
                    period_label=p.period_label,
                    start_date=p.start_date,
                    end_date=p.end_date,
                    file_path=p.file_path,
                    row_count=p.row_count,
                    size_bytes=p.size_bytes,
                )

    partitions = [merged[k] for k in sorted(merged.keys())]
    total_rows = sum(p.row_count for p in partitions)
    total_size = sum(p.size_bytes for p in partitions)

    return PartitionReport(
        partitions=partitions,
        total_partitions=len(partitions),
        total_rows=total_rows,
        total_size_bytes=total_size,
    )
