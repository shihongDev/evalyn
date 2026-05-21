"""
Aggregation queries: cost and usage analytics over in-memory record sets.

Pure Python - operates on lists of dicts with grouping, filtering, and formatting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AggregationQuery:
    """Describes a group-by aggregation over a record set."""

    group_by: list[str]
    metric: str
    operation: str  # sum, mean, count, min, max
    filters: dict[str, Any] = field(default_factory=dict)
    date_field: str = ""
    date_from: str = ""
    date_to: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.group_by, str):
            self.group_by = [self.group_by]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_by": list(self.group_by),
            "metric": self.metric,
            "operation": self.operation,
            "filters": dict(self.filters),
            "date_field": self.date_field,
            "date_from": self.date_from,
            "date_to": self.date_to,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregationQuery:
        return cls(
            group_by=data.get("group_by", []),
            metric=data.get("metric", ""),
            operation=data.get("operation", "count"),
            filters=data.get("filters", {}),
            date_field=data.get("date_field", ""),
            date_from=data.get("date_from", ""),
            date_to=data.get("date_to", ""),
        )


@dataclass
class AggregationResult:
    """Result of an aggregation query."""

    groups: list[dict[str, Any]]
    total_records: int
    query: AggregationQuery

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": [dict(g) for g in self.groups],
            "total_records": self.total_records,
            "query": self.query.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregationResult:
        return cls(
            groups=data.get("groups", []),
            total_records=data.get("total_records", 0),
            query=AggregationQuery.from_dict(data.get("query", {})),
        )


@dataclass
class ProjectStats:
    """Aggregated statistics for a single project."""

    project_name: str
    total_traces: int
    total_tokens: int
    total_cost: float
    model_breakdown: dict[str, int]
    date_range: tuple[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "total_traces": self.total_traces,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model_breakdown": dict(self.model_breakdown),
            "date_range": list(self.date_range),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectStats:
        dr = data.get("date_range", ["", ""])
        return cls(
            project_name=data.get("project_name", ""),
            total_traces=data.get("total_traces", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            model_breakdown=data.get("model_breakdown", {}),
            date_range=(dr[0] if dr else "", dr[1] if len(dr) > 1 else ""),
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_records(
    records: list[dict],
    filters: dict[str, Any],
    date_field: str = "",
    date_from: str = "",
    date_to: str = "",
) -> list[dict]:
    """Apply key=value filters and optional date range to records.

    Missing fields in a record cause that record to be excluded.
    Date comparisons use lexicographic string ordering (ISO-8601 safe).
    """
    result: list[dict] = []
    for rec in records:
        # Key-value filters
        skip = False
        for k, v in filters.items():
            if rec.get(k) != v:
                skip = True
                break
        if skip:
            continue

        # Date range
        if date_field and (date_from or date_to):
            dt_val = rec.get(date_field, "")
            if not dt_val:
                continue
            dt_str = str(dt_val)
            if date_from and dt_str < date_from:
                continue
            if date_to and dt_str > date_to:
                continue

        result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _group_key(record: dict, group_by: list[str]) -> tuple:
    """Build a hashable group key from a record."""
    return tuple(record.get(f) for f in group_by)


def _aggregate_values(values: list[float], operation: str) -> float:
    """Apply an aggregation operation to a list of numeric values."""
    if not values:
        return 0.0
    if operation == "sum":
        return sum(values)
    if operation == "mean":
        return sum(values) / len(values)
    if operation == "count":
        return float(len(values))
    if operation == "min":
        return min(values)
    if operation == "max":
        return max(values)
    return 0.0


# ---------------------------------------------------------------------------
# Core Query Execution
# ---------------------------------------------------------------------------


def execute_query(records: list[dict], query: AggregationQuery) -> AggregationResult:
    """Filter records, group by fields, aggregate a metric.

    Missing metric values are silently skipped (not counted as 0).
    For the ``count`` operation the metric value is not required.
    """
    filtered = filter_records(
        records,
        query.filters,
        date_field=query.date_field,
        date_from=query.date_from,
        date_to=query.date_to,
    )

    groups: dict[tuple, list[float]] = defaultdict(list)
    for rec in filtered:
        key = _group_key(rec, query.group_by)
        if query.operation == "count":
            groups[key].append(1.0)
        else:
            val = rec.get(query.metric)
            if val is None:
                continue
            try:
                groups[key].append(float(val))
            except (TypeError, ValueError):
                continue

    result_groups: list[dict] = []
    for key, values in groups.items():
        group_dict: dict[str, Any] = {}
        for i, field_name in enumerate(query.group_by):
            group_dict[field_name] = key[i]
        group_dict["value"] = _aggregate_values(values, query.operation)
        result_groups.append(group_dict)

    return AggregationResult(
        groups=result_groups,
        total_records=len(filtered),
        query=query,
    )


# ---------------------------------------------------------------------------
# Convenience Aggregations
# ---------------------------------------------------------------------------


def aggregate_by_project(records: list[dict], project_field: str = "project") -> list[ProjectStats]:
    """Compute per-project statistics."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        proj = rec.get(project_field, "unknown")
        buckets[str(proj)].append(rec)

    stats: list[ProjectStats] = []
    for proj_name, recs in sorted(buckets.items()):
        total_tokens = 0
        total_cost = 0.0
        model_counts: dict[str, int] = defaultdict(int)
        dates: list[str] = []

        for r in recs:
            total_tokens += _safe_int(r.get("tokens", 0))
            total_cost += _safe_float(r.get("cost", 0.0))
            model = r.get("model", "")
            if model:
                model_counts[str(model)] += 1
            ts = r.get("timestamp", "")
            if ts:
                dates.append(str(ts))

        dates.sort()
        date_range = (dates[0], dates[-1]) if dates else ("", "")

        stats.append(
            ProjectStats(
                project_name=proj_name,
                total_traces=len(recs),
                total_tokens=total_tokens,
                total_cost=round(total_cost, 6),
                model_breakdown=dict(model_counts),
                date_range=date_range,
            )
        )
    return stats


def _date_key(timestamp: str, interval: str) -> str:
    """Extract a date bucket key from an ISO-ish timestamp string."""
    # Expect at least YYYY-MM-DD prefix
    date_part = timestamp[:10] if len(timestamp) >= 10 else timestamp
    if interval == "day":
        return date_part
    if interval == "week":
        # ISO week: truncate to Monday of the week
        try:
            from datetime import date as _date

            parts = date_part.split("-")
            d = _date(int(parts[0]), int(parts[1]), int(parts[2]))
            monday = d.fromisocalendar(d.isocalendar()[0], d.isocalendar()[1], 1)
            return monday.isoformat()
        except Exception:
            return date_part
    if interval == "month":
        return date_part[:7]  # YYYY-MM
    return date_part


def aggregate_by_date(
    records: list[dict],
    date_field: str = "timestamp",
    interval: str = "day",
) -> dict[str, dict]:
    """Group records by date interval.

    Returns {date_key: {count, total_cost, total_tokens}}.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        ts = rec.get(date_field, "")
        if not ts:
            continue
        key = _date_key(str(ts), interval)
        buckets[key].append(rec)

    result: dict[str, dict] = {}
    for key in sorted(buckets):
        recs = buckets[key]
        result[key] = {
            "count": len(recs),
            "total_cost": round(sum(_safe_float(r.get("cost", 0.0)) for r in recs), 6),
            "total_tokens": sum(_safe_int(r.get("tokens", 0)) for r in recs),
        }
    return result


def aggregate_by_model(records: list[dict], model_field: str = "model") -> dict[str, dict]:
    """Per-model statistics: count, total_tokens, total_cost, avg_latency."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        model = rec.get(model_field, "")
        if not model:
            continue
        buckets[str(model)].append(rec)

    result: dict[str, dict] = {}
    for model_name in sorted(buckets):
        recs = buckets[model_name]
        latencies = [_safe_float(r.get("latency", 0.0)) for r in recs if "latency" in r]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        result[model_name] = {
            "count": len(recs),
            "total_tokens": sum(_safe_int(r.get("tokens", 0)) for r in recs),
            "total_cost": round(sum(_safe_float(r.get("cost", 0.0)) for r in recs), 6),
            "avg_latency": round(avg_lat, 4),
        }
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_aggregation_result(result: AggregationResult) -> str:
    """Human-readable table of aggregation result."""
    if not result.groups:
        return f"No results ({result.total_records} records matched)"

    # Determine columns
    columns = list(result.groups[0].keys())
    col_widths: dict[str, int] = {c: len(c) for c in columns}
    rows: list[dict[str, str]] = []
    for g in result.groups:
        row = {}
        for c in columns:
            val = g.get(c, "")
            s = f"{val:.4f}" if isinstance(val, float) else str(val)
            row[c] = s
            if len(s) > col_widths[c]:
                col_widths[c] = len(s)
        rows.append(row)

    header = "  ".join(c.ljust(col_widths[c]) for c in columns)
    sep = "  ".join("-" * col_widths[c] for c in columns)
    lines = [header, sep]
    for row in rows:
        lines.append("  ".join(row[c].ljust(col_widths[c]) for c in columns))
    lines.append(f"\nTotal records: {result.total_records}")
    return "\n".join(lines)


def format_project_stats(stats: list[ProjectStats]) -> str:
    """Project summary table."""
    if not stats:
        return "No project data"

    lines = [
        "Project Summary",
        "-" * 70,
    ]
    for s in stats:
        dr = f"{s.date_range[0]} - {s.date_range[1]}" if s.date_range[0] else "n/a"
        lines.append(f"  {s.project_name}")
        lines.append(
            f"    Traces: {s.total_traces}  Tokens: {s.total_tokens}  Cost: {s.total_cost:.4f}"
        )
        lines.append(f"    Models: {s.model_breakdown}")
        lines.append(f"    Date range: {dr}")
    return "\n".join(lines)
