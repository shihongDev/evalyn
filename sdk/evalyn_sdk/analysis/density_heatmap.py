from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class HeatmapCell:
    """Single cell in the heatmap grid."""

    day: int  # 0-6, Monday=0
    hour: int  # 0-23
    count: int
    value: float  # can be count, cost, or error rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "hour": self.hour,
            "count": self.count,
            "value": self.value,
        }


@dataclass
class HeatmapData:
    """Full heatmap dataset."""

    cells: list[HeatmapCell] = field(default_factory=list)
    total_traces: int = 0
    metric: str = "count"  # "count", "cost", "error_rate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": [c.as_dict() for c in self.cells],
            "total_traces": self.total_traces,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeatmapData:
        return cls(
            cells=[
                HeatmapCell(
                    day=c["day"],
                    hour=c["hour"],
                    count=c["count"],
                    value=c["value"],
                )
                for c in data.get("cells", [])
            ],
            total_traces=data.get("total_traces", 0),
            metric=data.get("metric", "count"),
        )


def build_heatmap(spans: list[Span], metric: str = "count") -> HeatmapData:
    """Group root spans by (day_of_week, hour_of_day).

    Only considers root spans (parent_id is None).

    Metric options:
    - "count": trace count per cell
    - "cost": sum of attributes.cost per cell
    - "error_rate": fraction with status=error per cell
    """
    root_spans = [s for s in spans if s.parent_id is None]

    # Buckets keyed by (day, hour)
    buckets: dict[tuple, list[Span]] = {}
    for s in root_spans:
        day = s.start_time.weekday()  # Monday=0
        hour = s.start_time.hour
        key = (day, hour)
        buckets.setdefault(key, []).append(s)

    cells: list[HeatmapCell] = []
    for (day, hour), bucket in sorted(buckets.items()):
        count = len(bucket)
        if metric == "cost":
            value = sum(
                float(s.attributes.get("cost", 0)) for s in bucket
            )
        elif metric == "error_rate":
            errors = sum(1 for s in bucket if s.status == "error")
            value = errors / count if count else 0.0
        else:
            value = float(count)
        cells.append(HeatmapCell(day=day, hour=hour, count=count, value=value))

    return HeatmapData(cells=cells, total_traces=len(root_spans), metric=metric)


_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Characters ordered by intensity
_CHARS = " .oO#"


def _intensity_char(value: float, thresholds: list[float]) -> str:
    """Pick a character based on percentile thresholds.

    thresholds has 4 values (p25, p50, p75, p90).
    """
    if value <= 0:
        return _CHARS[0]
    if not thresholds or thresholds[-1] <= 0:
        return _CHARS[4] if value > 0 else _CHARS[0]
    if value <= thresholds[0]:
        return _CHARS[1]
    if value <= thresholds[1]:
        return _CHARS[2]
    if value <= thresholds[2]:
        return _CHARS[3]
    return _CHARS[4]


def _percentile_thresholds(values: list[float]) -> list[float]:
    """Return [p25, p50, p75, p90] from nonzero values."""
    nonzero = sorted(v for v in values if v > 0)
    if not nonzero:
        return [0.0, 0.0, 0.0, 0.0]
    n = len(nonzero)

    def _p(pct: float) -> float:
        idx = int(pct / 100 * (n - 1))
        return nonzero[min(idx, n - 1)]

    return [_p(25), _p(50), _p(75), _p(90)]


def render_ascii_heatmap(data: HeatmapData, width: int = 80) -> str:
    """Render an ASCII heatmap grid.

    Days as rows (Mon-Sun), hours as columns (0-23).
    Characters: " " (zero), "." (low), "o" (medium), "O" (high), "#" (very high).
    """
    # Build lookup
    lookup: dict[tuple, float] = {}
    for cell in data.cells:
        lookup[(cell.day, cell.hour)] = cell.value

    all_values = [cell.value for cell in data.cells]
    thresholds = _percentile_thresholds(all_values)

    lines: list[str] = []
    # Header
    header = "     " + "".join(f"{h:>3}" for h in range(24))
    lines.append(header)

    for day in range(7):
        label = _DAY_LABELS[day]
        row_chars = []
        for hour in range(24):
            val = lookup.get((day, hour), 0.0)
            row_chars.append(f"  {_intensity_char(val, thresholds)}")
        lines.append(f"{label:>4} " + "".join(row_chars))

    # Legend
    lines.append("")
    lines.append(f"Metric: {data.metric}  Total traces: {data.total_traces}")
    lines.append("Legend: ' '=0  '.'=low  'o'=med  'O'=high  '#'=very high")

    return "\n".join(lines)


def render_html_heatmap(data: HeatmapData) -> str:
    """Render a simple HTML table with background-color intensity."""
    lookup: dict[tuple, float] = {}
    for cell in data.cells:
        lookup[(cell.day, cell.hour)] = cell.value

    all_values = [cell.value for cell in data.cells]
    max_val = max(all_values) if all_values else 1.0
    if max_val <= 0:
        max_val = 1.0

    rows: list[str] = []
    rows.append("<table style='border-collapse:collapse;font-family:monospace'>")
    # Header row
    rows.append("<tr><th></th>")
    for h in range(24):
        rows.append(f"<th style='padding:2px 4px'>{h}</th>")
    rows.append("</tr>")

    for day in range(7):
        rows.append(f"<tr><td style='padding:2px 6px;font-weight:bold'>{_DAY_LABELS[day]}</td>")
        for hour in range(24):
            val = lookup.get((day, hour), 0.0)
            intensity = min(val / max_val, 1.0)
            r = int(255 * (1 - intensity * 0.7))
            g = int(255 * (1 - intensity * 0.3))
            b = int(255 * (1 - intensity * 0.7))
            color = f"rgb({r},{g},{b})"
            display = f"{val:.1f}" if val else ""
            rows.append(
                f"<td style='padding:2px 4px;background:{color};text-align:center'>"
                f"{display}</td>"
            )
        rows.append("</tr>")

    rows.append("</table>")
    rows.append(f"<p>Metric: {data.metric} - Total traces: {data.total_traces}</p>")

    return "\n".join(rows)
