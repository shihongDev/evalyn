"""Comparative heatmap for metric scores across items and runs.

Provides data structures and rendering for visual heatmaps of
evaluation results.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HeatmapCell:
    """A single cell in the heatmap grid."""

    row: str
    col: str
    value: float
    color_intensity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "color_intensity": self.color_intensity,
        }


@dataclass
class HeatmapData:
    """Full heatmap grid with rows, columns, and cells."""

    cells: list[HeatmapCell] = field(default_factory=list)
    rows: list[str] = field(default_factory=list)
    cols: list[str] = field(default_factory=list)
    title: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": [c.as_dict() for c in self.cells],
            "rows": list(self.rows),
            "cols": list(self.cols),
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeatmapData:
        cells = [
            HeatmapCell(
                row=c["row"],
                col=c["col"],
                value=c["value"],
                color_intensity=c.get("color_intensity", 0.0),
            )
            for c in data.get("cells", [])
        ]
        return cls(
            cells=cells,
            rows=data.get("rows", []),
            cols=data.get("cols", []),
            title=data.get("title", ""),
        )


@dataclass
class HeatmapReport:
    """Heatmap with hotspot and coldspot analysis."""

    data: HeatmapData = field(default_factory=HeatmapData)
    hotspots: list[str] = field(default_factory=list)
    coldspots: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.as_dict(),
            "hotspots": list(self.hotspots),
            "coldspots": list(self.coldspots),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeatmapReport:
        return cls(
            data=HeatmapData.from_dict(data.get("data", {})),
            hotspots=data.get("hotspots", []),
            coldspots=data.get("coldspots", []),
        )

    def format_text(self) -> str:
        """Human-readable text summary."""
        lines: list[str] = []
        title = self.data.title or "Heatmap Report"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"Rows: {len(self.data.rows)}, Cols: {len(self.data.cols)}")
        lines.append(f"Cells: {len(self.data.cells)}")
        lines.append("")
        if self.hotspots:
            lines.append("Hotspots (high performers):")
            for h in self.hotspots:
                lines.append(f"  - {h}")
            lines.append("")
        if self.coldspots:
            lines.append("Coldspots (need attention):")
            for c in self.coldspots:
                lines.append(f"  - {c}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heatmap builders
# ---------------------------------------------------------------------------


def _collect_all_values(cells: list[HeatmapCell]) -> tuple:
    """Return (min_val, max_val) across all cells."""
    if not cells:
        return (0.0, 1.0)
    values = [c.value for c in cells]
    return (min(values), max(values))


def build_item_metric_heatmap(
    scores: dict[str, dict[str, float]],
) -> HeatmapData:
    """Build heatmap from scores[item_id][metric_id] = score.

    Rows = items, cols = metrics.
    """
    if not scores:
        return HeatmapData(title="Item-Metric Heatmap")

    rows = sorted(scores.keys())
    col_set: set = set()
    for item_scores in scores.values():
        col_set.update(item_scores.keys())
    cols = sorted(col_set)

    # Collect all values for normalization
    all_values = []
    for item_scores in scores.values():
        all_values.extend(item_scores.values())
    min_val = min(all_values) if all_values else 0.0
    max_val = max(all_values) if all_values else 1.0

    cells: list[HeatmapCell] = []
    for row in rows:
        for col in cols:
            value = scores[row].get(col, 0.0)
            intensity = compute_color_intensity(value, min_val, max_val)
            cells.append(HeatmapCell(
                row=row, col=col, value=value, color_intensity=intensity,
            ))

    return HeatmapData(
        cells=cells, rows=rows, cols=cols, title="Item-Metric Heatmap",
    )


def build_run_metric_heatmap(
    run_scores: dict[str, dict[str, float]],
) -> HeatmapData:
    """Build heatmap from run_scores[run_id][metric_id] = score.

    Rows = runs, cols = metrics.
    """
    if not run_scores:
        return HeatmapData(title="Run-Metric Heatmap")

    rows = sorted(run_scores.keys())
    col_set: set = set()
    for rs in run_scores.values():
        col_set.update(rs.keys())
    cols = sorted(col_set)

    all_values = []
    for rs in run_scores.values():
        all_values.extend(rs.values())
    min_val = min(all_values) if all_values else 0.0
    max_val = max(all_values) if all_values else 1.0

    cells: list[HeatmapCell] = []
    for row in rows:
        for col in cols:
            value = run_scores[row].get(col, 0.0)
            intensity = compute_color_intensity(value, min_val, max_val)
            cells.append(HeatmapCell(
                row=row, col=col, value=value, color_intensity=intensity,
            ))

    return HeatmapData(
        cells=cells, rows=rows, cols=cols, title="Run-Metric Heatmap",
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_color_intensity(
    value: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """Normalize value to 0-1 for coloring."""
    if max_val == min_val:
        return 0.5
    raw = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, raw))


def find_hotspots(data: HeatmapData, threshold: float = 0.9) -> list[str]:
    """Rows with all values above threshold."""
    if not data.cells:
        return []

    row_values: dict[str, list[float]] = {}
    for cell in data.cells:
        row_values.setdefault(cell.row, []).append(cell.value)

    hotspots = []
    for row in data.rows:
        values = row_values.get(row, [])
        if values and all(v > threshold for v in values):
            hotspots.append(row)
    return hotspots


def find_coldspots(data: HeatmapData, threshold: float = 0.3) -> list[str]:
    """Rows with any value below threshold."""
    if not data.cells:
        return []

    row_values: dict[str, list[float]] = {}
    for cell in data.cells:
        row_values.setdefault(cell.row, []).append(cell.value)

    coldspots = []
    for row in data.rows:
        values = row_values.get(row, [])
        if values and any(v < threshold for v in values):
            coldspots.append(row)
    return coldspots


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_heatmap_ascii(data: HeatmapData, width: int = 80) -> str:
    """ASCII rendering using characters for intensity."""
    if not data.cells:
        return "(empty heatmap)"

    # Intensity characters from low to high
    chars = " .:;+=xX#@"

    # Build lookup
    lookup: dict[str, dict[str, HeatmapCell]] = {}
    for cell in data.cells:
        lookup.setdefault(cell.row, {})[cell.col] = cell

    # Column label width
    col_width = max(8, max((len(c) for c in data.cols), default=8))
    row_label_width = max(8, max((len(r) for r in data.rows), default=8))

    # Clamp col_width so we fit in the requested width
    max_cols_width = (width - row_label_width - 2) // max(len(data.cols), 1)
    col_width = max(3, min(col_width, max_cols_width))

    lines: list[str] = []

    if data.title:
        lines.append(data.title)
        lines.append("")

    # Header row
    header = " " * row_label_width + " "
    for col in data.cols:
        header += col[:col_width].center(col_width)
    lines.append(header)
    lines.append("-" * len(header))

    for row in data.rows:
        label = row[:row_label_width].ljust(row_label_width)
        parts = [label, " "]
        for col in data.cols:
            cell = lookup.get(row, {}).get(col)
            if cell is None:
                ch = " "
            else:
                idx = int(cell.color_intensity * (len(chars) - 1))
                idx = max(0, min(len(chars) - 1, idx))
                ch = chars[idx]
            cell_str = f"[{ch}]".center(col_width)
            parts.append(cell_str)
        lines.append("".join(parts))

    return "\n".join(lines)


def render_heatmap_html(data: HeatmapData) -> str:
    """HTML table with background-color based on value."""
    if not data.cells:
        return "<p>No heatmap data.</p>"

    lookup: dict[str, dict[str, HeatmapCell]] = {}
    for cell in data.cells:
        lookup.setdefault(cell.row, {})[cell.col] = cell

    parts: list[str] = []
    parts.append("<table>")

    if data.title:
        parts.append(
            f"<caption>{_html.escape(data.title)}</caption>"
        )

    # Header
    parts.append("<tr><th></th>")
    for col in data.cols:
        parts.append(f"<th>{_html.escape(col)}</th>")
    parts.append("</tr>")

    # Body
    for row in data.rows:
        parts.append(f"<tr><td>{_html.escape(row)}</td>")
        for col in data.cols:
            cell = lookup.get(row, {}).get(col)
            if cell is None:
                parts.append("<td></td>")
            else:
                # Green intensity: higher = more green
                g = int(100 + cell.color_intensity * 155)
                r = int(255 - cell.color_intensity * 155)
                bg = f"rgb({r},{g},100)"
                parts.append(
                    f"<td style=\"background-color:{bg}\">"
                    f"{cell.value:.2f}</td>"
                )
        parts.append("</tr>")

    parts.append("</table>")
    return "\n".join(parts)
