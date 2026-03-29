"""Comparison overlay dashboard for visual evaluation run comparison.

Overlay two evaluation runs on the same charts. Pure Python, no external
dependencies. Uses html.escape for all dynamic content in HTML/SVG output.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RunOverlayData:
    """Data for a single evaluation run to overlay."""

    run_id: str
    label: str
    metrics: Dict[str, float]
    color: str = "#4A90D9"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "metrics": dict(self.metrics),
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunOverlayData:
        return cls(
            run_id=data["run_id"],
            label=data["label"],
            metrics=dict(data.get("metrics", {})),
            color=data.get("color", "#4A90D9"),
        )


@dataclass
class OverlayConfig:
    """Configuration for overlay chart rendering."""

    width: int = 600
    height: int = 400
    show_legend: bool = True
    show_delta: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "show_legend": self.show_legend,
            "show_delta": self.show_delta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OverlayConfig:
        return cls(
            width=data.get("width", 600),
            height=data.get("height", 400),
            show_legend=data.get("show_legend", True),
            show_delta=data.get("show_delta", True),
        )


@dataclass
class OverlayChart:
    """A generated overlay chart."""

    chart_type: str
    svg: str
    title: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "svg": self.svg,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OverlayChart:
        return cls(
            chart_type=data["chart_type"],
            svg=data["svg"],
            title=data["title"],
        )


def compute_deltas(
    run_a: RunOverlayData, run_b: RunOverlayData
) -> Dict[str, float]:
    """Compute metric deltas (b - a) for all shared metrics."""
    all_keys = sorted(set(run_a.metrics) | set(run_b.metrics))
    result: Dict[str, float] = {}
    for key in all_keys:
        val_a = run_a.metrics.get(key, 0.0)
        val_b = run_b.metrics.get(key, 0.0)
        result[key] = val_b - val_a
    return result


def generate_dual_bar_svg(
    run_a: RunOverlayData,
    run_b: RunOverlayData,
    config: OverlayConfig | None = None,
) -> str:
    """Generate SVG with side-by-side bars for each metric."""
    if config is None:
        config = OverlayConfig()

    metrics = sorted(set(run_a.metrics) | set(run_b.metrics))
    if not metrics:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{config.width}" height="{config.height}"></svg>'
        )

    w = config.width
    h = config.height
    margin_top = 40
    margin_bottom = 80
    margin_left = 60
    margin_right = 20
    legend_height = 30 if config.show_legend else 0

    chart_w = w - margin_left - margin_right
    chart_h = h - margin_top - margin_bottom - legend_height

    all_vals = []
    for m in metrics:
        all_vals.append(run_a.metrics.get(m, 0.0))
        all_vals.append(run_b.metrics.get(m, 0.0))
    max_val = max(abs(v) for v in all_vals) if all_vals else 1.0
    if max_val == 0:
        max_val = 1.0

    group_w = chart_w / len(metrics)
    bar_w = group_w * 0.35
    gap = group_w * 0.05

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
    )
    parts.append(
        f'<rect width="{w}" height="{h}" fill="white"/>'
    )

    # Title
    title_text = html.escape(f"{run_a.label} vs {run_b.label}")
    parts.append(
        f'<text x="{w / 2}" y="20" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14" font-weight="bold">'
        f"{title_text}</text>"
    )

    deltas = compute_deltas(run_a, run_b) if config.show_delta else {}

    for i, m in enumerate(metrics):
        val_a = run_a.metrics.get(m, 0.0)
        val_b = run_b.metrics.get(m, 0.0)
        x_group = margin_left + i * group_w

        # Bar A
        bar_h_a = (abs(val_a) / max_val) * chart_h
        x_a = x_group + gap
        y_a = margin_top + chart_h - bar_h_a
        esc_color_a = html.escape(run_a.color)
        parts.append(
            f'<rect class="run-a" x="{x_a:.1f}" y="{y_a:.1f}" '
            f'width="{bar_w:.1f}" height="{bar_h_a:.1f}" '
            f'fill="{esc_color_a}" opacity="0.8"/>'
        )
        # Value label A
        parts.append(
            f'<text x="{x_a + bar_w / 2:.1f}" y="{y_a - 4:.1f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="10">'
            f"{html.escape(f'{val_a:.2f}')}</text>"
        )

        # Bar B
        bar_h_b = (abs(val_b) / max_val) * chart_h
        x_b = x_group + gap + bar_w + gap
        y_b = margin_top + chart_h - bar_h_b
        esc_color_b = html.escape(run_b.color)
        parts.append(
            f'<rect class="run-b" x="{x_b:.1f}" y="{y_b:.1f}" '
            f'width="{bar_w:.1f}" height="{bar_h_b:.1f}" '
            f'fill="{esc_color_b}" opacity="0.8"/>'
        )
        # Value label B
        parts.append(
            f'<text x="{x_b + bar_w / 2:.1f}" y="{y_b - 4:.1f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="10">'
            f"{html.escape(f'{val_b:.2f}')}</text>"
        )

        # Metric label
        label_x = x_group + group_w / 2
        label_y = margin_top + chart_h + 16
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="11">'
            f"{html.escape(m)}</text>"
        )

        # Delta label
        if config.show_delta and m in deltas:
            delta = deltas[m]
            sign = "+" if delta > 0 else ""
            delta_color = "#2E7D32" if delta > 0 else "#C62828" if delta < 0 else "#666"
            delta_y = label_y + 16
            parts.append(
                f'<text x="{label_x:.1f}" y="{delta_y:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10" '
                f'fill="{delta_color}">'
                f"{html.escape(f'{sign}{delta:.2f}')}</text>"
            )

    # Legend
    if config.show_legend:
        ly = h - 16
        parts.append(
            f'<rect x="{w / 2 - 100}" y="{ly - 8}" width="12" height="12" '
            f'fill="{html.escape(run_a.color)}"/>'
        )
        parts.append(
            f'<text x="{w / 2 - 84}" y="{ly + 2}" '
            f'font-family="sans-serif" font-size="11">'
            f"{html.escape(run_a.label)}</text>"
        )
        parts.append(
            f'<rect x="{w / 2 + 20}" y="{ly - 8}" width="12" height="12" '
            f'fill="{html.escape(run_b.color)}"/>'
        )
        parts.append(
            f'<text x="{w / 2 + 36}" y="{ly + 2}" '
            f'font-family="sans-serif" font-size="11">'
            f"{html.escape(run_b.label)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_radar_svg(
    run_a: RunOverlayData,
    run_b: RunOverlayData,
    config: OverlayConfig | None = None,
) -> str:
    """Generate SVG radar/spider chart with two overlapping polygons.

    Requires 3+ metrics for a meaningful radar chart. Falls back to a
    simple message SVG if fewer than 3 metrics are available.
    """
    if config is None:
        config = OverlayConfig()

    metrics = sorted(set(run_a.metrics) | set(run_b.metrics))
    w = config.width
    h = config.height
    cx = w / 2
    cy = h / 2
    margin = 80
    radius = min(w, h) / 2 - margin

    if len(metrics) < 3:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}">'
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">'
            f"Radar chart requires 3+ metrics</text></svg>"
        )

    n = len(metrics)
    angle_step = 2 * math.pi / n

    all_vals = []
    for m in metrics:
        all_vals.append(run_a.metrics.get(m, 0.0))
        all_vals.append(run_b.metrics.get(m, 0.0))
    max_val = max(abs(v) for v in all_vals) if all_vals else 1.0
    if max_val == 0:
        max_val = 1.0

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    )
    parts.append(f'<rect width="{w}" height="{h}" fill="white"/>')

    # Title
    title_text = html.escape(f"{run_a.label} vs {run_b.label} - Radar")
    parts.append(
        f'<text x="{cx}" y="20" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14" font-weight="bold">'
        f"{title_text}</text>"
    )

    # Grid lines (3 concentric polygons)
    for level in [0.25, 0.5, 0.75, 1.0]:
        grid_points: List[str] = []
        for i in range(n):
            angle = -math.pi / 2 + i * angle_step
            gx = cx + radius * level * math.cos(angle)
            gy = cy + radius * level * math.sin(angle)
            grid_points.append(f"{gx:.1f},{gy:.1f}")
        parts.append(
            f'<polygon points="{" ".join(grid_points)}" '
            f'fill="none" stroke="#ddd" stroke-width="1"/>'
        )

    # Axis lines and labels
    for i, m in enumerate(metrics):
        angle = -math.pi / 2 + i * angle_step
        ax = cx + radius * math.cos(angle)
        ay = cy + radius * math.sin(angle)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{ax:.1f}" y2="{ay:.1f}" '
            f'stroke="#ccc" stroke-width="1"/>'
        )
        # Label position slightly beyond the axis end
        lx = cx + (radius + 18) * math.cos(angle)
        ly = cy + (radius + 18) * math.sin(angle)
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-family="sans-serif" '
            f'font-size="10">{html.escape(m)}</text>'
        )

    def _polygon_points(run: RunOverlayData) -> str:
        pts: List[str] = []
        for i, m in enumerate(metrics):
            val = run.metrics.get(m, 0.0)
            norm = abs(val) / max_val
            angle = -math.pi / 2 + i * angle_step
            px = cx + radius * norm * math.cos(angle)
            py = cy + radius * norm * math.sin(angle)
            pts.append(f"{px:.1f},{py:.1f}")
        return " ".join(pts)

    # Run A polygon
    parts.append(
        f'<polygon class="run-a" points="{_polygon_points(run_a)}" '
        f'fill="{html.escape(run_a.color)}" fill-opacity="0.2" '
        f'stroke="{html.escape(run_a.color)}" stroke-width="2"/>'
    )

    # Run B polygon
    parts.append(
        f'<polygon class="run-b" points="{_polygon_points(run_b)}" '
        f'fill="{html.escape(run_b.color)}" fill-opacity="0.2" '
        f'stroke="{html.escape(run_b.color)}" stroke-width="2"/>'
    )

    # Legend
    if config.show_legend:
        ly = h - 16
        parts.append(
            f'<rect x="{cx - 100}" y="{ly - 8}" width="12" height="12" '
            f'fill="{html.escape(run_a.color)}"/>'
        )
        parts.append(
            f'<text x="{cx - 84}" y="{ly + 2}" '
            f'font-family="sans-serif" font-size="11">'
            f"{html.escape(run_a.label)}</text>"
        )
        parts.append(
            f'<rect x="{cx + 20}" y="{ly - 8}" width="12" height="12" '
            f'fill="{html.escape(run_b.color)}"/>'
        )
        parts.append(
            f'<text x="{cx + 36}" y="{ly + 2}" '
            f'font-family="sans-serif" font-size="11">'
            f"{html.escape(run_b.label)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_delta_table_html(
    run_a: RunOverlayData, run_b: RunOverlayData
) -> str:
    """Generate HTML table showing metric deltas between two runs."""
    metrics = sorted(set(run_a.metrics) | set(run_b.metrics))
    deltas = compute_deltas(run_a, run_b)

    rows: List[str] = []
    for m in metrics:
        val_a = run_a.metrics.get(m, 0.0)
        val_b = run_b.metrics.get(m, 0.0)
        delta = deltas[m]
        if delta > 0:
            direction = "up"
            arrow = "&#9650;"
            color = "#2E7D32"
        elif delta < 0:
            direction = "down"
            arrow = "&#9660;"
            color = "#C62828"
        else:
            direction = "same"
            arrow = "&#9644;"
            color = "#666"

        rows.append(
            f"<tr>"
            f"<td>{html.escape(m)}</td>"
            f"<td>{html.escape(f'{val_a:.4f}')}</td>"
            f"<td>{html.escape(f'{val_b:.4f}')}</td>"
            f"<td>{html.escape(f'{delta:+.4f}')}</td>"
            f'<td style="color:{color}" data-direction="{direction}">'
            f"{arrow} {html.escape(direction)}</td>"
            f"</tr>"
        )

    label_a = html.escape(run_a.label)
    label_b = html.escape(run_b.label)

    return (
        '<table class="delta-table" style="border-collapse:collapse;'
        'width:100%;font-family:sans-serif;font-size:13px;">'
        "<thead><tr>"
        '<th style="text-align:left;padding:8px;border-bottom:2px solid #ccc;">Metric</th>'
        f'<th style="text-align:right;padding:8px;border-bottom:2px solid #ccc;">{label_a}</th>'
        f'<th style="text-align:right;padding:8px;border-bottom:2px solid #ccc;">{label_b}</th>'
        '<th style="text-align:right;padding:8px;border-bottom:2px solid #ccc;">Delta</th>'
        '<th style="text-align:center;padding:8px;border-bottom:2px solid #ccc;">Direction</th>'
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def generate_overlay_dashboard_html(
    run_a: RunOverlayData,
    run_b: RunOverlayData,
    config: OverlayConfig | None = None,
) -> str:
    """Generate a full self-contained HTML dashboard page.

    Includes dual bar chart, radar plot, delta table, and toggle
    visibility controls (checkboxes to show/hide each run's data).
    All CSS is inline; no external dependencies.
    """
    if config is None:
        config = OverlayConfig()

    bar_svg = generate_dual_bar_svg(run_a, run_b, config)
    radar_svg = generate_radar_svg(run_a, run_b, config)
    delta_table = generate_delta_table_html(run_a, run_b)

    label_a = html.escape(run_a.label)
    label_b = html.escape(run_b.label)
    title = html.escape(f"Comparison: {run_a.label} vs {run_b.label}")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #fafafa; color: #333; }}
h1 {{ font-size: 1.4em; }}
h2 {{ font-size: 1.1em; margin-top: 1.5em; }}
.chart-section {{ margin: 20px 0; background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.controls {{ margin: 12px 0; padding: 8px; background: #f0f0f0; border-radius: 4px; }}
.controls label {{ margin-right: 16px; cursor: pointer; }}
table.delta-table td, table.delta-table th {{ padding: 6px 10px; }}
table.delta-table tbody tr:nth-child(even) {{ background: #f8f8f8; }}
</style>
</head>
<body>
<h1>{title}</h1>

<div class="controls">
<label><input type="checkbox" id="toggle-run-a" checked onchange="toggleRun('a', this.checked)"/> Show {label_a}</label>
<label><input type="checkbox" id="toggle-run-b" checked onchange="toggleRun('b', this.checked)"/> Show {label_b}</label>
</div>

<div class="chart-section" id="bar-chart-section">
<h2>Bar Chart Comparison</h2>
{bar_svg}
</div>

<div class="chart-section" id="radar-chart-section">
<h2>Radar Chart Comparison</h2>
{radar_svg}
</div>

<div class="chart-section" id="delta-table-section">
<h2>Metric Deltas</h2>
{delta_table}
</div>

<script>
function toggleRun(run, visible) {{
    var cls = 'run-' + run;
    var elems = document.querySelectorAll('.' + cls);
    for (var i = 0; i < elems.length; i++) {{
        elems[i].style.display = visible ? '' : 'none';
    }}
}}
</script>
</body>
</html>"""
