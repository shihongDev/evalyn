"""Self-contained HTML dashboard generator for evaluation results.

Pure Python, no external dependencies. Uses html module for escaping.
Generates responsive dashboards with inline CSS/JS, dark/light theme support,
metric cards, tables, and SVG bar charts.
"""

from __future__ import annotations

import html
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DashboardConfig:
    """Configuration for dashboard rendering."""

    title: str = "Evalyn Dashboard"
    theme: str = "dark"
    refresh_interval: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "theme": self.theme,
            "refresh_interval": self.refresh_interval,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardConfig:
        return cls(
            title=data.get("title", "Evalyn Dashboard"),
            theme=data.get("theme", "dark"),
            refresh_interval=data.get("refresh_interval", 0),
        )


@dataclass
class DashboardPanel:
    """A single panel in the dashboard."""

    panel_id: str
    title: str
    content_html: str
    panel_type: str = "metrics"

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "content_html": self.content_html,
            "panel_type": self.panel_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardPanel:
        return cls(
            panel_id=data.get("panel_id", ""),
            title=data.get("title", ""),
            content_html=data.get("content_html", ""),
            panel_type=data.get("panel_type", "metrics"),
        )


@dataclass
class Dashboard:
    """A complete dashboard with config and panels."""

    config: DashboardConfig
    panels: list[DashboardPanel] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "panels": [p.as_dict() for p in self.panels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dashboard:
        config = DashboardConfig.from_dict(data.get("config", {}))
        panels = [DashboardPanel.from_dict(p) for p in data.get("panels", [])]
        return cls(config=config, panels=panels)


def _panel_id() -> str:
    """Generate a unique panel ID."""
    return f"panel-{uuid.uuid4().hex[:8]}"


def generate_metrics_panel(metrics: dict[str, float]) -> DashboardPanel:
    """Generate an HTML panel showing metric cards with scores.

    Each metric is rendered as a card with the metric name and its value.
    """
    cards: list[str] = []
    for name, value in metrics.items():
        escaped_name = html.escape(str(name))
        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        color = _score_color(value)
        cards.append(
            f'<div class="metric-card">'
            f'<div class="metric-name">{escaped_name}</div>'
            f'<div class="metric-value" style="color:{color}">{formatted_value}</div>'
            f"</div>"
        )
    content = f'<div class="metrics-grid">{"".join(cards)}</div>'
    return DashboardPanel(
        panel_id=_panel_id(),
        title="Metrics",
        content_html=content,
        panel_type="metrics",
    )


def generate_table_panel(
    title: str, headers: list[str], rows: list[list[str]]
) -> DashboardPanel:
    """Generate an HTML table panel. All content is html.escaped."""
    header_cells = "".join(
        f"<th>{html.escape(h)}</th>" for h in headers
    )
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    content = (
        f'<table class="dashboard-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>"
    )
    return DashboardPanel(
        panel_id=_panel_id(),
        title=title,
        content_html=content,
        panel_type="table",
    )


def generate_chart_panel(title: str, data: dict[str, float]) -> DashboardPanel:
    """Generate an SVG bar chart panel."""
    if not data:
        svg = '<svg width="400" height="200"><text x="10" y="100">No data</text></svg>'
        return DashboardPanel(
            panel_id=_panel_id(),
            title=title,
            content_html=svg,
            panel_type="chart",
        )

    max_val = max(abs(v) for v in data.values()) if data.values() else 1.0
    if max_val == 0:
        max_val = 1.0

    bar_height = 30
    gap = 10
    label_width = 120
    chart_width = 400
    bar_area_width = chart_width - label_width - 20
    svg_height = len(data) * (bar_height + gap) + gap

    bars: list[str] = []
    for i, (label, value) in enumerate(data.items()):
        y = gap + i * (bar_height + gap)
        bar_width = max(1, int(abs(value) / max_val * bar_area_width))
        color = _score_color(value)
        escaped_label = html.escape(str(label))
        formatted = f"{value:.2f}"
        bars.append(
            f'<text x="5" y="{y + bar_height // 2 + 5}" '
            f'font-size="12" fill="currentColor">{escaped_label}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_width}" '
            f'height="{bar_height}" fill="{color}" rx="3"/>'
            f'<text x="{label_width + bar_width + 5}" y="{y + bar_height // 2 + 5}" '
            f'font-size="11" fill="currentColor">{formatted}</text>'
        )

    svg = (
        f'<svg width="{chart_width}" height="{svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f"{''.join(bars)}</svg>"
    )
    return DashboardPanel(
        panel_id=_panel_id(),
        title=title,
        content_html=svg,
        panel_type="chart",
    )


def _score_color(value: float) -> str:
    """Return a color based on a 0-1 score range."""
    if value >= 0.8:
        return "#4caf50"
    if value >= 0.5:
        return "#ff9800"
    return "#f44336"


def render_dashboard(dashboard: Dashboard) -> str:
    """Render a full self-contained HTML page with inline CSS/JS.

    Supports dark/light theme, responsive layout, and optional auto-refresh.
    """
    config = dashboard.config
    is_dark = config.theme == "dark"

    bg = "#1e1e2e" if is_dark else "#f5f5f5"
    fg = "#cdd6f4" if is_dark else "#333333"
    card_bg = "#313244" if is_dark else "#ffffff"
    border = "#45475a" if is_dark else "#dddddd"

    panels_html: list[str] = []
    for panel in dashboard.panels:
        escaped_title = html.escape(panel.title)
        panels_html.append(
            f'<div class="panel panel-{html.escape(panel.panel_type)}">'
            f'<h2 class="panel-title">{escaped_title}</h2>'
            f'<div class="panel-content">{panel.content_html}</div>'
            f"</div>"
        )

    refresh_meta = ""
    refresh_js = ""
    if config.refresh_interval > 0:
        refresh_meta = (
            f'<meta http-equiv="refresh" content="{config.refresh_interval}">'
        )
        refresh_js = (
            f"setTimeout(function(){{location.reload();}}, "
            f"{config.refresh_interval * 1000});"
        )

    escaped_title = html.escape(config.title)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
{refresh_meta}
<title>{escaped_title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: {bg};
    color: {fg};
    padding: 20px;
}}
h1 {{ text-align: center; margin-bottom: 20px; font-size: 1.5rem; }}
.dashboard-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 16px;
    max-width: 1200px;
    margin: 0 auto;
}}
.panel {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 16px;
    overflow: auto;
}}
.panel-title {{
    font-size: 1rem;
    margin-bottom: 12px;
    border-bottom: 1px solid {border};
    padding-bottom: 8px;
}}
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
}}
.metric-card {{
    text-align: center;
    padding: 12px;
    background: {bg};
    border-radius: 6px;
}}
.metric-name {{ font-size: 0.8rem; opacity: 0.8; margin-bottom: 4px; }}
.metric-value {{ font-size: 1.4rem; font-weight: bold; }}
.dashboard-table {{
    width: 100%;
    border-collapse: collapse;
}}
.dashboard-table th, .dashboard-table td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid {border};
}}
.dashboard-table th {{
    font-weight: 600;
    opacity: 0.9;
}}
svg {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
<div class="dashboard-grid">
{''.join(panels_html)}
</div>
<script>{refresh_js}</script>
</body>
</html>"""


def build_eval_dashboard(
    metrics: dict[str, float],
    failed_items: list[dict[str, Any]],
    run_info: dict[str, Any],
) -> Dashboard:
    """Build a complete dashboard from evaluation results.

    Parameters
    ----------
    metrics : dict mapping metric names to float scores
    failed_items : list of dicts, each with at least "id" and "reason" keys
    run_info : dict with run metadata (e.g. "run_id", "dataset", "timestamp")
    """
    config = DashboardConfig(
        title=f"Evalyn - {run_info.get('dataset', 'Evaluation')} Results",
    )
    panels: list[DashboardPanel] = []

    # Info panel
    info_rows = [[str(k), str(v)] for k, v in run_info.items()]
    if info_rows:
        panels.append(
            generate_table_panel("Run Info", ["Property", "Value"], info_rows)
        )

    # Metrics panel
    if metrics:
        panels.append(generate_metrics_panel(metrics))

    # Chart panel
    if metrics:
        panels.append(generate_chart_panel("Score Distribution", metrics))

    # Failed items panel
    if failed_items:
        headers = ["ID", "Reason"]
        rows = [
            [str(item.get("id", "?")), str(item.get("reason", "unknown"))]
            for item in failed_items
        ]
        panels.append(generate_table_panel("Failed Items", headers, rows))

    return Dashboard(config=config, panels=panels)
