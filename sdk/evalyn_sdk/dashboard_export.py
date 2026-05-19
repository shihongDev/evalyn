"""In-dashboard data export for HTML evaluation dashboards.

Pure Python, no external dependencies. Generates CSV/JSON export buttons
and copy-to-clipboard functionality for dashboard chart data.
"""

from __future__ import annotations

import csv
import html
import io
import json
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class ExportableChart:
    """A chart's data in a format suitable for export."""

    chart_id: str
    title: str
    data: List[Dict[str, Any]]
    columns: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "data": [dict(row) for row in self.data],
            "columns": list(self.columns),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExportableChart:
        return cls(
            chart_id=data.get("chart_id", ""),
            title=data.get("title", ""),
            data=data.get("data", []),
            columns=data.get("columns", []),
        )


@dataclass
class ExportBundle:
    """A collection of exportable charts with summary data."""

    charts: List[ExportableChart]
    summary: Dict[str, Any]
    generated_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "charts": [c.as_dict() for c in self.charts],
            "summary": dict(self.summary),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExportBundle:
        return cls(
            charts=[
                ExportableChart.from_dict(c) for c in data.get("charts", [])
            ],
            summary=data.get("summary", {}),
            generated_at=data.get("generated_at", ""),
        )


def chart_to_csv(chart: ExportableChart) -> str:
    """Convert chart data to a CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=chart.columns)
    writer.writeheader()
    for row in chart.data:
        writer.writerow({col: row.get(col, "") for col in chart.columns})
    return output.getvalue()


def chart_to_json(chart: ExportableChart) -> str:
    """Convert chart data to a JSON string."""
    return json.dumps(
        {"chart_id": chart.chart_id, "title": chart.title, "data": chart.data},
        indent=2,
    )


def generate_download_button_html(
    chart: ExportableChart, fmt: str = "csv"
) -> str:
    """Generate an HTML button that triggers a JavaScript download.

    Uses data: URI scheme. All content is html.escaped.
    """
    if fmt == "csv":
        content = chart_to_csv(chart)
        mime = "text/csv"
        ext = "csv"
    else:
        content = chart_to_json(chart)
        mime = "application/json"
        ext = "json"

    encoded = b64encode(content.encode("utf-8")).decode("ascii")
    data_uri = f"data:{mime};base64,{encoded}"
    safe_title = html.escape(chart.title)
    safe_id = html.escape(chart.chart_id)
    filename = f"{safe_id}.{ext}"

    return (
        f'<a href="{data_uri}" download="{html.escape(filename)}" '
        f'class="export-btn" '
        f'title="Download {safe_title} as {ext.upper()}">'
        f"Download {safe_title} ({ext.upper()})"
        f"</a>"
    )


def generate_copy_button_html(text: str, label: str = "Copy") -> str:
    """Generate an HTML button that copies text to clipboard via JS."""
    # JS-escape first: backslash, single quote, newline, carriage return
    js_escaped = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    # Then HTML-escape the entire JS expression for the attribute context
    attr_safe = html.escape(f"navigator.clipboard.writeText('{js_escaped}')")
    safe_label = html.escape(label)
    return (
        f'<button class="copy-btn" '
        f'onclick="{attr_safe}">'
        f"{safe_label}"
        f"</button>"
    )


def generate_export_panel_html(bundle: ExportBundle) -> str:
    """Generate a full HTML export panel with download and copy buttons.

    Includes CSV and JSON download buttons for each chart plus a summary
    copy button.
    """
    parts: List[str] = []
    parts.append('<div class="export-panel">')
    parts.append(f"  <h3>{html.escape('Data Export')}</h3>")
    parts.append(f"  <p>Generated at: {html.escape(bundle.generated_at)}</p>")

    for chart in bundle.charts:
        safe_title = html.escape(chart.title)
        parts.append('  <div class="export-chart-section">')
        parts.append(f"    <h4>{safe_title}</h4>")
        parts.append(f"    {generate_download_button_html(chart, 'csv')}")
        parts.append(f"    {generate_download_button_html(chart, 'json')}")
        parts.append("  </div>")

    summary_text = json.dumps(bundle.summary, indent=2)
    parts.append('  <div class="export-summary-section">')
    parts.append("    <h4>Summary</h4>")
    parts.append(
        f"    {generate_copy_button_html(summary_text, 'Copy Summary')}"
    )
    parts.append("  </div>")
    parts.append("</div>")

    return "\n".join(parts)


def build_export_bundle(
    metrics: Dict[str, float],
    failed_items: List[Dict[str, Any]],
    per_metric_data: Dict[str, List[Dict[str, Any]]],
) -> ExportBundle:
    """Build an export bundle from typical dashboard data.

    Creates charts for: metrics summary, failed items, and per-metric
    detail tables.
    """
    charts: List[ExportableChart] = []

    # Metrics overview chart
    metrics_rows = [
        {"metric": name, "score": score} for name, score in metrics.items()
    ]
    charts.append(
        ExportableChart(
            chart_id="metrics_overview",
            title="Metrics Overview",
            data=metrics_rows,
            columns=["metric", "score"],
        )
    )

    # Failed items chart
    if failed_items:
        columns = list(failed_items[0].keys()) if failed_items else []
        charts.append(
            ExportableChart(
                chart_id="failed_items",
                title="Failed Items",
                data=failed_items,
                columns=columns,
            )
        )

    # Per-metric detail charts
    for metric_name, rows in per_metric_data.items():
        columns = list(rows[0].keys()) if rows else []
        safe_id = metric_name.replace(" ", "_").lower()
        charts.append(
            ExportableChart(
                chart_id=f"metric_{safe_id}",
                title=f"{metric_name} Details",
                data=rows,
                columns=columns,
            )
        )

    generated_at = datetime.now(timezone.utc).isoformat()

    summary: Dict[str, Any] = {
        "total_metrics": len(metrics),
        "failed_count": len(failed_items),
        "metrics": dict(metrics),
    }

    return ExportBundle(
        charts=charts,
        summary=summary,
        generated_at=generated_at,
    )
