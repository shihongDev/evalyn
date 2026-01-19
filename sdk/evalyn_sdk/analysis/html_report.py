"""
HTML report generation for evaluation dashboards.

Generates self-contained HTML reports with embedded Chart.js visualizations.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import RunAnalysis


def _format_io_content(content: Any, max_length: int = 10000) -> str:
    """Format input/output content for HTML display with escaping and truncation."""
    if content is None:
        return "No data available"

    # Convert to string and format
    if isinstance(content, dict):
        try:
            text = json.dumps(content, indent=2, ensure_ascii=False, default=str)
        except Exception:
            text = str(content)
    else:
        text = str(content)

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"

    # HTML escape to prevent XSS and preserve formatting
    return html.escape(text)


def _render_failed_items(analysis: RunAnalysis, item_details: dict) -> str:
    """Render the failed items HTML section."""
    parts = []
    for i, item_id in enumerate(analysis.failed_items[:30]):
        item_stats = analysis.item_stats[item_id]
        failed_count = sum(
            1 for m, r in item_stats.metric_results.items() if r["passed"] is False
        )
        total_with_pass_fail = sum(
            1 for m, r in item_stats.metric_results.items() if r["passed"] is not None
        )

        # Get input/output from item_details
        item_info = item_details.get(item_id, {})
        input_content = _format_io_content(
            item_info.get("input", "No input data available")
        )
        output_content = _format_io_content(
            item_info.get("output", "No output data available")
        )

        # Render failed metrics with reasoning
        metric_details_html = []
        for m, r in item_stats.metric_results.items():
            if r["passed"] is False:
                score = r.get("score", 0) or 0
                reason = (
                    r.get("reason")
                    or (r.get("details") or {}).get("error")
                    or "No reasoning available"
                )
                reason_escaped = html.escape(str(reason))
                metric_details_html.append(f"""<div class="metric-detail">
                    <div class="metric-detail-header">
                        <span class="metric-detail-name">✗ {html.escape(m)}</span>
                        <span class="metric-detail-score">({score:.2f})</span>
                    </div>
                    <div class="metric-detail-reason">{reason_escaped}</div>
                </div>""")

        parts.append(f"""<div class="failed-item" id="item-{i}">
            <div class="failed-item-header">
                <div>
                    <span class="failed-item-id">{html.escape(item_id[:24])}...</span>
                    <span class="failed-item-summary">{failed_count}/{total_with_pass_fail} failed</span>
                </div>
                <button class="expand-btn" onclick="toggleExpand({i})">Expand</button>
            </div>
            <div class="failed-item-content">
                <div class="io-section">
                    <div class="io-block input">
                        <div class="io-label">INPUT</div>
                        <pre class="io-content">{input_content}</pre>
                    </div>
                    <div class="io-block output">
                        <div class="io-label">OUTPUT</div>
                        <pre class="io-content">{output_content}</pre>
                    </div>
                </div>
                <div class="metric-details-section">
                    <div class="io-label">FAILED METRICS</div>
                    {"".join(metric_details_html)}
                </div>
            </div>
        </div>""")

    return "".join(parts)


def generate_html_report(
    analysis: RunAnalysis, verbose: bool = False, item_details: Optional[dict] = None
) -> str:
    """Generate a warm-themed evaluation dashboard.

    Uses Chart.js for interactive charts. No external images - everything is embedded.
    Styled with warm cream backgrounds and terracotta accents.

    Args:
        analysis: The run analysis data
        verbose: Include additional details
        item_details: Optional dict mapping item_id -> {"input": ..., "output": ...}
    """
    item_details = item_details or {}

    # Prepare metric data sorted by pass rate (metrics without pass/fail go to end)
    metrics_by_pass_rate = sorted(
        analysis.metric_stats.values(),
        key=lambda m: -(m.pass_rate if m.pass_rate is not None else -1),
    )
    metric_labels = json.dumps([m.metric_id for m in metrics_by_pass_rate])
    # For charts: use 0 for None pass rates (they'll be shown as N/A in table)
    pass_rates = json.dumps(
        [
            round(m.pass_rate * 100, 1) if m.pass_rate is not None else 0
            for m in metrics_by_pass_rate
        ]
    )
    passed_counts = json.dumps([m.passed for m in metrics_by_pass_rate])
    failed_counts = json.dumps([m.failed for m in metrics_by_pass_rate])

    # Color coding for pass rates (warm theme)
    def get_pass_rate_color(m):
        if m.pass_rate is None:
            return "#A89F97"  # Muted for metrics without pass/fail
        if m.pass_rate >= 0.8:
            return "#6B8E8E"  # Sage green for good
        if m.pass_rate >= 0.5:
            return "#D4A27F"  # Terracotta for warning
        return "#C97B63"  # Coral for bad

    pass_rate_colors = json.dumps(
        [get_pass_rate_color(m) for m in metrics_by_pass_rate]
    )

    # Prepare per-item data for detailed view
    item_data_rows = []
    for item_id, item in analysis.item_stats.items():
        failed_metrics = [
            m for m, r in item.metric_results.items() if r["passed"] is False
        ]
        status = "pass" if item.all_passed else "fail"
        item_data_rows.append(
            {
                "id": item_id[:12] + "...",
                "full_id": item_id,
                "passed": item.metrics_passed,
                "failed": item.metrics_failed,
                "status": status,
                "failed_metrics": ", ".join(failed_metrics) if failed_metrics else "-",
            }
        )

    # Score distribution data (for box plot simulation)
    score_dist_data = []
    for ms in metrics_by_pass_rate:
        if ms.scores:
            sorted_scores = sorted(ms.scores)
            n = len(sorted_scores)
            q1_idx = n // 4
            q3_idx = (3 * n) // 4
            score_dist_data.append(
                {
                    "metric": ms.metric_id,
                    "min": ms.min_score,
                    "q1": sorted_scores[q1_idx] if n > 0 else 0,
                    "median": sorted_scores[n // 2] if n > 0 else 0,
                    "q3": sorted_scores[q3_idx] if n > 0 else 0,
                    "max": ms.max_score,
                    "avg": ms.avg_score,
                }
            )

    # Calculate correlations if we have enough data
    correlation_data = None
    if len(analysis.item_stats) > 2 and len(analysis.metric_stats) > 1:
        metric_ids = [m.metric_id for m in metrics_by_pass_rate]
        # Build score matrix
        score_matrix = []
        for item_id in analysis.item_stats:
            row = []
            for metric_id in metric_ids:
                if metric_id in analysis.item_stats[item_id].metric_results:
                    result = analysis.item_stats[item_id].metric_results[metric_id]
                    row.append(result["score"])
                else:
                    row.append(None)
            score_matrix.append(row)

        # Compute correlation matrix
        correlations = []
        for i in range(len(metric_ids)):
            row = []
            for j in range(len(metric_ids)):
                if i == j:
                    row.append(1.0)
                else:
                    # Pearson correlation
                    x_vals = [
                        score_matrix[k][i]
                        for k in range(len(score_matrix))
                        if score_matrix[k][i] is not None
                        and score_matrix[k][j] is not None
                    ]
                    y_vals = [
                        score_matrix[k][j]
                        for k in range(len(score_matrix))
                        if score_matrix[k][i] is not None
                        and score_matrix[k][j] is not None
                    ]
                    if len(x_vals) > 1:
                        x_mean = sum(x_vals) / len(x_vals)
                        y_mean = sum(y_vals) / len(y_vals)
                        numerator = sum(
                            (x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)
                        )
                        denom_x = sum((x - x_mean) ** 2 for x in x_vals) ** 0.5
                        denom_y = sum((y - y_mean) ** 2 for y in y_vals) ** 0.5
                        if denom_x > 0 and denom_y > 0:
                            row.append(round(numerator / (denom_x * denom_y), 2))
                        else:
                            row.append(0)
                    else:
                        row.append(0)
            correlations.append(row)
        correlation_data = {"labels": metric_ids, "matrix": correlations}

    all_passed_count = sum(
        1 for item in analysis.item_stats.values() if item.all_passed
    )

    # Import the HTML template
    return _generate_html_content(
        analysis=analysis,
        metrics_by_pass_rate=metrics_by_pass_rate,
        metric_labels=metric_labels,
        pass_rates=pass_rates,
        pass_rate_colors=pass_rate_colors,
        score_dist_data=score_dist_data,
        passed_counts=passed_counts,
        failed_counts=failed_counts,
        correlation_data=correlation_data,
        all_passed_count=all_passed_count,
        item_details=item_details,
    )


def _generate_html_content(
    analysis: RunAnalysis,
    metrics_by_pass_rate: List,
    metric_labels: str,
    pass_rates: str,
    pass_rate_colors: str,
    score_dist_data: List[Dict],
    passed_counts: str,
    failed_counts: str,
    correlation_data: Optional[Dict],
    all_passed_count: int,
    item_details: dict,
) -> str:
    """Generate the actual HTML content. Split out for readability."""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval Dashboard - {analysis.dataset_name}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {_get_css_styles()}
</head>
<body>
    <div class="dashboard">
        {_render_header(analysis)}
        {_render_kpi_bar(analysis, metrics_by_pass_rate, all_passed_count)}
        {_render_charts_section(metric_labels, pass_rates, pass_rate_colors, score_dist_data)}
        {_render_metrics_table(analysis, metrics_by_pass_rate)}
        {_render_failed_items_section(analysis, item_details)}
        {_render_footer(analysis, all_passed_count)}
    </div>
    {_get_javascript(metric_labels, pass_rates, pass_rate_colors, score_dist_data, passed_counts, failed_counts, correlation_data)}
</body>
</html>
"""
    return html_content


def _get_css_styles() -> str:
    """Return the CSS styles for the HTML report."""
    return """<style>
        :root {
            /* Backgrounds - warm cream tones */
            --bg-primary: #FFFBF7;
            --bg-secondary: #FBF7F3;
            --bg-tertiary: #F5EDE6;
            --bg-hover: #F0E8E0;

            /* Accents - terracotta and warm tones */
            --accent-primary: #D4A27F;
            --accent-secondary: #C4836A;
            --accent-muted: #A89F97;
            --accent-purple: #9B8AA6;

            /* Status */
            --status-pass: #6B8E8E;
            --status-fail: #C97B63;
            --status-warn: #D4A27F;

            /* Text */
            --text-primary: #1A1A1A;
            --text-secondary: #555555;
            --text-muted: #888888;

            /* Borders */
            --border-subtle: rgba(212, 162, 127, 0.2);
            --border-strong: rgba(212, 162, 127, 0.4);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            font-size: 14px;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }

        .dashboard {
            max-width: 1600px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Header Bar */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 24px;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 24px;
        }

        .header-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .header-meta {
            display: flex;
            gap: 16px;
            font-size: 13px;
            color: var(--text-muted);
        }

        .header-meta span {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .header-meta .value {
            color: var(--text-secondary);
            font-family: 'DM Mono', 'SF Mono', 'Fira Code', Menlo, Monaco, monospace;
        }

        /* KPI Bar - horizontal, no cards */
        .kpi-bar {
            display: flex;
            align-items: stretch;
            gap: 0;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 24px;
            overflow-x: auto;
        }

        .kpi-item {
            flex: 1;
            min-width: 120px;
            padding: 0 24px;
            border-right: 1px solid var(--border-subtle);
            text-align: center;
        }

        .kpi-item:last-child {
            border-right: none;
        }

        .kpi-value {
            font-size: 32px;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 4px;
            font-variant-numeric: tabular-nums;
        }

        .kpi-value.pass { color: var(--status-pass); }
        .kpi-value.warn { color: var(--status-warn); }
        .kpi-value.fail { color: var(--status-fail); }
        .kpi-value.neutral { color: var(--text-primary); }

        .kpi-unit {
            font-size: 16px;
            font-weight: 400;
            color: var(--text-muted);
            margin-left: 2px;
        }

        .kpi-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        /* Section layout */
        .section {
            margin-bottom: 24px;
        }

        .section-header {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-subtle);
        }

        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        .chart-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            padding: 20px;
        }

        .chart-title {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }

        .chart-container {
            position: relative;
            height: 300px;
        }

        /* Dense Table */
        .table-wrapper {
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            font-variant-numeric: tabular-nums;
        }

        th {
            padding: 10px 12px;
            text-align: left;
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: var(--text-muted);
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-strong);
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
        }

        th:hover {
            color: var(--text-secondary);
        }

        /* Pinned average row */
        tr.avg-row {
            background: var(--bg-tertiary);
            font-weight: 600;
        }

        tr.avg-row td {
            border-bottom: 2px solid var(--border-strong);
            color: var(--accent-primary);
        }

        td {
            padding: 8px 12px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-secondary);
        }

        tr:hover {
            background: var(--bg-hover);
        }

        /* Mini score bar */
        .score-bar {
            display: inline-block;
            width: 40px;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
            vertical-align: middle;
            margin-right: 8px;
        }

        .score-bar-fill {
            height: 100%;
            border-radius: 3px;
        }

        .score-bar-fill.high { background: var(--status-pass); }
        .score-bar-fill.mid { background: var(--status-warn); }
        .score-bar-fill.low { background: var(--status-fail); }

        /* Status indicators */
        .status-pass { color: var(--status-pass); }
        .status-fail { color: var(--status-fail); }
        .status-warn { color: var(--status-warn); }

        /* Failed Items Section */
        .failed-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
        }

        .failed-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .failed-header h3 {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .failed-count {
            font-size: 12px;
            color: var(--status-fail);
            font-weight: 600;
        }

        .failed-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .failed-item {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .failed-item:last-child {
            border-bottom: none;
        }

        .failed-item-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .failed-item-id {
            font-family: 'DM Mono', 'SF Mono', 'Fira Code', Menlo, Monaco, monospace;
            font-size: 12px;
            color: var(--text-muted);
        }

        .failed-item-summary {
            font-size: 12px;
            color: var(--status-fail);
        }

        .failed-item-content {
            display: none;
        }

        .failed-item.expanded .failed-item-content {
            display: block;
        }

        .io-block {
            margin-top: 12px;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            font-family: 'DM Mono', 'SF Mono', 'Fira Code', Menlo, Monaco, monospace;
            font-size: 12px;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .io-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 8px;
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        }

        .io-block.input {
            border-left: 3px solid var(--accent-purple);
        }

        .io-block.output {
            border-left: 3px solid var(--accent-secondary);
        }

        .expand-btn {
            background: none;
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 4px 10px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
        }

        .expand-btn:hover {
            border-color: var(--text-muted);
            color: var(--text-secondary);
        }

        /* Enhanced failed item expansion */
        .io-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }

        @media (max-width: 900px) {
            .io-section {
                grid-template-columns: 1fr;
            }
        }

        .io-content {
            margin: 0;
            padding: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 12px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
            color: var(--text-secondary);
        }

        .metric-details-section {
            margin-top: 16px;
        }

        .metric-detail {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--status-fail);
            border-radius: 4px;
            padding: 12px;
            margin-top: 8px;
        }

        .metric-detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .metric-detail-name {
            font-weight: 600;
            color: var(--status-fail);
            font-size: 13px;
        }

        .metric-detail-score {
            font-size: 12px;
            color: var(--text-muted);
            font-family: 'DM Mono', 'SF Mono', 'Fira Code', Menlo, Monaco, monospace;
        }

        .metric-detail-reason {
            font-size: 12px;
            line-height: 1.5;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            padding: 10px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 150px;
            overflow-y: auto;
        }

        /* Metadata footer */
        .footer {
            margin-top: 24px;
            padding: 16px 0;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            gap: 32px;
            font-size: 12px;
            color: var(--text-muted);
        }

        .footer-item {
            display: flex;
            gap: 8px;
        }

        .footer-label {
            color: var(--text-muted);
        }

        .footer-value {
            color: var(--text-secondary);
            font-family: 'DM Mono', 'SF Mono', 'Fira Code', Menlo, Monaco, monospace;
        }

        /* Responsive */
        @media (max-width: 1024px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .kpi-bar {
                flex-wrap: wrap;
            }
            .kpi-item {
                flex: 1 1 45%;
                border-right: none;
                border-bottom: 1px solid var(--border-subtle);
                padding: 16px;
            }
            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-strong);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>"""


def _render_header(analysis: RunAnalysis) -> str:
    """Render the header section."""
    return f"""<!-- Header Bar -->
        <div class="header">
            <div class="header-left">
                <div class="header-title">Evaluation Dashboard</div>
                <div class="header-meta">
                    <span>Dataset: <span class="value">{analysis.dataset_name}</span></span>
                    <span>Run: <span class="value">{analysis.run_id[:12]}...</span></span>
                    <span>Time: <span class="value">{analysis.created_at[:19] if analysis.created_at else "N/A"}</span></span>
                </div>
            </div>
        </div>"""


def _render_kpi_bar(
    analysis: RunAnalysis, metrics_by_pass_rate: List, all_passed_count: int
) -> str:
    """Render the KPI bar section."""
    pass_class = (
        "pass"
        if analysis.overall_pass_rate >= 0.8
        else "warn"
        if analysis.overall_pass_rate >= 0.5
        else "fail"
    )
    failed_class = "pass" if len(analysis.failed_items) == 0 else "fail"

    metric_kpis = ""
    for ms in list(metrics_by_pass_rate)[:5]:
        if ms.has_pass_fail:
            kpi_class = (
                "pass"
                if ms.pass_rate is not None and ms.pass_rate >= 0.8
                else "warn"
                if ms.pass_rate is not None and ms.pass_rate >= 0.5
                else "neutral"
                if ms.pass_rate is None
                else "fail"
            )
            value = f"{ms.pass_rate * 100:.0f}" if ms.pass_rate is not None else "N/A"
            unit = "%" if ms.pass_rate is not None else ""
            metric_kpis += f"""<div class="kpi-item">
                <div class="kpi-value {kpi_class}">{value}<span class="kpi-unit">{unit}</span></div>
                <div class="kpi-label">{ms.metric_id[:12]}</div>
            </div>"""

    return f"""<!-- KPI Bar -->
        <div class="kpi-bar">
            <div class="kpi-item">
                <div class="kpi-value neutral">{analysis.total_items}</div>
                <div class="kpi-label">Items</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-value neutral">{analysis.total_metrics}</div>
                <div class="kpi-label">Metrics</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-value {pass_class}">{analysis.overall_pass_rate * 100:.1f}<span class="kpi-unit">%</span></div>
                <div class="kpi-label">Pass Rate</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-value {failed_class}">{len(analysis.failed_items)}</div>
                <div class="kpi-label">Failed</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-value pass">{all_passed_count}</div>
                <div class="kpi-label">All Passed</div>
            </div>
            {metric_kpis}
        </div>"""


def _render_charts_section(
    metric_labels: str,
    pass_rates: str,
    pass_rate_colors: str,
    score_dist_data: List[Dict],
) -> str:
    """Render the charts grid section."""
    return """<!-- Charts Grid -->
        <div class="charts-grid">
            <div class="chart-box">
                <div class="chart-title">Pass Rate by Metric</div>
                <div class="chart-container">
                    <canvas id="passRateChart"></canvas>
                </div>
            </div>
            <div class="chart-box">
                <div class="chart-title">Score Distribution</div>
                <div class="chart-container">
                    <canvas id="scoreDistChart"></canvas>
                </div>
            </div>
        </div>"""


def _render_metrics_table(analysis: RunAnalysis, metrics_by_pass_rate: List) -> str:
    """Render the metrics details table."""
    # Calculate averages
    pass_rate_metrics = [m for m in metrics_by_pass_rate if m.pass_rate is not None]
    avg_pass_rate = (
        f"{sum(m.pass_rate for m in pass_rate_metrics) / len(pass_rate_metrics) * 100:.1f}%"
        if pass_rate_metrics
        else "N/A"
    )
    avg_score = (
        sum(m.avg_score for m in metrics_by_pass_rate) / len(metrics_by_pass_rate)
        if metrics_by_pass_rate
        else 0
    )
    total_passed = sum(m.passed for m in metrics_by_pass_rate if m.has_pass_fail)
    total_failed = sum(m.failed for m in metrics_by_pass_rate if m.has_pass_fail)

    rows = ""
    for ms in metrics_by_pass_rate:
        type_class = "status-pass" if ms.metric_type == "objective" else "status-warn"
        if ms.pass_rate is not None:
            bar_class = (
                "high"
                if ms.pass_rate >= 0.8
                else "mid"
                if ms.pass_rate >= 0.5
                else "low"
            )
            status_class = (
                "status-pass"
                if ms.pass_rate >= 0.8
                else "status-warn"
                if ms.pass_rate >= 0.5
                else "status-fail"
            )
            pass_rate_cell = f'<span class="score-bar"><span class="score-bar-fill {bar_class}" style="width: {ms.pass_rate * 100}%"></span></span><span class="{status_class}">{ms.pass_rate * 100:.1f}%</span>'
        else:
            pass_rate_cell = '<span style="color: var(--text-muted);">N/A</span>'

        passed_cell = ms.passed if ms.has_pass_fail else "-"
        passed_class = "status-pass" if ms.has_pass_fail else ""
        failed_cell = ms.failed if ms.has_pass_fail else "-"
        failed_class = "status-fail" if ms.failed > 0 else ""

        rows += f'''<tr>
            <td style="color: var(--text-primary); font-weight: 500;">{ms.metric_id}</td>
            <td><span class="{type_class}">{ms.metric_type[:3]}</span></td>
            <td>{pass_rate_cell}</td>
            <td>{ms.avg_score:.3f}</td>
            <td>{ms.min_score:.3f}</td>
            <td>{ms.max_score:.3f}</td>
            <td>{ms.std_dev:.3f}</td>
            <td class="{passed_class}">{passed_cell}</td>
            <td class="{failed_class}">{failed_cell}</td>
        </tr>'''

    return f"""<!-- Metric Details Table -->
        <div class="section">
            <div class="section-header">Metric Details</div>
            <div class="table-wrapper">
                <table id="metricsTable">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Type</th>
                            <th>Pass Rate</th>
                            <th>Avg Score</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Std Dev</th>
                            <th>Passed</th>
                            <th>Failed</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Average row pinned at top -->
                        <tr class="avg-row">
                            <td>AVG</td>
                            <td>-</td>
                            <td>{avg_pass_rate}</td>
                            <td>{avg_score:.3f}</td>
                            <td>-</td>
                            <td>-</td>
                            <td>-</td>
                            <td>{total_passed}</td>
                            <td>{total_failed}</td>
                        </tr>
                        {rows}
                    </tbody>
                </table>
            </div>
        </div>"""


def _render_failed_items_section(analysis: RunAnalysis, item_details: dict) -> str:
    """Render the failed items section."""
    if not analysis.failed_items:
        return ""

    more_text = (
        f'<div class="failed-item" style="color: var(--text-muted); text-align: center;">...and {len(analysis.failed_items) - 30} more</div>'
        if len(analysis.failed_items) > 30
        else ""
    )

    return f"""<!-- Failed Items Section -->
        <div class="section">
            <div class="section-header">Failed Items</div>
            <div class="failed-section">
                <div class="failed-header">
                    <h3>Items with metric failures</h3>
                    <span class="failed-count">{len(analysis.failed_items)} items</span>
                </div>
                <div class="failed-list">
                    {_render_failed_items(analysis, item_details)}
                    {more_text}
                </div>
            </div>
        </div>"""


def _render_footer(analysis: RunAnalysis, all_passed_count: int) -> str:
    """Render the footer section."""
    return f"""<!-- Footer -->
        <div class="footer">
            <div class="footer-item">
                <span class="footer-label">Run ID:</span>
                <span class="footer-value">{analysis.run_id}</span>
            </div>
            <div class="footer-item">
                <span class="footer-label">Dataset:</span>
                <span class="footer-value">{analysis.dataset_name}</span>
            </div>
            <div class="footer-item">
                <span class="footer-label">Created:</span>
                <span class="footer-value">{analysis.created_at}</span>
            </div>
            <div class="footer-item">
                <span class="footer-label">Passing All:</span>
                <span class="footer-value">{all_passed_count}/{analysis.total_items}</span>
            </div>
        </div>"""


def _get_javascript(
    metric_labels: str,
    pass_rates: str,
    pass_rate_colors: str,
    score_dist_data: List[Dict],
    passed_counts: str,
    failed_counts: str,
    correlation_data: Optional[Dict],
) -> str:
    """Return the JavaScript for the HTML report."""
    correlation_js = ""
    if correlation_data and len(correlation_data["labels"]) <= 10:
        correlation_js = f"""
        // Correlation Heatmap - dark theme colors
        const corrData = {json.dumps(correlation_data)};
        if (corrData) {{
            const corrCtx = document.getElementById('correlationChart')?.getContext('2d');
            if (corrCtx) {{
                const heatmapData = [];
                for (let i = 0; i < corrData.labels.length; i++) {{
                    for (let j = 0; j < corrData.labels.length; j++) {{
                        heatmapData.push({{
                            x: corrData.labels[j],
                            y: corrData.labels[i],
                            v: corrData.matrix[i][j]
                        }});
                    }}
                }}
            }}
        }}
        """

    return f"""<script>
        // Warm theme colors - Anthropic style
        const colors = {{
            accent: '#D4A27F',
            accentDim: 'rgba(212, 162, 127, 0.3)',
            secondary: '#C4836A',
            error: '#C97B63',
            errorDim: 'rgba(201, 123, 99, 0.3)',
            warning: '#D4A27F',
            success: '#6B8E8E',
            muted: '#A89F97',
            border: 'rgba(212, 162, 127, 0.2)',
            gridLine: 'rgba(212, 162, 127, 0.15)',
            bg: '#FFFBF7',
            text: '#1A1A1A',
            textMuted: '#888888'
        }};

        // Chart.js defaults for warm theme
        Chart.defaults.font.family = "'DM Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif";
        Chart.defaults.color = colors.textMuted;
        Chart.defaults.borderColor = colors.border;

        // Toggle expand for failed items
        function toggleExpand(idx) {{
            const item = document.getElementById('item-' + idx);
            if (item) {{
                const content = item.querySelector('.failed-item-content');
                const btn = item.querySelector('.expand-btn');
                if (content.style.display === 'none' || !content.style.display) {{
                    content.style.display = 'block';
                    btn.textContent = 'Collapse';
                    item.classList.add('expanded');
                }} else {{
                    content.style.display = 'none';
                    btn.textContent = 'Expand';
                    item.classList.remove('expanded');
                }}
            }}
        }}

        // Pass Rate Chart - horizontal bars with thin styling
        const passRateCtx = document.getElementById('passRateChart').getContext('2d');
        new Chart(passRateCtx, {{
            type: 'bar',
            data: {{
                labels: {metric_labels},
                datasets: [{{
                    data: {pass_rates},
                    backgroundColor: {pass_rate_colors},
                    borderColor: {pass_rate_colors},
                    borderWidth: 1,
                    borderRadius: 2,
                    borderSkipped: false,
                    barThickness: 16,
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        backgroundColor: colors.bg,
                        titleColor: colors.text,
                        bodyColor: colors.text,
                        borderColor: colors.border,
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {{
                            label: (ctx) => ctx.raw.toFixed(1) + '%'
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{
                            color: colors.gridLine,
                            lineWidth: 1
                        }},
                        ticks: {{ color: colors.textMuted }},
                        title: {{ display: true, text: 'Pass Rate (%)', color: colors.muted }}
                    }},
                    y: {{
                        grid: {{ display: false }},
                        ticks: {{ color: colors.text }}
                    }}
                }}
            }}
        }});

        // Score Distribution Chart
        const scoreDistCtx = document.getElementById('scoreDistChart').getContext('2d');
        const scoreDistData = {json.dumps(score_dist_data)};
        new Chart(scoreDistCtx, {{
            type: 'bar',
            data: {{
                labels: scoreDistData.map(d => d.metric),
                datasets: [
                    {{
                        label: 'Score Range',
                        data: scoreDistData.map(d => [d.min, d.max]),
                        backgroundColor: colors.accentDim,
                        borderColor: colors.accent,
                        borderWidth: 2,
                        borderRadius: 2,
                        borderSkipped: false,
                        barThickness: 8,
                    }},
                    {{
                        label: 'Average',
                        data: scoreDistData.map(d => d.avg),
                        type: 'scatter',
                        backgroundColor: colors.accent,
                        borderColor: colors.bg,
                        borderWidth: 2,
                        pointRadius: 6,
                        pointStyle: 'circle',
                    }}
                ]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        backgroundColor: colors.bg,
                        titleColor: colors.text,
                        bodyColor: colors.text,
                        borderColor: colors.border,
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {{
                            label: (ctx) => {{
                                if (ctx.datasetIndex === 0) {{
                                    return `Range: ${{ctx.raw[0].toFixed(3)}} - ${{ctx.raw[1].toFixed(3)}}`;
                                }}
                                return `Avg: ${{ctx.raw.toFixed(3)}}`;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 1.1,
                        grid: {{
                            color: colors.gridLine,
                            lineWidth: 1
                        }},
                        ticks: {{ color: colors.textMuted }},
                        title: {{ display: true, text: 'Score', color: colors.muted }}
                    }},
                    y: {{
                        grid: {{ display: false }},
                        ticks: {{ color: colors.text }}
                    }}
                }}
            }}
        }});

        {correlation_js}
    </script>"""


def _render_cluster_scatter(clustering_result: Any, metric_id: str) -> str:
    """Render an embedded cluster scatter plot section for the HTML report.

    This function generates a Plotly scatter plot showing misalignment clusters.
    Each point represents a disagreement case (LLM judge vs human).

    Args:
        clustering_result: ClusteringResult with 2D coordinates
        metric_id: Name of the metric being analyzed

    Returns:
        HTML string with the cluster scatter plot section, or empty string if
        visualization is not possible.
    """
    if clustering_result is None:
        return ""

    # Check if we have visualization data
    if not getattr(clustering_result, "coordinates_2d", None):
        return ""

    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return ""

    coords_2d = clustering_result.coordinates_2d
    case_labels = clustering_result.case_labels or []
    case_types = clustering_result.case_types or []
    case_reasons = clustering_result.case_reasons or []

    if not coords_2d:
        return ""

    x_coords = [c[0] for c in coords_2d]
    y_coords = [c[1] for c in coords_2d]

    # Assign colors by label and type
    unique_labels = sorted(set(case_labels))
    fp_colors = ["#ef4444", "#f87171", "#fca5a5", "#fecaca", "#fee2e2"]
    fn_colors = ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"]

    color_map: Dict[str, str] = {}
    fp_idx, fn_idx = 0, 0
    for label in unique_labels:
        for i, l in enumerate(case_labels):
            if l == label:
                if case_types[i] == "false_positive":
                    color_map[label] = fp_colors[fp_idx % len(fp_colors)]
                    fp_idx += 1
                else:
                    color_map[label] = fn_colors[fn_idx % len(fn_colors)]
                    fn_idx += 1
                break

    colors = [color_map.get(l, "#888888") for l in case_labels]

    # Build hover text
    hover_texts = []
    for i in range(len(x_coords)):
        label = case_labels[i] if case_labels else "Unknown"
        case_type = case_types[i] if case_types else "unknown"
        reason = case_reasons[i] if case_reasons else ""
        reason_short = reason[:120] + "..." if len(reason) > 120 else reason
        type_label = "FP (too lenient)" if case_type == "false_positive" else "FN (too strict)"
        hover_texts.append(f"<b>{label}</b><br>Type: {type_label}<br>{reason_short}")

    fig = go.Figure()

    # Add FP trace
    fp_mask = [t == "false_positive" for t in case_types]
    if any(fp_mask):
        fig.add_trace(
            go.Scatter(
                x=[x for x, m in zip(x_coords, fp_mask) if m],
                y=[y for y, m in zip(y_coords, fp_mask) if m],
                mode="markers",
                marker=dict(
                    size=10,
                    color=[c for c, m in zip(colors, fp_mask) if m],
                    symbol="circle",
                    line=dict(width=1, color="#0a1210"),
                ),
                text=[t for t, m in zip(hover_texts, fp_mask) if m],
                hoverinfo="text",
                name="False Positive",
            )
        )

    # Add FN trace
    fn_mask = [t == "false_negative" for t in case_types]
    if any(fn_mask):
        fig.add_trace(
            go.Scatter(
                x=[x for x, m in zip(x_coords, fn_mask) if m],
                y=[y for y, m in zip(y_coords, fn_mask) if m],
                mode="markers",
                marker=dict(
                    size=10,
                    color=[c for c, m in zip(colors, fn_mask) if m],
                    symbol="diamond",
                    line=dict(width=1, color="#0a1210"),
                ),
                text=[t for t, m in zip(hover_texts, fn_mask) if m],
                hoverinfo="text",
                name="False Negative",
            )
        )

    fig.update_layout(
        title=dict(text=f"Misalignment Clusters: {metric_id}", font=dict(size=14, color="#e5e7eb")),
        paper_bgcolor="#0f1a16",
        plot_bgcolor="#152420",
        font=dict(color="#e5e7eb", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1f2d28", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#1f2d28", zeroline=False, showticklabels=False),
        legend=dict(bgcolor="#0f1a16", bordercolor="#1f2d28", borderwidth=1),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    plot_html = to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

    fp_count = sum(1 for t in case_types if t == "false_positive")
    fn_count = sum(1 for t in case_types if t == "false_negative")

    return f"""
        <div class="section">
            <div class="section-header">Misalignment Clusters</div>
            <div class="chart-box">
                <div style="display: flex; gap: 24px; margin-bottom: 12px;">
                    <span style="color: #ef4444;">False Positives: {fp_count}</span>
                    <span style="color: #3b82f6;">False Negatives: {fn_count}</span>
                </div>
                {plot_html}
            </div>
        </div>
    """


def generate_report(
    analysis: RunAnalysis, output_path: Path, format: str = "html"
) -> Path:
    """Generate a report file.

    Args:
        analysis: The run analysis data
        output_path: Path to save the report
        format: "html" or "text"

    Returns:
        Path to the generated report
    """
    from .reports import generate_text_report

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "html":
        content = generate_html_report(analysis)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".html")
    else:
        content = generate_text_report(analysis, verbose=True)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path
