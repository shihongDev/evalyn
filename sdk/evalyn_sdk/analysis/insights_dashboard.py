"""
HTML dashboard generation for insights analysis.

Generates a self-contained HTML report with Chart.js visualizations,
following the warm theme from html_report.py. Includes:
- KPI bar, metric pass rate bars, radar chart
- Score distribution histograms, item-metric heatmap
- Input length vs score scatter, correlation matrix
- Regression waterfall, recommendations cards
- Expert panel discussion (if available)
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

from .core import RunAnalysis
from .insights import InsightsReport


def generate_insights_html(
    run_analysis: RunAnalysis,
    insights_report: InsightsReport,
    panel_discussion: Any | None = None,
    dataset_items: list[dict[str, Any]] | None = None,
) -> str:
    """Generate self-contained HTML insights dashboard.

    Args:
        run_analysis: Analyzed run data
        insights_report: Deterministic insights report
        panel_discussion: Optional PanelDiscussion from expert panel
        dataset_items: Optional raw dataset items for input analysis charts
    """
    metric_ids = sorted(run_analysis.metric_stats.keys())
    metrics_by_pass_rate = sorted(
        run_analysis.metric_stats.values(),
        key=lambda m: -(m.pass_rate if m.pass_rate is not None else -1),
    )

    # Prepare chart data
    pass_rate_data = _build_pass_rate_data(metrics_by_pass_rate)
    radar_data = _build_radar_data(metrics_by_pass_rate)
    histogram_data = _build_histogram_data(run_analysis)
    heatmap_html = _build_heatmap_html(run_analysis)
    scatter_data = _build_scatter_data(run_analysis, dataset_items)
    correlation_data = _build_correlation_data(run_analysis, metric_ids)
    regression_data = _build_regression_data(insights_report)

    all_passed = sum(1 for ist in run_analysis.item_stats.values() if ist.all_passed)
    if run_analysis.overall_pass_rate >= 0.8:
        health = "Healthy"
    elif run_analysis.overall_pass_rate >= 0.5:
        health = "Needs Attention"
    else:
        health = "Critical"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insights Dashboard - {html.escape(run_analysis.dataset_name)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {_get_css()}
</head>
<body>
<div class="dashboard">
    {_render_header(run_analysis)}
    {_render_kpi_bar(run_analysis, all_passed, health)}
    {_render_metric_overview(pass_rate_data, radar_data)}
    {_render_distributions(histogram_data)}
    {heatmap_html}
    {_render_scatter(scatter_data)}
    {_render_correlation_matrix(correlation_data)}
    {_render_regression_panel(regression_data)}
    {_render_recommendations(insights_report)}
    {_render_expert_panel(panel_discussion)}
    {_render_footer()}
</div>
{_get_js(pass_rate_data, radar_data, histogram_data, scatter_data, correlation_data, regression_data)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_pass_rate_data(metrics: list) -> dict[str, Any]:
    labels = []
    values = []
    colors = []
    for m in metrics:
        labels.append(m.metric_id)
        pr = m.pass_rate
        if pr is not None:
            values.append(round(pr * 100, 1))
            if pr >= 0.8:
                colors.append("#6B8E8E")
            elif pr >= 0.5:
                colors.append("#D4A27F")
            else:
                colors.append("#C97B63")
        else:
            values.append(0)
            colors.append("#A89F97")
    return {"labels": labels, "values": values, "colors": colors}


def _build_radar_data(metrics: list) -> dict[str, Any]:
    labels = [m.metric_id for m in metrics]
    values = [
        round(m.pass_rate * 100, 1) if m.pass_rate is not None else 0
        for m in metrics
    ]
    return {"labels": labels, "values": values}


def _build_histogram_data(run_analysis: RunAnalysis) -> list[dict[str, Any]]:
    result = []
    for mid in sorted(run_analysis.metric_stats.keys()):
        ms = run_analysis.metric_stats[mid]
        if not ms.scores:
            continue
        # Build 10-bin histogram
        bins = [0] * 10
        for s in ms.scores:
            idx = min(int(s * 10), 9)
            bins[idx] += 1
        result.append({
            "metric_id": mid,
            "bins": bins,
            "labels": [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)],
            "avg": round(ms.avg_score, 3),
            "std": round(ms.std_dev, 3),
        })
    return result


def _build_scatter_data(
    run_analysis: RunAnalysis,
    dataset_items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not dataset_items:
        return []

    item_lookup = {}
    for item in dataset_items:
        iid = item.get("id")
        if iid:
            item_lookup[iid] = item

    points = []
    for item_id, ist in run_analysis.item_stats.items():
        if item_id not in item_lookup:
            continue
        raw = item_lookup[item_id]
        # Compute input length
        inp = raw.get("input") or raw.get("inputs")
        if inp is None:
            continue
        if isinstance(inp, str):
            length = len(inp)
        elif isinstance(inp, dict):
            length = len(json.dumps(inp))
        else:
            length = len(str(inp))

        # Average score across metrics
        scores = [
            r["score"] for r in ist.metric_results.values()
            if r.get("score") is not None
        ]
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        points.append({"x": length, "y": round(avg, 3), "id": item_id[:12]})

    return points


def _build_correlation_data(
    run_analysis: RunAnalysis,
    metric_ids: list[str],
) -> dict[str, Any] | None:
    if len(metric_ids) < 2:
        return None

    # Build score vectors per metric per item
    from .insights import _pearson_r

    matrix = []
    for i, m_a in enumerate(metric_ids):
        row = []
        for j, m_b in enumerate(metric_ids):
            if i == j:
                row.append(1.0)
                continue
            xs, ys = [], []
            for ist in run_analysis.item_stats.values():
                sa = ist.metric_results.get(m_a, {}).get("score")
                sb = ist.metric_results.get(m_b, {}).get("score")
                if sa is not None and sb is not None:
                    xs.append(sa)
                    ys.append(sb)
            r = _pearson_r(xs, ys) if len(xs) >= 3 else None
            row.append(round(r, 2) if r is not None else 0)
        matrix.append(row)

    return {"labels": metric_ids, "matrix": matrix}


def _build_regression_data(insights_report: InsightsReport) -> list[dict[str, Any]]:
    return [
        {
            "metric_id": r.metric_id,
            "previous": round(r.previous_pass_rate * 100, 1),
            "current": round(r.current_pass_rate * 100, 1),
            "delta": round(r.delta * 100, 1),
            "severity": r.severity,
        }
        for r in insights_report.regressions
    ]


# ---------------------------------------------------------------------------
# HTML section renderers
# ---------------------------------------------------------------------------


def _render_header(analysis: RunAnalysis) -> str:
    return f"""<div class="header">
    <div class="header-left">
        <div class="header-title">Insights Dashboard</div>
        <div class="header-meta">
            <span>Dataset: <span class="value">{html.escape(analysis.dataset_name)}</span></span>
            <span>Run: <span class="value">{html.escape(analysis.run_id[:12])}...</span></span>
        </div>
    </div>
</div>"""


def _render_kpi_bar(analysis: RunAnalysis, all_passed: int, health: str) -> str:
    health_color = {
        "Healthy": "#6B8E8E",
        "Needs Attention": "#D4A27F",
        "Critical": "#C97B63",
    }.get(health, "#A89F97")

    return f"""<div class="kpi-bar">
    <div class="kpi-card">
        <div class="kpi-value">{analysis.total_items}</div>
        <div class="kpi-label">Total Items</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{analysis.overall_pass_rate:.0%}</div>
        <div class="kpi-label">Overall Pass Rate</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{analysis.total_metrics}</div>
        <div class="kpi-label">Metrics</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color: {health_color}">{health}</div>
        <div class="kpi-label">Health Status</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{all_passed}</div>
        <div class="kpi-label">All Passed</div>
    </div>
</div>"""


def _render_metric_overview(pass_rate_data: dict, radar_data: dict) -> str:
    return """<div class="section">
    <h2 class="section-title">Metric Overview</h2>
    <div class="chart-row">
        <div class="chart-container">
            <h3 class="chart-title">Pass Rates</h3>
            <canvas id="passRateChart"></canvas>
        </div>
        <div class="chart-container">
            <h3 class="chart-title">Metric Balance</h3>
            <canvas id="radarChart"></canvas>
        </div>
    </div>
</div>"""


def _render_distributions(histogram_data: list) -> str:
    if not histogram_data:
        return ""
    canvases = []
    for i, h in enumerate(histogram_data):
        canvases.append(
            f'<div class="chart-container chart-small">'
            f'<h3 class="chart-title">{html.escape(h["metric_id"])} '
            f'(avg={h["avg"]}, std={h["std"]})</h3>'
            f'<canvas id="histChart{i}"></canvas></div>'
        )
    return f"""<div class="section">
    <h2 class="section-title">Score Distributions</h2>
    <div class="chart-grid">{"".join(canvases)}</div>
</div>"""


def _build_heatmap_html(run_analysis: RunAnalysis) -> str:
    metric_ids = sorted(run_analysis.metric_stats.keys())
    if not metric_ids or not run_analysis.item_stats:
        return ""

    # Limit to first 50 items for readability
    item_ids = list(run_analysis.item_stats.keys())[:50]

    header_cells = "".join(
        f'<th class="heatmap-header">{html.escape(m[:12])}</th>'
        for m in metric_ids
    )

    rows = []
    for iid in item_ids:
        ist = run_analysis.item_stats[iid]
        cells = []
        for mid in metric_ids:
            r = ist.metric_results.get(mid)
            if r is None:
                cells.append('<td class="heatmap-cell heatmap-na">-</td>')
            else:
                score = r.get("score", 0) or 0
                passed = r.get("passed")
                if passed is True:
                    cls = "heatmap-pass"
                elif passed is False:
                    cls = "heatmap-fail"
                else:
                    cls = "heatmap-na"
                cells.append(
                    f'<td class="heatmap-cell {cls}" '
                    f'title="{html.escape(iid[:20])} / {html.escape(mid)}: {score:.2f}">'
                    f'{score:.1f}</td>'
                )
        rows.append(
            f'<tr><td class="heatmap-item">{html.escape(iid[:12])}...</td>'
            f'{"".join(cells)}</tr>'
        )

    return f"""<div class="section">
    <h2 class="section-title">Item-Metric Matrix</h2>
    <div class="heatmap-wrapper">
        <table class="heatmap-table">
            <thead><tr><th></th>{header_cells}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
</div>"""


def _render_scatter(scatter_data: list) -> str:
    if not scatter_data:
        return ""
    return """<div class="section">
    <h2 class="section-title">Input Analysis</h2>
    <div class="chart-container">
        <h3 class="chart-title">Input Length vs Average Score</h3>
        <canvas id="scatterChart"></canvas>
    </div>
</div>"""


def _render_correlation_matrix(correlation_data: dict | None) -> str:
    if not correlation_data:
        return ""

    labels = correlation_data["labels"]
    matrix = correlation_data["matrix"]

    header = "".join(
        f'<th class="corr-header">{html.escape(label[:10])}</th>'
        for label in labels
    )

    rows = []
    for i, row in enumerate(matrix):
        cells = []
        for j, val in enumerate(row):
            intensity = abs(val)
            if val > 0:
                bg = f"rgba(107, 142, 142, {intensity * 0.7})"
            elif val < 0:
                bg = f"rgba(201, 123, 99, {intensity * 0.7})"
            else:
                bg = "transparent"
            cells.append(
                f'<td class="corr-cell" style="background:{bg}">{val:.2f}</td>'
            )
        rows.append(
            f'<tr><td class="corr-label">{html.escape(labels[i][:10])}</td>'
            f'{"".join(cells)}</tr>'
        )

    return f"""<div class="section">
    <h2 class="section-title">Correlation Matrix</h2>
    <div class="heatmap-wrapper">
        <table class="heatmap-table">
            <thead><tr><th></th>{header}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
</div>"""


def _render_regression_panel(regression_data: list) -> str:
    if not regression_data:
        return ""
    return """<div class="section">
    <h2 class="section-title">Regression Analysis</h2>
    <div class="chart-container">
        <h3 class="chart-title">Pass Rate Changes vs Previous Run</h3>
        <canvas id="regressionChart"></canvas>
    </div>
</div>"""


def _render_recommendations(report: InsightsReport) -> str:
    if not report.recommendations:
        return ""

    cards = []
    for rec in report.recommendations:
        badge_color = {
            "regression": "#C97B63",
            "metric_config": "#D4A27F",
            "calibration": "#9B8AA6",
            "data_quality": "#6B8E8E",
        }.get(rec.category, "#A89F97")

        cards.append(f"""<div class="rec-card">
    <div class="rec-header">
        <span class="rec-priority">#{rec.priority}</span>
        <span class="rec-badge" style="background:{badge_color}">{html.escape(rec.category)}</span>
    </div>
    <div class="rec-message">{html.escape(rec.message)}</div>
    <div class="rec-action"><code>{html.escape(rec.action)}</code></div>
</div>""")

    return f"""<div class="section">
    <h2 class="section-title">Recommendations</h2>
    <div class="rec-grid">{"".join(cards)}</div>
</div>"""


def _render_expert_panel(panel_discussion: Any | None) -> str:
    if panel_discussion is None:
        return ""

    expert_sections = []
    for expert in panel_discussion.experts:
        findings_html = "".join(
            f"<li>{html.escape(f)}</li>" for f in expert.findings
        )
        concerns_html = "".join(
            f"<li>{html.escape(c)}</li>" for c in expert.concerns
        )
        suggestions_html = "".join(
            f"<li>{html.escape(s)}</li>" for s in expert.suggestions
        )

        role_display = expert.role.replace("_", " ").title()
        expert_sections.append(f"""<div class="expert-card">
    <div class="expert-header" onclick="this.parentElement.classList.toggle('expanded')">
        <span class="expert-role">{html.escape(role_display)}</span>
        <span class="expert-confidence">confidence: {expert.confidence:.0%}</span>
        <span class="expand-indicator">+</span>
    </div>
    <div class="expert-summary">{html.escape(expert.summary)}</div>
    <div class="expert-details">
        {f'<div class="expert-section"><h4>Findings</h4><ul>{findings_html}</ul></div>' if findings_html else ''}
        {f'<div class="expert-section"><h4>Concerns</h4><ul>{concerns_html}</ul></div>' if concerns_html else ''}
        {f'<div class="expert-section"><h4>Suggestions</h4><ul>{suggestions_html}</ul></div>' if suggestions_html else ''}
    </div>
</div>""")

    synthesis_html = html.escape(panel_discussion.synthesis) if panel_discussion.synthesis else ""

    action_items = "".join(
        f"<li>{html.escape(a)}</li>" for a in panel_discussion.action_plan
    )
    dissenting = "".join(
        f"<li>{html.escape(d)}</li>" for d in panel_discussion.dissenting_views
    )

    tokens_info = ""
    if panel_discussion.total_input_tokens or panel_discussion.total_output_tokens:
        tokens_info = (
            f'<div class="token-info">Tokens: '
            f'{panel_discussion.total_input_tokens} in / '
            f'{panel_discussion.total_output_tokens} out</div>'
        )

    return f"""<div class="section">
    <h2 class="section-title">Expert Panel Analysis</h2>
    <div class="panel-synthesis">
        <h3 class="chart-title">Synthesis</h3>
        <p>{synthesis_html}</p>
        {f'<h4>Action Plan</h4><ol>{action_items}</ol>' if action_items else ''}
        {f'<h4>Dissenting Views</h4><ul>{dissenting}</ul>' if dissenting else ''}
        {tokens_info}
    </div>
    <h3 class="chart-title" style="margin-top:24px">Expert Opinions</h3>
    {"".join(expert_sections)}
</div>"""


def _render_footer() -> str:
    return """<div class="footer">
    <span>Generated by Evalyn Insights</span>
</div>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def _get_css() -> str:
    return """<style>
    :root {
        --bg-primary: #FFFBF7;
        --bg-secondary: #FBF7F3;
        --bg-tertiary: #F5EDE6;
        --bg-hover: #F0E8E0;
        --accent-primary: #D4A27F;
        --accent-secondary: #C4836A;
        --accent-muted: #A89F97;
        --accent-purple: #9B8AA6;
        --status-pass: #6B8E8E;
        --status-fail: #C97B63;
        --status-warn: #D4A27F;
        --text-primary: #1A1A1A;
        --text-secondary: #555555;
        --text-muted: #888888;
        --border-subtle: rgba(212, 162, 127, 0.2);
        --border-strong: rgba(212, 162, 127, 0.4);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        line-height: 1.6;
        font-size: 14px;
        -webkit-font-smoothing: antialiased;
    }

    .dashboard { max-width: 1600px; margin: 0 auto; padding: 24px; }

    .header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 16px 0; border-bottom: 1px solid var(--border-subtle); margin-bottom: 24px;
    }
    .header-left { display: flex; align-items: center; gap: 24px; }
    .header-title { font-size: 16px; font-weight: 600; }
    .header-meta { display: flex; gap: 16px; font-size: 13px; color: var(--text-muted); }
    .header-meta .value {
        color: var(--text-secondary);
        font-family: 'DM Mono', monospace;
    }

    /* KPI Bar */
    .kpi-bar {
        display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap;
    }
    .kpi-card {
        flex: 1; min-width: 140px; background: var(--bg-secondary);
        border: 1px solid var(--border-subtle); border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .kpi-value { font-size: 24px; font-weight: 700; font-family: 'DM Mono', monospace; }
    .kpi-label { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

    /* Sections */
    .section {
        margin-bottom: 32px; background: var(--bg-secondary);
        border: 1px solid var(--border-subtle); border-radius: 12px; padding: 24px;
    }
    .section-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; }

    /* Charts */
    .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
    .chart-container {
        background: var(--bg-primary); border: 1px solid var(--border-subtle);
        border-radius: 8px; padding: 16px;
    }
    .chart-small { min-height: 220px; }
    .chart-title { font-size: 13px; font-weight: 500; color: var(--text-secondary); margin-bottom: 12px; }

    /* Heatmap */
    .heatmap-wrapper { overflow-x: auto; }
    .heatmap-table {
        border-collapse: collapse; font-size: 12px; width: 100%;
        font-family: 'DM Mono', monospace;
    }
    .heatmap-table th, .heatmap-table td {
        padding: 6px 10px; text-align: center; border: 1px solid var(--border-subtle);
    }
    .heatmap-header {
        background: var(--bg-tertiary); font-weight: 500;
        position: sticky; top: 0; white-space: nowrap;
    }
    .heatmap-item {
        text-align: left; white-space: nowrap; font-size: 11px;
        color: var(--text-secondary); position: sticky; left: 0;
        background: var(--bg-secondary);
    }
    .heatmap-cell { font-size: 11px; }
    .heatmap-pass { background: rgba(107, 142, 142, 0.2); color: var(--status-pass); }
    .heatmap-fail { background: rgba(201, 123, 99, 0.2); color: var(--status-fail); }
    .heatmap-na { background: var(--bg-tertiary); color: var(--text-muted); }

    /* Correlation matrix */
    .corr-header { background: var(--bg-tertiary); font-weight: 500; white-space: nowrap; }
    .corr-label {
        text-align: left; font-weight: 500; white-space: nowrap;
        background: var(--bg-secondary); position: sticky; left: 0;
    }
    .corr-cell { font-size: 11px; font-family: 'DM Mono', monospace; }

    /* Recommendations */
    .rec-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
    .rec-card {
        background: var(--bg-primary); border: 1px solid var(--border-subtle);
        border-radius: 8px; padding: 16px;
    }
    .rec-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .rec-priority { font-weight: 700; font-size: 16px; color: var(--accent-secondary); }
    .rec-badge {
        font-size: 11px; color: white; padding: 2px 8px;
        border-radius: 4px; font-weight: 500;
    }
    .rec-message { font-size: 13px; color: var(--text-primary); margin-bottom: 8px; }
    .rec-action { font-size: 12px; color: var(--text-muted); }
    .rec-action code {
        background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px;
        font-family: 'DM Mono', monospace;
    }

    /* Expert Panel */
    .panel-synthesis {
        background: var(--bg-primary); border: 1px solid var(--border-subtle);
        border-radius: 8px; padding: 20px;
    }
    .panel-synthesis p { color: var(--text-secondary); margin-bottom: 12px; }
    .panel-synthesis h4 { font-size: 13px; margin: 12px 0 8px; }
    .panel-synthesis ol, .panel-synthesis ul {
        padding-left: 20px; color: var(--text-secondary); font-size: 13px;
    }
    .panel-synthesis li { margin-bottom: 4px; }
    .token-info {
        font-size: 11px; color: var(--text-muted); margin-top: 12px;
        font-family: 'DM Mono', monospace;
    }

    .expert-card {
        background: var(--bg-primary); border: 1px solid var(--border-subtle);
        border-radius: 8px; margin-top: 8px; overflow: hidden;
    }
    .expert-header {
        display: flex; align-items: center; gap: 12px; padding: 12px 16px;
        cursor: pointer; user-select: none;
    }
    .expert-header:hover { background: var(--bg-hover); }
    .expert-role { font-weight: 600; font-size: 14px; }
    .expert-confidence { font-size: 12px; color: var(--text-muted); font-family: 'DM Mono', monospace; }
    .expand-indicator { margin-left: auto; color: var(--text-muted); font-size: 18px; }
    .expert-summary {
        padding: 0 16px 12px; font-size: 13px; color: var(--text-secondary);
    }
    .expert-details { display: none; padding: 0 16px 16px; }
    .expert-card.expanded .expert-details { display: block; }
    .expert-card.expanded .expand-indicator::after { content: ''; }
    .expert-card.expanded .expand-indicator { transform: rotate(45deg); }
    .expert-section { margin-bottom: 12px; }
    .expert-section h4 { font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 4px; }
    .expert-section ul { padding-left: 16px; font-size: 13px; color: var(--text-secondary); }
    .expert-section li { margin-bottom: 2px; }

    /* Footer */
    .footer {
        text-align: center; padding: 24px; font-size: 12px;
        color: var(--text-muted); border-top: 1px solid var(--border-subtle);
    }

    @media (max-width: 768px) {
        .chart-row { grid-template-columns: 1fr; }
        .kpi-bar { flex-direction: column; }
    }
</style>"""


# ---------------------------------------------------------------------------
# JavaScript (Chart.js)
# ---------------------------------------------------------------------------


def _get_js(
    pass_rate_data: dict,
    radar_data: dict,
    histogram_data: list,
    scatter_data: list,
    correlation_data: dict | None,
    regression_data: list,
) -> str:
    return f"""<script>
const warmColors = {{
    pass: '#6B8E8E',
    fail: '#C97B63',
    warn: '#D4A27F',
    muted: '#A89F97',
    purple: '#9B8AA6',
    bg: 'rgba(212, 162, 127, 0.1)',
    grid: 'rgba(212, 162, 127, 0.15)',
}};

const defaultOpts = {{
    responsive: true,
    maintainAspectRatio: true,
    plugins: {{
        legend: {{ display: false }},
    }},
    scales: {{
        x: {{ grid: {{ color: warmColors.grid }} }},
        y: {{ grid: {{ color: warmColors.grid }} }},
    }},
}};

// Pass Rate Bar Chart
new Chart(document.getElementById('passRateChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(pass_rate_data['labels'])},
        datasets: [{{
            data: {json.dumps(pass_rate_data['values'])},
            backgroundColor: {json.dumps(pass_rate_data['colors'])},
            borderRadius: 4,
        }}],
    }},
    options: {{
        ...defaultOpts,
        indexAxis: 'y',
        scales: {{
            x: {{ max: 100, grid: {{ color: warmColors.grid }}, title: {{ display: true, text: 'Pass Rate %' }} }},
            y: {{ grid: {{ display: false }} }},
        }},
    }},
}});

// Radar Chart
new Chart(document.getElementById('radarChart'), {{
    type: 'radar',
    data: {{
        labels: {json.dumps(radar_data['labels'])},
        datasets: [{{
            data: {json.dumps(radar_data['values'])},
            backgroundColor: 'rgba(212, 162, 127, 0.2)',
            borderColor: '#D4A27F',
            borderWidth: 2,
            pointBackgroundColor: '#C4836A',
        }}],
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            r: {{
                min: 0, max: 100,
                ticks: {{ stepSize: 20 }},
                grid: {{ color: warmColors.grid }},
                angleLines: {{ color: warmColors.grid }},
            }},
        }},
    }},
}});

// Histogram Charts
const histData = {json.dumps(histogram_data)};
histData.forEach((h, i) => {{
    const el = document.getElementById('histChart' + i);
    if (!el) return;
    new Chart(el, {{
        type: 'bar',
        data: {{
            labels: h.labels,
            datasets: [{{
                data: h.bins,
                backgroundColor: warmColors.warn,
                borderRadius: 2,
            }}],
        }},
        options: {{
            ...defaultOpts,
            scales: {{
                x: {{ grid: {{ display: false }}, title: {{ display: true, text: 'Score' }} }},
                y: {{ grid: {{ color: warmColors.grid }}, title: {{ display: true, text: 'Count' }}, beginAtZero: true }},
            }},
        }},
    }});
}});

// Scatter Chart
const scatterData = {json.dumps(scatter_data)};
if (scatterData.length > 0 && document.getElementById('scatterChart')) {{
    new Chart(document.getElementById('scatterChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                data: scatterData,
                backgroundColor: 'rgba(212, 162, 127, 0.5)',
                borderColor: '#C4836A',
                pointRadius: 4,
            }}],
        }},
        options: {{
            ...defaultOpts,
            scales: {{
                x: {{ grid: {{ color: warmColors.grid }}, title: {{ display: true, text: 'Input Length (chars)' }} }},
                y: {{ grid: {{ color: warmColors.grid }}, title: {{ display: true, text: 'Average Score' }}, min: 0, max: 1 }},
            }},
        }},
    }});
}}

// Regression Chart
const regressionData = {json.dumps(regression_data)};
if (regressionData.length > 0 && document.getElementById('regressionChart')) {{
    const severityColors = {{
        critical: warmColors.fail,
        warning: warmColors.warn,
        info: warmColors.muted,
    }};
    new Chart(document.getElementById('regressionChart'), {{
        type: 'bar',
        data: {{
            labels: regressionData.map(r => r.metric_id),
            datasets: [{{
                data: regressionData.map(r => r.delta),
                backgroundColor: regressionData.map(r => severityColors[r.severity] || warmColors.muted),
                borderRadius: 4,
            }}],
        }},
        options: {{
            ...defaultOpts,
            scales: {{
                x: {{ grid: {{ display: false }} }},
                y: {{ grid: {{ color: warmColors.grid }}, title: {{ display: true, text: 'Change (pp)' }} }},
            }},
        }},
    }});
}}
</script>"""


__all__ = ["generate_insights_html"]
