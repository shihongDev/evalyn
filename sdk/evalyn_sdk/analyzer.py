"""
Analyzer module for comprehensive eval results analysis and visualization.

Generates a single self-contained HTML report with embedded Chart.js visualizations.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MetricStats:
    """Statistics for a single metric across items."""

    metric_id: str
    metric_type: str  # objective or subjective
    count: int = 0
    passed: int = 0
    failed: int = 0
    scores: List[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.count if self.count > 0 else 0.0

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def min_score(self) -> float:
        return min(self.scores) if self.scores else 0.0

    @property
    def max_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def std_dev(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        avg = self.avg_score
        variance = sum((s - avg) ** 2 for s in self.scores) / len(self.scores)
        return math.sqrt(variance)


@dataclass
class ItemStats:
    """Statistics for a single dataset item across metrics."""

    item_id: str
    metrics_passed: int = 0
    metrics_failed: int = 0
    metric_results: Dict[str, Tuple[bool, float]] = field(
        default_factory=dict
    )  # metric_id -> (passed, score)

    @property
    def all_passed(self) -> bool:
        return self.metrics_failed == 0 and self.metrics_passed > 0


@dataclass
class RunAnalysis:
    """Complete analysis of an eval run."""

    run_id: str
    dataset_name: str
    created_at: str
    total_items: int
    total_metrics: int
    metric_stats: Dict[str, MetricStats]
    item_stats: Dict[str, ItemStats]
    failed_items: List[str]

    @property
    def overall_pass_rate(self) -> float:
        """Percentage of items that passed ALL metrics."""
        if self.total_items == 0:
            return 0.0
        all_passed = sum(1 for item in self.item_stats.values() if item.all_passed)
        return all_passed / self.total_items


def load_eval_run(path: Path) -> Dict[str, Any]:
    """Load a single eval run JSON file.

    Args:
        path: Path to results.json file, folder containing results.json, or legacy .json file
    """
    path = Path(path)
    # Handle folder path
    if path.is_dir():
        path = path / "results.json"
    with open(path) as f:
        return json.load(f)


def find_eval_runs(dataset_dir: Path) -> List[Path]:
    """Find all eval run folders/files in a dataset directory.

    Returns paths to results.json files (new structure) or .json files (legacy).
    """
    eval_runs_dir = Path(dataset_dir) / "eval_runs"
    if not eval_runs_dir.exists():
        return []

    runs = []

    # New folder structure: look for folders with results.json
    for item in eval_runs_dir.iterdir():
        if item.is_dir():
            results_file = item / "results.json"
            if results_file.exists():
                runs.append(results_file)

    # Legacy flat JSON files
    for json_file in eval_runs_dir.glob("*.json"):
        runs.append(json_file)

    # Sort by modification time, newest first
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def analyze_run(run_data: Dict[str, Any]) -> RunAnalysis:
    """Analyze a single eval run and compute statistics."""
    metric_stats: Dict[str, MetricStats] = {}
    item_stats: Dict[str, ItemStats] = defaultdict(lambda: ItemStats(item_id=""))

    # Build metric type lookup from run data
    metric_types = {}
    for m in run_data.get("metrics", []):
        metric_types[m["id"]] = m.get("type", "unknown")

    # Process each result
    for result in run_data.get("metric_results", []):
        metric_id = result["metric_id"]
        item_id = result["item_id"]
        score = result.get("score")
        # Handle None scores (API errors, etc.)
        if score is None:
            score = 0.0
        passed = result.get("passed", False)

        # Update metric stats
        if metric_id not in metric_stats:
            metric_stats[metric_id] = MetricStats(
                metric_id=metric_id, metric_type=metric_types.get(metric_id, "unknown")
            )
        ms = metric_stats[metric_id]
        ms.count += 1
        ms.scores.append(score)
        if passed:
            ms.passed += 1
        else:
            ms.failed += 1

        # Update item stats
        if item_stats[item_id].item_id == "":
            item_stats[item_id].item_id = item_id
        item_stats[item_id].metric_results[metric_id] = (passed, score)
        if passed:
            item_stats[item_id].metrics_passed += 1
        else:
            item_stats[item_id].metrics_failed += 1

    # Get failed items
    failed_items = [
        item_id for item_id, stats in item_stats.items() if not stats.all_passed
    ]

    return RunAnalysis(
        run_id=run_data.get("id", "unknown"),
        dataset_name=run_data.get("dataset_name", "unknown"),
        created_at=run_data.get("created_at", ""),
        total_items=len(item_stats),
        total_metrics=len(metric_stats),
        metric_stats=dict(metric_stats),
        item_stats=dict(item_stats),
        failed_items=failed_items,
    )


# =============================================================================
# ASCII Visualization Helpers (for terminal output)
# =============================================================================


def ascii_bar(
    value: float, max_width: int = 30, fill: str = "█", empty: str = "░"
) -> str:
    """Create an ASCII progress bar."""
    filled = int(value * max_width)
    return fill * filled + empty * (max_width - filled)


def ascii_score_distribution(scores: List[float], metric_id: str) -> str:
    """Create a compact score distribution visualization."""
    if not scores:
        return f"  {metric_id}: (no data)"

    # Bucket scores into ranges: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    buckets = [0, 0, 0, 0, 0]
    for s in scores:
        idx = min(int(s * 5), 4)
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1

    # Create mini bar chart
    bars = []
    for count in buckets:
        height = int((count / max_count) * 5) if max_count > 0 else 0
        bars.append("▁▂▃▄▅▆▇█"[min(height, 7)])

    return f"  {metric_id:30} [{''.join(bars)}] avg={sum(scores) / len(scores):.2f}"


def format_pass_rate_bar(
    metric_id: str, pass_rate: float, count: int, width: int = 25
) -> str:
    """Format a metric pass rate with a bar chart."""
    bar = ascii_bar(pass_rate, max_width=width)
    pct = f"{pass_rate * 100:5.1f}%"
    return f"  {metric_id:30} {bar} {pct} (n={count})"


# =============================================================================
# Text Report Generation
# =============================================================================


def generate_text_report(analysis: RunAnalysis, verbose: bool = False) -> str:
    """Generate a comprehensive text report."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("  EVAL RUN ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Run ID:     {analysis.run_id[:8]}...")
    lines.append(f"  Dataset:    {analysis.dataset_name}")
    lines.append(
        f"  Created:    {analysis.created_at[:19] if analysis.created_at else 'unknown'}"
    )
    lines.append(f"  Items:      {analysis.total_items}")
    lines.append(f"  Metrics:    {analysis.total_metrics}")
    lines.append("")

    # Overall summary
    lines.append("-" * 70)
    lines.append("  OVERALL SUMMARY")
    lines.append("-" * 70)
    all_passed = sum(1 for item in analysis.item_stats.values() if item.all_passed)
    lines.append(
        f"  Items passing all metrics: {all_passed}/{analysis.total_items} ({analysis.overall_pass_rate * 100:.1f}%)"
    )
    lines.append(f"  Items with failures:       {len(analysis.failed_items)}")
    lines.append("")

    # Metric pass rates
    lines.append("-" * 70)
    lines.append("  METRIC PASS RATES")
    lines.append("-" * 70)

    # Sort by pass rate (lowest first to highlight problems)
    sorted_metrics = sorted(analysis.metric_stats.values(), key=lambda m: m.pass_rate)

    for ms in sorted_metrics:
        lines.append(format_pass_rate_bar(ms.metric_id, ms.pass_rate, ms.count))
    lines.append("")

    # Score statistics
    lines.append("-" * 70)
    lines.append("  SCORE STATISTICS")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<30} {'Avg':>8} {'Min':>8} {'Max':>8} {'StdDev':>8}")
    lines.append(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for ms in sorted_metrics:
        lines.append(
            f"  {ms.metric_id:<30} {ms.avg_score:>8.3f} {ms.min_score:>8.3f} "
            f"{ms.max_score:>8.3f} {ms.std_dev:>8.3f}"
        )
    lines.append("")

    # Score distributions (compact)
    lines.append("-" * 70)
    lines.append("  SCORE DISTRIBUTIONS (0.0 → 1.0)")
    lines.append("-" * 70)
    for ms in sorted_metrics:
        lines.append(ascii_score_distribution(ms.scores, ms.metric_id))
    lines.append("")

    # Failed items (if verbose)
    if verbose and analysis.failed_items:
        lines.append("-" * 70)
        lines.append(f"  FAILED ITEMS ({len(analysis.failed_items)})")
        lines.append("-" * 70)
        for item_id in analysis.failed_items[:20]:  # Limit to 20
            item = analysis.item_stats[item_id]
            failed_metrics = [
                m for m, (passed, _) in item.metric_results.items() if not passed
            ]
            lines.append(f"  {item_id[:12]}... failed: {', '.join(failed_metrics)}")
        if len(analysis.failed_items) > 20:
            lines.append(f"  ... and {len(analysis.failed_items) - 20} more")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def generate_comparison_report(analyses: List[RunAnalysis]) -> str:
    """Generate a comparison report across multiple runs."""
    if len(analyses) < 2:
        return "  Need at least 2 runs to compare."

    lines = []
    lines.append("=" * 70)
    lines.append("  EVAL RUN COMPARISON")
    lines.append("=" * 70)
    lines.append("")

    # Run info
    lines.append(f"  {'Run':<12} {'Date':<20} {'Items':>8} {'Overall':>10}")
    lines.append(f"  {'-' * 12} {'-' * 20} {'-' * 8} {'-' * 10}")

    for a in analyses:
        date = a.created_at[:16] if a.created_at else "unknown"
        lines.append(
            f"  {a.run_id[:12]} {date:<20} {a.total_items:>8} "
            f"{a.overall_pass_rate * 100:>9.1f}%"
        )
    lines.append("")

    # Metric comparison
    lines.append("-" * 70)
    lines.append("  PASS RATE BY METRIC")
    lines.append("-" * 70)

    # Get all metrics
    all_metrics = set()
    for a in analyses:
        all_metrics.update(a.metric_stats.keys())

    # Header
    header = f"  {'Metric':<25}"
    for i, a in enumerate(analyses):
        header += f" {'Run' + str(i + 1):>10}"
    header += f" {'Delta':>10}"
    lines.append(header)
    lines.append(f"  {'-' * 25}" + f" {'-' * 10}" * (len(analyses) + 1))

    for metric_id in sorted(all_metrics):
        row = f"  {metric_id:<25}"
        rates = []
        for a in analyses:
            if metric_id in a.metric_stats:
                rate = a.metric_stats[metric_id].pass_rate
                rates.append(rate)
                row += f" {rate * 100:>9.1f}%"
            else:
                row += f" {'N/A':>10}"

        # Delta (newest - oldest)
        if len(rates) >= 2:
            delta = rates[-1] - rates[0]
            sign = "+" if delta >= 0 else ""
            row += f" {sign}{delta * 100:>8.1f}%"
        else:
            row += f" {'N/A':>10}"

        lines.append(row)

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# HTML Report Generation (Single Self-Contained File)
# =============================================================================


def generate_html_report(analysis: RunAnalysis, verbose: bool = False) -> str:
    """Generate a comprehensive single-page HTML report with all visualizations.

    Uses Chart.js for interactive charts. No external images - everything is embedded.
    Styled with Anthropic research paper aesthetic (light background, blue/coral colors).
    """

    # Prepare metric data sorted by pass rate
    metrics_by_pass_rate = sorted(
        analysis.metric_stats.values(), key=lambda m: -m.pass_rate
    )
    metric_labels = json.dumps([m.metric_id for m in metrics_by_pass_rate])
    pass_rates = json.dumps([round(m.pass_rate * 100, 1) for m in metrics_by_pass_rate])
    avg_scores = json.dumps([round(m.avg_score, 3) for m in metrics_by_pass_rate])
    min_scores = json.dumps([round(m.min_score, 3) for m in metrics_by_pass_rate])
    max_scores = json.dumps([round(m.max_score, 3) for m in metrics_by_pass_rate])
    passed_counts = json.dumps([m.passed for m in metrics_by_pass_rate])
    failed_counts = json.dumps([m.failed for m in metrics_by_pass_rate])

    # Color coding for pass rates
    pass_rate_colors = json.dumps(
        [
            "#4a90a4"
            if m.pass_rate >= 0.8
            else "#e0a030"
            if m.pass_rate >= 0.5
            else "#d65a4a"
            for m in metrics_by_pass_rate
        ]
    )

    # Prepare per-item data for detailed view
    item_data_rows = []
    for item_id, item in analysis.item_stats.items():
        failed_metrics = [
            m for m, (passed, _) in item.metric_results.items() if not passed
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
                    _, score = analysis.item_stats[item_id].metric_results[metric_id]
                    row.append(score)
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

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval Analysis - {analysis.dataset_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #fafaf8;
            --bg-card: #ffffff;
            --bg-alt: #f5f5f3;
            --text: #1a1a1a;
            --text-muted: #6b6b6b;
            --border: #e5e5e0;
            --accent: #4a90a4;
            --success: #4a90a4;
            --warning: #e0a030;
            --error: #d65a4a;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 48px;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 8px;
        }}

        .header .subtitle {{
            font-size: 14px;
            color: var(--text-muted);
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 24px;
            margin-bottom: 48px;
        }}

        .stat-card {{
            text-align: center;
            padding: 32px 16px;
        }}

        .stat-value {{
            font-size: 48px;
            font-weight: 500;
            color: var(--accent);
            line-height: 1;
            margin-bottom: 8px;
        }}

        .stat-value.success {{ color: var(--success); }}
        .stat-value.warning {{ color: var(--warning); }}
        .stat-value.error {{ color: var(--error); }}

        .stat-label {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Cards */
        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 32px;
            border: 1px solid var(--border);
        }}

        .card h2 {{
            font-size: 16px;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}

        /* Charts */
        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 16px;
        }}

        .chart-container.tall {{
            height: 500px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}

        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            font-weight: 500;
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: var(--bg-alt);
        }}

        tr:hover {{
            background: var(--bg-alt);
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }}

        .badge.pass {{
            background: rgba(74, 144, 164, 0.1);
            color: var(--success);
        }}

        .badge.fail {{
            background: rgba(214, 90, 74, 0.1);
            color: var(--error);
        }}

        .badge.warn {{
            background: rgba(224, 160, 48, 0.1);
            color: var(--warning);
        }}

        /* Grid layouts */
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
        }}

        /* Metadata */
        .metadata {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            padding: 20px;
            background: var(--bg-alt);
            border-radius: 8px;
            font-size: 13px;
        }}

        .metadata-item {{
            display: flex;
            gap: 8px;
        }}

        .metadata-label {{
            color: var(--text-muted);
        }}

        .metadata-value {{
            font-weight: 500;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .two-col {{
                grid-template-columns: 1fr;
            }}
            .stat-value {{
                font-size: 36px;
            }}
        }}

        /* Correlation matrix */
        .correlation-grid {{
            display: grid;
            gap: 2px;
            font-size: 11px;
        }}

        .corr-cell {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px;
            font-weight: 500;
        }}

        .corr-label {{
            padding: 8px;
            font-size: 11px;
            color: var(--text-muted);
            text-align: right;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        /* Failed items section */
        .failed-items {{
            max-height: 300px;
            overflow-y: auto;
        }}

        .failed-item {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }}

        .failed-item:last-child {{
            border-bottom: none;
        }}

        .failed-item-id {{
            font-family: monospace;
            font-size: 12px;
            color: var(--text-muted);
        }}

        .failed-item-metrics {{
            margin-top: 4px;
            color: var(--error);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Evaluation Results</h1>
            <p class="subtitle">{analysis.dataset_name} · {
        analysis.total_items
    } items · {analysis.total_metrics} metrics</p>
        </div>

        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{analysis.total_items}</div>
                <div class="stat-label">Items</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis.total_metrics}</div>
                <div class="stat-label">Metrics</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {
        "success"
        if analysis.overall_pass_rate >= 0.8
        else "warning"
        if analysis.overall_pass_rate >= 0.5
        else "error"
    }">{analysis.overall_pass_rate * 100:.0f}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {
        "success" if len(analysis.failed_items) == 0 else "error"
    }">{len(analysis.failed_items)}</div>
                <div class="stat-label">Failed Items</div>
            </div>
        </div>

        <!-- Pass Rate Chart -->
        <div class="card">
            <h2>Metric Pass Rates</h2>
            <div class="chart-container">
                <canvas id="passRateChart"></canvas>
            </div>
        </div>

        <!-- Score Distribution & Results by Metric -->
        <div class="two-col">
            <div class="card">
                <h2>Score Distribution</h2>
                <div class="chart-container">
                    <canvas id="scoreDistChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>Pass/Fail by Metric</h2>
                <div class="chart-container">
                    <canvas id="passFailChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Metric Details Table -->
        <div class="card">
            <h2>Metric Details</h2>
            <table>
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
                    {
        "".join(
            f'''
                    <tr>
                        <td><strong>{ms.metric_id}</strong></td>
                        <td>{ms.metric_type}</td>
                        <td><span class="badge {'pass' if ms.pass_rate >= 0.8 else 'warn' if ms.pass_rate >= 0.5 else 'fail'}">{ms.pass_rate * 100:.1f}%</span></td>
                        <td>{ms.avg_score:.3f}</td>
                        <td>{ms.min_score:.3f}</td>
                        <td>{ms.max_score:.3f}</td>
                        <td>{ms.std_dev:.3f}</td>
                        <td>{ms.passed}</td>
                        <td>{ms.failed if ms.failed == 0 else f'<span style="color: var(--error)">{ms.failed}</span>'}</td>
                    </tr>
                    '''
            for ms in metrics_by_pass_rate
        )
    }
                </tbody>
            </table>
        </div>

        {
        "<!-- Correlation Matrix -->"
        if correlation_data and len(correlation_data["labels"]) <= 10
        else ""
    }
        {
        f'''
        <div class="card">
            <h2>Metric Correlations</h2>
            <div class="chart-container">
                <canvas id="correlationChart"></canvas>
            </div>
        </div>
        '''
        if correlation_data and len(correlation_data["labels"]) <= 10
        else ""
    }

        <!-- Failed Items -->
        {
        f'''
        <div class="card">
            <h2>Failed Items ({len(analysis.failed_items)})</h2>
            <div class="failed-items">
                {"".join(f'<div class="failed-item"><div class="failed-item-id">{item_id[:20]}...</div><div class="failed-item-metrics">Failed: {", ".join(m for m, (p, _) in analysis.item_stats[item_id].metric_results.items() if not p)}</div></div>' for item_id in analysis.failed_items[:30])}
                {f'<div class="failed-item" style="color: var(--text-muted)">...and {len(analysis.failed_items) - 30} more</div>' if len(analysis.failed_items) > 30 else ''}
            </div>
        </div>
        '''
        if analysis.failed_items
        else ""
    }

        <!-- Metadata -->
        <div class="card">
            <h2>Run Metadata</h2>
            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Run ID:</span>
                    <span class="metadata-value">{analysis.run_id}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Dataset:</span>
                    <span class="metadata-value">{analysis.dataset_name}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Created:</span>
                    <span class="metadata-value">{analysis.created_at}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Items Passing All:</span>
                    <span class="metadata-value">{all_passed_count}/{
        analysis.total_items
    }</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Anthropic-style colors
        const colors = {{
            accent: '#4a90a4',
            error: '#d65a4a',
            warning: '#e0a030',
            success: '#4a90a4',
            muted: '#6b6b6b',
            border: '#e5e5e0',
            bg: '#fafaf8'
        }};

        // Chart.js defaults
        Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif";
        Chart.defaults.color = '#6b6b6b';

        // Pass Rate Chart
        const passRateCtx = document.getElementById('passRateChart').getContext('2d');
        new Chart(passRateCtx, {{
            type: 'bar',
            data: {{
                labels: {metric_labels},
                datasets: [{{
                    data: {pass_rates},
                    backgroundColor: {pass_rate_colors},
                    borderRadius: 4,
                    borderSkipped: false,
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => ctx.raw.toFixed(1) + '%'
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{ color: colors.border }},
                        title: {{ display: true, text: 'Pass Rate (%)', color: colors.muted }}
                    }},
                    y: {{
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});

        // Score Distribution Chart (floating bars for min-max range with avg point)
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
                        backgroundColor: 'rgba(74, 144, 164, 0.3)',
                        borderColor: colors.accent,
                        borderWidth: 1,
                        borderRadius: 4,
                        borderSkipped: false,
                    }},
                    {{
                        label: 'Average',
                        data: scoreDistData.map(d => d.avg),
                        type: 'scatter',
                        backgroundColor: colors.accent,
                        borderColor: '#fff',
                        borderWidth: 2,
                        pointRadius: 8,
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
                        grid: {{ color: colors.border }},
                        title: {{ display: true, text: 'Score', color: colors.muted }}
                    }},
                    y: {{
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});

        // Pass/Fail Chart
        const passFailCtx = document.getElementById('passFailChart').getContext('2d');
        new Chart(passFailCtx, {{
            type: 'bar',
            data: {{
                labels: {metric_labels},
                datasets: [
                    {{
                        label: 'Passed',
                        data: {passed_counts},
                        backgroundColor: colors.accent,
                        borderRadius: 4,
                        borderSkipped: false,
                    }},
                    {{
                        label: 'Failed',
                        data: {failed_counts},
                        backgroundColor: colors.error,
                        borderRadius: 4,
                        borderSkipped: false,
                    }}
                ]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{ usePointStyle: true, pointStyle: 'rect' }}
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: true,
                        grid: {{ color: colors.border }},
                        title: {{ display: true, text: 'Count', color: colors.muted }}
                    }},
                    y: {{
                        stacked: true,
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});

        {
        f'''
        // Correlation Heatmap
        const corrData = {json.dumps(correlation_data)};
        if (corrData) {{
            const corrCtx = document.getElementById('correlationChart').getContext('2d');
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

            new Chart(corrCtx, {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        data: heatmapData.map(d => ({{
                            x: corrData.labels.indexOf(d.x),
                            y: corrData.labels.indexOf(d.y),
                            v: d.v
                        }})),
                        backgroundColor: heatmapData.map(d => {{
                            const v = d.v;
                            if (v >= 0.7) return 'rgba(74, 144, 164, 0.9)';
                            if (v >= 0.3) return 'rgba(74, 144, 164, 0.5)';
                            if (v >= -0.3) return 'rgba(200, 200, 200, 0.5)';
                            if (v >= -0.7) return 'rgba(214, 90, 74, 0.5)';
                            return 'rgba(214, 90, 74, 0.9)';
                        }}),
                        pointRadius: heatmapData.map(() => 20),
                        pointStyle: 'rect'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: (ctx) => {{
                                    const idx = ctx.dataIndex;
                                    const d = heatmapData[idx];
                                    return `${{d.x}} vs ${{d.y}}: ${{d.v.toFixed(2)}}`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'linear',
                            position: 'bottom',
                            min: -0.5,
                            max: corrData.labels.length - 0.5,
                            ticks: {{
                                stepSize: 1,
                                callback: (v) => corrData.labels[v] || ''
                            }},
                            grid: {{ display: false }}
                        }},
                        y: {{
                            type: 'linear',
                            min: -0.5,
                            max: corrData.labels.length - 0.5,
                            ticks: {{
                                stepSize: 1,
                                callback: (v) => corrData.labels[v] || ''
                            }},
                            grid: {{ display: false }}
                        }}
                    }}
                }}
            }});
        }}
        '''
        if correlation_data and len(correlation_data["labels"]) <= 10
        else ""
    }
    </script>
</body>
</html>
"""

    return html


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

    with open(output_path, "w") as f:
        f.write(content)

    return output_path
