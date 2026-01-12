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
import html


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


def _render_failed_items(analysis: "RunAnalysis", item_details: dict) -> str:
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


@dataclass
class MetricStats:
    """Statistics for a single metric across items."""

    metric_id: str
    metric_type: str  # objective or subjective
    count: int = 0
    passed: int = 0
    failed: int = 0
    scores: List[float] = field(default_factory=list)
    has_pass_fail: bool = False  # True if metric has pass/fail semantics

    @property
    def pass_rate(self) -> Optional[float]:
        """Return pass rate or None if metric doesn't have pass/fail semantics."""
        if not self.has_pass_fail:
            return None
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
    metric_results: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # metric_id -> {passed, score, reason, details}

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
        # Handle None scores (API errors, etc.) - keep as 0.0 for averaging
        numeric_score = score if score is not None else 0.0
        passed = result.get("passed")  # Can be True, False, or None

        # Update metric stats
        if metric_id not in metric_stats:
            metric_stats[metric_id] = MetricStats(
                metric_id=metric_id, metric_type=metric_types.get(metric_id, "unknown")
            )
        ms = metric_stats[metric_id]
        ms.count += 1
        ms.scores.append(numeric_score)

        # Only track pass/fail if the metric has pass/fail semantics (passed is not None)
        if passed is not None:
            ms.has_pass_fail = True
            if passed:
                ms.passed += 1
            else:
                ms.failed += 1

        # Update item stats with full details
        if item_stats[item_id].item_id == "":
            item_stats[item_id].item_id = item_id
        details = result.get("details", {})
        item_stats[item_id].metric_results[metric_id] = {
            "passed": passed,
            "score": numeric_score,
            "reason": details.get("reason"),
            "details": details,
        }
        # Only count pass/fail for metrics that have pass/fail semantics
        if passed is True:
            item_stats[item_id].metrics_passed += 1
        elif passed is False:
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
                m for m, r in item.metric_results.items() if r["passed"] is False
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
# Trend Analysis
# =============================================================================


@dataclass
class TrendAnalysis:
    """Analysis of trends across multiple evaluation runs for a project."""

    project_name: str
    runs: List[RunAnalysis]  # Ordered oldest to newest
    metric_trends: Dict[str, List[Optional[float]]]  # metric_id -> [pass_rate per run]
    overall_trends: List[float]  # overall pass rate per run
    item_count_trends: List[int]  # items per run
    timestamps: List[str]  # created_at per run
    run_ids: List[str]  # run IDs

    @property
    def metric_deltas(self) -> Dict[str, Optional[float]]:
        """Change in pass rate from oldest to newest run per metric."""
        deltas = {}
        for metric_id, rates in self.metric_trends.items():
            valid_rates = [r for r in rates if r is not None]
            if len(valid_rates) >= 2:
                deltas[metric_id] = valid_rates[-1] - valid_rates[0]
            else:
                deltas[metric_id] = None
        return deltas

    @property
    def overall_delta(self) -> float:
        """Change in overall pass rate from oldest to newest."""
        if len(self.overall_trends) >= 2:
            return self.overall_trends[-1] - self.overall_trends[0]
        return 0.0

    @property
    def improving_metrics(self) -> List[str]:
        return [m for m, d in self.metric_deltas.items() if d is not None and d > 0.001]

    @property
    def regressing_metrics(self) -> List[str]:
        return [m for m, d in self.metric_deltas.items() if d is not None and d < -0.001]

    @property
    def stable_metrics(self) -> List[str]:
        return [
            m
            for m, d in self.metric_deltas.items()
            if d is not None and abs(d) <= 0.001
        ]


def analyze_trends(runs: List["EvalRun"]) -> TrendAnalysis:
    """Analyze trends across multiple evaluation runs.

    Args:
        runs: List of EvalRun objects (can be in any order, will be sorted by created_at)

    Returns:
        TrendAnalysis with trend data
    """
    from .models import EvalRun

    if not runs:
        return TrendAnalysis(
            project_name="unknown",
            runs=[],
            metric_trends={},
            overall_trends=[],
            item_count_trends=[],
            timestamps=[],
            run_ids=[],
        )

    # Sort runs by created_at (oldest first for trend direction)
    sorted_runs = sorted(runs, key=lambda r: r.created_at)

    # Analyze each run
    analyses = [analyze_run(run.as_dict()) for run in sorted_runs]

    # Extract all metric IDs across all runs
    all_metrics: set = set()
    for a in analyses:
        all_metrics.update(a.metric_stats.keys())

    # Build trend data
    metric_trends: Dict[str, List[Optional[float]]] = {m: [] for m in all_metrics}
    overall_trends = []
    item_counts = []
    timestamps = []
    run_ids = []

    for a in analyses:
        overall_trends.append(a.overall_pass_rate)
        item_counts.append(a.total_items)
        timestamps.append(a.created_at)
        run_ids.append(a.run_id)

        for metric_id in all_metrics:
            if metric_id in a.metric_stats:
                metric_trends[metric_id].append(a.metric_stats[metric_id].pass_rate)
            else:
                metric_trends[metric_id].append(None)

    return TrendAnalysis(
        project_name=sorted_runs[0].dataset_name if sorted_runs else "unknown",
        runs=analyses,
        metric_trends=metric_trends,
        overall_trends=overall_trends,
        item_count_trends=item_counts,
        timestamps=timestamps,
        run_ids=run_ids,
    )


def generate_trend_text_report(trend: TrendAnalysis) -> str:
    """Generate an ASCII text report showing evaluation trends over time."""
    if not trend.runs:
        return "  No runs found for analysis."

    lines = []
    lines.append("=" * 70)
    lines.append(f"  EVALUATION TRENDS - {trend.project_name}")
    lines.append("=" * 70)
    lines.append("")

    # Summary info
    lines.append(f"  Runs analyzed: {len(trend.runs)} (oldest to newest)")
    if len(trend.timestamps) >= 2:
        first_date = trend.timestamps[0][:10] if trend.timestamps[0] else "unknown"
        last_date = trend.timestamps[-1][:10] if trend.timestamps[-1] else "unknown"
        lines.append(f"  Time range: {first_date} to {last_date}")
    lines.append("")

    # Run overview table
    lines.append("-" * 70)
    lines.append("  RUN OVERVIEW")
    lines.append("-" * 70)
    lines.append(
        f"  {'Run ID':<14} {'Date':<18} {'Items':>8} {'Pass Rate':>12} {'Delta':>10}"
    )
    lines.append(f"  {'-' * 14} {'-' * 18} {'-' * 8} {'-' * 12} {'-' * 10}")

    prev_rate = None
    for i, run in enumerate(trend.runs):
        run_id = run.run_id[:12] + ".." if len(run.run_id) > 12 else run.run_id
        date = run.created_at[:16] if run.created_at else "unknown"
        items = run.total_items
        rate = run.overall_pass_rate * 100

        # Calculate delta from previous run
        delta_str = ""
        if prev_rate is not None:
            delta = rate - prev_rate
            if delta > 0.1:
                delta_str = f"+{delta:.1f}%"
            elif delta < -0.1:
                delta_str = f"{delta:.1f}%"
            else:
                delta_str = "="
        prev_rate = rate

        lines.append(f"  {run_id:<14} {date:<18} {items:>8} {rate:>11.1f}% {delta_str:>10}")

    lines.append("")

    # Metric trends table
    lines.append("-" * 70)
    lines.append("  METRIC TRENDS (Pass Rate %)")
    lines.append("-" * 70)

    # Build dynamic header based on number of runs
    num_runs = len(trend.runs)
    if num_runs <= 5:
        # Show all runs
        header = f"  {'Metric':<22}"
        for i in range(num_runs):
            header += f" {'R' + str(i + 1):>8}"
        header += f" {'Delta':>10}"
        lines.append(header)
        lines.append(f"  {'-' * 22}" + f" {'-' * 8}" * num_runs + f" {'-' * 10}")

        for metric_id in sorted(trend.metric_trends.keys()):
            rates = trend.metric_trends[metric_id]
            metric_name = metric_id[:20] + ".." if len(metric_id) > 20 else metric_id
            row = f"  {metric_name:<22}"

            valid_rates = []
            for rate in rates:
                if rate is not None:
                    row += f" {rate * 100:>7.1f}%"
                    valid_rates.append(rate)
                else:
                    row += f" {'N/A':>8}"

            # Delta (first to last valid rate)
            if len(valid_rates) >= 2:
                delta = (valid_rates[-1] - valid_rates[0]) * 100
                if delta > 0.1:
                    row += f" {'+' + f'{delta:.1f}%':>10}"
                elif delta < -0.1:
                    row += f" {f'{delta:.1f}%':>10}"
                else:
                    row += f" {'=':>10}"
            else:
                row += f" {'N/A':>10}"

            lines.append(row)
    else:
        # Show first, last, and delta for many runs
        header = f"  {'Metric':<22} {'First':>10} {'Latest':>10} {'Delta':>10}"
        lines.append(header)
        lines.append(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10}")

        for metric_id in sorted(trend.metric_trends.keys()):
            rates = trend.metric_trends[metric_id]
            metric_name = metric_id[:20] + ".." if len(metric_id) > 20 else metric_id

            valid_rates = [r for r in rates if r is not None]
            if valid_rates:
                first = valid_rates[0] * 100
                last = valid_rates[-1] * 100
                delta = last - first

                if delta > 0.1:
                    delta_str = f"+{delta:.1f}%"
                elif delta < -0.1:
                    delta_str = f"{delta:.1f}%"
                else:
                    delta_str = "="

                lines.append(
                    f"  {metric_name:<22} {first:>9.1f}% {last:>9.1f}% {delta_str:>10}"
                )
            else:
                lines.append(f"  {metric_name:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    lines.append("")

    # Summary
    lines.append("-" * 70)
    lines.append("  SUMMARY")
    lines.append("-" * 70)

    if len(trend.overall_trends) >= 2:
        first_rate = trend.overall_trends[0] * 100
        last_rate = trend.overall_trends[-1] * 100
        overall_delta = last_rate - first_rate

        if overall_delta > 0.1:
            change_str = f"+{overall_delta:.1f}%"
        elif overall_delta < -0.1:
            change_str = f"{overall_delta:.1f}%"
        else:
            change_str = "no change"

        lines.append(f"  Overall change: {change_str} ({first_rate:.1f}% to {last_rate:.1f}%)")
    else:
        lines.append(f"  Overall pass rate: {trend.overall_trends[0] * 100:.1f}%")

    lines.append("")

    # Metric summary
    if trend.improving_metrics:
        lines.append(f"  Metrics improving ({len(trend.improving_metrics)}):  {', '.join(sorted(trend.improving_metrics)[:5])}")
        if len(trend.improving_metrics) > 5:
            lines.append(f"    ... and {len(trend.improving_metrics) - 5} more")

    if trend.regressing_metrics:
        lines.append(f"  Metrics regressing ({len(trend.regressing_metrics)}): {', '.join(sorted(trend.regressing_metrics)[:5])}")
        if len(trend.regressing_metrics) > 5:
            lines.append(f"    ... and {len(trend.regressing_metrics) - 5} more")

    if trend.stable_metrics:
        lines.append(f"  Metrics stable ({len(trend.stable_metrics)}):     {', '.join(sorted(trend.stable_metrics)[:5])}")
        if len(trend.stable_metrics) > 5:
            lines.append(f"    ... and {len(trend.stable_metrics) - 5} more")

    lines.append("")

    # Item count change
    if len(trend.item_count_trends) >= 2:
        first_items = trend.item_count_trends[0]
        last_items = trend.item_count_trends[-1]
        item_delta = last_items - first_items
        if item_delta != 0:
            sign = "+" if item_delta > 0 else ""
            lines.append(f"  Item count change: {sign}{item_delta} ({first_items} to {last_items})")

    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# HTML Report Generation (Single Self-Contained File)
# =============================================================================


def generate_html_report(
    analysis: RunAnalysis, verbose: bool = False, item_details: dict = None
) -> str:
    """Generate a high-density dark-themed evaluation dashboard.

    Uses Chart.js for interactive charts. No external images - everything is embedded.
    Styled with dark observability aesthetic (dark green background, neon accents).

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
    avg_scores = json.dumps([round(m.avg_score, 3) for m in metrics_by_pass_rate])
    min_scores = json.dumps([round(m.min_score, 3) for m in metrics_by_pass_rate])
    max_scores = json.dumps([round(m.max_score, 3) for m in metrics_by_pass_rate])
    passed_counts = json.dumps([m.passed for m in metrics_by_pass_rate])
    failed_counts = json.dumps([m.failed for m in metrics_by_pass_rate])

    # Color coding for pass rates (dark theme)
    def get_pass_rate_color(m):
        if m.pass_rate is None:
            return "#6b7280"  # Muted gray for metrics without pass/fail
        if m.pass_rate >= 0.8:
            return "#39ff14"  # Neon green for good
        if m.pass_rate >= 0.5:
            return "#ff9f1c"  # Amber for warning
        return "#ef4444"  # Red for bad

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

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval Dashboard - {analysis.dataset_name}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            /* Backgrounds - deep dark green */
            --bg-primary: #0a1210;
            --bg-secondary: #0f1a16;
            --bg-tertiary: #152420;
            --bg-hover: #1a2d28;

            /* Accents */
            --accent-primary: #39ff14;
            --accent-secondary: #ff9f1c;
            --accent-muted: #6b7280;
            --accent-purple: #8b5cf6;

            /* Status */
            --status-pass: #39ff14;
            --status-fail: #ef4444;
            --status-warn: #ff9f1c;

            /* Text */
            --text-primary: #e5e7eb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;

            /* Borders */
            --border-subtle: #1f2d28;
            --border-strong: #2d403a;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            font-size: 14px;
            min-height: 100vh;
        }}

        .dashboard {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 24px;
        }}

        /* Header Bar */
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 24px;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 24px;
        }}

        .header-title {{
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .header-meta {{
            display: flex;
            gap: 16px;
            font-size: 13px;
            color: var(--text-muted);
        }}

        .header-meta span {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .header-meta .value {{
            color: var(--text-secondary);
            font-family: monospace;
        }}

        /* KPI Bar - horizontal, no cards */
        .kpi-bar {{
            display: flex;
            align-items: stretch;
            gap: 0;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 24px;
            overflow-x: auto;
        }}

        .kpi-item {{
            flex: 1;
            min-width: 120px;
            padding: 0 24px;
            border-right: 1px solid var(--border-subtle);
            text-align: center;
        }}

        .kpi-item:last-child {{
            border-right: none;
        }}

        .kpi-value {{
            font-size: 32px;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 4px;
            font-variant-numeric: tabular-nums;
        }}

        .kpi-value.pass {{ color: var(--status-pass); }}
        .kpi-value.warn {{ color: var(--status-warn); }}
        .kpi-value.fail {{ color: var(--status-fail); }}
        .kpi-value.neutral {{ color: var(--text-primary); }}

        .kpi-unit {{
            font-size: 16px;
            font-weight: 400;
            color: var(--text-muted);
            margin-left: 2px;
        }}

        .kpi-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        .kpi-delta {{
            font-size: 12px;
            margin-top: 4px;
        }}

        .kpi-delta.up {{ color: var(--status-pass); }}
        .kpi-delta.down {{ color: var(--status-fail); }}

        /* Section layout */
        .section {{
            margin-bottom: 24px;
        }}

        .section-header {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}

        .chart-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            padding: 20px;
        }}

        .chart-title {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }}

        .chart-container {{
            position: relative;
            height: 300px;
        }}

        /* Dense Table */
        .table-wrapper {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            font-variant-numeric: tabular-nums;
        }}

        th {{
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
        }}

        th:hover {{
            color: var(--text-secondary);
        }}

        th.sort-asc,
        th.sort-desc {{
            color: var(--accent-primary);
        }}

        th.sort-asc::after {{
            content: ' ↑';
            color: var(--accent-primary);
        }}

        th.sort-desc::after {{
            content: ' ↓';
            color: var(--accent-primary);
        }}

        /* Pinned average row */
        tr.avg-row {{
            background: var(--bg-tertiary);
            font-weight: 600;
        }}

        tr.avg-row td {{
            border-bottom: 2px solid var(--border-strong);
            color: var(--accent-primary);
        }}

        td {{
            padding: 8px 12px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-secondary);
        }}

        tr:hover {{
            background: var(--bg-hover);
        }}

        /* Mini score bar */
        .score-bar {{
            display: inline-block;
            width: 40px;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
            vertical-align: middle;
            margin-right: 8px;
        }}

        .score-bar-fill {{
            height: 100%;
            border-radius: 3px;
        }}

        .score-bar-fill.high {{ background: var(--status-pass); }}
        .score-bar-fill.mid {{ background: var(--status-warn); }}
        .score-bar-fill.low {{ background: var(--status-fail); }}

        /* Status indicators */
        .status-pass {{
            color: var(--status-pass);
        }}

        .status-fail {{
            color: var(--status-fail);
        }}

        .status-warn {{
            color: var(--status-warn);
        }}

        /* Failed Items Section */
        .failed-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
        }}

        .failed-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .failed-header h3 {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .failed-count {{
            font-size: 12px;
            color: var(--status-fail);
            font-weight: 600;
        }}

        .failed-list {{
            max-height: 400px;
            overflow-y: auto;
        }}

        .failed-item {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .failed-item:last-child {{
            border-bottom: none;
        }}

        .failed-item-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }}

        .failed-item-id {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 12px;
            color: var(--text-muted);
        }}

        .failed-item-summary {{
            font-size: 12px;
            color: var(--status-fail);
        }}

        .failed-item-content {{
            display: none;
        }}

        .failed-item.expanded .failed-item-content {{
            display: block;
        }}

        .io-block {{
            margin-top: 12px;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 12px;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .io-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 8px;
            font-family: 'Inter', sans-serif;
        }}

        .io-block.input {{
            border-left: 3px solid var(--accent-purple);
        }}

        .io-block.output {{
            border-left: 3px solid var(--accent-secondary);
        }}

        .failed-metrics {{
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .failed-metric-tag {{
            font-size: 11px;
            padding: 4px 8px;
            background: rgba(239, 68, 68, 0.15);
            color: var(--status-fail);
            border-radius: 4px;
        }}

        .expand-btn {{
            background: none;
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 4px 10px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
        }}

        .expand-btn:hover {{
            border-color: var(--text-muted);
            color: var(--text-secondary);
        }}

        /* Enhanced failed item expansion */
        .io-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }}

        @media (max-width: 900px) {{
            .io-section {{
                grid-template-columns: 1fr;
            }}
        }}

        .io-content {{
            margin: 0;
            padding: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 12px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
            color: var(--text-secondary);
        }}

        .metric-details-section {{
            margin-top: 16px;
        }}

        .metric-detail {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--status-fail);
            border-radius: 4px;
            padding: 12px;
            margin-top: 8px;
        }}

        .metric-detail-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .metric-detail-name {{
            font-weight: 600;
            color: var(--status-fail);
            font-size: 13px;
        }}

        .metric-detail-score {{
            font-size: 12px;
            color: var(--text-muted);
            font-family: 'SF Mono', Monaco, monospace;
        }}

        .metric-detail-reason {{
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
        }}

        /* Metadata footer */
        .footer {{
            margin-top: 24px;
            padding: 16px 0;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            gap: 32px;
            font-size: 12px;
            color: var(--text-muted);
        }}

        .footer-item {{
            display: flex;
            gap: 8px;
        }}

        .footer-label {{
            color: var(--text-muted);
        }}

        .footer-value {{
            color: var(--text-secondary);
            font-family: monospace;
        }}

        /* Responsive */
        @media (max-width: 1024px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        @media (max-width: 768px) {{
            .kpi-bar {{
                flex-wrap: wrap;
            }}
            .kpi-item {{
                flex: 1 1 45%;
                border-right: none;
                border-bottom: 1px solid var(--border-subtle);
                padding: 16px;
            }}
            .header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }}
        }}

        /* Tooltip */
        .tooltip {{
            position: absolute;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-strong);
            padding: 8px 12px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: var(--bg-secondary);
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border-strong);
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <!-- Header Bar -->
        <div class="header">
            <div class="header-left">
                <div class="header-title">Evaluation Dashboard</div>
                <div class="header-meta">
                    <span>Dataset: <span class="value">{
        analysis.dataset_name
    }</span></span>
                    <span>Run: <span class="value">{
        analysis.run_id[:12]
    }...</span></span>
                    <span>Time: <span class="value">{
        analysis.created_at[:19] if analysis.created_at else "N/A"
    }</span></span>
                </div>
            </div>
        </div>

        <!-- KPI Bar -->
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
                <div class="kpi-value {
        "pass"
        if analysis.overall_pass_rate >= 0.8
        else "warn"
        if analysis.overall_pass_rate >= 0.5
        else "fail"
    }">{analysis.overall_pass_rate * 100:.1f}<span class="kpi-unit">%</span></div>
                <div class="kpi-label">Pass Rate</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-value {
        "pass" if len(analysis.failed_items) == 0 else "fail"
    }">{len(analysis.failed_items)}</div>
                <div class="kpi-label">Failed</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-value pass">{all_passed_count}</div>
                <div class="kpi-label">All Passed</div>
            </div>
            {
        "".join(
            f'''<div class="kpi-item">
                <div class="kpi-value {'pass' if ms.pass_rate is not None and ms.pass_rate >= 0.8 else 'warn' if ms.pass_rate is not None and ms.pass_rate >= 0.5 else 'neutral' if ms.pass_rate is None else 'fail'}">{f"{ms.pass_rate * 100:.0f}" if ms.pass_rate is not None else "N/A"}<span class="kpi-unit">{"%" if ms.pass_rate is not None else ""}</span></div>
                <div class="kpi-label">{ms.metric_id[:12]}</div>
            </div>'''
            for ms in list(metrics_by_pass_rate)[:5]
            if ms.has_pass_fail  # Only show metrics with pass/fail in KPI bar
        )
    }
        </div>

        <!-- Charts Grid -->
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
        </div>

        <!-- Metric Details Table -->
        <div class="section">
            <div class="section-header">Metric Details</div>
            <div class="table-wrapper">
                <table id="metricsTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Metric</th>
                            <th onclick="sortTable(1)">Type</th>
                            <th onclick="sortTable(2)">Pass Rate</th>
                            <th onclick="sortTable(3)">Avg Score</th>
                            <th onclick="sortTable(4)">Min</th>
                            <th onclick="sortTable(5)">Max</th>
                            <th onclick="sortTable(6)">Std Dev</th>
                            <th onclick="sortTable(7)">Passed</th>
                            <th onclick="sortTable(8)">Failed</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Average row pinned at top -->
                        <tr class="avg-row">
                            <td>AVG</td>
                            <td>-</td>
                            <td>{
        f"{sum(m.pass_rate for m in metrics_by_pass_rate if m.pass_rate is not None) / max(1, sum(1 for m in metrics_by_pass_rate if m.pass_rate is not None)) * 100:.1f}%"
        if any(m.pass_rate is not None for m in metrics_by_pass_rate)
        else "N/A"
    }</td>
                            <td>{
        sum(m.avg_score for m in metrics_by_pass_rate)
        / len(metrics_by_pass_rate):.3f}</td>
                            <td>-</td>
                            <td>-</td>
                            <td>-</td>
                            <td>{
        sum(m.passed for m in metrics_by_pass_rate if m.has_pass_fail)
    }</td>
                            <td>{
        sum(m.failed for m in metrics_by_pass_rate if m.has_pass_fail)
    }</td>
                        </tr>
                        {
        "".join(
            f'''<tr>
                            <td style="color: var(--text-primary); font-weight: 500;">{ms.metric_id}</td>
                            <td><span class="{'status-pass' if ms.metric_type == 'objective' else 'status-warn'}">{ms.metric_type[:3]}</span></td>
                            <td>
                                {f'<span class="score-bar"><span class="score-bar-fill {"high" if ms.pass_rate >= 0.8 else "mid" if ms.pass_rate >= 0.5 else "low"}" style="width: {ms.pass_rate * 100}%"></span></span><span class="{"status-pass" if ms.pass_rate >= 0.8 else "status-warn" if ms.pass_rate >= 0.5 else "status-fail"}">{ms.pass_rate * 100:.1f}%</span>' if ms.pass_rate is not None else '<span style="color: var(--text-muted);">N/A</span>'}
                            </td>
                            <td>{ms.avg_score:.3f}</td>
                            <td>{ms.min_score:.3f}</td>
                            <td>{ms.max_score:.3f}</td>
                            <td>{ms.std_dev:.3f}</td>
                            <td class="{'status-pass' if ms.has_pass_fail else ''}">{ms.passed if ms.has_pass_fail else '-'}</td>
                            <td class="{'status-fail' if ms.failed > 0 else ''}">{ms.failed if ms.has_pass_fail else '-'}</td>
                        </tr>'''
            for ms in metrics_by_pass_rate
        )
    }
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Failed Items Section -->
        {
        f'''
        <div class="section">
            <div class="section-header">Failed Items</div>
            <div class="failed-section">
                <div class="failed-header">
                    <h3>Items with metric failures</h3>
                    <span class="failed-count">{len(analysis.failed_items)} items</span>
                </div>
                <div class="failed-list">
                    {_render_failed_items(analysis, item_details)}
                    {
            f'<div class="failed-item" style="color: var(--text-muted); text-align: center;">...and {len(analysis.failed_items) - 30} more</div>'
            if len(analysis.failed_items) > 30
            else ''
        }
                </div>
            </div>
        </div>
        '''
        if analysis.failed_items
        else ""
    }

        <!-- Footer -->
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
                <span class="footer-value">{all_passed_count}/{
        analysis.total_items
    }</span>
            </div>
        </div>
    </div>

    <script>
        // Dark theme colors - observability style
        const colors = {{
            accent: '#39ff14',      // Neon lime green
            accentDim: 'rgba(57, 255, 20, 0.3)',
            secondary: '#ff9f1c',   // Amber/orange
            error: '#ef4444',       // Red
            errorDim: 'rgba(239, 68, 68, 0.3)',
            warning: '#ff9f1c',
            success: '#39ff14',
            muted: '#6b7280',
            border: '#1f2d28',
            gridLine: 'rgba(31, 45, 40, 0.5)',
            bg: '#0a1210',
            text: '#e5e7eb',
            textMuted: '#9ca3af'
        }};

        // Chart.js defaults for dark theme
        Chart.defaults.font.family = "'Inter', 'SF Pro Display', -apple-system, system-ui, sans-serif";
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

        // Table sorting
        let sortState = {{ column: null, ascending: true }};
        function sortTable(colIndex) {{
            const table = document.querySelector('.results-table');
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Toggle direction if same column
            if (sortState.column === colIndex) {{
                sortState.ascending = !sortState.ascending;
            }} else {{
                sortState.column = colIndex;
                sortState.ascending = true;
            }}

            rows.sort((a, b) => {{
                const aCell = a.cells[colIndex];
                const bCell = b.cells[colIndex];
                let aVal = aCell?.textContent?.trim() || '';
                let bVal = bCell?.textContent?.trim() || '';

                // Try numeric comparison
                const aNum = parseFloat(aVal.replace('%', ''));
                const bNum = parseFloat(bVal.replace('%', ''));
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return sortState.ascending ? aNum - bNum : bNum - aNum;
                }}

                // String comparison
                return sortState.ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            rows.forEach(row => tbody.appendChild(row));

            // Update header indicators
            table.querySelectorAll('th').forEach((th, i) => {{
                th.classList.remove('sort-asc', 'sort-desc');
                if (i === colIndex) {{
                    th.classList.add(sortState.ascending ? 'sort-asc' : 'sort-desc');
                }}
            }});
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

        // Score Distribution Chart - thin lines with circular avg points
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

        // Pass/Fail Chart - stacked horizontal bars
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
                        borderColor: colors.accent,
                        borderWidth: 1,
                        borderRadius: 2,
                        borderSkipped: false,
                        barThickness: 16,
                    }},
                    {{
                        label: 'Failed',
                        data: {failed_counts},
                        backgroundColor: colors.error,
                        borderColor: colors.error,
                        borderWidth: 1,
                        borderRadius: 2,
                        borderSkipped: false,
                        barThickness: 16,
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
                        labels: {{
                            usePointStyle: true,
                            pointStyle: 'rect',
                            color: colors.text
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: colors.bg,
                        titleColor: colors.text,
                        bodyColor: colors.text,
                        borderColor: colors.border,
                        borderWidth: 1,
                        padding: 10
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: true,
                        grid: {{
                            color: colors.gridLine,
                            lineWidth: 1
                        }},
                        ticks: {{ color: colors.textMuted }},
                        title: {{ display: true, text: 'Count', color: colors.muted }}
                    }},
                    y: {{
                        stacked: true,
                        grid: {{ display: false }},
                        ticks: {{ color: colors.text }}
                    }}
                }}
            }}
        }});

        {
        f'''
        // Correlation Heatmap - dark theme colors
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
                            // Dark theme: neon green for positive, red for negative
                            if (v >= 0.7) return 'rgba(57, 255, 20, 0.9)';
                            if (v >= 0.3) return 'rgba(57, 255, 20, 0.5)';
                            if (v >= -0.3) return 'rgba(107, 114, 128, 0.5)';
                            if (v >= -0.7) return 'rgba(239, 68, 68, 0.5)';
                            return 'rgba(239, 68, 68, 0.9)';
                        }}),
                        pointRadius: heatmapData.map(() => 18),
                        pointStyle: 'rect'
                    }}]
                }},
                options: {{
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
                                callback: (v) => corrData.labels[v] || '',
                                color: colors.textMuted
                            }},
                            grid: {{ display: false }}
                        }},
                        y: {{
                            type: 'linear',
                            min: -0.5,
                            max: corrData.labels.length - 0.5,
                            ticks: {{
                                stepSize: 1,
                                callback: (v) => corrData.labels[v] || '',
                                color: colors.textMuted
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
