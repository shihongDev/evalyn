"""
Trend analysis across multiple evaluation runs.

Provides tools to analyze how metrics change over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from .core import RunAnalysis, analyze_run

if TYPE_CHECKING:
    from ..models import EvalRun


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
