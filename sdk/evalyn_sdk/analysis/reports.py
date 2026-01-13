"""
Text and ASCII report generation.

Provides terminal-friendly text reports with ASCII visualizations.
"""

from __future__ import annotations

from typing import List, Optional

from .core import RunAnalysis


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
    metric_id: str, pass_rate: Optional[float], count: int, width: int = 25
) -> str:
    """Format a metric pass rate with a bar chart."""
    if pass_rate is None:
        return f"  {metric_id:30} {'N/A (no pass/fail)':>{width + 12}}"
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

    # Sort by pass rate (lowest first to highlight problems, None values at end)
    sorted_metrics = sorted(
        analysis.metric_stats.values(),
        key=lambda m: m.pass_rate if m.pass_rate is not None else 2.0,
    )

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
                if rate is not None:
                    rates.append(rate)
                    row += f" {rate * 100:>9.1f}%"
                else:
                    row += f" {'N/A':>10}"
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
