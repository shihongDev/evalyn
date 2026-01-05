"""
Analyzer module for comprehensive eval results analysis and visualization.

Supports optional seaborn/matplotlib visualizations and sklearn analysis.
"""
from __future__ import annotations

import json
import math
import io
import base64
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional imports for advanced visualization
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Optional sklearn for clustering analysis
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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
    metric_results: Dict[str, Tuple[bool, float]] = field(default_factory=dict)  # metric_id -> (passed, score)

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
    """Load a single eval run JSON file."""
    with open(path) as f:
        return json.load(f)


def find_eval_runs(dataset_dir: Path) -> List[Path]:
    """Find all eval run JSON files in a dataset directory."""
    eval_runs_dir = dataset_dir / "eval_runs"
    if not eval_runs_dir.exists():
        return []

    runs = list(eval_runs_dir.glob("*.json"))
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
        score = result.get("score", 0.0)
        passed = result.get("passed", False)

        # Update metric stats
        if metric_id not in metric_stats:
            metric_stats[metric_id] = MetricStats(
                metric_id=metric_id,
                metric_type=metric_types.get(metric_id, "unknown")
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
        item_id for item_id, stats in item_stats.items()
        if not stats.all_passed
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
# ASCII Visualization Helpers
# =============================================================================

def ascii_bar(value: float, max_width: int = 30, fill: str = "█", empty: str = "░") -> str:
    """Create an ASCII progress bar."""
    filled = int(value * max_width)
    return fill * filled + empty * (max_width - filled)


def ascii_histogram(values: List[float], bins: int = 10, width: int = 40, height: int = 8) -> str:
    """Create an ASCII histogram of values."""
    if not values:
        return "  (no data)"

    min_val = min(values)
    max_val = max(values)

    # Handle case where all values are the same
    if min_val == max_val:
        bin_counts = [len(values)] + [0] * (bins - 1)
    else:
        bin_size = (max_val - min_val) / bins
        bin_counts = [0] * bins
        for v in values:
            bin_idx = min(int((v - min_val) / bin_size), bins - 1)
            bin_counts[bin_idx] += 1

    max_count = max(bin_counts) if bin_counts else 1

    lines = []

    # Draw histogram bars (vertical)
    bar_width = width // bins
    for row in range(height, 0, -1):
        threshold = (row / height) * max_count
        line = "  "
        for count in bin_counts:
            if count >= threshold:
                line += "█" * bar_width
            else:
                line += " " * bar_width
        lines.append(line)

    # X-axis
    lines.append("  " + "─" * (bar_width * bins))

    # Labels
    label_line = f"  {min_val:.2f}" + " " * (bar_width * bins - 10) + f"{max_val:.2f}"
    lines.append(label_line)

    return "\n".join(lines)


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

    return f"  {metric_id:30} [{''.join(bars)}] avg={sum(scores)/len(scores):.2f}"


def format_pass_rate_bar(metric_id: str, pass_rate: float, count: int, width: int = 25) -> str:
    """Format a metric pass rate with a bar chart."""
    bar = ascii_bar(pass_rate, max_width=width)
    pct = f"{pass_rate*100:5.1f}%"
    return f"  {metric_id:30} {bar} {pct} (n={count})"


# =============================================================================
# Report Generation
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
    lines.append(f"  Created:    {analysis.created_at[:19] if analysis.created_at else 'unknown'}")
    lines.append(f"  Items:      {analysis.total_items}")
    lines.append(f"  Metrics:    {analysis.total_metrics}")
    lines.append("")

    # Overall summary
    lines.append("-" * 70)
    lines.append("  OVERALL SUMMARY")
    lines.append("-" * 70)
    all_passed = sum(1 for item in analysis.item_stats.values() if item.all_passed)
    lines.append(f"  Items passing all metrics: {all_passed}/{analysis.total_items} ({analysis.overall_pass_rate*100:.1f}%)")
    lines.append(f"  Items with failures:       {len(analysis.failed_items)}")
    lines.append("")

    # Metric pass rates
    lines.append("-" * 70)
    lines.append("  METRIC PASS RATES")
    lines.append("-" * 70)

    # Sort by pass rate (lowest first to highlight problems)
    sorted_metrics = sorted(
        analysis.metric_stats.values(),
        key=lambda m: m.pass_rate
    )

    for ms in sorted_metrics:
        lines.append(format_pass_rate_bar(ms.metric_id, ms.pass_rate, ms.count))
    lines.append("")

    # Score statistics
    lines.append("-" * 70)
    lines.append("  SCORE STATISTICS")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<30} {'Avg':>8} {'Min':>8} {'Max':>8} {'StdDev':>8}")
    lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

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
    lines.append(f"  {'-'*12} {'-'*20} {'-'*8} {'-'*10}")

    for a in analyses:
        date = a.created_at[:16] if a.created_at else "unknown"
        lines.append(
            f"  {a.run_id[:12]} {date:<20} {a.total_items:>8} "
            f"{a.overall_pass_rate*100:>9.1f}%"
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
        header += f" {'Run'+str(i+1):>10}"
    header += f" {'Delta':>10}"
    lines.append(header)
    lines.append(f"  {'-'*25}" + f" {'-'*10}" * (len(analyses) + 1))

    for metric_id in sorted(all_metrics):
        row = f"  {metric_id:<25}"
        rates = []
        for a in analyses:
            if metric_id in a.metric_stats:
                rate = a.metric_stats[metric_id].pass_rate
                rates.append(rate)
                row += f" {rate*100:>9.1f}%"
            else:
                row += f" {'N/A':>10}"

        # Delta (newest - oldest)
        if len(rates) >= 2:
            delta = rates[-1] - rates[0]
            sign = "+" if delta >= 0 else ""
            row += f" {sign}{delta*100:>8.1f}%"
        else:
            row += f" {'N/A':>10}"

        lines.append(row)

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_html_report(analysis: RunAnalysis) -> str:
    """Generate an HTML report with charts."""

    # Prepare data for charts
    metric_data = []
    for ms in sorted(analysis.metric_stats.values(), key=lambda m: -m.pass_rate):
        metric_data.append({
            "id": ms.metric_id,
            "type": ms.metric_type,
            "pass_rate": ms.pass_rate * 100,
            "avg_score": ms.avg_score,
            "count": ms.count,
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Eval Run Analysis - {analysis.run_id[:8]}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-top: 0; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        .stat {{ display: inline-block; margin-right: 40px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9fafb; }}
        .pass {{ color: #059669; }}
        .fail {{ color: #dc2626; }}
        .chart-container {{ height: 400px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Eval Run Analysis</h1>
            <div class="stat">
                <div class="stat-value">{analysis.total_items}</div>
                <div class="stat-label">Items</div>
            </div>
            <div class="stat">
                <div class="stat-value">{analysis.total_metrics}</div>
                <div class="stat-label">Metrics</div>
            </div>
            <div class="stat">
                <div class="stat-value">{analysis.overall_pass_rate*100:.1f}%</div>
                <div class="stat-label">Overall Pass Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(analysis.failed_items)}</div>
                <div class="stat-label">Failed Items</div>
            </div>
            <p style="color: #666; margin-top: 16px;">
                <strong>Run ID:</strong> {analysis.run_id}<br>
                <strong>Dataset:</strong> {analysis.dataset_name}<br>
                <strong>Created:</strong> {analysis.created_at}
            </p>
        </div>

        <div class="card">
            <h2>Pass Rates by Metric</h2>
            <div class="chart-container">
                <canvas id="passRateChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Score Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Type</th>
                        <th>Pass Rate</th>
                        <th>Avg Score</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''<tr>
                        <td>{m["id"]}</td>
                        <td>{m["type"]}</td>
                        <td class="{"pass" if m["pass_rate"] >= 80 else "fail"}">{m["pass_rate"]:.1f}%</td>
                        <td>{m["avg_score"]:.3f}</td>
                        <td>{m["count"]}</td>
                    </tr>''' for m in metric_data)}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('passRateChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([m["id"] for m in metric_data])},
                datasets: [{{
                    label: 'Pass Rate (%)',
                    data: {json.dumps([m["pass_rate"] for m in metric_data])},
                    backgroundColor: {json.dumps([
                        '#059669' if m["pass_rate"] >= 80 else '#f59e0b' if m["pass_rate"] >= 50 else '#dc2626'
                        for m in metric_data
                    ])},
                    borderRadius: 4,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        title: {{ display: true, text: 'Pass Rate (%)' }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


# =============================================================================
# Seaborn/Matplotlib Visualizations
# =============================================================================

def _setup_modern_style():
    """Setup modern OpenAI-inspired plot style."""
    # Modern color palette
    colors = {
        'bg': '#0d1117',           # Dark background
        'bg_light': '#161b22',     # Slightly lighter bg
        'text': '#e6edf3',         # Light text
        'text_muted': '#8b949e',   # Muted text
        'grid': '#30363d',         # Grid lines
        'accent': '#58a6ff',       # Blue accent
        'success': '#3fb950',      # Green
        'warning': '#d29922',      # Yellow/Orange
        'error': '#f85149',        # Red
        'purple': '#a371f7',       # Purple accent
        'cyan': '#39c5cf',         # Cyan accent
    }

    # Set style
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': colors['bg'],
        'axes.facecolor': colors['bg_light'],
        'axes.edgecolor': colors['grid'],
        'axes.labelcolor': colors['text'],
        'axes.titlecolor': colors['text'],
        'text.color': colors['text'],
        'xtick.color': colors['text_muted'],
        'ytick.color': colors['text_muted'],
        'grid.color': colors['grid'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })

    return colors


def generate_visualizations(analysis: RunAnalysis, output_dir: Path) -> Dict[str, Path]:
    """Generate modern OpenAI-style visualizations and save as PNG files.

    Returns dict mapping chart name to file path.
    """
    if not HAS_VISUALIZATION:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = {}
    colors = _setup_modern_style()

    metrics = sorted(analysis.metric_stats.values(), key=lambda m: m.pass_rate, reverse=True)

    # 1. Pass Rate Bar Chart - Modern horizontal bars with gradient effect
    fig, ax = plt.subplots(figsize=(14, max(6, len(metrics) * 0.6)))

    metric_ids = [m.metric_id for m in metrics]
    pass_rates = [m.pass_rate * 100 for m in metrics]

    # Color based on pass rate
    bar_colors = [
        colors['success'] if r >= 80 else colors['warning'] if r >= 50 else colors['error']
        for r in pass_rates
    ]

    y_pos = np.arange(len(metric_ids))
    bars = ax.barh(y_pos, pass_rates, height=0.7, color=bar_colors, alpha=0.9,
                   edgecolor='none', zorder=3)

    # Add subtle background bars
    ax.barh(y_pos, [100] * len(metric_ids), height=0.7, color=colors['grid'],
            alpha=0.3, zorder=1)

    # Threshold lines
    ax.axvline(x=80, color=colors['success'], linestyle='--', alpha=0.5,
               linewidth=1.5, zorder=2, label='Target (80%)')
    ax.axvline(x=50, color=colors['warning'], linestyle='--', alpha=0.5,
               linewidth=1.5, zorder=2, label='Minimum (50%)')

    # Value labels
    for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
        label_color = colors['text'] if rate < 90 else colors['bg']
        x_pos = min(rate + 2, 95) if rate < 85 else rate - 8
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{rate:.0f}%', va='center', ha='left' if rate < 85 else 'right',
                fontsize=11, fontweight='bold', color=label_color, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_ids, fontsize=11)
    ax.set_xlabel('Pass Rate (%)', fontsize=12, labelpad=10)
    ax.set_xlim(0, 105)
    ax.set_title('Metric Performance', fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.8, facecolor=colors['bg_light'])
    ax.invert_yaxis()

    plt.tight_layout()
    path = output_dir / 'pass_rates.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close()
    generated['pass_rates'] = path

    # 2. Score Distribution - Modern violin/strip plot
    fig, ax = plt.subplots(figsize=(14, max(6, len(metrics) * 0.5)))

    score_data = []
    score_labels = []
    for ms in metrics:
        if ms.scores and max(ms.scores) <= 1.5:  # Include normalized-ish scores
            score_data.extend(ms.scores)
            score_labels.extend([ms.metric_id] * len(ms.scores))

    if score_data:
        df_scores = pd.DataFrame({'Metric': score_labels, 'Score': score_data})

        # Create gradient palette
        n_metrics = len(set(score_labels))
        palette = sns.color_palette([colors['cyan'], colors['accent'],
                                    colors['purple'], colors['success']], n_colors=n_metrics)

        # Strip plot with jitter
        sns.stripplot(data=df_scores, x='Score', y='Metric', ax=ax,
                     palette=palette, alpha=0.7, size=10, jitter=0.2, zorder=3)

        # Add mean markers
        means = df_scores.groupby('Metric')['Score'].mean()
        for i, (metric, mean) in enumerate(means.items()):
            ax.scatter([mean], [i], color=colors['text'], s=150, marker='|',
                      linewidths=3, zorder=4)

        ax.set_xlabel('Score', fontsize=12, labelpad=10)
        ax.set_xlim(-0.05, 1.1)
        ax.set_title('Score Distribution', fontsize=18, fontweight='bold', pad=20)
        ax.axvline(x=0.5, color=colors['warning'], linestyle=':', alpha=0.4, linewidth=1)
        ax.axvline(x=0.8, color=colors['success'], linestyle=':', alpha=0.4, linewidth=1)

    plt.tight_layout()
    path = output_dir / 'score_distributions.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close()
    generated['score_distributions'] = path

    # 3. Metric Correlation Heatmap - Modern style
    if len(analysis.item_stats) > 1 and len(metrics) > 1:
        item_ids = list(analysis.item_stats.keys())
        metric_ids = [m.metric_id for m in metrics]

        score_matrix = []
        for item_id in item_ids:
            row = []
            for metric_id in metric_ids:
                if metric_id in analysis.item_stats[item_id].metric_results:
                    _, score = analysis.item_stats[item_id].metric_results[metric_id]
                    row.append(score)
                else:
                    row.append(np.nan)
            score_matrix.append(row)

        df = pd.DataFrame(score_matrix, columns=metric_ids, index=item_ids)
        corr = df.corr()

        if not corr.empty and not corr.isna().all().all():
            fig, ax = plt.subplots(figsize=(12, 10))

            # Custom diverging colormap
            cmap = sns.diverging_palette(250, 130, s=80, l=55, as_cmap=True)

            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                       center=0, vmin=-1, vmax=1, ax=ax,
                       square=True, linewidths=2, linecolor=colors['bg'],
                       cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                       annot_kws={'size': 11, 'weight': 'bold'})

            ax.set_title('Metric Correlations', fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            path = output_dir / 'correlation_heatmap.png'
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
            plt.close()
            generated['correlation_heatmap'] = path

    # 4. Pass/Fail Donut Chart - Modern style
    fig, ax = plt.subplots(figsize=(10, 10))
    all_passed = sum(1 for item in analysis.item_stats.values() if item.all_passed)
    failed = len(analysis.failed_items)

    if all_passed + failed > 0:
        sizes = [all_passed, failed]
        chart_colors = [colors['success'], colors['error']]

        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            sizes, colors=chart_colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor=colors['bg'], linewidth=3),
            textprops={'fontsize': 14, 'fontweight': 'bold', 'color': colors['text']}
        )

        # Center text
        ax.text(0, 0.1, f'{all_passed + failed}', ha='center', va='center',
                fontsize=48, fontweight='bold', color=colors['text'])
        ax.text(0, -0.15, 'Total Items', ha='center', va='center',
                fontsize=14, color=colors['text_muted'])

        # Legend
        ax.legend(wedges, [f'Passed ({all_passed})', f'Failed ({failed})'],
                 loc='upper right', bbox_to_anchor=(1.15, 1),
                 fontsize=12, framealpha=0.8, facecolor=colors['bg_light'])

        ax.set_title('Overall Results', fontsize=18, fontweight='bold', pad=20, y=1.02)

    plt.tight_layout()
    path = output_dir / 'pass_fail_pie.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close()
    generated['pass_fail_pie'] = path

    # 5. Per-Metric Pass/Fail - Modern stacked bar
    fig, ax = plt.subplots(figsize=(14, max(6, len(metrics) * 0.6)))

    metric_ids = [m.metric_id for m in metrics]
    passed_counts = [m.passed for m in metrics]
    failed_counts = [m.failed for m in metrics]
    total_counts = [p + f for p, f in zip(passed_counts, failed_counts)]

    y_pos = np.arange(len(metric_ids))

    # Stacked bars
    bars1 = ax.barh(y_pos, passed_counts, height=0.7, color=colors['success'],
                    alpha=0.9, label='Passed', edgecolor='none', zorder=3)
    bars2 = ax.barh(y_pos, failed_counts, height=0.7, left=passed_counts,
                    color=colors['error'], alpha=0.9, label='Failed', edgecolor='none', zorder=3)

    # Labels
    for i, (passed, failed, total) in enumerate(zip(passed_counts, failed_counts, total_counts)):
        if passed > 0:
            ax.text(passed/2, i, str(passed), ha='center', va='center',
                   fontsize=11, fontweight='bold', color=colors['bg'])
        if failed > 0:
            ax.text(passed + failed/2, i, str(failed), ha='center', va='center',
                   fontsize=11, fontweight='bold', color=colors['bg'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_ids, fontsize=11)
    ax.set_xlabel('Count', fontsize=12, labelpad=10)
    ax.set_title('Results by Metric', fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.8, facecolor=colors['bg_light'])
    ax.invert_yaxis()

    plt.tight_layout()
    path = output_dir / 'pass_fail_stacked.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close()
    generated['pass_fail_stacked'] = path

    # 6. Summary Dashboard Card
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor(colors['bg'])

    # Create grid
    gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)

    # Stat cards
    stats = [
        ('Total Items', analysis.total_items, colors['accent']),
        ('Metrics', analysis.total_metrics, colors['purple']),
        ('Pass Rate', f'{analysis.overall_pass_rate*100:.1f}%',
         colors['success'] if analysis.overall_pass_rate >= 0.8 else colors['warning'] if analysis.overall_pass_rate >= 0.5 else colors['error']),
        ('Failed', len(analysis.failed_items),
         colors['success'] if len(analysis.failed_items) == 0 else colors['error']),
    ]

    for i, (label, value, color) in enumerate(stats):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(colors['bg_light'])

        # Add rounded rectangle effect
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(0.5, 0.65, str(value), ha='center', va='center',
               fontsize=42, fontweight='bold', color=color,
               transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center',
               fontsize=14, color=colors['text_muted'],
               transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Border
        rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                             edgecolor=colors['grid'], linewidth=2,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

    fig.suptitle('Evaluation Summary', fontsize=20, fontweight='bold',
                color=colors['text'], y=1.02)

    plt.tight_layout()
    path = output_dir / 'summary_dashboard.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close()
    generated['summary_dashboard'] = path

    # 7. Unified Dashboard - Single page with all key visualizations
    unified_path = generate_unified_dashboard(analysis, output_dir, colors)
    if unified_path:
        generated['unified_dashboard'] = unified_path

    return generated


def generate_unified_dashboard(analysis: RunAnalysis, output_dir: Path, colors: dict) -> Optional[Path]:
    """Generate Anthropic-style unified single-page dashboard."""
    if not HAS_VISUALIZATION:
        return None

    metrics = sorted(analysis.metric_stats.values(), key=lambda m: m.pass_rate, reverse=True)

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor(colors['bg'])

    # Title
    fig.suptitle('Evaluation Report', fontsize=28, fontweight='bold',
                color=colors['text'], y=0.98)

    # Subtitle with metadata
    subtitle = f"Dataset: {analysis.dataset_name}  •  {analysis.total_items} items  •  {analysis.total_metrics} metrics  •  {analysis.created_at[:10] if analysis.created_at else 'N/A'}"
    fig.text(0.5, 0.955, subtitle, ha='center', fontsize=12, color=colors['text_muted'])

    # Create grid spec for layout
    gs = fig.add_gridspec(4, 3, height_ratios=[0.8, 1.2, 1.2, 1.2],
                         hspace=0.25, wspace=0.25,
                         left=0.06, right=0.94, top=0.93, bottom=0.04)

    # =========================================================================
    # Row 1: Summary Stats Cards
    # =========================================================================
    all_passed = sum(1 for item in analysis.item_stats.values() if item.all_passed)
    stats = [
        ('Total Items', str(analysis.total_items), colors['accent']),
        ('Metrics Evaluated', str(analysis.total_metrics), colors['purple']),
        ('Overall Pass Rate', f'{analysis.overall_pass_rate*100:.1f}%',
         colors['success'] if analysis.overall_pass_rate >= 0.8 else colors['warning'] if analysis.overall_pass_rate >= 0.5 else colors['error']),
        ('Items Passed All', f'{all_passed}/{analysis.total_items}',
         colors['success'] if all_passed == analysis.total_items else colors['warning'] if all_passed > 0 else colors['error']),
        ('Failed Items', str(len(analysis.failed_items)),
         colors['success'] if len(analysis.failed_items) == 0 else colors['error']),
        ('Avg Metrics/Item', f'{sum(len(item.metric_results) for item in analysis.item_stats.values()) / max(len(analysis.item_stats), 1):.1f}', colors['cyan']),
    ]

    # Create 6 stat cards in row 1
    for i, (label, value, color) in enumerate(stats):
        # Calculate position for 6 cards across 3 columns
        row_idx = i // 3
        col_idx = i % 3
        if row_idx == 0:
            ax = fig.add_subplot(gs[0, col_idx])
        else:
            # For second row of stats, we need to split the first grid row
            ax = fig.add_axes([0.06 + col_idx * 0.3, 0.89, 0.26, 0.035])

        ax.set_facecolor(colors['bg_light'])
        for spine in ax.spines.values():
            spine.set_edgecolor(colors['grid'])
            spine.set_linewidth(1.5)

        ax.text(0.5, 0.6, value, ha='center', va='center',
               fontsize=28 if row_idx == 0 else 20, fontweight='bold', color=color,
               transform=ax.transAxes)
        ax.text(0.5, 0.2, label, ha='center', va='center',
               fontsize=11, color=colors['text_muted'],
               transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    # =========================================================================
    # Row 2: Pass Rate Bar Chart (left 2 cols) + Donut Chart (right col)
    # =========================================================================

    # Pass Rate Bar Chart
    ax_bars = fig.add_subplot(gs[1, :2])
    ax_bars.set_facecolor(colors['bg_light'])

    metric_ids = [m.metric_id for m in metrics]
    pass_rates = [m.pass_rate * 100 for m in metrics]
    bar_colors = [
        colors['success'] if r >= 80 else colors['warning'] if r >= 50 else colors['error']
        for r in pass_rates
    ]

    y_pos = np.arange(len(metric_ids))
    bars = ax_bars.barh(y_pos, pass_rates, height=0.65, color=bar_colors, alpha=0.9, zorder=3)
    ax_bars.barh(y_pos, [100] * len(metric_ids), height=0.65, color=colors['grid'], alpha=0.3, zorder=1)

    ax_bars.axvline(x=80, color=colors['success'], linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    ax_bars.axvline(x=50, color=colors['warning'], linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)

    for bar, rate in zip(bars, pass_rates):
        x_pos = min(rate + 2, 92) if rate < 80 else rate - 6
        ax_bars.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{rate:.0f}%', va='center', ha='left' if rate < 80 else 'right',
                    fontsize=10, fontweight='bold', color=colors['text'], zorder=4)

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(metric_ids, fontsize=10)
    ax_bars.set_xlabel('Pass Rate (%)', fontsize=11, color=colors['text_muted'])
    ax_bars.set_xlim(0, 105)
    ax_bars.set_title('Metric Performance', fontsize=14, fontweight='bold', pad=15, loc='left')
    ax_bars.invert_yaxis()

    for spine in ax_bars.spines.values():
        spine.set_edgecolor(colors['grid'])

    # Donut Chart
    ax_donut = fig.add_subplot(gs[1, 2])
    ax_donut.set_facecolor(colors['bg'])

    failed = len(analysis.failed_items)
    if all_passed + failed > 0:
        sizes = [all_passed, failed] if failed > 0 else [all_passed]
        chart_colors = [colors['success'], colors['error']] if failed > 0 else [colors['success']]

        wedges, texts, autotexts = ax_donut.pie(
            sizes, colors=chart_colors,
            autopct=lambda p: f'{p:.0f}%' if p > 0 else '',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor=colors['bg'], linewidth=2),
            textprops={'fontsize': 12, 'fontweight': 'bold', 'color': colors['text']}
        )

        ax_donut.text(0, 0.05, f'{analysis.overall_pass_rate*100:.0f}%', ha='center', va='center',
                     fontsize=32, fontweight='bold', color=colors['text'])
        ax_donut.text(0, -0.15, 'Pass Rate', ha='center', va='center',
                     fontsize=11, color=colors['text_muted'])

    ax_donut.set_title('Overall Results', fontsize=14, fontweight='bold', pad=15)

    # =========================================================================
    # Row 3: Correlation Heatmap (left 2 cols) + Score Distribution (right col)
    # =========================================================================

    # Correlation Heatmap
    ax_corr = fig.add_subplot(gs[2, :2])
    ax_corr.set_facecolor(colors['bg_light'])

    if len(analysis.item_stats) > 1 and len(metrics) > 1:
        item_ids = list(analysis.item_stats.keys())
        metric_ids_corr = [m.metric_id for m in metrics]

        score_matrix = []
        for item_id in item_ids:
            row = []
            for metric_id in metric_ids_corr:
                if metric_id in analysis.item_stats[item_id].metric_results:
                    _, score = analysis.item_stats[item_id].metric_results[metric_id]
                    row.append(score)
                else:
                    row.append(np.nan)
            score_matrix.append(row)

        df = pd.DataFrame(score_matrix, columns=metric_ids_corr, index=item_ids)
        corr = df.corr()

        if not corr.empty and not corr.isna().all().all():
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(250, 130, s=80, l=55, as_cmap=True)
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                       center=0, vmin=-1, vmax=1, ax=ax_corr,
                       square=True, linewidths=1.5, linecolor=colors['bg'],
                       cbar_kws={'shrink': 0.6, 'label': 'Correlation'},
                       annot_kws={'size': 9, 'weight': 'bold'})

    ax_corr.set_title('Metric Correlations', fontsize=14, fontweight='bold', pad=15, loc='left')

    # Score Stats Table (right column)
    ax_table = fig.add_subplot(gs[2, 2])
    ax_table.set_facecolor(colors['bg_light'])
    ax_table.axis('off')

    # Create a simple score summary
    table_data = []
    for ms in metrics[:8]:  # Top 8
        table_data.append([
            ms.metric_id[:20],
            f'{ms.avg_score:.2f}',
            f'{ms.std_dev:.2f}'
        ])

    if table_data:
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Metric', 'Avg', 'StdDev'],
            loc='center',
            cellLoc='center',
            colColours=[colors['bg_light']] * 3,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        for key, cell in table.get_celld().items():
            cell.set_edgecolor(colors['grid'])
            cell.set_facecolor(colors['bg_light'])
            cell.set_text_props(color=colors['text'])
            if key[0] == 0:  # Header
                cell.set_text_props(weight='bold', color=colors['accent'])

    ax_table.set_title('Score Statistics', fontsize=14, fontweight='bold', pad=15)

    # =========================================================================
    # Row 4: Pass/Fail Stacked (left 2 cols) + Failed Items List (right col)
    # =========================================================================

    # Stacked Bar
    ax_stacked = fig.add_subplot(gs[3, :2])
    ax_stacked.set_facecolor(colors['bg_light'])

    passed_counts = [m.passed for m in metrics]
    failed_counts = [m.failed for m in metrics]

    y_pos = np.arange(len(metrics))
    ax_stacked.barh(y_pos, passed_counts, height=0.65, color=colors['success'],
                   alpha=0.9, label='Passed', zorder=3)
    ax_stacked.barh(y_pos, failed_counts, height=0.65, left=passed_counts,
                   color=colors['error'], alpha=0.9, label='Failed', zorder=3)

    for i, (passed, failed) in enumerate(zip(passed_counts, failed_counts)):
        total = passed + failed
        if passed > 0:
            ax_stacked.text(passed/2, i, str(passed), ha='center', va='center',
                          fontsize=9, fontweight='bold', color=colors['bg'])
        if failed > 0:
            ax_stacked.text(passed + failed/2, i, str(failed), ha='center', va='center',
                          fontsize=9, fontweight='bold', color=colors['bg'])

    ax_stacked.set_yticks(y_pos)
    ax_stacked.set_yticklabels([m.metric_id for m in metrics], fontsize=10)
    ax_stacked.set_xlabel('Count', fontsize=11, color=colors['text_muted'])
    ax_stacked.set_title('Results by Metric', fontsize=14, fontweight='bold', pad=15, loc='left')
    ax_stacked.legend(loc='lower right', framealpha=0.8, facecolor=colors['bg_light'])
    ax_stacked.invert_yaxis()

    for spine in ax_stacked.spines.values():
        spine.set_edgecolor(colors['grid'])

    # Failed Items Summary
    ax_failed = fig.add_subplot(gs[3, 2])
    ax_failed.set_facecolor(colors['bg_light'])
    ax_failed.axis('off')

    if analysis.failed_items:
        failed_text = "Failed Items:\n\n"
        for i, item_id in enumerate(analysis.failed_items[:8]):
            item = analysis.item_stats[item_id]
            failed_metrics = [m for m, (p, _) in item.metric_results.items() if not p]
            failed_text += f"• {item_id[:12]}...\n"
            failed_text += f"  ↳ {', '.join(failed_metrics[:2])}\n"
            if len(failed_metrics) > 2:
                failed_text += f"     +{len(failed_metrics)-2} more\n"
            failed_text += "\n"

        if len(analysis.failed_items) > 8:
            failed_text += f"... and {len(analysis.failed_items) - 8} more"

        ax_failed.text(0.05, 0.95, failed_text, transform=ax_failed.transAxes,
                      fontsize=9, color=colors['text'], verticalalignment='top',
                      fontfamily='monospace')
    else:
        ax_failed.text(0.5, 0.5, '✓ All items passed!', transform=ax_failed.transAxes,
                      fontsize=14, color=colors['success'], ha='center', va='center',
                      fontweight='bold')

    ax_failed.set_title('Failure Details', fontsize=14, fontweight='bold', pad=15)

    for spine in ax_failed.spines.values():
        spine.set_edgecolor(colors['grid'])
        spine.set_visible(True)

    # Save
    path = output_dir / 'unified_dashboard.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close()

    return path


# =============================================================================
# Sklearn Analysis
# =============================================================================

def analyze_failure_patterns(analysis: RunAnalysis) -> Dict[str, Any]:
    """Use sklearn to analyze patterns in failed items.

    Returns clustering and pattern analysis results.
    """
    if not HAS_SKLEARN or not HAS_VISUALIZATION:
        return {"error": "sklearn or numpy/pandas not available"}

    if len(analysis.failed_items) < 3:
        return {"error": "Need at least 3 failed items for pattern analysis"}

    # Build feature matrix from failed items
    metrics = list(analysis.metric_stats.keys())
    feature_matrix = []
    item_ids = []

    for item_id in analysis.failed_items:
        item = analysis.item_stats[item_id]
        row = []
        for metric_id in metrics:
            if metric_id in item.metric_results:
                passed, score = item.metric_results[metric_id]
                row.append(score if not np.isnan(score) else 0.0)
            else:
                row.append(0.0)
        feature_matrix.append(row)
        item_ids.append(item_id)

    X = np.array(feature_matrix)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal number of clusters (max 5 or n_samples-1)
    max_clusters = min(5, len(X) - 1, 3)
    if max_clusters < 2:
        return {"error": "Not enough samples for clustering"}

    # K-means clustering
    kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Analyze each cluster
    cluster_analysis = {}
    for cluster_id in range(max_clusters):
        cluster_mask = clusters == cluster_id
        cluster_items = [item_ids[i] for i, m in enumerate(cluster_mask) if m]

        # Find common failure patterns in this cluster
        common_failures = defaultdict(int)
        for item_id in cluster_items:
            item = analysis.item_stats[item_id]
            for metric_id, (passed, _) in item.metric_results.items():
                if not passed:
                    common_failures[metric_id] += 1

        # Sort by frequency
        sorted_failures = sorted(common_failures.items(), key=lambda x: -x[1])

        cluster_analysis[f"cluster_{cluster_id}"] = {
            "size": len(cluster_items),
            "items": cluster_items[:5],  # Sample
            "common_failures": sorted_failures[:3],  # Top 3
        }

    # PCA for visualization (if we have enough features)
    pca_result = None
    if len(metrics) >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_result = {
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "components": X_pca.tolist(),
            "clusters": clusters.tolist(),
        }

    return {
        "n_clusters": max_clusters,
        "cluster_analysis": cluster_analysis,
        "pca": pca_result,
    }


def generate_cluster_visualization(analysis: RunAnalysis, output_dir: Path) -> Optional[Path]:
    """Generate modern PCA cluster visualization of failed items."""
    if not HAS_SKLEARN or not HAS_VISUALIZATION:
        return None

    pattern_analysis = analyze_failure_patterns(analysis)
    if "error" in pattern_analysis:
        return None

    pca_data = pattern_analysis.get("pca")
    if not pca_data:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = _setup_modern_style()

    fig, ax = plt.subplots(figsize=(12, 10))

    components = np.array(pca_data["components"])
    clusters = np.array(pca_data["clusters"])

    # Modern cluster colors
    cluster_colors = [colors['accent'], colors['purple'], colors['cyan'],
                     colors['warning'], colors['success']]
    point_colors = [cluster_colors[c % len(cluster_colors)] for c in clusters]

    # Scatter with glow effect
    ax.scatter(components[:, 0], components[:, 1],
              c=point_colors, s=200, alpha=0.3, zorder=1)  # Glow
    scatter = ax.scatter(components[:, 0], components[:, 1],
                        c=point_colors, s=100, alpha=0.9, edgecolors='white',
                        linewidths=1.5, zorder=2)

    # Add cluster labels
    for i in range(len(components)):
        ax.annotate(f'{clusters[i]}', (components[i, 0], components[i, 1]),
                   fontsize=10, fontweight='bold', color=colors['text'],
                   ha='center', va='center', zorder=3)

    ax.set_xlabel(f'PC1 ({pca_data["explained_variance"][0]*100:.1f}% variance)',
                 fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({pca_data["explained_variance"][1]*100:.1f}% variance)',
                 fontsize=12, labelpad=10)
    ax.set_title('Failure Pattern Clusters', fontsize=18, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--')

    # Legend
    unique_clusters = sorted(set(clusters))
    legend_handles = [plt.scatter([], [], c=cluster_colors[c % len(cluster_colors)],
                                  s=100, label=f'Cluster {c}')
                     for c in unique_clusters]
    ax.legend(handles=legend_handles, loc='upper right',
             framealpha=0.8, facecolor=colors['bg_light'])

    plt.tight_layout()
    path = output_dir / 'failure_clusters.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close()

    return path


def generate_full_report(analysis: RunAnalysis, output_dir: Path) -> Dict[str, Any]:
    """Generate complete analysis report with all visualizations.

    Returns dict with paths to generated files and analysis results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "text_report": generate_text_report(analysis, verbose=True),
        "visualizations": {},
        "pattern_analysis": None,
    }

    # Generate visualizations
    if HAS_VISUALIZATION:
        viz_paths = generate_visualizations(analysis, output_dir)
        result["visualizations"] = {k: str(v) for k, v in viz_paths.items()}

        # Cluster visualization
        cluster_path = generate_cluster_visualization(analysis, output_dir)
        if cluster_path:
            result["visualizations"]["failure_clusters"] = str(cluster_path)

    # Pattern analysis
    if HAS_SKLEARN:
        result["pattern_analysis"] = analyze_failure_patterns(analysis)

    # Save text report
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(result["text_report"])
    result["report_path"] = str(report_path)

    # Save HTML with embedded images
    html_path = output_dir / "report.html"
    html = generate_html_report_with_images(analysis, result["visualizations"])
    with open(html_path, "w") as f:
        f.write(html)
    result["html_path"] = str(html_path)

    return result


def generate_html_report_with_images(analysis: RunAnalysis, viz_paths: Dict[str, str]) -> str:
    """Generate HTML report with embedded visualization images."""

    # Prepare data for charts
    metric_data = []
    for ms in sorted(analysis.metric_stats.values(), key=lambda m: -m.pass_rate):
        metric_data.append({
            "id": ms.metric_id,
            "type": ms.metric_type,
            "pass_rate": ms.pass_rate * 100,
            "avg_score": ms.avg_score,
            "min_score": ms.min_score,
            "max_score": ms.max_score,
            "std_dev": ms.std_dev,
            "count": ms.count,
            "passed": ms.passed,
            "failed": ms.failed,
        })

    # Encode images as base64
    image_tags = []
    for name, path in viz_paths.items():
        try:
            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            title = name.replace("_", " ").title()
            image_tags.append(f'''
                <div class="card">
                    <h2>{title}</h2>
                    <img src="data:image/png;base64,{img_data}" alt="{title}" style="max-width: 100%;">
                </div>
            ''')
        except Exception:
            pass

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Eval Run Analysis - {analysis.run_id[:8]}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-top: 0; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 20px; background: #f9fafb; border-radius: 8px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #666; font-size: 14px; margin-top: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        .pass {{ color: #059669; font-weight: bold; }}
        .warn {{ color: #f59e0b; font-weight: bold; }}
        .fail {{ color: #dc2626; font-weight: bold; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; }}
        img {{ border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Evaluation Analysis Report</h1>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-value">{analysis.total_items}</div>
                    <div class="stat-label">Total Items</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{analysis.total_metrics}</div>
                    <div class="stat-label">Metrics</div>
                </div>
                <div class="stat">
                    <div class="stat-value {"pass" if analysis.overall_pass_rate >= 0.8 else "warn" if analysis.overall_pass_rate >= 0.5 else "fail"}">{analysis.overall_pass_rate*100:.1f}%</div>
                    <div class="stat-label">Overall Pass Rate</div>
                </div>
                <div class="stat">
                    <div class="stat-value {"pass" if len(analysis.failed_items) == 0 else "fail"}">{len(analysis.failed_items)}</div>
                    <div class="stat-label">Failed Items</div>
                </div>
            </div>
            <p style="color: #666;">
                <strong>Run ID:</strong> {analysis.run_id}<br>
                <strong>Dataset:</strong> {analysis.dataset_name}<br>
                <strong>Created:</strong> {analysis.created_at}
            </p>
        </div>

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
                    {"".join(f'''<tr>
                        <td><strong>{m["id"]}</strong></td>
                        <td>{m["type"]}</td>
                        <td class="{"pass" if m["pass_rate"] >= 80 else "warn" if m["pass_rate"] >= 50 else "fail"}">{m["pass_rate"]:.1f}%</td>
                        <td>{m["avg_score"]:.3f}</td>
                        <td>{m["min_score"]:.3f}</td>
                        <td>{m["max_score"]:.3f}</td>
                        <td>{m["std_dev"]:.3f}</td>
                        <td class="pass">{m["passed"]}</td>
                        <td class="{"fail" if m["failed"] > 0 else ""}">{m["failed"]}</td>
                    </tr>''' for m in metric_data)}
                </tbody>
            </table>
        </div>

        <div class="viz-grid">
            {"".join(image_tags)}
        </div>
    </div>
</body>
</html>
"""
    return html
