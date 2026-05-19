"""
Metric volatility index: measure historical stability of each metric across runs.

Provides functions to compute volatility indices, detect trends, and render
ASCII charts for quick visual assessment of metric stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class VolatilityMetric:
    """Volatility analysis for a single metric."""

    metric_id: str
    volatility_index: float  # 0-1: 0 = perfectly stable, 1 = maximally volatile
    trend: str = "stable"  # "stable" / "increasing" / "decreasing"
    run_count: int = 0
    max_swing: float = 0.0  # largest single-run change

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "volatility_index": self.volatility_index,
            "trend": self.trend,
            "run_count": self.run_count,
            "max_swing": self.max_swing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VolatilityMetric:
        return cls(
            metric_id=data["metric_id"],
            volatility_index=data["volatility_index"],
            trend=data.get("trend", "stable"),
            run_count=data.get("run_count", 0),
            max_swing=data.get("max_swing", 0.0),
        )


@dataclass
class VolatilityReport:
    """Aggregated volatility report across all metrics."""

    metrics: list[VolatilityMetric] = field(default_factory=list)
    most_volatile: str = ""
    least_volatile: str = ""
    avg_volatility: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.as_dict() for m in self.metrics],
            "most_volatile": self.most_volatile,
            "least_volatile": self.least_volatile,
            "avg_volatility": self.avg_volatility,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VolatilityReport:
        metrics = [VolatilityMetric.from_dict(m) for m in data.get("metrics", [])]
        return cls(
            metrics=metrics,
            most_volatile=data.get("most_volatile", ""),
            least_volatile=data.get("least_volatile", ""),
            avg_volatility=data.get("avg_volatility", 0.0),
        )

    def format_text(self) -> str:
        """Format the report as human-readable text."""
        lines: list[str] = []
        lines.append("METRIC VOLATILITY REPORT")
        lines.append("-" * 40)
        if not self.metrics:
            lines.append("No metrics to report.")
            return "\n".join(lines)

        lines.append(f"Average volatility: {self.avg_volatility:.3f}")
        lines.append(f"Most volatile:  {self.most_volatile}")
        lines.append(f"Least volatile: {self.least_volatile}")
        lines.append("")
        lines.append(f"{'Metric':<30} {'Index':>6} {'Trend':<12} {'Swing':>6}")
        lines.append("-" * 60)
        for m in sorted(self.metrics, key=lambda x: x.volatility_index, reverse=True):
            lines.append(
                f"{m.metric_id:<30} {m.volatility_index:>6.3f} {m.trend:<12} {m.max_swing:>6.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def compute_volatility_index(scores: list[float]) -> float:
    """Mean of absolute differences between consecutive scores, normalized by range.

    Returns 0.0 for constant or insufficient data, up to 1.0 for maximally volatile.
    """
    if len(scores) < 2:
        return 0.0
    diffs = [abs(scores[i + 1] - scores[i]) for i in range(len(scores) - 1)]
    mean_diff = sum(diffs) / len(diffs)
    score_range = max(scores) - min(scores)
    if score_range == 0.0:
        return 0.0
    return min(mean_diff / score_range, 1.0)


def detect_volatility_trend(scores: list[float]) -> str:
    """Detect whether volatility is increasing, decreasing, or stable.

    Compares volatility in the first half vs second half of the score sequence.
    """
    if len(scores) < 4:
        return "stable"
    diffs = [abs(scores[i + 1] - scores[i]) for i in range(len(scores) - 1)]
    mid = len(diffs) // 2
    first_half = sum(diffs[:mid]) / mid if mid > 0 else 0.0
    second_half = sum(diffs[mid:]) / (len(diffs) - mid) if (len(diffs) - mid) > 0 else 0.0
    ratio = second_half / first_half if first_half > 0 else 1.0
    if ratio > 1.5:
        return "increasing"
    if ratio < 0.67:
        return "decreasing"
    return "stable"


def compute_max_swing(scores: list[float]) -> float:
    """Largest absolute change between consecutive scores."""
    if len(scores) < 2:
        return 0.0
    return max(abs(scores[i + 1] - scores[i]) for i in range(len(scores) - 1))


def analyze_metric_volatility(metric_id: str, scores: list[float]) -> VolatilityMetric:
    """Full volatility analysis for a single metric."""
    return VolatilityMetric(
        metric_id=metric_id,
        volatility_index=compute_volatility_index(scores),
        trend=detect_volatility_trend(scores),
        run_count=len(scores),
        max_swing=compute_max_swing(scores),
    )


def analyze_all_volatility(metric_scores: dict[str, list[float]]) -> VolatilityReport:
    """Analyze volatility for all metrics and build a report."""
    if not metric_scores:
        return VolatilityReport()

    metrics = [
        analyze_metric_volatility(mid, scores)
        for mid, scores in metric_scores.items()
    ]

    avg_vol = sum(m.volatility_index for m in metrics) / len(metrics)
    most = max(metrics, key=lambda m: m.volatility_index)
    least = min(metrics, key=lambda m: m.volatility_index)

    return VolatilityReport(
        metrics=metrics,
        most_volatile=most.metric_id,
        least_volatile=least.metric_id,
        avg_volatility=avg_vol,
    )


def render_volatility_chart(report: VolatilityReport, width: int = 60) -> str:
    """Render an ASCII bar chart of volatility per metric."""
    if not report.metrics:
        return "No metrics to chart."

    lines: list[str] = []
    lines.append("Volatility Chart")
    lines.append("=" * width)

    # Find longest metric name for alignment
    max_name = max(len(m.metric_id) for m in report.metrics)
    bar_width = width - max_name - 10  # leave room for label and value

    sorted_metrics = sorted(
        report.metrics, key=lambda m: m.volatility_index, reverse=True
    )
    for m in sorted_metrics:
        filled = int(m.volatility_index * bar_width)
        bar = "#" * filled + "." * (bar_width - filled)
        lines.append(f"{m.metric_id:<{max_name}} |{bar}| {m.volatility_index:.3f}")

    return "\n".join(lines)
