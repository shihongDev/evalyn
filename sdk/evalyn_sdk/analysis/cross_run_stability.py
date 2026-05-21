"""
Cross-run stability analysis: measure how stable metric scores are across repeated runs.

Helps identify flaky metrics whose scores vary significantly between runs,
and quantify overall evaluation reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class StabilityMetric:
    """Stability statistics for a single metric across runs."""

    metric_id: str
    mean_score: float = 0.0
    std_dev: float = 0.0
    coefficient_of_variation: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    num_runs: int = 0
    stability_label: str = "stable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "mean_score": self.mean_score,
            "std_dev": self.std_dev,
            "coefficient_of_variation": self.coefficient_of_variation,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "num_runs": self.num_runs,
            "stability_label": self.stability_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StabilityMetric:
        return cls(
            metric_id=data["metric_id"],
            mean_score=data.get("mean_score", 0.0),
            std_dev=data.get("std_dev", 0.0),
            coefficient_of_variation=data.get("coefficient_of_variation", 0.0),
            min_score=data.get("min_score", 0.0),
            max_score=data.get("max_score", 0.0),
            num_runs=data.get("num_runs", 0),
            stability_label=data.get("stability_label", "stable"),
        )


@dataclass
class StabilityReport:
    """Full cross-run stability report."""

    metrics: list[StabilityMetric] = field(default_factory=list)
    overall_stability: str = "stable"
    most_stable: str = ""
    least_stable: str = ""
    avg_cv: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.as_dict() for m in self.metrics],
            "overall_stability": self.overall_stability,
            "most_stable": self.most_stable,
            "least_stable": self.least_stable,
            "avg_cv": self.avg_cv,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StabilityReport:
        metrics = [StabilityMetric.from_dict(m) for m in data.get("metrics", [])]
        return cls(
            metrics=metrics,
            overall_stability=data.get("overall_stability", "stable"),
            most_stable=data.get("most_stable", ""),
            least_stable=data.get("least_stable", ""),
            avg_cv=data.get("avg_cv", 0.0),
        )

    def format_text(self) -> str:
        """Render a human-readable text summary."""
        lines = [
            "CROSS-RUN STABILITY REPORT",
            "=" * 50,
            f"Overall stability: {self.overall_stability}",
            f"Most stable: {self.most_stable}",
            f"Least stable: {self.least_stable}",
            f"Average CV: {self.avg_cv:.4f}",
            "",
            f"  {'Metric':<25} {'Mean':>7} {'StdDev':>8} {'CV':>7} {'Label':>10}",
            f"  {'-' * 25} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 10}",
        ]
        for m in self.metrics:
            lines.append(
                f"  {m.metric_id:<25} {m.mean_score:>7.4f} {m.std_dev:>8.4f} "
                f"{m.coefficient_of_variation:>7.4f} {m.stability_label:>10}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stability label thresholds
# ---------------------------------------------------------------------------

CV_STABLE_THRESHOLD = 0.05
CV_MODERATE_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _label_from_cv(cv: float) -> str:
    """Classify stability from coefficient of variation."""
    if cv < CV_STABLE_THRESHOLD:
        return "stable"
    if cv < CV_MODERATE_THRESHOLD:
        return "moderate"
    return "unstable"


def compute_stability(metric_id: str, scores: list[float]) -> StabilityMetric:
    """Compute stability statistics for a single metric across runs.

    Args:
        metric_id: Metric identifier.
        scores: List of metric scores (one per run).

    Returns:
        StabilityMetric with mean, std_dev, CV, min, max, and label.
    """
    n = len(scores)
    if n == 0:
        return StabilityMetric(metric_id=metric_id)

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance)
    cv = (std / mean) if mean != 0 else 0.0

    return StabilityMetric(
        metric_id=metric_id,
        mean_score=mean,
        std_dev=std,
        coefficient_of_variation=cv,
        min_score=min(scores),
        max_score=max(scores),
        num_runs=n,
        stability_label=_label_from_cv(cv),
    )


def analyze_stability(
    metric_scores: dict[str, list[float]],
) -> StabilityReport:
    """Analyze stability across all metrics.

    Args:
        metric_scores: {metric_id: [score_per_run, ...]}.

    Returns:
        StabilityReport summarizing all metrics.
    """
    metrics: list[StabilityMetric] = []
    for mid in sorted(metric_scores.keys()):
        metrics.append(compute_stability(mid, metric_scores[mid]))

    if not metrics:
        return StabilityReport()

    avg_cv = sum(m.coefficient_of_variation for m in metrics) / len(metrics)
    overall_label = _label_from_cv(avg_cv)

    # most stable = lowest CV, least stable = highest CV
    sorted_by_cv = sorted(metrics, key=lambda m: m.coefficient_of_variation)
    most_stable = sorted_by_cv[0].metric_id
    least_stable = sorted_by_cv[-1].metric_id

    return StabilityReport(
        metrics=metrics,
        overall_stability=overall_label,
        most_stable=most_stable,
        least_stable=least_stable,
        avg_cv=avg_cv,
    )


def detect_flaky_metrics(report: StabilityReport) -> list[str]:
    """Return metric IDs labeled 'unstable'."""
    return [m.metric_id for m in report.metrics if m.stability_label == "unstable"]


def compute_reproducibility_score(report: StabilityReport) -> float:
    """Compute overall reproducibility: 1 - avg_cv, clamped to [0, 1].

    Higher means more reproducible.
    """
    score = 1.0 - report.avg_cv
    return max(0.0, min(1.0, score))


def render_stability_chart(report: StabilityReport, width: int = 60) -> str:
    """ASCII chart of coefficient of variation per metric."""
    if not report.metrics:
        return "No metrics to chart."

    max_cv = max(m.coefficient_of_variation for m in report.metrics)
    if max_cv == 0:
        max_cv = 1.0

    lines: list[str] = ["STABILITY CHART (CV)", "=" * width]
    for m in report.metrics:
        bar_len = int((m.coefficient_of_variation / max_cv) * (width - 30))
        bar_len = max(bar_len, 0)
        bar = "#" * bar_len
        lines.append(
            f"  {m.metric_id:<20} |{bar} {m.coefficient_of_variation:.4f} [{m.stability_label}]"
        )
    return "\n".join(lines)


def suggest_stability_improvements(report: StabilityReport) -> list[str]:
    """Return suggestions for unstable metrics."""
    suggestions: list[str] = []
    for m in report.metrics:
        if m.stability_label == "unstable":
            suggestions.append(
                f"{m.metric_id}: CV={m.coefficient_of_variation:.4f} - "
                f"consider fixing non-deterministic inputs or increasing sample size"
            )
        elif m.stability_label == "moderate":
            suggestions.append(
                f"{m.metric_id}: CV={m.coefficient_of_variation:.4f} - "
                f"moderate variability, monitor across future runs"
            )
    return suggestions
