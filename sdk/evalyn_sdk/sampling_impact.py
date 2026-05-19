"""Sampling impact analysis: estimate how sample size affects metric confidence intervals.

Pure Python, no external dependencies. Provides functions to compute
CI widths at various sample sizes, recommend minimum sample sizes for
a target precision, and format human-readable impact reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Z-score lookup (normal distribution critical values)
# ---------------------------------------------------------------------------

_Z_SCORES: dict[float, float] = {
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}


def _z_score(confidence: float) -> float:
    """Return z-score for the given confidence level.

    Falls back to 1.96 (95%) for unsupported levels.
    """
    return _Z_SCORES.get(confidence, 1.96)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ImpactEstimate:
    """Single estimate of CI width for a given sample size."""

    sample_size: int
    expected_ci_width: float
    precision_level: str  # "high", "medium", "low"
    estimated_cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "expected_ci_width": self.expected_ci_width,
            "precision_level": self.precision_level,
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImpactEstimate:
        return cls(
            sample_size=data["sample_size"],
            expected_ci_width=data["expected_ci_width"],
            precision_level=data["precision_level"],
            estimated_cost=data.get("estimated_cost", 0.0),
        )


@dataclass
class ImpactReport:
    """Full impact report with estimates across sample sizes."""

    estimates: list[ImpactEstimate] = field(default_factory=list)
    metric_std: float = 0.0
    recommended_size: int = 0
    target_ci_width: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimates": [e.as_dict() for e in self.estimates],
            "metric_std": self.metric_std,
            "recommended_size": self.recommended_size,
            "target_ci_width": self.target_ci_width,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImpactReport:
        return cls(
            estimates=[ImpactEstimate.from_dict(e) for e in data.get("estimates", [])],
            metric_std=data.get("metric_std", 0.0),
            recommended_size=data.get("recommended_size", 0),
            target_ci_width=data.get("target_ci_width", 0.05),
        )


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

_DEFAULT_SIZES: list[int] = [20, 50, 100, 200, 500, 1000]


def estimate_ci_width(
    std_dev: float,
    sample_size: int,
    confidence: float = 0.95,
) -> float:
    """Estimate confidence interval width: 2 * z * std / sqrt(n).

    Args:
        std_dev: Standard deviation of the metric scores.
        sample_size: Number of samples.
        confidence: Confidence level (default 0.95).

    Returns:
        Full CI width (not half-width).
    """
    if sample_size <= 0:
        return float("inf")
    z = _z_score(confidence)
    return 2.0 * z * std_dev / math.sqrt(sample_size)


def _precision_level(ci_width: float) -> str:
    """Classify CI width into a precision level."""
    if ci_width < 0.05:
        return "high"
    if ci_width < 0.1:
        return "medium"
    return "low"


def compute_impact_curve(
    scores: list[float],
    sizes: list[int],
    confidence: float = 0.95,
    cost_per_item: float = 0.0,
) -> list[ImpactEstimate]:
    """Compute CI width estimates for each sample size.

    Args:
        scores: Observed metric scores used to estimate std dev.
        sizes: List of sample sizes to evaluate.
        confidence: Confidence level.
        cost_per_item: Cost per evaluation item.

    Returns:
        List of ImpactEstimate, one per size.
    """
    std = _std_dev(scores)
    results: list[ImpactEstimate] = []
    for n in sizes:
        width = estimate_ci_width(std, n, confidence)
        results.append(
            ImpactEstimate(
                sample_size=n,
                expected_ci_width=width,
                precision_level=_precision_level(width),
                estimated_cost=cost_per_item * n,
            )
        )
    return results


def recommend_sample_size(
    std_dev: float,
    target_width: float = 0.05,
    confidence: float = 0.95,
) -> int:
    """Return minimum sample size such that CI width <= target_width.

    Uses the formula: n >= (2 * z * std / target_width)^2,
    rounded up to the nearest integer.

    Args:
        std_dev: Standard deviation of the metric.
        target_width: Desired maximum CI width.
        confidence: Confidence level.

    Returns:
        Minimum sample size (at least 1).
    """
    if target_width <= 0:
        return 1
    if std_dev <= 0:
        return 1
    z = _z_score(confidence)
    n = (2.0 * z * std_dev / target_width) ** 2
    return max(1, math.ceil(n))


def build_impact_report(
    scores: list[float],
    sizes: list[int] | None = None,
    target_width: float = 0.05,
    cost_per_item: float = 0.0,
) -> ImpactReport:
    """Build a full impact report.

    Args:
        scores: Observed metric scores.
        sizes: Sample sizes to evaluate (default [20, 50, 100, 200, 500, 1000]).
        target_width: Target CI width for recommendation.
        cost_per_item: Cost per evaluation item.

    Returns:
        ImpactReport with estimates and recommendation.
    """
    if sizes is None:
        sizes = list(_DEFAULT_SIZES)
    std = _std_dev(scores)
    estimates = compute_impact_curve(scores, sizes, cost_per_item=cost_per_item)
    rec = recommend_sample_size(std, target_width)
    return ImpactReport(
        estimates=estimates,
        metric_std=std,
        recommended_size=rec,
        target_ci_width=target_width,
    )


def format_impact_report(report: ImpactReport) -> str:
    """Format an ImpactReport as a human-readable table.

    Args:
        report: The report to format.

    Returns:
        Multi-line string with header, separator, and rows.
    """
    lines: list[str] = []
    lines.append(f"Metric std dev: {report.metric_std:.4f}")
    lines.append(f"Target CI width: {report.target_ci_width:.4f}")
    lines.append(f"Recommended sample size: {report.recommended_size}")
    lines.append("")

    # Table header
    header = f"{'Size':>8}  {'CI Width':>10}  {'Precision':>10}  {'Cost':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for est in report.estimates:
        lines.append(
            f"{est.sample_size:>8}  {est.expected_ci_width:>10.4f}  "
            f"{est.precision_level:>10}  {est.estimated_cost:>10.2f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _std_dev(scores: list[float]) -> float:
    """Population standard deviation of a list of scores."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return math.sqrt(variance)
