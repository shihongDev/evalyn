"""Bootstrap resampling for confidence interval estimation.

Generate bootstrap samples and compute percentile-based confidence intervals
for evaluation metrics. Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap resampling."""

    n_iterations: int = 1000
    confidence_level: float = 0.95
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BootstrapConfig:
        return cls(
            n_iterations=data.get("n_iterations", 1000),
            confidence_level=data.get("confidence_level", 0.95),
            seed=data.get("seed", 42),
        )


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a single metric."""

    metric_id: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "std_error": self.std_error,
            "n_iterations": self.n_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BootstrapCI:
        return cls(
            metric_id=data["metric_id"],
            point_estimate=data["point_estimate"],
            ci_lower=data["ci_lower"],
            ci_upper=data["ci_upper"],
            std_error=data["std_error"],
            n_iterations=data["n_iterations"],
        )


@dataclass
class BootstrapReport:
    """Report containing confidence intervals for multiple metrics."""

    intervals: list[BootstrapCI] = field(default_factory=list)
    confidence_level: float = 0.95
    n_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "intervals": [ci.as_dict() for ci in self.intervals],
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BootstrapReport:
        return cls(
            intervals=[BootstrapCI.from_dict(ci) for ci in data.get("intervals", [])],
            confidence_level=data.get("confidence_level", 0.95),
            n_samples=data.get("n_samples", 0),
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-1) of pre-sorted values with linear interpolation.

    For p outside [0, 1], clamps to the boundary values.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    p = max(0.0, min(1.0, p))
    idx = p * (n - 1)
    lo = math.floor(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def resample(scores: list[float], rng: random.Random) -> list[float]:
    """Draw len(scores) values with replacement from scores."""
    n = len(scores)
    if n == 0:
        return []
    return [scores[rng.randint(0, n - 1)] for _ in range(n)]


def compute_bootstrap_distribution(
    scores: list[float],
    statistic_fn: Callable[[list[float]], float],
    config: BootstrapConfig,
) -> list[float]:
    """Run n_iterations resamples, compute statistic on each, return distribution."""
    rng = random.Random(config.seed)
    distribution: list[float] = []
    for _ in range(config.n_iterations):
        sample = resample(scores, rng)
        distribution.append(statistic_fn(sample))
    return distribution


def compute_confidence_interval(
    distribution: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute percentile-based confidence interval from a bootstrap distribution.

    Returns (lower, upper) bounds.
    """
    if not distribution:
        return (0.0, 0.0)
    sorted_dist = sorted(distribution)
    alpha = 1.0 - confidence
    lower = percentile(sorted_dist, alpha / 2.0)
    upper = percentile(sorted_dist, 1.0 - alpha / 2.0)
    return (lower, upper)


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std_error(distribution: list[float]) -> float:
    """Compute standard error (standard deviation of the bootstrap distribution)."""
    if len(distribution) < 2:
        return 0.0
    m = _mean(distribution)
    variance = sum((x - m) ** 2 for x in distribution) / (len(distribution) - 1)
    return math.sqrt(variance)


def bootstrap_metric(
    metric_id: str,
    scores: list[float],
    config: BootstrapConfig,
) -> BootstrapCI:
    """Full bootstrap CI computation for one metric using mean as the statistic."""
    point_estimate = _mean(scores)
    distribution = compute_bootstrap_distribution(scores, _mean, config)
    ci_lower, ci_upper = compute_confidence_interval(distribution, config.confidence_level)
    std_error = _std_error(distribution)
    return BootstrapCI(
        metric_id=metric_id,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_iterations=config.n_iterations,
    )


def bootstrap_all_metrics(
    metrics: dict[str, list[float]],
    config: BootstrapConfig | None = None,
) -> BootstrapReport:
    """Compute bootstrap CIs for all metrics in the dictionary."""
    if config is None:
        config = BootstrapConfig()
    intervals: list[BootstrapCI] = []
    n_samples = 0
    for metric_id in sorted(metrics.keys()):
        scores = metrics[metric_id]
        if scores:
            n_samples = max(n_samples, len(scores))
        ci = bootstrap_metric(metric_id, scores, config)
        intervals.append(ci)
    return BootstrapReport(
        intervals=intervals,
        confidence_level=config.confidence_level,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_bootstrap_report(report: BootstrapReport) -> str:
    """Format a bootstrap report as a human-readable table."""
    lines: list[str] = []
    lines.append(f"Bootstrap Report ({report.confidence_level:.0%} CI, n={report.n_samples})")
    lines.append("-" * 72)
    header = f"{'Metric':<20} {'Estimate':>10} {'CI Lower':>10} {'CI Upper':>10} {'Std Err':>10}"
    lines.append(header)
    lines.append("-" * 72)
    for ci in report.intervals:
        row = (
            f"{ci.metric_id:<20} "
            f"{ci.point_estimate:>10.4f} "
            f"{ci.ci_lower:>10.4f} "
            f"{ci.ci_upper:>10.4f} "
            f"{ci.std_error:>10.4f}"
        )
        lines.append(row)
    lines.append("-" * 72)
    return "\n".join(lines)
