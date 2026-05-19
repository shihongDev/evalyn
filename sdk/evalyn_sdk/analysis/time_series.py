"""Time series decomposition for metric trends.

Separates metric pass rates across runs into trend, seasonality, and
noise components. Uses simple moving average decomposition (no scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DecompositionResult:
    """Time series decomposition for a single metric."""

    metric_id: str
    values: list[float] = field(default_factory=list)
    trend: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    trend_slope: float = 0.0
    noise_level: float = 0.0  # std dev of residuals

    def as_dict(self) -> dict[str, Any]:
        """Summary dict (excludes raw trend/residual arrays)."""
        return {
            "metric_id": self.metric_id,
            "num_points": len(self.values),
            "trend_direction": self.trend_direction,
            "trend_slope": round(self.trend_slope, 6),
            "noise_level": round(self.noise_level, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecompositionResult:
        """Reconstruct from summary dict (trend/residual not restored)."""
        return cls(
            metric_id=data["metric_id"],
            trend_direction=data.get("trend_direction", "stable"),
            trend_slope=data.get("trend_slope", 0.0),
            noise_level=data.get("noise_level", 0.0),
        )

    def format_text(self) -> str:
        return (
            f"{self.metric_id}: {self.trend_direction} "
            f"(slope={self.trend_slope:+.4f}, noise={self.noise_level:.3f}, "
            f"n={len(self.values)})"
        )


@dataclass
class TimeSeriesReport:
    """Report of time series analysis across all metrics."""

    results: list[DecompositionResult] = field(default_factory=list)

    @property
    def improving_metrics(self) -> list[str]:
        return [r.metric_id for r in self.results if r.trend_direction == "improving"]

    @property
    def declining_metrics(self) -> list[str]:
        return [r.metric_id for r in self.results if r.trend_direction == "declining"]

    @property
    def noisy_metrics(self) -> list[str]:
        """Metrics with high noise (residual std > 0.1)."""
        return [r.metric_id for r in self.results if r.noise_level > 0.1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "improving": self.improving_metrics,
            "declining": self.declining_metrics,
            "noisy": self.noisy_metrics,
            "results": [r.as_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeSeriesReport:
        results = [
            DecompositionResult.from_dict(r) for r in data.get("results", [])
        ]
        return cls(results=results)

    def format_text(self) -> str:
        lines = ["Time Series Analysis:"]
        for r in sorted(self.results, key=lambda x: x.trend_slope):
            lines.append(f"  {r.format_text()}")
        if self.declining_metrics:
            lines.append(f"\n  Declining: {', '.join(self.declining_metrics)}")
        if self.noisy_metrics:
            lines.append(f"  Noisy: {', '.join(self.noisy_metrics)}")
        return "\n".join(lines)


def decompose_metric(
    metric_id: str,
    values: list[float],
    window: int = 3,
    slope_threshold: float = 0.01,
) -> DecompositionResult:
    """Decompose a metric time series into trend and residual.

    Uses simple moving average for trend extraction.

    Args:
        metric_id: Metric identifier.
        values: Pass rates over time (oldest to newest).
        window: Moving average window size.
        slope_threshold: Min |slope| to classify as improving/declining.

    Returns:
        DecompositionResult with trend, residual, and classification.
    """
    if len(values) < 3:
        return DecompositionResult(
            metric_id=metric_id,
            values=list(values),
            trend_direction="insufficient_data",
        )

    # Compute moving average trend
    trend = _moving_average(values, window)

    # Residuals = original - trend
    residual = [v - t for v, t in zip(values, trend)]

    # Trend slope (linear regression on trend values)
    slope = _linear_slope(trend)

    # Noise level = std dev of residuals
    noise = _std(residual)

    # Classify trend
    if slope > slope_threshold:
        direction = "improving"
    elif slope < -slope_threshold:
        direction = "declining"
    else:
        direction = "stable"

    return DecompositionResult(
        metric_id=metric_id,
        values=list(values),
        trend=trend,
        residual=residual,
        trend_direction=direction,
        trend_slope=slope,
        noise_level=noise,
    )


def decompose_all(
    series_by_metric: dict[str, list[float]],
    window: int = 3,
) -> TimeSeriesReport:
    """Decompose all metric time series.

    Args:
        series_by_metric: Dict of metric_id -> list of values over time.
        window: Moving average window size.

    Returns:
        TimeSeriesReport with per-metric decomposition.
    """
    results = []
    for mid, values in sorted(series_by_metric.items()):
        results.append(decompose_metric(mid, values, window))
    return TimeSeriesReport(results=results)


def _moving_average(values: list[float], window: int) -> list[float]:
    """Compute centered moving average. Window must be odd."""
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    n = len(values)
    if n <= window:
        avg = sum(values) / n
        return [avg] * n

    result = []
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def _linear_slope(values: list[float]) -> float:
    """Compute slope of a linear fit through values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def _std(values: list[float]) -> float:
    """Population std dev (N denominator). Noise thresholds are calibrated to this."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
