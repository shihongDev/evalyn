"""Trend forecasting for metric time series.

Predicts future metric values using linear regression and exponential
smoothing. Generates confidence bands and threshold alerts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ForecastPoint:
    """A single forecasted value with confidence band."""

    step: int  # 1-indexed: step=1 is next run after observed data
    value: float
    lower: float  # lower bound of confidence band
    upper: float  # upper bound of confidence band

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "value": round(self.value, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ForecastPoint:
        return cls(
            step=data["step"],
            value=data["value"],
            lower=data["lower"],
            upper=data["upper"],
        )


@dataclass
class MetricForecast:
    """Forecast for a single metric."""

    metric_id: str
    method: str  # "linear" or "exponential_smoothing"
    observed: List[float] = field(default_factory=list)
    forecasted: List[ForecastPoint] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "method": self.method,
            "num_observed": len(self.observed),
            "num_forecasted": len(self.forecasted),
            "forecasted": [f.as_dict() for f in self.forecasted],
            "alerts": list(self.alerts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricForecast:
        forecasted = [
            ForecastPoint.from_dict(f) for f in data.get("forecasted", [])
        ]
        return cls(
            metric_id=data["metric_id"],
            method=data.get("method", "linear"),
            alerts=data.get("alerts", []),
            forecasted=forecasted,
        )

    def format_text(self) -> str:
        lines = [f"{self.metric_id} ({self.method}):"]
        for fp in self.forecasted:
            lines.append(
                f"  +{fp.step}: {fp.value:.4f} [{fp.lower:.4f}, {fp.upper:.4f}]"
            )
        for alert in self.alerts:
            lines.append(f"  ALERT: {alert}")
        return "\n".join(lines)


@dataclass
class ForecastReport:
    """Forecast results for all metrics."""

    results: List[MetricForecast] = field(default_factory=list)

    @property
    def alerts(self) -> List[str]:
        out = []
        for r in self.results:
            out.extend(r.alerts)
        return out

    @property
    def metrics_at_risk(self) -> List[str]:
        return [r.metric_id for r in self.results if r.alerts]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alerts": self.alerts,
            "metrics_at_risk": self.metrics_at_risk,
            "results": [r.as_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ForecastReport:
        results = [MetricForecast.from_dict(r) for r in data.get("results", [])]
        return cls(results=results)

    def format_text(self) -> str:
        lines = ["Trend Forecast:"]
        for r in self.results:
            for line in r.format_text().splitlines():
                lines.append(f"  {line}")
        if self.metrics_at_risk:
            lines.append(f"\n  At risk: {', '.join(self.metrics_at_risk)}")
        return "\n".join(lines)


def forecast_metric(
    metric_id: str,
    values: List[float],
    horizon: int = 3,
    method: str = "linear",
    confidence: float = 0.9,
    threshold: Optional[float] = None,
) -> MetricForecast:
    """Forecast future values for a metric time series.

    Args:
        metric_id: Metric identifier.
        values: Historical pass rates (oldest to newest).
        horizon: Number of future steps to predict.
        method: "linear" or "exponential_smoothing".
        confidence: Confidence level for bands (0 to 1).
        threshold: If set, alert when forecast drops below this value.

    Returns:
        MetricForecast with predicted values and alerts.
    """
    if len(values) < 3:
        return MetricForecast(
            metric_id=metric_id,
            method=method,
            observed=list(values),
            alerts=["insufficient data for forecasting (need >= 3 points)"],
        )

    if method == "exponential_smoothing":
        points = _exponential_smoothing_forecast(values, horizon, confidence)
    else:
        points = _linear_forecast(values, horizon, confidence)

    alerts = []
    if threshold is not None:
        for fp in points:
            if fp.value < threshold:
                alerts.append(
                    f"{metric_id} forecast to drop below {threshold:.2f} "
                    f"at step +{fp.step} (predicted: {fp.value:.4f})"
                )
                break  # one alert per metric

    return MetricForecast(
        metric_id=metric_id,
        method=method,
        observed=list(values),
        forecasted=points,
        alerts=alerts,
    )


def forecast_all(
    series_by_metric: Dict[str, List[float]],
    horizon: int = 3,
    method: str = "linear",
    confidence: float = 0.9,
    threshold: Optional[float] = None,
) -> ForecastReport:
    """Forecast all metric time series.

    Args:
        series_by_metric: Dict of metric_id -> values over time.
        horizon: Steps to forecast.
        method: Forecasting method.
        confidence: Confidence level for bands (0 to 1).
        threshold: Alert threshold.

    Returns:
        ForecastReport with per-metric forecasts.
    """
    results = []
    for mid, values in sorted(series_by_metric.items()):
        results.append(forecast_metric(
            mid, values, horizon, method,
            confidence=confidence, threshold=threshold,
        ))
    return ForecastReport(results=results)


# ---- Internal helpers ----

def _linear_slope_intercept(values: List[float]) -> tuple:
    """Return (slope, intercept) of linear fit."""
    n = len(values)
    if n < 2:
        return 0.0, (values[0] if values else 0.0)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0.0
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _residual_std(values: List[float], slope: float, intercept: float) -> float:
    """Standard error of residuals around the regression line."""
    n = len(values)
    if n < 3:
        return 0.0
    ss = sum((v - (intercept + slope * i)) ** 2 for i, v in enumerate(values))
    return math.sqrt(ss / (n - 2))


def _linear_forecast(
    values: List[float],
    horizon: int,
    confidence: float,
) -> List[ForecastPoint]:
    """Linear regression forecast with prediction intervals."""
    n = len(values)
    slope, intercept = _linear_slope_intercept(values)
    se = _residual_std(values, slope, intercept)

    # Use z-score approximation for confidence bands
    z = _z_score(confidence)
    x_mean = (n - 1) / 2
    ss_x = sum((i - x_mean) ** 2 for i in range(n))

    points = []
    for step in range(1, horizon + 1):
        x = n - 1 + step
        y = intercept + slope * x

        # Prediction interval width grows with distance from mean
        if ss_x > 0:
            margin = z * se * math.sqrt(1 + 1 / n + (x - x_mean) ** 2 / ss_x)
        else:
            margin = z * se

        points.append(ForecastPoint(
            step=step,
            value=y,
            lower=y - margin,
            upper=y + margin,
        ))
    return points


def _exponential_smoothing_forecast(
    values: List[float],
    horizon: int,
    confidence: float,
    alpha: float = 0.3,
) -> List[ForecastPoint]:
    """Simple exponential smoothing forecast.

    Uses Holt's linear method (double exponential smoothing) with
    fixed smoothing parameters.
    """
    n = len(values)
    beta = 0.1  # trend smoothing

    # Initialize
    level = values[0]
    trend = (values[-1] - values[0]) / (n - 1) if n > 1 else 0.0

    # Fit
    residuals = []
    for v in values:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        residuals.append(v - (prev_level + trend))

    # Residual std for confidence bands
    if len(residuals) >= 2:
        r_mean = sum(residuals) / len(residuals)
        se = math.sqrt(sum((r - r_mean) ** 2 for r in residuals) / (len(residuals) - 1))
    else:
        se = 0.0

    z = _z_score(confidence)

    points = []
    for step in range(1, horizon + 1):
        y = level + trend * step
        # Confidence band widens with forecast horizon
        margin = z * se * math.sqrt(step)
        points.append(ForecastPoint(
            step=step,
            value=y,
            lower=y - margin,
            upper=y + margin,
        ))
    return points


def _z_score(confidence: float) -> float:
    """Approximate z-score for common confidence levels."""
    # Common lookup to avoid scipy dependency
    table = {
        0.80: 1.282,
        0.85: 1.440,
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    # Find closest match
    closest = min(table, key=lambda k: abs(k - confidence))
    return table[closest]
