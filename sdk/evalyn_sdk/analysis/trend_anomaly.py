"""
Trend anomaly detection: alert on unusual metric patterns.

Provides z-score based anomaly detection, trend break detection,
sudden drop detection, and ASCII chart rendering for metric time series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnomalyPoint:
    """A single data point with anomaly scoring."""

    index: int
    value: float
    expected: float
    z_score: float
    is_anomaly: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "value": self.value,
            "expected": self.expected,
            "z_score": self.z_score,
            "is_anomaly": self.is_anomaly,
        }


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""

    z_threshold: float = 2.0
    min_data_points: int = 5
    window_size: int = 5

    def as_dict(self) -> Dict[str, Any]:
        return {
            "z_threshold": self.z_threshold,
            "min_data_points": self.min_data_points,
            "window_size": self.window_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyConfig":
        return cls(
            z_threshold=data.get("z_threshold", 2.0),
            min_data_points=data.get("min_data_points", 5),
            window_size=data.get("window_size", 5),
        )


@dataclass
class AnomalyReport:
    """Full anomaly detection report for a metric."""

    anomalies: List[AnomalyPoint] = field(default_factory=list)
    total_points: int = 0
    anomaly_count: int = 0
    anomaly_rate: float = 0.0
    metric_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anomalies": [a.as_dict() for a in self.anomalies],
            "total_points": self.total_points,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": self.anomaly_rate,
            "metric_id": self.metric_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyReport":
        return cls(
            anomalies=[
                AnomalyPoint(**a) for a in data.get("anomalies", [])
            ],
            total_points=data.get("total_points", 0),
            anomaly_count=data.get("anomaly_count", 0),
            anomaly_rate=data.get("anomaly_rate", 0.0),
            metric_id=data.get("metric_id", ""),
        )

    def format_text(self) -> str:
        """Format as human-readable text."""
        lines = []
        lines.append(f"Anomaly Report: {self.metric_id}")
        lines.append(f"  Total points: {self.total_points}")
        lines.append(f"  Anomalies: {self.anomaly_count}")
        lines.append(f"  Anomaly rate: {self.anomaly_rate:.1%}")
        if self.anomalies:
            lines.append("  Details:")
            for a in self.anomalies:
                flag = "*" if a.is_anomaly else " "
                lines.append(
                    f"    [{flag}] idx={a.index} value={a.value:.3f}"
                    f" expected={a.expected:.3f} z={a.z_score:.2f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_z_scores(values: List[float], window_size: int = 5) -> List[float]:
    """Compute z-score for each value relative to the moving window average.

    Requires at least 2 points in the window to compute a meaningful z-score.
    Points with insufficient window data get z=0. When the window has zero
    standard deviation but the value differs from the mean, returns a large
    z-score (+/-100) to flag it as anomalous.
    """
    n = len(values)
    if n < 2:
        return [0.0] * n

    z_scores: List[float] = []
    for i in range(n):
        # Window: up to window_size points before index i
        start = max(0, i - window_size)
        window = values[start:i]
        if len(window) < 2:
            z_scores.append(0.0)
            continue
        mean = sum(window) / len(window)
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        std = math.sqrt(variance)
        if std == 0:
            diff = values[i] - mean
            if diff == 0:
                z_scores.append(0.0)
            else:
                # All window values identical but current differs - strong anomaly
                z_scores.append(100.0 if diff > 0 else -100.0)
        else:
            z_scores.append((values[i] - mean) / std)
    return z_scores


def detect_anomalies(
    values: List[float], config: Optional[AnomalyConfig] = None
) -> List[AnomalyPoint]:
    """Find values where |z_score| > z_threshold.

    Returns a list of AnomalyPoint for every data point, with is_anomaly
    set to True for those exceeding the threshold. Returns empty list if
    there are fewer data points than config.min_data_points.
    """
    if config is None:
        config = AnomalyConfig()

    if len(values) < config.min_data_points:
        return []

    z_scores = compute_z_scores(values, window_size=config.window_size)
    points: List[AnomalyPoint] = []
    for i, (val, z) in enumerate(zip(values, z_scores)):
        # Compute expected (window mean)
        start = max(0, i - config.window_size)
        window = values[start:i]
        expected = sum(window) / len(window) if window else val
        points.append(AnomalyPoint(
            index=i,
            value=val,
            expected=expected,
            z_score=z,
            is_anomaly=abs(z) > config.z_threshold,
        ))
    return points


def detect_trend_break(values: List[float], window_size: int = 5) -> Optional[int]:
    """Find the index where the trend reverses (slope sign change).

    Computes a simple moving average slope over the given window. Returns
    the index where the slope sign changes, or None if no break is found.
    Requires at least window_size + 1 data points.
    """
    n = len(values)
    if n < window_size + 1:
        return None

    # Compute slopes between consecutive moving averages
    slopes: List[float] = []
    for i in range(n - window_size + 1):
        window = values[i : i + window_size]
        avg = sum(window) / len(window)
        slopes.append(avg)

    # Find sign changes in the slope differences
    diffs = [slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)]
    for i in range(len(diffs) - 1):
        if diffs[i] != 0 and diffs[i + 1] != 0:
            if (diffs[i] > 0 and diffs[i + 1] < 0) or (diffs[i] < 0 and diffs[i + 1] > 0):
                # The break point is at the index corresponding to the peak/trough
                return i + window_size
    return None


def detect_sudden_drop(
    values: List[float], threshold_pct: float = 0.2
) -> List[int]:
    """Find indices where value drops > threshold% from previous.

    A drop is defined as (previous - current) / |previous| > threshold_pct,
    where previous is nonzero. Returns list of indices where drops occur.
    """
    drops: List[int] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev == 0:
            continue
        drop_pct = (prev - values[i]) / abs(prev)
        if drop_pct > threshold_pct:
            drops.append(i)
    return drops


def build_anomaly_report(
    metric_id: str,
    values: List[float],
    config: Optional[AnomalyConfig] = None,
) -> AnomalyReport:
    """Full anomaly analysis for a metric time series."""
    if config is None:
        config = AnomalyConfig()

    points = detect_anomalies(values, config)
    anomaly_count = sum(1 for p in points if p.is_anomaly)
    total = len(values)
    rate = anomaly_count / total if total > 0 else 0.0

    return AnomalyReport(
        anomalies=points,
        total_points=total,
        anomaly_count=anomaly_count,
        anomaly_rate=rate,
        metric_id=metric_id,
    )


def render_anomaly_chart(
    values: List[float],
    anomaly_indices: List[int],
    width: int = 60,
) -> str:
    """ASCII chart highlighting anomalies with markers.

    Normal values are shown as '-', anomalies as '!'. The chart is
    scaled to fit within the given width.
    """
    if not values:
        return ""

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val
    if val_range == 0:
        val_range = 1.0

    anomaly_set = set(anomaly_indices)
    lines: List[str] = []

    for i, v in enumerate(values):
        pos = int((v - min_val) / val_range * (width - 1))
        marker = "!" if i in anomaly_set else "-"
        bar = "." * pos + marker
        label = f"{i:3d} |{bar}"
        lines.append(label)

    return "\n".join(lines)
