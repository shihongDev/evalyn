"""Metric score curve fitting: fit parametric distributions to scores.

Analyzes the shape of metric score distributions to detect anomalies
and characterize metric behavior (bimodal, skewed, normal, etc).
No external dependencies - uses basic statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class DistributionFit:
    """Result of fitting a distribution to metric scores."""

    metric_id: str
    sample_count: int
    mean: float
    std_dev: float
    median: float
    skewness: float       # 0 = symmetric, >0 = right-tailed, <0 = left-tailed
    kurtosis: float       # 3 = normal, >3 = heavy-tailed, <3 = light-tailed
    distribution_type: str  # "normal", "bimodal", "uniform", "skewed_left", "skewed_right"
    is_bimodal: bool = False
    mode_1: float | None = None
    mode_2: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "metric_id": self.metric_id,
            "sample_count": self.sample_count,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "median": round(self.median, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "distribution_type": self.distribution_type,
            "is_bimodal": self.is_bimodal,
        }
        if self.mode_1 is not None:
            d["mode_1"] = round(self.mode_1, 4)
        if self.mode_2 is not None:
            d["mode_2"] = round(self.mode_2, 4)
        return d

    def format_text(self) -> str:
        lines = [
            f"{self.metric_id}: {self.distribution_type} "
            f"(mean={self.mean:.3f}, std={self.std_dev:.3f}, "
            f"skew={self.skewness:.2f}, n={self.sample_count})"
        ]
        if self.is_bimodal:
            lines.append(f"  Bimodal: modes at {self.mode_1:.2f} and {self.mode_2:.2f}")
        return "\n".join(lines)


def fit_distribution(
    metric_id: str,
    scores: list[float],
) -> DistributionFit:
    """Fit a distribution to a metric's scores.

    Classifies the distribution shape:
    - normal: symmetric, bell-shaped (|skew| < 0.5, 2 < kurtosis < 5)
    - bimodal: two peaks (detected via histogram gaps)
    - skewed_left: left-tailed (skew < -0.5)
    - skewed_right: right-tailed (skew > 0.5)
    - uniform: low kurtosis, flat distribution
    - degenerate: all same value

    Args:
        metric_id: Metric identifier.
        scores: List of score values.

    Returns:
        DistributionFit with shape classification.
    """
    if not scores:
        return DistributionFit(
            metric_id=metric_id, sample_count=0,
            mean=0.0, std_dev=0.0, median=0.0,
            skewness=0.0, kurtosis=0.0,
            distribution_type="empty",
        )

    n = len(scores)
    mean = sum(scores) / n
    sorted_scores = sorted(scores)
    median = sorted_scores[n // 2] if n % 2 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    if n < 3:
        return DistributionFit(
            metric_id=metric_id, sample_count=n,
            mean=mean, std_dev=0.0, median=median,
            skewness=0.0, kurtosis=0.0,
            distribution_type="insufficient_data",
        )

    # Variance and std dev
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance)

    if std == 0:
        return DistributionFit(
            metric_id=metric_id, sample_count=n,
            mean=mean, std_dev=0.0, median=median,
            skewness=0.0, kurtosis=0.0,
            distribution_type="degenerate",
        )

    # Skewness (Fisher's)
    m3 = sum((s - mean) ** 3 for s in scores) / n
    skewness = m3 / (std ** 3) if std > 0 else 0.0

    # Kurtosis (excess, Fisher's)
    m4 = sum((s - mean) ** 4 for s in scores) / n
    kurtosis = (m4 / (variance ** 2)) if variance > 0 else 0.0

    # Bimodality detection via histogram
    is_bimodal, mode_1, mode_2 = _detect_bimodality(scores)

    # Classify distribution
    if is_bimodal:
        dist_type = "bimodal"
    elif abs(skewness) < 0.5 and 1.5 < kurtosis < 5.0:
        dist_type = "normal"
    elif skewness < -0.5:
        dist_type = "skewed_left"
    elif skewness > 0.5:
        dist_type = "skewed_right"
    elif kurtosis < 2.0:
        dist_type = "uniform"
    else:
        dist_type = "normal"

    return DistributionFit(
        metric_id=metric_id,
        sample_count=n,
        mean=mean,
        std_dev=std,
        median=median,
        skewness=skewness,
        kurtosis=kurtosis,
        distribution_type=dist_type,
        is_bimodal=is_bimodal,
        mode_1=mode_1,
        mode_2=mode_2,
    )


def fit_all_metrics(
    scores_by_metric: dict[str, list[float]],
) -> dict[str, DistributionFit]:
    """Fit distributions for all metrics.

    Args:
        scores_by_metric: Dict of metric_id -> score list.

    Returns:
        Dict of metric_id -> DistributionFit.
    """
    return {
        mid: fit_distribution(mid, scores)
        for mid, scores in sorted(scores_by_metric.items())
    }


def _detect_bimodality(scores: list[float], num_bins: int = 10) -> tuple:
    """Detect bimodal distribution using histogram analysis.

    Returns (is_bimodal, mode_1, mode_2).
    """
    if len(scores) < 10:
        return False, None, None

    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        return False, None, None

    bin_width = (s_max - s_min) / num_bins
    bins = [0] * num_bins

    for s in scores:
        idx = min(int((s - s_min) / bin_width), num_bins - 1)
        bins[idx] += 1

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, num_bins - 1):
        if bins[i] > bins[i - 1] and bins[i] > bins[i + 1]:
            peak_center = s_min + (i + 0.5) * bin_width
            peaks.append((bins[i], peak_center))

    # Check edges
    if bins[0] > bins[1]:
        peaks.append((bins[0], s_min + 0.5 * bin_width))
    if bins[-1] > bins[-2]:
        peaks.append((bins[-1], s_max - 0.5 * bin_width))

    if len(peaks) >= 2:
        # Sort by height and check if top 2 peaks are separated
        peaks.sort(reverse=True)
        mode_1 = peaks[0][1]
        mode_2 = peaks[1][1]
        separation = abs(mode_1 - mode_2) / (s_max - s_min)

        # Valley between peaks: check if there's a significant dip
        if separation > 0.3:
            return True, min(mode_1, mode_2), max(mode_1, mode_2)

    return False, None, None
