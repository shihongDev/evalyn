"""Metric score normalization for fair cross-metric comparison.

Supports z-score and min-max normalization of metric scores so
different metrics (with different natural scales) can be compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedScores:
    """Normalized scores for a single metric."""

    metric_id: str
    method: str  # "z_score" or "min_max"
    original_scores: list[float] = field(default_factory=list)
    normalized_scores: list[float] = field(default_factory=list)
    original_mean: float = 0.0
    original_std: float = 0.0
    original_min: float = 0.0
    original_max: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "method": self.method,
            "original_mean": round(self.original_mean, 4),
            "original_std": round(self.original_std, 4),
            "normalized_mean": round(
                sum(self.normalized_scores) / len(self.normalized_scores), 4
            ) if self.normalized_scores else 0.0,
            "count": len(self.normalized_scores),
        }


def normalize_z_score(
    scores_by_metric: dict[str, list[float]],
) -> dict[str, NormalizedScores]:
    """Z-score normalize metric scores (mean=0, std=1).

    Useful for comparing metrics on different scales. A z-score of 2
    means "2 standard deviations above the mean for this metric."

    Args:
        scores_by_metric: Dict of metric_id -> raw score list.

    Returns:
        Dict of metric_id -> NormalizedScores.
    """
    results = {}
    for metric_id, scores in scores_by_metric.items():
        if len(scores) < 2:
            results[metric_id] = NormalizedScores(
                metric_id=metric_id, method="z_score",
                original_scores=list(scores),
                normalized_scores=list(scores),
                original_mean=scores[0] if scores else 0.0,
            )
            continue

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)

        if std == 0:
            normalized = [0.0] * len(scores)
        else:
            normalized = [(s - mean) / std for s in scores]

        results[metric_id] = NormalizedScores(
            metric_id=metric_id, method="z_score",
            original_scores=list(scores),
            normalized_scores=normalized,
            original_mean=mean,
            original_std=std,
            original_min=min(scores),
            original_max=max(scores),
        )

    return results


def normalize_min_max(
    scores_by_metric: dict[str, list[float]],
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> dict[str, NormalizedScores]:
    """Min-max normalize scores to [target_min, target_max] range.

    Maps the lowest score to target_min and highest to target_max.

    Args:
        scores_by_metric: Dict of metric_id -> raw score list.
        target_min: Lower bound of normalized range (default 0.0).
        target_max: Upper bound of normalized range (default 1.0).

    Returns:
        Dict of metric_id -> NormalizedScores.
    """
    results = {}
    target_range = target_max - target_min

    for metric_id, scores in scores_by_metric.items():
        if not scores:
            results[metric_id] = NormalizedScores(
                metric_id=metric_id, method="min_max",
            )
            continue

        s_min = min(scores)
        s_max = max(scores)
        s_range = s_max - s_min

        if s_range == 0:
            normalized = [target_min + target_range / 2] * len(scores)
        else:
            normalized = [
                target_min + (s - s_min) / s_range * target_range
                for s in scores
            ]

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0
        std = math.sqrt(variance)

        results[metric_id] = NormalizedScores(
            metric_id=metric_id, method="min_max",
            original_scores=list(scores),
            normalized_scores=normalized,
            original_mean=mean,
            original_std=std,
            original_min=s_min,
            original_max=s_max,
        )

    return results


def normalize_run_scores(
    metric_results: list,
    method: str = "min_max",
) -> dict[str, NormalizedScores]:
    """Normalize scores from an evaluation run.

    Convenience function that extracts scores from MetricResult objects
    and applies the specified normalization method.

    Args:
        metric_results: List of MetricResult objects.
        method: "z_score" or "min_max" (default).

    Returns:
        Dict of metric_id -> NormalizedScores.
    """
    scores_by_metric: dict[str, list[float]] = {}
    for mr in metric_results:
        if mr.score is not None:
            scores_by_metric.setdefault(mr.metric_id, []).append(mr.score)

    if method == "z_score":
        return normalize_z_score(scores_by_metric)
    return normalize_min_max(scores_by_metric)
