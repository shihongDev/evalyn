"""K-fold cross-validation evaluation.

Split dataset items into K folds, run evaluation on each fold, and
aggregate metric scores with mean/std to assess stability.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Result for a single fold."""

    fold_index: int
    train_size: int
    test_size: int
    metric_scores: dict[str, float] = field(default_factory=dict)  # metric_id -> score

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "metric_scores": {k: round(v, 4) for k, v in self.metric_scores.items()},
        }


@dataclass
class CVReport:
    """Cross-validation report aggregating fold results."""

    num_folds: int
    fold_results: list[FoldResult] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)  # metric_id -> mean
    score_std_devs: dict[str, float] = field(default_factory=dict)   # metric_id -> std dev
    high_variance_items: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_folds": self.num_folds,
            "fold_results": [fr.as_dict() for fr in self.fold_results],
            "aggregate_scores": {k: round(v, 4) for k, v in self.aggregate_scores.items()},
            "score_std_devs": {k: round(v, 4) for k, v in self.score_std_devs.items()},
            "high_variance_items": self.high_variance_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CVReport:
        fold_results = [
            FoldResult(
                fold_index=fr["fold_index"],
                train_size=fr["train_size"],
                test_size=fr["test_size"],
                metric_scores=fr.get("metric_scores", {}),
            )
            for fr in data.get("fold_results", [])
        ]
        return cls(
            num_folds=data.get("num_folds", 0),
            fold_results=fold_results,
            aggregate_scores=data.get("aggregate_scores", {}),
            score_std_devs=data.get("score_std_devs", {}),
            high_variance_items=data.get("high_variance_items", []),
        )

    def format_text(self) -> str:
        lines = [
            "Cross-Validation Report",
            f"  Folds: {self.num_folds}",
        ]
        for metric_id in sorted(self.aggregate_scores):
            mean = self.aggregate_scores[metric_id]
            std = self.score_std_devs.get(metric_id, 0.0)
            lines.append(f"  {metric_id}: {mean:.4f} +/- {std:.4f}")
        if self.high_variance_items:
            lines.append(f"  High variance metrics: {', '.join(self.high_variance_items)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_folds(
    items: list[Any],
    num_folds: int = 5,
    seed: int | None = None,
    stratify_key: Callable | None = None,
) -> list[tuple[list[Any], list[Any]]]:
    """Split items into K folds. Each entry is (train_items, test_items).

    Args:
        items: The dataset items to split.
        num_folds: Number of folds (K).
        seed: Optional random seed for reproducibility.
        stratify_key: Optional callable returning a stratum label per item.

    Returns:
        List of (train, test) tuples, one per fold.
    """
    if num_folds < 1:
        raise ValueError("num_folds must be >= 1")

    rng = random.Random(seed)

    if stratify_key is not None:
        # Group items by stratum, shuffle each group, then distribute
        strata: dict[Any, list[Any]] = defaultdict(list)
        for item in items:
            strata[stratify_key(item)].append(item)
        # Assign each stratum's items round-robin across folds
        fold_buckets: list[list[Any]] = [[] for _ in range(num_folds)]
        for _key in sorted(strata.keys(), key=str):
            group = strata[_key]
            rng.shuffle(group)
            for i, item in enumerate(group):
                fold_buckets[i % num_folds].append(item)
    else:
        shuffled = list(items)
        rng.shuffle(shuffled)
        fold_buckets = [[] for _ in range(num_folds)]
        for i, item in enumerate(shuffled):
            fold_buckets[i % num_folds].append(item)

    # Build (train, test) pairs
    folds: list[tuple[list[Any], list[Any]]] = []
    for fold_idx in range(num_folds):
        test = fold_buckets[fold_idx]
        train = []
        for j in range(num_folds):
            if j != fold_idx:
                train.extend(fold_buckets[j])
        folds.append((train, test))

    return folds


def aggregate_fold_results(
    fold_results: list[FoldResult],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean and std dev per metric across folds.

    Returns:
        (means, std_devs) dicts keyed by metric_id.
    """
    if not fold_results:
        return {}, {}

    # Collect scores per metric
    scores_by_metric: dict[str, list[float]] = defaultdict(list)
    for fr in fold_results:
        for metric_id, score in fr.metric_scores.items():
            scores_by_metric[metric_id].append(score)

    means: dict[str, float] = {}
    std_devs: dict[str, float] = {}
    for metric_id, scores in scores_by_metric.items():
        n = len(scores)
        mean = sum(scores) / n
        means[metric_id] = mean
        if n > 1:
            variance = sum((s - mean) ** 2 for s in scores) / n
            std_devs[metric_id] = math.sqrt(variance)
        else:
            std_devs[metric_id] = 0.0

    return means, std_devs


def detect_high_variance_items(
    fold_results: list[FoldResult],
    threshold: float = 0.1,
) -> list[str]:
    """Find metrics with high score variance across folds.

    A metric is flagged if its std dev across folds exceeds the threshold.

    Returns:
        Sorted list of metric_ids with high variance.
    """
    _, std_devs = aggregate_fold_results(fold_results)
    return sorted(m for m, sd in std_devs.items() if sd > threshold)


def build_cv_report(
    fold_results: list[FoldResult],
    high_variance_threshold: float = 0.1,
) -> CVReport:
    """Build a cross-validation report from fold results.

    Args:
        fold_results: List of per-fold results.
        high_variance_threshold: Std dev threshold for flagging metrics.

    Returns:
        CVReport with aggregated scores and high-variance flags.
    """
    means, std_devs = aggregate_fold_results(fold_results)
    high_var = detect_high_variance_items(fold_results, high_variance_threshold)

    return CVReport(
        num_folds=len(fold_results),
        fold_results=fold_results,
        aggregate_scores=means,
        score_std_devs=std_devs,
        high_variance_items=high_var,
    )
