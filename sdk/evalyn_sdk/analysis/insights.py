"""
Insights engine: diagnostic, prescriptive, and proactive analysis.

Provides pure functions operating on RunAnalysis data to surface
actionable intelligence about evaluation results.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from .core import RunAnalysis


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CorrelationResult:
    """Correlation between two metrics' scores across items."""
    metric_a: str
    metric_b: str
    pearson: float
    relationship: Literal["redundant", "tradeoff", "independent"]


@dataclass
class RegressionAlert:
    """Alert for a metric whose pass rate dropped between runs."""
    metric_id: str
    previous_pass_rate: float
    current_pass_rate: float
    delta: float
    severity: Literal["critical", "warning", "info"]


@dataclass
class FeatureInsight:
    """Insight linking an input/output feature to pass/fail rates."""
    feature_name: str
    finding: str
    affected_items: int
    pass_rate_low: float
    pass_rate_high: float


@dataclass
class DistributionInsight:
    """Insight about unusual score distribution shape for a metric."""
    metric_id: str
    shape: Literal["normal", "bimodal", "skewed_low", "skewed_high", "cliff", "uniform"]
    finding: str


@dataclass
class Recommendation:
    """Prioritized actionable recommendation."""
    priority: int
    category: str
    message: str
    action: str


@dataclass
class InsightsReport:
    """Complete insights report combining all analysis types."""
    correlations: List[CorrelationResult] = field(default_factory=list)
    regressions: List[RegressionAlert] = field(default_factory=list)
    feature_insights: List[FeatureInsight] = field(default_factory=list)
    distribution_insights: List[DistributionInsight] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric Correlations
# ---------------------------------------------------------------------------

MIN_ITEMS_FOR_CORRELATION = 5
REDUNDANT_THRESHOLD = 0.7
TRADEOFF_THRESHOLD = -0.5


def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient. Returns None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def _classify_correlation(r: float) -> Literal["redundant", "tradeoff", "independent"]:
    if r >= REDUNDANT_THRESHOLD:
        return "redundant"
    if r <= TRADEOFF_THRESHOLD:
        return "tradeoff"
    return "independent"


def compute_metric_correlations(run_analysis: RunAnalysis) -> List[CorrelationResult]:
    """Compute pairwise Pearson correlation between metric scores.

    For each pair of metrics, collects items that have scores for both,
    computes Pearson r, and classifies the relationship.

    Returns only non-independent pairs (|r| > 0.5) sorted by |r| descending.
    Requires at least MIN_ITEMS_FOR_CORRELATION items with scores for both metrics.
    """
    # Build per-item score vectors: {metric_id: {item_id: score}}
    metric_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for item_id, item_stats in run_analysis.item_stats.items():
        for metric_id, result in item_stats.metric_results.items():
            score = result.get("score")
            if score is not None:
                metric_scores[metric_id][item_id] = score

    metric_ids = sorted(metric_scores.keys())
    results: List[CorrelationResult] = []

    for i, m_a in enumerate(metric_ids):
        for m_b in metric_ids[i + 1:]:
            # Find common items
            common_items = set(metric_scores[m_a].keys()) & set(metric_scores[m_b].keys())
            if len(common_items) < MIN_ITEMS_FOR_CORRELATION:
                continue

            xs = [metric_scores[m_a][item] for item in common_items]
            ys = [metric_scores[m_b][item] for item in common_items]

            r = _pearson_r(xs, ys)
            if r is None:
                continue

            rel = _classify_correlation(r)
            if rel != "independent":
                results.append(CorrelationResult(
                    metric_a=m_a,
                    metric_b=m_b,
                    pearson=round(r, 3),
                    relationship=rel,
                ))

    results.sort(key=lambda c: abs(c.pearson), reverse=True)
    return results
