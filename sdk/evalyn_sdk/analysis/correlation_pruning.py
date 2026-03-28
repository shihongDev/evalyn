"""Metric correlation pruning: auto-suggest removing redundant metrics.

Analyzes score correlations between metrics and recommends dropping
highly correlated (redundant) metrics to save evaluation cost without
losing signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CorrelationPair:
    """A pair of correlated metrics."""

    metric_a: str
    metric_b: str
    correlation: float  # Pearson r
    relationship: str   # "redundant", "tradeoff", "independent"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "correlation": round(self.correlation, 4),
            "relationship": self.relationship,
        }


@dataclass
class PruningRecommendation:
    """Recommendation for which metrics to keep/drop."""

    keep: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)
    redundant_pairs: List[CorrelationPair] = field(default_factory=list)
    estimated_cost_savings: float = 0.0  # fraction of subjective cost saved

    def as_dict(self) -> Dict[str, Any]:
        return {
            "keep": list(self.keep),
            "drop": list(self.drop),
            "redundant_pairs": [p.as_dict() for p in self.redundant_pairs],
            "estimated_cost_savings": round(self.estimated_cost_savings, 4),
        }

    def format_text(self) -> str:
        lines = ["Metric Pruning Recommendation"]
        lines.append(f"  Keep: {len(self.keep)} metrics")
        lines.append(f"  Drop: {len(self.drop)} metrics")
        if self.estimated_cost_savings > 0:
            lines.append(f"  Estimated cost savings: {self.estimated_cost_savings:.0%}")
        if self.drop:
            lines.append("")
            lines.append("  Suggested to drop (redundant with kept metrics):")
            for mid in self.drop:
                # Find which kept metric it's redundant with
                for pair in self.redundant_pairs:
                    if mid in (pair.metric_a, pair.metric_b):
                        other = pair.metric_b if mid == pair.metric_a else pair.metric_a
                        lines.append(f"    - {mid} (r={pair.correlation:.2f} with {other})")
                        break
                else:
                    lines.append(f"    - {mid}")
        return "\n".join(lines)


def compute_correlations(
    metric_scores: Dict[str, List[float]],
    min_samples: int = 5,
) -> List[CorrelationPair]:
    """Compute pairwise Pearson correlations between metrics.

    Args:
        metric_scores: Dict of metric_id -> list of scores (one per item).
            All lists must have the same length.
        min_samples: Minimum items needed for correlation (default 5).

    Returns:
        List of CorrelationPair for each unique metric pair.
    """
    metric_ids = sorted(metric_scores.keys())
    pairs = []

    for i, a in enumerate(metric_ids):
        for b in metric_ids[i + 1:]:
            scores_a = metric_scores[a]
            scores_b = metric_scores[b]

            n = min(len(scores_a), len(scores_b))
            if n < min_samples:
                continue

            r = _pearson_r(scores_a[:n], scores_b[:n])
            if r is None:
                continue

            if r >= 0.7:
                relationship = "redundant"
            elif r <= -0.5:
                relationship = "tradeoff"
            else:
                relationship = "independent"

            pairs.append(CorrelationPair(
                metric_a=a, metric_b=b,
                correlation=r, relationship=relationship,
            ))

    return pairs


def recommend_pruning(
    correlations: List[CorrelationPair],
    metric_types: Optional[Dict[str, str]] = None,
    redundancy_threshold: float = 0.7,
) -> PruningRecommendation:
    """Recommend which metrics to drop based on correlations.

    Strategy: for each redundant pair, keep the metric that is more
    "central" (correlated with more other metrics) or prefer objective
    over subjective (cheaper).

    Args:
        correlations: List of CorrelationPair from compute_correlations().
        metric_types: Optional dict of metric_id -> "objective"/"subjective".
        redundancy_threshold: Correlation threshold for "redundant" (default 0.7).

    Returns:
        PruningRecommendation with keep/drop lists.
    """
    metric_types = metric_types or {}

    # Find redundant pairs
    redundant = [p for p in correlations if p.correlation >= redundancy_threshold]

    if not redundant:
        all_metrics = set()
        for p in correlations:
            all_metrics.add(p.metric_a)
            all_metrics.add(p.metric_b)
        return PruningRecommendation(keep=sorted(all_metrics))

    # Count how many redundant pairs each metric appears in
    redundancy_count: Dict[str, int] = {}
    for p in redundant:
        redundancy_count[p.metric_a] = redundancy_count.get(p.metric_a, 0) + 1
        redundancy_count[p.metric_b] = redundancy_count.get(p.metric_b, 0) + 1

    # For each redundant pair, decide which to drop
    drop_set: set = set()
    for pair in redundant:
        a, b = pair.metric_a, pair.metric_b
        if a in drop_set or b in drop_set:
            continue  # already handled

        # Prefer objective (cheaper) over subjective
        a_type = metric_types.get(a, "objective")
        b_type = metric_types.get(b, "objective")

        if a_type == "objective" and b_type == "subjective":
            drop_set.add(b)  # drop the expensive one
        elif b_type == "objective" and a_type == "subjective":
            drop_set.add(a)
        else:
            # Same type: drop the one with fewer other correlations (less central)
            if redundancy_count.get(a, 0) >= redundancy_count.get(b, 0):
                drop_set.add(b)  # keep the more connected one
            else:
                drop_set.add(a)

    all_metrics = set()
    for p in correlations:
        all_metrics.add(p.metric_a)
        all_metrics.add(p.metric_b)

    keep = sorted(all_metrics - drop_set)
    drop = sorted(drop_set)

    # Estimate cost savings (fraction of subjective metrics dropped)
    subjective_total = sum(1 for m in all_metrics if metric_types.get(m) == "subjective")
    subjective_dropped = sum(1 for m in drop if metric_types.get(m) == "subjective")
    cost_savings = subjective_dropped / subjective_total if subjective_total > 0 else 0.0

    return PruningRecommendation(
        keep=keep,
        drop=drop,
        redundant_pairs=redundant,
        estimated_cost_savings=cost_savings,
    )


def _pearson_r(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None

    return cov_xy / denom
