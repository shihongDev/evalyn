"""Adversarial sampling: select items most likely to trigger model failures.

Computes weights from failure history, boundary proximity, and cohort
performance, then samples without replacement using those weights.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AdversarialWeight:
    """Weight assigned to a single item for adversarial sampling."""

    item_id: str
    weight: float
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "weight": self.weight,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdversarialWeight:
        return cls(
            item_id=data["item_id"],
            weight=data["weight"],
            reason=data["reason"],
        )


@dataclass
class AdversarialConfig:
    """Configuration for adversarial sampling."""

    failure_boost: float = 3.0
    boundary_boost: float = 2.0
    boundary_width: float = 0.1
    sample_size: int = 50
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_boost": self.failure_boost,
            "boundary_boost": self.boundary_boost,
            "boundary_width": self.boundary_width,
            "sample_size": self.sample_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdversarialConfig:
        return cls(
            failure_boost=data.get("failure_boost", 3.0),
            boundary_boost=data.get("boundary_boost", 2.0),
            boundary_width=data.get("boundary_width", 0.1),
            sample_size=data.get("sample_size", 50),
            seed=data.get("seed"),
        )


@dataclass
class AdversarialResult:
    """Result of an adversarial sampling run."""

    selected_ids: List[str] = field(default_factory=list)
    weights: List[AdversarialWeight] = field(default_factory=list)
    failure_count: int = 0
    boundary_count: int = 0
    total_pool: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "weights": [w.as_dict() for w in self.weights],
            "failure_count": self.failure_count,
            "boundary_count": self.boundary_count,
            "total_pool": self.total_pool,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdversarialResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            weights=[
                AdversarialWeight.from_dict(w) for w in data.get("weights", [])
            ],
            failure_count=data.get("failure_count", 0),
            boundary_count=data.get("boundary_count", 0),
            total_pool=data.get("total_pool", 0),
        )


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def compute_failure_weights(
    scores: Dict[str, float],
    threshold: float = 0.5,
    boost: float = 3.0,
) -> List[AdversarialWeight]:
    """Boost items that scored below the threshold.

    Items below the threshold receive a weight equal to boost; items at or
    above the threshold receive a weight of 1.0.
    """
    weights: List[AdversarialWeight] = []
    for item_id in sorted(scores):
        score = scores[item_id]
        if score < threshold:
            weights.append(
                AdversarialWeight(
                    item_id=item_id,
                    weight=boost,
                    reason=f"failure (score {score:.3f} < {threshold})",
                )
            )
        else:
            weights.append(
                AdversarialWeight(
                    item_id=item_id,
                    weight=1.0,
                    reason="pass",
                )
            )
    return weights


def compute_boundary_weights(
    scores: Dict[str, float],
    threshold: float = 0.5,
    width: float = 0.1,
    boost: float = 2.0,
) -> List[AdversarialWeight]:
    """Boost items whose score falls within width of the threshold.

    Items with score in [threshold - width, threshold + width] receive the
    boost weight; all others receive 1.0.
    """
    low = threshold - width
    high = threshold + width
    weights: List[AdversarialWeight] = []
    for item_id in sorted(scores):
        score = scores[item_id]
        if low <= score <= high:
            weights.append(
                AdversarialWeight(
                    item_id=item_id,
                    weight=boost,
                    reason=f"boundary (score {score:.3f} near {threshold})",
                )
            )
        else:
            weights.append(
                AdversarialWeight(
                    item_id=item_id,
                    weight=1.0,
                    reason="non-boundary",
                )
            )
    return weights


def compute_cohort_weights(
    scores: Dict[str, float],
    cohorts: Dict[str, List[str]],
) -> List[AdversarialWeight]:
    """Boost items from the weakest-performing cohort.

    Computes mean score per cohort, then gives a 2x boost to items in the
    cohort with the lowest mean. Items not in any cohort get weight 1.0.
    """
    if not cohorts:
        return [
            AdversarialWeight(item_id=item_id, weight=1.0, reason="no cohorts")
            for item_id in sorted(scores)
        ]

    # Compute mean score per cohort
    cohort_means: Dict[str, float] = {}
    for name, members in cohorts.items():
        member_scores = [scores[m] for m in members if m in scores]
        if member_scores:
            cohort_means[name] = sum(member_scores) / len(member_scores)

    if not cohort_means:
        return [
            AdversarialWeight(item_id=item_id, weight=1.0, reason="no scored cohorts")
            for item_id in sorted(scores)
        ]

    worst_cohort = min(cohort_means, key=lambda k: cohort_means[k])
    worst_members = set(cohorts[worst_cohort])

    weights: List[AdversarialWeight] = []
    for item_id in sorted(scores):
        if item_id in worst_members:
            weights.append(
                AdversarialWeight(
                    item_id=item_id,
                    weight=2.0,
                    reason=f"weak cohort '{worst_cohort}' (mean {cohort_means[worst_cohort]:.3f})",
                )
            )
        else:
            weights.append(
                AdversarialWeight(
                    item_id=item_id,
                    weight=1.0,
                    reason="non-weak cohort",
                )
            )
    return weights


# ---------------------------------------------------------------------------
# Weight merging
# ---------------------------------------------------------------------------


def merge_weights(
    weight_lists: List[List[AdversarialWeight]],
) -> List[AdversarialWeight]:
    """Combine multiple weight lists by summing weights for the same item_id.

    Reasons are joined with "; ". If an item appears in multiple lists, its
    final weight is the sum across all lists.
    """
    combined: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}

    for wlist in weight_lists:
        for w in wlist:
            if w.item_id not in combined:
                combined[w.item_id] = 0.0
                reasons[w.item_id] = []
            combined[w.item_id] += w.weight
            reasons[w.item_id].append(w.reason)

    merged: List[AdversarialWeight] = []
    for item_id in sorted(combined):
        merged.append(
            AdversarialWeight(
                item_id=item_id,
                weight=combined[item_id],
                reason="; ".join(reasons[item_id]),
            )
        )
    return merged


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _weighted_sample_without_replacement(
    ids: List[str],
    weights: List[float],
    k: int,
    rng: random.Random,
) -> List[str]:
    """Sample k items without replacement, weighted by weight.

    Uses the exponential sort trick: for each item, draw u ~ Uniform(0,1)
    and compute key = u^(1/weight). Select the top-k items by key.
    This gives exact weighted sampling without replacement.
    """
    if k >= len(ids):
        return list(ids)
    if k <= 0:
        return []

    keyed = []
    for item_id, w in zip(ids, weights):
        u = rng.random()
        # Avoid log(0); clamp u away from 0
        if u == 0.0:
            u = 1e-15
        # key = u^(1/w) is equivalent to exp(log(u)/w)
        key = math.exp(math.log(u) / max(w, 1e-15))
        keyed.append((key, item_id))

    keyed.sort(key=lambda x: x[0], reverse=True)
    return [item_id for _, item_id in keyed[:k]]


def run_adversarial_sampling(
    scores: Dict[str, float],
    config: AdversarialConfig,
    cohorts: Optional[Dict[str, List[str]]] = None,
) -> AdversarialResult:
    """Full adversarial sampling pipeline.

    1. Compute failure weights
    2. Compute boundary weights
    3. Optionally compute cohort weights
    4. Merge all weight lists
    5. Sample without replacement using merged weights
    """
    if not scores:
        return AdversarialResult(
            selected_ids=[],
            weights=[],
            failure_count=0,
            boundary_count=0,
            total_pool=0,
        )

    threshold = 0.5

    failure_w = compute_failure_weights(
        scores, threshold=threshold, boost=config.failure_boost
    )
    boundary_w = compute_boundary_weights(
        scores,
        threshold=threshold,
        width=config.boundary_width,
        boost=config.boundary_boost,
    )

    all_weights = [failure_w, boundary_w]

    if cohorts:
        cohort_w = compute_cohort_weights(scores, cohorts)
        all_weights.append(cohort_w)

    merged = merge_weights(all_weights)

    # Build lookup for sampling
    ids = [w.item_id for w in merged]
    weight_values = [w.weight for w in merged]

    rng = random.Random(config.seed)
    k = min(config.sample_size, len(ids))
    selected = _weighted_sample_without_replacement(ids, weight_values, k, rng)

    # Count failures and boundary items
    failure_count = sum(1 for s in scores.values() if s < threshold)
    boundary_count = sum(
        1
        for s in scores.values()
        if (threshold - config.boundary_width)
        <= s
        <= (threshold + config.boundary_width)
    )

    return AdversarialResult(
        selected_ids=selected,
        weights=merged,
        failure_count=failure_count,
        boundary_count=boundary_count,
        total_pool=len(scores),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_adversarial_report(result: AdversarialResult) -> str:
    """Format a human-readable adversarial sampling report."""
    lines: List[str] = []
    lines.append("Adversarial Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Selected count: {len(result.selected_ids)}")
    lines.append(f"Failure items: {result.failure_count}")
    lines.append(f"Boundary items: {result.boundary_count}")
    lines.append("")
    lines.append("Top weights:")
    top = sorted(result.weights, key=lambda w: w.weight, reverse=True)[:10]
    for w in top:
        lines.append(f"  {w.item_id}: {w.weight:.2f} ({w.reason})")
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
