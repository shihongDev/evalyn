"""Importance sampling for weighted evaluation item selection.

Weight sample selection by item difficulty or model uncertainty.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SampleWeight:
    """Weight assigned to a single evaluation item."""

    item_id: str
    weight: float
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "weight": self.weight,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SampleWeight:
        return cls(
            item_id=data["item_id"],
            weight=data["weight"],
            reason=data.get("reason", ""),
        )


@dataclass
class SamplingConfig:
    """Configuration for importance sampling."""

    strategy: str  # "inverse_pass_rate", "confidence", "custom"
    sample_size: int = 100
    min_weight: float = 0.01
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "sample_size": self.sample_size,
            "min_weight": self.min_weight,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SamplingConfig:
        return cls(
            strategy=data["strategy"],
            sample_size=data.get("sample_size", 100),
            min_weight=data.get("min_weight", 0.01),
            seed=data.get("seed"),
        )


@dataclass
class SamplingResult:
    """Result of an importance sampling run."""

    selected_ids: List[str] = field(default_factory=list)
    weights_used: List[SampleWeight] = field(default_factory=list)
    total_pool: int = 0
    effective_sample_size: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "weights_used": [w.as_dict() for w in self.weights_used],
            "total_pool": self.total_pool,
            "effective_sample_size": self.effective_sample_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SamplingResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            weights_used=[
                SampleWeight.from_dict(w) for w in data.get("weights_used", [])
            ],
            total_pool=data.get("total_pool", 0),
            effective_sample_size=data.get("effective_sample_size", 0.0),
        )


def compute_inverse_pass_rate_weights(
    scores: Dict[str, float],
    threshold: float = 0.5,
    min_weight: float = 0.01,
) -> List[SampleWeight]:
    """Compute weights inversely proportional to score.

    Lower scores get higher weights: weight = 1 / max(score, min_weight).
    The threshold parameter is used only for labeling (below/above) in the
    reason field, not for weight computation.
    """
    weights: List[SampleWeight] = []
    for item_id, score in scores.items():
        clamped = max(score, min_weight)
        w = 1.0 / clamped
        passed = score >= threshold
        reason = "passed" if passed else "failed"
        weights.append(SampleWeight(item_id=item_id, weight=w, reason=reason))
    return weights


def compute_confidence_weights(
    confidences: Dict[str, float],
) -> List[SampleWeight]:
    """Compute weights from model confidence scores.

    Low confidence items get higher weights: weight = 1 / max(confidence, 0.01).
    """
    weights: List[SampleWeight] = []
    for item_id, conf in confidences.items():
        clamped = max(conf, 0.01)
        w = 1.0 / clamped
        reason = "low_confidence" if conf < 0.5 else "high_confidence"
        weights.append(SampleWeight(item_id=item_id, weight=w, reason=reason))
    return weights


def compute_custom_weights(
    items: Dict[str, float],
    weight_fn: Callable[[float], float],
) -> List[SampleWeight]:
    """Apply a custom weight function to item values."""
    weights: List[SampleWeight] = []
    for item_id, value in items.items():
        w = weight_fn(value)
        weights.append(SampleWeight(item_id=item_id, weight=w, reason="custom"))
    return weights


def weighted_sample(
    weights: List[SampleWeight],
    sample_size: int,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample item_ids proportional to weight without replacement."""
    if not weights:
        return []
    sample_size = min(sample_size, len(weights))
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    remaining_ids = [w.item_id for w in weights]
    remaining_ws = [w.weight for w in weights]
    selected: List[str] = []
    for _ in range(sample_size):
        if not remaining_ids:
            break
        pick = rng.choices(remaining_ids, weights=remaining_ws, k=1)[0]
        selected.append(pick)
        idx = remaining_ids.index(pick)
        remaining_ids.pop(idx)
        remaining_ws.pop(idx)
    return selected


def compute_effective_sample_size(weights: List[float]) -> float:
    """Compute effective sample size: ESS = (sum(w))^2 / sum(w^2).

    Measures how many "effective" independent samples the weighted set
    represents. ESS equals n when all weights are equal.
    """
    if not weights:
        return 0.0
    total = sum(weights)
    sum_sq = sum(w * w for w in weights)
    if sum_sq == 0.0:
        return 0.0
    return (total * total) / sum_sq


def run_importance_sampling(
    pool: Dict[str, float],
    config: SamplingConfig,
) -> SamplingResult:
    """Run the full importance sampling pipeline.

    1. Compute weights based on strategy
    2. Sample items proportional to weight
    3. Compute effective sample size
    """
    if config.strategy == "inverse_pass_rate":
        weights = compute_inverse_pass_rate_weights(
            pool, min_weight=config.min_weight
        )
    elif config.strategy == "confidence":
        weights = compute_confidence_weights(pool)
    elif config.strategy == "custom":
        # For custom strategy, use identity weight function as default
        weights = compute_custom_weights(pool, lambda x: x)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy!r}")

    selected = weighted_sample(weights, config.sample_size, seed=config.seed)
    ess = compute_effective_sample_size([w.weight for w in weights])

    return SamplingResult(
        selected_ids=selected,
        weights_used=weights,
        total_pool=len(pool),
        effective_sample_size=ess,
    )


def format_sampling_report(result: SamplingResult) -> str:
    """Format a human-readable sampling report."""
    lines: List[str] = []
    lines.append("Importance Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Selected samples: {len(result.selected_ids)}")
    lines.append(f"Effective sample size: {result.effective_sample_size:.2f}")
    lines.append("")

    if result.weights_used:
        sorted_weights = sorted(
            result.weights_used, key=lambda w: w.weight, reverse=True
        )
        lines.append("Top weighted items:")
        for w in sorted_weights[:10]:
            lines.append(f"  {w.item_id}: {w.weight:.4f} ({w.reason})")

    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")

    return "\n".join(lines)
