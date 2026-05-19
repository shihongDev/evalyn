"""Time-weighted sampling: prefer recent traces over older ones.

Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class TimeWeight:
    """Weight assigned to a single item based on its age."""

    item_id: str
    timestamp: str  # ISO format
    age_days: float
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "timestamp": self.timestamp,
            "age_days": self.age_days,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeWeight:
        return cls(
            item_id=data["item_id"],
            timestamp=data["timestamp"],
            age_days=data["age_days"],
            weight=data["weight"],
        )


@dataclass
class TimeWeightConfig:
    """Configuration for time-weighted sampling."""

    half_life_days: float = 7.0
    min_weight: float = 0.01
    reference_time: str = ""  # ISO format, defaults to now if empty

    def as_dict(self) -> dict[str, Any]:
        return {
            "half_life_days": self.half_life_days,
            "min_weight": self.min_weight,
            "reference_time": self.reference_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeWeightConfig:
        return cls(
            half_life_days=data.get("half_life_days", 7.0),
            min_weight=data.get("min_weight", 0.01),
            reference_time=data.get("reference_time", ""),
        )


@dataclass
class TimeWeightResult:
    """Result of a time-weighted sampling run."""

    selected_ids: list[str] = field(default_factory=list)
    weights: list[TimeWeight] = field(default_factory=list)
    total_pool: int = 0
    effective_days_covered: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "weights": [w.as_dict() for w in self.weights],
            "total_pool": self.total_pool,
            "effective_days_covered": self.effective_days_covered,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeWeightResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            weights=[TimeWeight.from_dict(w) for w in data.get("weights", [])],
            total_pool=data.get("total_pool", 0),
            effective_days_covered=data.get("effective_days_covered", 0.0),
        )


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def compute_age_days(timestamp: str, reference: str) -> float:
    """Parse ISO timestamps and return age in days.

    Both timestamp and reference must be ISO format strings.
    """
    ts = datetime.fromisoformat(timestamp)
    ref = datetime.fromisoformat(reference)
    # Ensure both are timezone-aware for comparison
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    delta = ref - ts
    return delta.total_seconds() / 86400.0


def compute_exponential_weight(
    age_days: float,
    half_life: float,
    min_weight: float = 0.01,
) -> float:
    """Compute exponential decay weight: max(2^(-age/half_life), min_weight)."""
    if half_life <= 0:
        return min_weight
    raw = math.pow(2, -age_days / half_life)
    return max(raw, min_weight)


def compute_time_weights(
    items: dict[str, str],
    config: TimeWeightConfig,
) -> list[TimeWeight]:
    """Compute time-based weights for all items.

    items is {item_id: ISO_timestamp}.
    """
    if not items:
        return []
    ref = config.reference_time
    if not ref:
        ref = datetime.now(timezone.utc).isoformat()
    results: list[TimeWeight] = []
    for item_id, ts in items.items():
        age = compute_age_days(ts, ref)
        weight = compute_exponential_weight(age, config.half_life_days, config.min_weight)
        results.append(
            TimeWeight(
                item_id=item_id,
                timestamp=ts,
                age_days=age,
                weight=weight,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_by_time(
    weights: list[TimeWeight],
    sample_size: int,
    seed: int | None = None,
) -> list[str]:
    """Weighted sampling without replacement.

    Draws one item at a time proportional to weight, removes it from the pool,
    and repeats until sample_size items are selected or the pool is exhausted.
    """
    if not weights or sample_size <= 0:
        return []
    rng = _random.Random(seed)
    pool = [(w.item_id, w.weight) for w in weights]
    selected: list[str] = []
    for _ in range(min(sample_size, len(pool))):
        total = sum(w for _, w in pool)
        if total <= 0:
            break
        r = rng.random() * total
        cumulative = 0.0
        chosen_idx = 0
        for idx, (_, w) in enumerate(pool):
            cumulative += w
            if cumulative >= r:
                chosen_idx = idx
                break
        selected.append(pool[chosen_idx][0])
        pool.pop(chosen_idx)
    return selected


def ensure_minimum_representation(
    weights: list[TimeWeight],
    min_per_period: int = 1,
    period_days: float = 7.0,
) -> list[TimeWeight]:
    """Boost weights of underrepresented time periods.

    Groups items by period (age_days // period_days) and boosts weights
    of items in periods that have fewer than min_per_period items with
    non-minimal weight. Boosted items get weight set to the median weight
    across all items.
    """
    if not weights:
        return []

    # Group by period
    periods: dict[int, list[int]] = {}
    for i, w in enumerate(weights):
        period = int(w.age_days // period_days) if period_days > 0 else 0
        periods.setdefault(period, []).append(i)

    # Compute median weight for boosting
    sorted_weights = sorted(w.weight for w in weights)
    mid = len(sorted_weights) // 2
    if len(sorted_weights) % 2 == 0 and len(sorted_weights) >= 2:
        median_weight = (sorted_weights[mid - 1] + sorted_weights[mid]) / 2.0
    else:
        median_weight = sorted_weights[mid]

    # Build new weight list
    boosted: list[TimeWeight] = []
    for w in weights:
        period = int(w.age_days // period_days) if period_days > 0 else 0
        period_items = periods[period]
        # Count items with weight above minimal threshold
        high_weight_count = sum(
            1 for idx in period_items if weights[idx].weight > median_weight * 0.1
        )
        if high_weight_count < min_per_period:
            # Boost this item
            new_weight = max(w.weight, median_weight)
            boosted.append(
                TimeWeight(
                    item_id=w.item_id,
                    timestamp=w.timestamp,
                    age_days=w.age_days,
                    weight=new_weight,
                )
            )
        else:
            boosted.append(
                TimeWeight(
                    item_id=w.item_id,
                    timestamp=w.timestamp,
                    age_days=w.age_days,
                    weight=w.weight,
                )
            )
    return boosted


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_time_weighted_sampling(
    items: dict[str, str],
    sample_size: int,
    config: TimeWeightConfig | None = None,
    seed: int | None = None,
) -> TimeWeightResult:
    """Full time-weighted sampling pipeline.

    items is {item_id: ISO_timestamp}. Returns selected ids and metadata.
    """
    if config is None:
        config = TimeWeightConfig()

    weights = compute_time_weights(items, config)
    selected = sample_by_time(weights, sample_size, seed=seed)

    # Compute effective days covered
    if weights:
        selected_set = set(selected)
        ages = [w.age_days for w in weights if w.item_id in selected_set]
        if ages:
            effective_days = max(ages) - min(ages)
        else:
            effective_days = 0.0
    else:
        effective_days = 0.0

    return TimeWeightResult(
        selected_ids=selected,
        weights=weights,
        total_pool=len(items),
        effective_days_covered=effective_days,
    )


def format_time_weight_report(result: TimeWeightResult) -> str:
    """Format a human-readable time-weighted sampling report."""
    lines: list[str] = []
    lines.append("Time-Weighted Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool: {result.total_pool}")
    lines.append(f"Selected: {len(result.selected_ids)}")
    lines.append(f"Effective days covered: {result.effective_days_covered:.1f}")
    lines.append("")

    if result.weights:
        lines.append("Weight distribution:")
        _selected_set = set(result.selected_ids)
        sorted_weights = sorted(result.weights, key=lambda w: w.age_days)
        for w in sorted_weights:
            marker = " *" if w.item_id in _selected_set else ""
            lines.append(
                f"  {w.item_id}: age={w.age_days:.1f}d weight={w.weight:.4f}{marker}"
            )
        lines.append("")

    return "\n".join(lines)
