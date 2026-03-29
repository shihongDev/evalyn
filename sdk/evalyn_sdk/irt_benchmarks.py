"""
IRT-based tiny benchmarks: use Item Response Theory to find minimal
representative subsets of evaluation items.

Pure Python, no external dependencies. Uses the 2PL model for Fisher
information calculations and greedy selection for subset optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class IRTItem:
    """An evaluation item with IRT parameters."""

    item_id: str
    difficulty: float
    discrimination: float
    guessing: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "difficulty": self.difficulty,
            "discrimination": self.discrimination,
            "guessing": self.guessing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IRTItem:
        return cls(
            item_id=data["item_id"],
            difficulty=data["difficulty"],
            discrimination=data["discrimination"],
            guessing=data.get("guessing", 0.0),
        )


@dataclass
class IRTConfig:
    """Configuration for IRT-based item selection."""

    target_size: int = 100
    ability_range: Tuple[float, float] = (-3.0, 3.0)
    n_ability_points: int = 21

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_size": self.target_size,
            "ability_range": list(self.ability_range),
            "n_ability_points": self.n_ability_points,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IRTConfig:
        ar = data.get("ability_range", [-3.0, 3.0])
        return cls(
            target_size=data.get("target_size", 100),
            ability_range=(ar[0], ar[1]),
            n_ability_points=data.get("n_ability_points", 21),
        )


@dataclass
class IRTResult:
    """Result of IRT-based item selection."""

    selected_items: List[IRTItem] = field(default_factory=list)
    total_pool: int = 0
    compression_ratio: float = 0.0
    information_retained: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_items": [item.as_dict() for item in self.selected_items],
            "total_pool": self.total_pool,
            "compression_ratio": self.compression_ratio,
            "information_retained": self.information_retained,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IRTResult:
        return cls(
            selected_items=[
                IRTItem.from_dict(d) for d in data.get("selected_items", [])
            ],
            total_pool=data.get("total_pool", 0),
            compression_ratio=data.get("compression_ratio", 0.0),
            information_retained=data.get("information_retained", 0.0),
        )


# ---------------------------------------------------------------------------
# Parameter Estimation
# ---------------------------------------------------------------------------


def estimate_item_parameters(item_id: str, scores: list[float]) -> IRTItem:
    """Estimate IRT parameters from observed scores for a single item.

    Difficulty: inverse of mean score (low mean = hard item).
    Discrimination: inverse of std dev (low variance = high discrimination,
    meaning the item sharply separates ability levels).

    Args:
        item_id: unique identifier for the item.
        scores: list of observed scores in [0, 1].

    Returns:
        IRTItem with estimated parameters.
    """
    if not scores:
        return IRTItem(item_id=item_id, difficulty=0.0, discrimination=1.0)

    n = len(scores)
    mean = sum(scores) / n

    # Difficulty: transform mean score to logit scale.
    # Clamp mean away from 0 and 1 to avoid infinite logit.
    clamped_mean = max(0.01, min(0.99, mean))
    difficulty = -math.log(clamped_mean / (1.0 - clamped_mean))

    # Discrimination from std dev: low std = high discrimination.
    if n < 2:
        discrimination = 1.0
    else:
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std = math.sqrt(variance)
        # Map std to discrimination: std near 0 -> high disc, std near 0.5 -> low disc.
        # Use inverse relationship with floor.
        if std < 0.01:
            discrimination = 3.0
        else:
            discrimination = min(3.0, 0.5 / std)

    return IRTItem(
        item_id=item_id,
        difficulty=difficulty,
        discrimination=discrimination,
    )


def estimate_all_parameters(
    item_scores: dict[str, list[float]],
) -> list[IRTItem]:
    """Estimate IRT parameters for all items.

    Args:
        item_scores: mapping item_id -> list of scores.

    Returns:
        List of IRTItem with estimated parameters.
    """
    return [
        estimate_item_parameters(item_id, scores)
        for item_id, scores in item_scores.items()
    ]


# ---------------------------------------------------------------------------
# Information Functions (2PL Model)
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def compute_item_information(item: IRTItem, ability: float) -> float:
    """Fisher information of a single item at a given ability level.

    Uses the 2PL model:
        P(ability) = 1 / (1 + exp(-D * (ability - difficulty)))
        I(ability) = D^2 * P * (1 - P)

    where D is the discrimination parameter.

    Args:
        item: IRT item with difficulty and discrimination.
        ability: ability level (theta).

    Returns:
        Fisher information value (non-negative).
    """
    d = item.discrimination
    p = _sigmoid(d * (ability - item.difficulty))
    return d * d * p * (1.0 - p)


def compute_total_information(
    items: list[IRTItem],
    ability_points: list[float],
) -> float:
    """Sum Fisher information across all items and ability points.

    Args:
        items: list of IRT items.
        ability_points: list of ability levels to evaluate at.

    Returns:
        Total information (sum over all items and all ability points).
    """
    total = 0.0
    for item in items:
        for theta in ability_points:
            total += compute_item_information(item, theta)
    return total


def _make_ability_grid(config: IRTConfig) -> list[float]:
    """Create evenly spaced ability points from config."""
    lo, hi = config.ability_range
    n = config.n_ability_points
    if n <= 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Greedy Selection
# ---------------------------------------------------------------------------


def select_informative_items(
    items: list[IRTItem],
    config: IRTConfig | None = None,
) -> IRTResult:
    """Select the most informative subset using greedy maximization.

    At each step, pick the item that adds the most Fisher information
    across the ability range. Stops when target_size is reached or the
    pool is exhausted.

    Args:
        items: full pool of IRT items.
        config: selection configuration (uses defaults if None).

    Returns:
        IRTResult with selected items and compression stats.
    """
    if config is None:
        config = IRTConfig()

    if not items:
        return IRTResult(
            selected_items=[],
            total_pool=0,
            compression_ratio=0.0,
            information_retained=0.0,
        )

    ability_points = _make_ability_grid(config)
    target = min(config.target_size, len(items))

    # Precompute per-item information at each ability point.
    item_info: dict[str, list[float]] = {}
    for item in items:
        item_info[item.item_id] = [
            compute_item_information(item, theta)
            for theta in ability_points
        ]

    total_pool_info = compute_total_information(items, ability_points)

    selected: list[IRTItem] = []
    selected_ids: set[str] = set()
    # Track cumulative information at each ability point.
    cumulative_info = [0.0] * len(ability_points)

    item_map = {item.item_id: item for item in items}

    for _ in range(target):
        best_id: str | None = None
        best_gain = -1.0

        for item in items:
            if item.item_id in selected_ids:
                continue
            # Marginal gain: sum of info this item adds at each ability point.
            gain = sum(item_info[item.item_id])
            if gain > best_gain:
                best_gain = gain
                best_id = item.item_id

        if best_id is None:
            break

        selected.append(item_map[best_id])
        selected_ids.add(best_id)
        info_vec = item_info[best_id]
        for j in range(len(ability_points)):
            cumulative_info[j] += info_vec[j]

    selected_info = sum(cumulative_info)
    info_retained = selected_info / total_pool_info if total_pool_info > 0 else 0.0

    return IRTResult(
        selected_items=selected,
        total_pool=len(items),
        compression_ratio=len(selected) / len(items) if items else 0.0,
        information_retained=info_retained,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_irt_report(result: IRTResult) -> str:
    """Format a human-readable IRT selection report.

    Args:
        result: IRT selection result.

    Returns:
        Multi-line report string.
    """
    lines: list[str] = []
    lines.append("IRT Benchmark Selection Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Selected items: {len(result.selected_items)}")
    lines.append(f"Compression ratio: {result.compression_ratio:.2%}")
    lines.append(f"Information retained: {result.information_retained:.2%}")
    lines.append("")

    if result.selected_items:
        lines.append("Selected Items (by selection order):")
        lines.append("-" * 40)
        for i, item in enumerate(result.selected_items, 1):
            lines.append(
                f"  {i:3d}. {item.item_id}"
                f"  diff={item.difficulty:+.2f}"
                f"  disc={item.discrimination:.2f}"
            )

    return "\n".join(lines)
