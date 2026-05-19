"""Balanced sampling for equal representation across metadata categories.

Ensures each category in a metadata field is represented fairly in the
sampled subset. Supports undersample, oversample, and proportional strategies.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BalanceConfig:
    """Configuration for balanced sampling."""

    balance_field: str
    strategy: str = "undersample"  # "undersample", "oversample", "proportional"
    target_size: int = 0  # 0 means auto
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "balance_field": self.balance_field,
            "strategy": self.strategy,
            "target_size": self.target_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BalanceConfig:
        return cls(
            balance_field=data["balance_field"],
            strategy=data.get("strategy", "undersample"),
            target_size=data.get("target_size", 0),
            seed=data.get("seed"),
        )


@dataclass
class CategoryStats:
    """Statistics for a single category in balanced sampling."""

    category: str
    count: int
    sampled_count: int
    ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "count": self.count,
            "sampled_count": self.sampled_count,
            "ratio": self.ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryStats:
        return cls(
            category=data["category"],
            count=data["count"],
            sampled_count=data["sampled_count"],
            ratio=data["ratio"],
        )


@dataclass
class BalanceResult:
    """Result of a balanced sampling run."""

    selected_ids: list[str] = field(default_factory=list)
    category_stats: list[CategoryStats] = field(default_factory=list)
    total_pool: int = 0
    total_selected: int = 0
    balance_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "category_stats": [s.as_dict() for s in self.category_stats],
            "total_pool": self.total_pool,
            "total_selected": self.total_selected,
            "balance_score": self.balance_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BalanceResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            category_stats=[
                CategoryStats.from_dict(s) for s in data.get("category_stats", [])
            ],
            total_pool=data.get("total_pool", 0),
            total_selected=data.get("total_selected", 0),
            balance_score=data.get("balance_score", 0.0),
        )


def compute_category_counts(
    items: list[dict[str, Any]], field_name: str
) -> dict[str, list[str]]:
    """Group item IDs by category field value.

    Each item dict must have an "id" key and a key matching field_name.
    Items missing the field are grouped under "<missing>".
    """
    groups: dict[str, list[str]] = {}
    for item in items:
        item_id = str(item.get("id", ""))
        category = str(item.get(field_name, "<missing>"))
        if category not in groups:
            groups[category] = []
        groups[category].append(item_id)
    return groups


def compute_balance_score(counts: dict[str, int]) -> float:
    """Compute balance score as 1 - Gini coefficient.

    Returns 1.0 for perfectly uniform distributions, approaching 0.0 for
    maximally imbalanced distributions. Returns 1.0 for empty or single-category
    inputs.
    """
    if not counts:
        return 1.0
    values = list(counts.values())
    n = len(values)
    if n <= 1:
        return 1.0
    total = sum(values)
    if total == 0:
        return 1.0

    # Gini coefficient: mean absolute difference / (2 * mean)
    abs_diff_sum = 0.0
    for i in range(n):
        for j in range(n):
            abs_diff_sum += abs(values[i] - values[j])

    gini = abs_diff_sum / (2 * n * total)
    return 1.0 - gini


def undersample(
    groups: dict[str, list[str]],
    target_per_category: int,
    seed: int | None = None,
) -> list[str]:
    """Sample min(target, available) items from each category.

    Produces at most target_per_category items per group. If a group has
    fewer items, all are included.
    """
    rng = random.Random(seed)
    selected: list[str] = []
    for category in sorted(groups.keys()):
        ids = list(groups[category])
        take = min(target_per_category, len(ids))
        if take >= len(ids):
            selected.extend(sorted(ids))
        else:
            sampled = rng.sample(ids, take)
            selected.extend(sampled)
    return selected


def oversample(
    groups: dict[str, list[str]],
    target_per_category: int,
    seed: int | None = None,
) -> list[str]:
    """Repeat items from small categories to match target count.

    Categories with fewer items than target_per_category will have items
    repeated (with random selection) to reach the target.
    """
    rng = random.Random(seed)
    selected: list[str] = []
    for category in sorted(groups.keys()):
        ids = list(groups[category])
        if not ids:
            continue
        if len(ids) >= target_per_category:
            sampled = rng.sample(ids, target_per_category)
            selected.extend(sampled)
        else:
            # Include all originals, then fill with random repeats
            result = list(ids)
            remaining = target_per_category - len(ids)
            for _ in range(remaining):
                result.append(rng.choice(ids))
            selected.extend(result)
    return selected


def proportional_sample(
    groups: dict[str, list[str]],
    total: int,
    seed: int | None = None,
) -> list[str]:
    """Sample proportionally but ensure at least 1 item per category.

    Distributes total budget proportionally to category sizes, ensuring
    every non-empty category gets at least 1 sample.
    """
    if not groups or total <= 0:
        return []

    rng = random.Random(seed)
    pool_size = sum(len(ids) for ids in groups.values())
    if pool_size == 0:
        return []

    sorted_cats = sorted(groups.keys())
    non_empty = [c for c in sorted_cats if groups[c]]

    # Calculate proportional allocation with minimum of 1
    allocation: dict[str, int] = {}
    # First pass: give each category at least 1
    remaining_budget = total
    for cat in non_empty:
        allocation[cat] = 1
        remaining_budget -= 1

    if remaining_budget < 0:
        # Not enough budget for all categories: pick a subset
        rng.shuffle(non_empty)
        chosen = non_empty[:total]
        selected: list[str] = []
        for cat in sorted(chosen):
            ids = groups[cat]
            selected.append(rng.choice(ids))
        return selected

    # Second pass: distribute remaining budget proportionally
    if remaining_budget > 0:
        proportions = {
            cat: len(groups[cat]) / pool_size for cat in non_empty
        }
        # Distribute remaining
        fractional: dict[str, float] = {}
        for cat in non_empty:
            fractional[cat] = proportions[cat] * remaining_budget

        # Floor allocation + remainder distribution
        for cat in non_empty:
            allocation[cat] += int(fractional[cat])
        leftover = remaining_budget - sum(
            int(fractional[cat]) for cat in non_empty
        )
        # Give leftover to categories with highest fractional parts
        remainders = [
            (fractional[cat] - int(fractional[cat]), cat) for cat in non_empty
        ]
        remainders.sort(key=lambda x: x[0], reverse=True)
        for i in range(leftover):
            allocation[remainders[i][1]] += 1

    # Sample from each category
    selected = []
    for cat in sorted(allocation.keys()):
        ids = groups[cat]
        take = min(allocation[cat], len(ids))
        if take >= len(ids):
            selected.extend(sorted(ids))
        else:
            selected.extend(rng.sample(ids, take))
    return selected


def run_balanced_sampling(
    items: list[dict[str, Any]], config: BalanceConfig
) -> BalanceResult:
    """Full balanced sampling pipeline.

    1. Group items by the balance field
    2. Apply the chosen strategy
    3. Compute balance score and category stats
    """
    if not items:
        return BalanceResult(
            selected_ids=[],
            category_stats=[],
            total_pool=0,
            total_selected=0,
            balance_score=1.0,
        )

    groups = compute_category_counts(items, config.balance_field)

    if config.strategy == "undersample":
        if config.target_size > 0:
            n_cats = len(groups)
            target_per = max(1, config.target_size // n_cats) if n_cats else 0
        else:
            # Auto: use the smallest category size
            target_per = min(len(ids) for ids in groups.values()) if groups else 0
        selected = undersample(groups, target_per, seed=config.seed)

    elif config.strategy == "oversample":
        if config.target_size > 0:
            n_cats = len(groups)
            target_per = max(1, config.target_size // n_cats) if n_cats else 0
        else:
            # Auto: use the largest category size
            target_per = max(len(ids) for ids in groups.values()) if groups else 0
        selected = oversample(groups, target_per, seed=config.seed)

    elif config.strategy == "proportional":
        total = config.target_size if config.target_size > 0 else len(items)
        selected = proportional_sample(groups, total, seed=config.seed)

    else:
        raise ValueError(f"Unknown strategy: {config.strategy!r}")

    # Build category stats
    category_stats: list[CategoryStats] = []
    total_selected = len(selected)
    for cat in sorted(groups.keys()):
        original_count = len(groups[cat])
        group_set = set(groups[cat])
        sampled = sum(1 for sid in selected if sid in group_set)
        ratio = sampled / total_selected if total_selected > 0 else 0.0
        category_stats.append(
            CategoryStats(
                category=cat,
                count=original_count,
                sampled_count=sampled,
                ratio=ratio,
            )
        )

    # Compute balance score on sampled counts
    sampled_counts = {s.category: s.sampled_count for s in category_stats}
    score = compute_balance_score(sampled_counts)

    return BalanceResult(
        selected_ids=selected,
        category_stats=category_stats,
        total_pool=len(items),
        total_selected=total_selected,
        balance_score=score,
    )


def format_balance_report(result: BalanceResult) -> str:
    """Format a human-readable balance sampling report with per-category breakdown."""
    lines: list[str] = []
    lines.append("Balanced Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Total selected: {result.total_selected}")
    lines.append(f"Balance score: {result.balance_score:.4f}")
    lines.append("")
    lines.append("Per-category breakdown:")
    lines.append(f"  {'Category':<20} {'Original':>10} {'Sampled':>10} {'Ratio':>10}")
    lines.append(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    for stat in result.category_stats:
        lines.append(
            f"  {stat.category:<20} {stat.count:>10} {stat.sampled_count:>10} {stat.ratio:>10.2%}"
        )
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)


def suggest_rebalancing(stats: list[CategoryStats]) -> list[str]:
    """Suggest which categories need more or fewer items.

    Compares each category's ratio to a uniform target ratio. Categories
    significantly over- or under-represented are flagged.
    """
    if not stats:
        return []

    n_cats = len(stats)
    target_ratio = 1.0 / n_cats
    threshold = target_ratio * 0.25  # 25% deviation threshold

    suggestions: list[str] = []
    for s in stats:
        diff = s.ratio - target_ratio
        if diff > threshold:
            suggestions.append(
                f"Category '{s.category}' is over-represented "
                f"({s.ratio:.1%} vs target {target_ratio:.1%}) - "
                f"consider reducing from {s.sampled_count} items"
            )
        elif diff < -threshold:
            suggestions.append(
                f"Category '{s.category}' is under-represented "
                f"({s.ratio:.1%} vs target {target_ratio:.1%}) - "
                f"consider adding more items (currently {s.count} available)"
            )

    if not suggestions:
        suggestions.append("Distribution is well-balanced across all categories")

    return suggestions
