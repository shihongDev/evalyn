"""Dataset interleaving: round-robin merge from multiple datasets for balanced evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InterleavingConfig:
    """Configuration for dataset interleaving."""

    strategy: str = "round_robin"  # "round_robin" / "weighted" / "random"
    weights: dict[str, float] = field(default_factory=dict)
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "weights": dict(self.weights),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterleavingConfig:
        return cls(
            strategy=data.get("strategy", "round_robin"),
            weights=data.get("weights", {}),
            seed=data.get("seed"),
        )


@dataclass
class InterleavedItem:
    """A single item in the interleaved result, tracking its source."""

    id: str
    source_dataset: str
    original_index: int
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_dataset": self.source_dataset,
            "original_index": self.original_index,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterleavedItem:
        return cls(
            id=data["id"],
            source_dataset=data["source_dataset"],
            original_index=data["original_index"],
            data=data.get("data", {}),
        )


@dataclass
class InterleavingResult:
    """Result of interleaving multiple datasets."""

    items: list[InterleavedItem] = field(default_factory=list)
    total_items: int = 0
    source_distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "total_items": self.total_items,
            "source_distribution": dict(self.source_distribution),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterleavingResult:
        return cls(
            items=[InterleavedItem.from_dict(i) for i in data.get("items", [])],
            total_items=data.get("total_items", 0),
            source_distribution=data.get("source_distribution", {}),
        )

    def format_text(self) -> str:
        lines = [
            f"Interleaved {self.total_items} items from {len(self.source_distribution)} datasets"
        ]
        for name, count in sorted(self.source_distribution.items()):
            pct = (count / self.total_items * 100) if self.total_items else 0.0
            lines.append(f"  {name}: {count} ({pct:.1f}%)")
        return "\n".join(lines)


def _build_result(items: list[InterleavedItem]) -> InterleavingResult:
    dist: dict[str, int] = {}
    for item in items:
        dist[item.source_dataset] = dist.get(item.source_dataset, 0) + 1
    return InterleavingResult(items=items, total_items=len(items), source_distribution=dist)


def interleave_round_robin(
    datasets: dict[str, list[dict[str, Any]]],
) -> InterleavingResult:
    """Round-robin merge. Take one item from each dataset in turn until all exhausted."""
    items: list[InterleavedItem] = []
    names = sorted(datasets.keys())
    if not names:
        return _build_result(items)
    max_len = max(len(datasets[n]) for n in names) if names else 0
    for round_idx in range(max_len):
        for name in names:
            ds = datasets[name]
            if round_idx < len(ds):
                items.append(
                    InterleavedItem(
                        id=f"{name}_{round_idx}",
                        source_dataset=name,
                        original_index=round_idx,
                        data=ds[round_idx],
                    )
                )
    return _build_result(items)


def interleave_weighted(
    datasets: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    seed: int | None = None,
) -> InterleavingResult:
    """Weighted random selection. Higher weight = more items from that dataset."""
    rng = random.Random(seed)
    items: list[InterleavedItem] = []
    names = sorted(datasets.keys())
    if not names:
        return _build_result(items)
    # Build pool of (name, index) pairs
    pool: list[tuple[str, int]] = []
    for name in names:
        for idx in range(len(datasets[name])):
            pool.append((name, idx))
    if not pool:
        return _build_result(items)
    # Assign selection probability based on weight
    total_weight = sum(weights.get(n, 1.0) for n in names)
    dataset_probs = {n: weights.get(n, 1.0) / total_weight for n in names}
    # Assign each pool entry a probability proportional to its dataset weight / dataset size
    entry_weights: list[float] = []
    for name, _idx in pool:
        ds_size = len(datasets[name])
        entry_weights.append(dataset_probs[name] / ds_size if ds_size else 0.0)
    # Sample all items without replacement using weighted shuffle
    order: list[int] = []
    remaining = list(range(len(pool)))
    remaining_weights = list(entry_weights)
    while remaining:
        chosen_indices = rng.choices(range(len(remaining)), weights=remaining_weights, k=1)
        chosen = chosen_indices[0]
        order.append(remaining[chosen])
        remaining.pop(chosen)
        remaining_weights.pop(chosen)
    for pos in order:
        name, idx = pool[pos]
        items.append(
            InterleavedItem(
                id=f"{name}_{idx}",
                source_dataset=name,
                original_index=idx,
                data=datasets[name][idx],
            )
        )
    return _build_result(items)


def interleave_random(
    datasets: dict[str, list[dict[str, Any]]],
    seed: int | None = None,
) -> InterleavingResult:
    """Random shuffle of all items."""
    rng = random.Random(seed)
    items: list[InterleavedItem] = []
    for name in sorted(datasets.keys()):
        for idx, data in enumerate(datasets[name]):
            items.append(
                InterleavedItem(
                    id=f"{name}_{idx}",
                    source_dataset=name,
                    original_index=idx,
                    data=data,
                )
            )
    rng.shuffle(items)
    return _build_result(items)


def interleave(
    datasets: dict[str, list[dict[str, Any]]],
    config: InterleavingConfig,
) -> InterleavingResult:
    """Route to appropriate strategy."""
    if config.strategy == "round_robin":
        return interleave_round_robin(datasets)
    elif config.strategy == "weighted":
        return interleave_weighted(datasets, config.weights, config.seed)
    elif config.strategy == "random":
        return interleave_random(datasets, config.seed)
    else:
        raise ValueError(f"Unknown interleaving strategy: {config.strategy}")


def validate_interleaving(result: InterleavingResult) -> tuple[bool, list[str]]:
    """Check all source datasets represented, no duplicates."""
    errors: list[str] = []
    # Check for duplicate IDs
    seen_ids: set = set()
    for item in result.items:
        if item.id in seen_ids:
            errors.append(f"Duplicate item ID: {item.id}")
        seen_ids.add(item.id)
    # Check total matches
    if result.total_items != len(result.items):
        errors.append(
            f"total_items ({result.total_items}) does not match actual count ({len(result.items)})"
        )
    # Check distribution sums to total
    dist_sum = sum(result.source_distribution.values())
    if dist_sum != result.total_items:
        errors.append(
            f"Distribution sum ({dist_sum}) does not match total_items ({result.total_items})"
        )
    return (len(errors) == 0, errors)


def compute_balance_score(result: InterleavingResult) -> float:
    """How evenly distributed sources are (0-1, 1 = perfectly even)."""
    if not result.source_distribution or result.total_items == 0:
        return 1.0
    n = len(result.source_distribution)
    if n == 1:
        return 1.0
    ideal = result.total_items / n
    max_deviation = result.total_items - ideal  # worst case: all items in one source
    actual_deviation = sum(abs(count - ideal) for count in result.source_distribution.values())
    if max_deviation == 0:
        return 1.0
    return 1.0 - (actual_deviation / (2 * max_deviation))
