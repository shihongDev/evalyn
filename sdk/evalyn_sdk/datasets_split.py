"""Dataset splitting utilities for train/test/validation splits.

Supports random and stratified splitting with deterministic seeding.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SplitResult:
    """Result of a dataset split operation."""

    train: list = field(default_factory=list)
    test: list = field(default_factory=list)
    validation: list = field(default_factory=list)

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "test": len(self.test),
            "validation": len(self.validation),
        }

    @property
    def ratios(self) -> dict[str, float]:
        total = len(self.train) + len(self.test) + len(self.validation)
        if total == 0:
            return {"train": 0.0, "test": 0.0, "validation": 0.0}
        return {
            "train": len(self.train) / total,
            "test": len(self.test) / total,
            "validation": len(self.validation) / total,
        }


def split_dataset(
    items: list,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    validation_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: str | None = None,
) -> SplitResult:
    """Split a dataset into train/test/validation sets.

    Args:
        items: List of DatasetItem objects.
        train_ratio: Fraction for training set (default 0.7).
        test_ratio: Fraction for test set (default 0.15).
        validation_ratio: Fraction for validation set (default 0.15).
        seed: Random seed for reproducibility.
        stratify_by: Metadata field to stratify by (optional). Ensures
            proportional representation of each value across splits.

    Returns:
        SplitResult with train, test, and validation lists.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or dataset is too small.
    """
    total_ratio = train_ratio + test_ratio + validation_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio:.2f} "
            f"(train={train_ratio}, test={test_ratio}, validation={validation_ratio})"
        )

    if not items:
        return SplitResult()

    if stratify_by:
        return _stratified_split(items, train_ratio, test_ratio, validation_ratio, seed, stratify_by)

    return _random_split(items, train_ratio, test_ratio, seed)


def _random_split(
    items: list,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitResult:
    """Simple random split."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)

    return SplitResult(
        train=shuffled[:train_end],
        test=shuffled[train_end:test_end],
        validation=shuffled[test_end:],
    )


def _stratified_split(
    items: list,
    train_ratio: float,
    test_ratio: float,
    validation_ratio: float,
    seed: int,
    stratify_by: str,
) -> SplitResult:
    """Stratified split maintaining proportional representation."""
    # Group items by stratification key
    groups: dict[str, list] = defaultdict(list)
    for item in items:
        key = str((item.metadata or {}).get(stratify_by, "_none_"))
        groups[key].append(item)

    result = SplitResult()
    rng = random.Random(seed)

    for _key, group_items in sorted(groups.items()):
        rng.shuffle(group_items)
        n = len(group_items)
        train_end = max(1, int(n * train_ratio)) if n >= 3 else n
        test_end = train_end + max(1, int(n * test_ratio)) if n >= 3 else n

        result.train.extend(group_items[:train_end])
        result.test.extend(group_items[train_end:test_end])
        result.validation.extend(group_items[test_end:])

    return result


def split_two_way(
    items: list,
    ratio: float = 0.8,
    seed: int = 42,
    stratify_by: str | None = None,
) -> tuple[list, list]:
    """Simple two-way split (e.g., for calibration train/validation).

    Args:
        items: Dataset items.
        ratio: Fraction for the first split.
        seed: Random seed.
        stratify_by: Optional stratification field.

    Returns:
        Tuple of (first_split, second_split).
    """
    result = split_dataset(
        items,
        train_ratio=ratio,
        test_ratio=1.0 - ratio,
        validation_ratio=0.0,
        seed=seed,
        stratify_by=stratify_by,
    )
    # Merge test + validation (validation may have rounding remainders)
    return result.train, result.test + result.validation
