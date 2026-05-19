"""
Dataset A/B split generator: create matched pairs for controlled model comparison.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ABPair:
    """A matched pair of items from the two groups."""

    pair_id: str
    item_a: dict[str, Any] = field(default_factory=dict)
    item_b: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "item_a": self.item_a,
            "item_b": self.item_b,
            "similarity": self.similarity,
        }


@dataclass
class ABSplitConfig:
    """Configuration for A/B splitting."""

    split_ratio: float = 0.5
    stratify_by: str = ""
    match_by_length: bool = True
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_ratio": self.split_ratio,
            "stratify_by": self.stratify_by,
            "match_by_length": self.match_by_length,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ABSplitConfig:
        return cls(
            split_ratio=data.get("split_ratio", 0.5),
            stratify_by=data.get("stratify_by", ""),
            match_by_length=data.get("match_by_length", True),
            seed=data.get("seed"),
        )


@dataclass
class ABSplitResult:
    """Result of an A/B split operation."""

    group_a: list[dict[str, Any]] = field(default_factory=list)
    group_b: list[dict[str, Any]] = field(default_factory=list)
    pairs: list[ABPair] = field(default_factory=list)
    balance_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_a": list(self.group_a),
            "group_b": list(self.group_b),
            "pairs": [p.as_dict() for p in self.pairs],
            "balance_score": self.balance_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ABSplitResult:
        pairs = [ABPair(**p) for p in data.get("pairs", [])]
        return cls(
            group_a=data.get("group_a", []),
            group_b=data.get("group_b", []),
            pairs=pairs,
            balance_score=data.get("balance_score", 0.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("A/B Split Result")
        lines.append("-" * 40)
        lines.append(f"  Group A: {len(self.group_a)} items")
        lines.append(f"  Group B: {len(self.group_b)} items")
        lines.append(f"  Pairs: {len(self.pairs)}")
        lines.append(f"  Balance score: {self.balance_score:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def split_ab(
    items: list[dict[str, Any]],
    config: ABSplitConfig,
) -> ABSplitResult:
    """Split items into two balanced groups.

    If stratify_by is set, maintain category proportions in both groups.
    """
    if not items:
        return ABSplitResult(balance_score=1.0)

    rng = random.Random(config.seed)

    if config.stratify_by:
        group_a, group_b = _stratified_split(items, config.stratify_by, config.split_ratio, rng)
    else:
        shuffled = list(items)
        rng.shuffle(shuffled)
        split_idx = max(1, round(len(shuffled) * config.split_ratio))
        group_a = shuffled[:split_idx]
        group_b = shuffled[split_idx:]

    balance = compute_balance_score(group_a, group_b)

    pairs: list[ABPair] = []
    if config.match_by_length:
        pairs = create_matched_pairs(items)

    return ABSplitResult(
        group_a=group_a,
        group_b=group_b,
        pairs=pairs,
        balance_score=balance,
    )


def _stratified_split(
    items: list[dict[str, Any]],
    stratify_field: str,
    ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split maintaining category proportions."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = str(item.get(stratify_field, "__none__"))
        buckets.setdefault(key, []).append(item)

    group_a: list[dict[str, Any]] = []
    group_b: list[dict[str, Any]] = []

    for _key, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        split_idx = max(1, round(len(bucket) * ratio))
        if len(bucket) == 1:
            # Single item - assign based on ratio
            if rng.random() < ratio:
                group_a.append(bucket[0])
            else:
                group_b.append(bucket[0])
        else:
            group_a.extend(bucket[:split_idx])
            group_b.extend(bucket[split_idx:])

    return group_a, group_b


def compute_balance_score(
    group_a: list[dict[str, Any]],
    group_b: list[dict[str, Any]],
    input_field: str = "input",
) -> float:
    """Score 0-1 how balanced the groups are.

    Considers size ratio and length distribution similarity. 1.0 = perfectly balanced.
    """
    if not group_a and not group_b:
        return 1.0
    if not group_a or not group_b:
        return 0.0

    # Size balance: 1.0 when equal, lower when imbalanced
    size_ratio = min(len(group_a), len(group_b)) / max(len(group_a), len(group_b))

    # Length distribution balance
    lengths_a = [len(str(item.get(input_field, ""))) for item in group_a]
    lengths_b = [len(str(item.get(input_field, ""))) for item in group_b]

    mean_a = sum(lengths_a) / len(lengths_a) if lengths_a else 0
    mean_b = sum(lengths_b) / len(lengths_b) if lengths_b else 0
    max_mean = max(mean_a, mean_b, 1.0)
    length_similarity = 1.0 - abs(mean_a - mean_b) / max_mean

    # Category balance (if items have categories)
    cats_a = Counter(str(item.get("category", "")) for item in group_a if "category" in item)
    cats_b = Counter(str(item.get("category", "")) for item in group_b if "category" in item)

    cat_score = 1.0
    if cats_a or cats_b:
        all_cats = set(cats_a.keys()) | set(cats_b.keys())
        total_a = sum(cats_a.values()) or 1
        total_b = sum(cats_b.values()) or 1
        diffs = []
        for cat in all_cats:
            pct_a = cats_a.get(cat, 0) / total_a
            pct_b = cats_b.get(cat, 0) / total_b
            diffs.append(abs(pct_a - pct_b))
        cat_score = max(0.0, 1.0 - sum(diffs))

    # Weighted combination
    score = 0.4 * size_ratio + 0.3 * length_similarity + 0.3 * cat_score
    return min(1.0, max(0.0, score))


def create_matched_pairs(
    items: list[dict[str, Any]],
    input_field: str = "input",
) -> list[ABPair]:
    """Create pairs of similar items using length-based matching."""
    if len(items) < 2:
        return []

    # Sort by input length
    sorted_items = sorted(items, key=lambda x: len(str(x.get(input_field, ""))))

    pairs: list[ABPair] = []
    used: set[int] = set()

    for i in range(0, len(sorted_items) - 1, 2):
        if i in used or i + 1 in used:
            continue
        item_a = sorted_items[i]
        item_b = sorted_items[i + 1]

        len_a = len(str(item_a.get(input_field, "")))
        len_b = len(str(item_b.get(input_field, "")))
        max_len = max(len_a, len_b, 1)
        similarity = 1.0 - abs(len_a - len_b) / max_len

        pairs.append(ABPair(
            pair_id=f"pair_{len(pairs)}",
            item_a=item_a,
            item_b=item_b,
            similarity=similarity,
        ))
        used.add(i)
        used.add(i + 1)

    return pairs


def validate_ab_split(
    result: ABSplitResult,
    min_balance: float = 0.7,
) -> tuple[bool, list[str]]:
    """Validate split quality. Returns (is_valid, list_of_issues)."""
    issues: list[str] = []

    if not result.group_a:
        issues.append("Group A is empty")
    if not result.group_b:
        issues.append("Group B is empty")

    if result.balance_score < min_balance:
        issues.append(
            f"Balance score {result.balance_score:.4f} below minimum {min_balance}"
        )

    # Check size ratio
    if result.group_a and result.group_b:
        size_ratio = min(len(result.group_a), len(result.group_b)) / max(
            len(result.group_a), len(result.group_b)
        )
        if size_ratio < 0.3:
            issues.append(f"Groups are very uneven (size ratio {size_ratio:.2f})")

    is_valid = len(issues) == 0
    return is_valid, issues


def render_split_summary(result: ABSplitResult) -> str:
    """ASCII summary of split statistics."""
    lines: list[str] = []
    lines.append("A/B Split Summary")
    lines.append("=" * 40)

    total = len(result.group_a) + len(result.group_b)
    pct_a = len(result.group_a) / total * 100 if total else 0
    pct_b = len(result.group_b) / total * 100 if total else 0

    lines.append(f"  Group A: {len(result.group_a)} items ({pct_a:.1f}%)")
    lines.append(f"  Group B: {len(result.group_b)} items ({pct_b:.1f}%)")
    lines.append(f"  Total:   {total} items")
    lines.append(f"  Pairs:   {len(result.pairs)}")
    lines.append(f"  Balance: {result.balance_score:.4f}")

    # Visual bar
    bar_width = 30
    if total:
        a_bar = round(bar_width * len(result.group_a) / total)
        b_bar = bar_width - a_bar
    else:
        a_bar = 0
        b_bar = 0
    lines.append(f"  [{'A' * a_bar}{'B' * b_bar}]")

    is_valid, issues = validate_ab_split(result)
    if is_valid:
        lines.append("  Status: PASS")
    else:
        lines.append("  Status: WARN")
        for issue in issues:
            lines.append(f"    - {issue}")

    return "\n".join(lines)
