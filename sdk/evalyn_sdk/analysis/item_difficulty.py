"""
Item difficulty estimation: compute per-item difficulty from cross-run fail rates.

Provides functions to compute fail rates, classify difficulty levels,
and generate difficulty reports across multiple evaluation runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ItemDifficulty:
    """Difficulty estimation for a single dataset item."""

    item_id: str
    fail_rate: float
    num_runs: int = 0
    difficulty_label: str = "medium"
    variance: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "fail_rate": self.fail_rate,
            "num_runs": self.num_runs,
            "difficulty_label": self.difficulty_label,
            "variance": self.variance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ItemDifficulty:
        return cls(
            item_id=data["item_id"],
            fail_rate=data["fail_rate"],
            num_runs=data.get("num_runs", 0),
            difficulty_label=data.get("difficulty_label", "medium"),
            variance=data.get("variance", 0.0),
        )


@dataclass
class DifficultyReport:
    """Aggregated difficulty report across all items."""

    items: List[ItemDifficulty] = field(default_factory=list)
    avg_difficulty: float = 0.0
    distribution: Dict[str, int] = field(default_factory=dict)
    hardest_items: List[str] = field(default_factory=list)
    easiest_items: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": [i.as_dict() for i in self.items],
            "avg_difficulty": self.avg_difficulty,
            "distribution": dict(self.distribution),
            "hardest_items": list(self.hardest_items),
            "easiest_items": list(self.easiest_items),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DifficultyReport:
        return cls(
            items=[ItemDifficulty.from_dict(i) for i in data.get("items", [])],
            avg_difficulty=data.get("avg_difficulty", 0.0),
            distribution=data.get("distribution", {}),
            hardest_items=data.get("hardest_items", []),
            easiest_items=data.get("easiest_items", []),
        )

    def format_text(self) -> str:
        """Format report as human-readable text."""
        lines: List[str] = []
        lines.append(f"Item Difficulty Report ({len(self.items)} items)")
        lines.append("-" * 50)
        lines.append(f"  Average fail rate: {self.avg_difficulty:.2%}")
        lines.append("  Distribution:")
        for label in ["easy", "medium", "hard", "very_hard"]:
            count = self.distribution.get(label, 0)
            if count > 0:
                lines.append(f"    {label}: {count}")
        if self.hardest_items:
            lines.append(f"  Hardest: {', '.join(self.hardest_items[:5])}")
        if self.easiest_items:
            lines.append(f"  Easiest: {', '.join(self.easiest_items[:5])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def compute_item_fail_rate(
    item_id: str, run_results: List[Dict[str, Any]]
) -> float:
    """Compute fail rate for a single item across runs.

    Each result dict has "item_id" and "passed" (bool).
    Returns fraction of runs where the item failed (0.0 to 1.0).
    Returns 0.0 if no matching results found.
    """
    matching = [r for r in run_results if r.get("item_id") == item_id]
    if not matching:
        return 0.0
    failed = sum(1 for r in matching if not r.get("passed", True))
    return failed / len(matching)


def classify_difficulty(fail_rate: float) -> str:
    """Classify difficulty based on fail rate.

    - easy: fail_rate < 0.1
    - medium: 0.1 <= fail_rate < 0.4
    - hard: 0.4 <= fail_rate < 0.7
    - very_hard: fail_rate >= 0.7
    """
    if fail_rate < 0.1:
        return "easy"
    if fail_rate < 0.4:
        return "medium"
    if fail_rate < 0.7:
        return "hard"
    return "very_hard"


def compute_difficulty_variance(results: List[bool]) -> float:
    """Compute variance of pass/fail outcomes across runs.

    Treats True as 1.0 and False as 0.0.
    Returns 0.0 for empty input.
    """
    if not results:
        return 0.0
    n = len(results)
    values = [1.0 if r else 0.0 for r in results]
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance


def estimate_difficulty(
    item_id: str, run_results: List[Dict[str, Any]]
) -> ItemDifficulty:
    """Full difficulty estimation for a single item.

    Computes fail rate, classifies difficulty, and calculates variance.
    """
    matching = [r for r in run_results if r.get("item_id") == item_id]
    fail_rate = compute_item_fail_rate(item_id, run_results)
    label = classify_difficulty(fail_rate)
    outcomes = [r.get("passed", True) for r in matching]
    variance = compute_difficulty_variance(outcomes)

    return ItemDifficulty(
        item_id=item_id,
        fail_rate=fail_rate,
        num_runs=len(matching),
        difficulty_label=label,
        variance=variance,
    )


def estimate_all_difficulties(
    items: List[str], run_results: List[Dict[str, Any]]
) -> DifficultyReport:
    """Estimate difficulty for all items and build a report."""
    if not items:
        return DifficultyReport()

    difficulties: List[ItemDifficulty] = []
    for item_id in items:
        d = estimate_difficulty(item_id, run_results)
        difficulties.append(d)

    # Average fail rate
    avg = sum(d.fail_rate for d in difficulties) / len(difficulties) if difficulties else 0.0

    # Distribution
    dist: Dict[str, int] = {}
    for d in difficulties:
        dist[d.difficulty_label] = dist.get(d.difficulty_label, 0) + 1

    # Sort for hardest/easiest
    sorted_items = sort_by_difficulty(difficulties, ascending=False)
    hardest = [d.item_id for d in sorted_items[:5]]

    sorted_easy = sort_by_difficulty(difficulties, ascending=True)
    easiest = [d.item_id for d in sorted_easy[:5]]

    return DifficultyReport(
        items=difficulties,
        avg_difficulty=avg,
        distribution=dist,
        hardest_items=hardest,
        easiest_items=easiest,
    )


def sort_by_difficulty(
    items: List[ItemDifficulty], ascending: bool = False
) -> List[ItemDifficulty]:
    """Sort items by fail rate. Hardest first by default."""
    return sorted(items, key=lambda d: d.fail_rate, reverse=not ascending)
