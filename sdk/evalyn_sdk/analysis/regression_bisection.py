"""
Regression bisection: binary search across dataset items to pinpoint
the exact subset causing a regression.

Provides functions for bisecting item lists, computing pass rates,
identifying regressed items, and ranking by impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BisectionStep:
    """A single step in the bisection search."""

    step_number: int
    subset_size: int
    pass_rate: float
    direction: str = ""
    item_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "subset_size": self.subset_size,
            "pass_rate": self.pass_rate,
            "direction": self.direction,
            "item_ids": list(self.item_ids),
        }


@dataclass
class BisectionResult:
    """Result of a bisection search for regression culprits."""

    culprit_items: list[str] = field(default_factory=list)
    total_steps: int = 0
    steps: list[BisectionStep] = field(default_factory=list)
    original_pass_rate: float = 0.0
    regression_magnitude: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "culprit_items": list(self.culprit_items),
            "total_steps": self.total_steps,
            "steps": [s.as_dict() for s in self.steps],
            "original_pass_rate": self.original_pass_rate,
            "regression_magnitude": self.regression_magnitude,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BisectionResult:
        steps = [
            BisectionStep(
                step_number=s["step_number"],
                subset_size=s["subset_size"],
                pass_rate=s["pass_rate"],
                direction=s.get("direction", ""),
                item_ids=s.get("item_ids", []),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            culprit_items=data.get("culprit_items", []),
            total_steps=data.get("total_steps", 0),
            steps=steps,
            original_pass_rate=data.get("original_pass_rate", 0.0),
            regression_magnitude=data.get("regression_magnitude", 0.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("BISECTION RESULT")
        lines.append(f"Original pass rate: {self.original_pass_rate:.2%}")
        lines.append(f"Regression magnitude: {self.regression_magnitude:.2%}")
        lines.append(f"Total steps: {self.total_steps}")
        lines.append(f"Culprit items: {len(self.culprit_items)}")
        for item_id in self.culprit_items:
            lines.append(f"  - {item_id}")
        if self.steps:
            lines.append("")
            lines.append("Steps:")
            for step in self.steps:
                lines.append(
                    f"  Step {step.step_number}: "
                    f"subset_size={step.subset_size} "
                    f"pass_rate={step.pass_rate:.2%} "
                    f"direction={step.direction}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_pass_rate(
    items: list[dict[str, Any]],
    score_field: str = "score",
    threshold: float = 0.5,
) -> float:
    """Compute pass rate for items based on score field and threshold."""
    if not items:
        return 0.0
    passed = sum(1 for item in items if item.get(score_field, 0.0) >= threshold)
    return passed / len(items)


def bisect_items(
    items: list[dict[str, Any]],
    score_fn: Callable[[list[dict[str, Any]]], float],
    target_drop: float = 0.1,
) -> BisectionResult:
    """Binary search to find the subset causing a regression.

    Splits items in half, evaluates each half with score_fn,
    recurses into the half with the lower pass rate.
    Stops when the subset is small enough (<=2 items) or
    when neither half shows a meaningful drop.

    Args:
        items: List of item dicts (must have "id" key).
        score_fn: Function that takes a list of items and returns a pass rate.
        target_drop: Minimum drop to consider a subset as containing the culprit.
    """
    if not items:
        return BisectionResult()

    original_rate = score_fn(items)
    steps: list[BisectionStep] = []
    current = list(items)
    step_num = 0

    while len(current) > 2:
        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]

        left_rate = score_fn(left)
        right_rate = score_fn(right)

        step_num += 1

        if left_rate <= right_rate:
            direction = "left"
            chosen = left
            chosen_rate = left_rate
        else:
            direction = "right"
            chosen = right
            chosen_rate = right_rate

        steps.append(BisectionStep(
            step_number=step_num,
            subset_size=len(chosen),
            pass_rate=chosen_rate,
            direction=direction,
            item_ids=[item.get("id", "") for item in chosen],
        ))

        # If neither half shows a meaningful drop, stop
        if min(left_rate, right_rate) >= original_rate - target_drop:
            break

        current = chosen

    # Final step for the remaining items
    if current:
        final_rate = score_fn(current)
        step_num += 1
        steps.append(BisectionStep(
            step_number=step_num,
            subset_size=len(current),
            pass_rate=final_rate,
            direction="found",
            item_ids=[item.get("id", "") for item in current],
        ))

    culprit_ids = [item.get("id", "") for item in current]
    regression_mag = original_rate - score_fn(current) if current else 0.0

    return BisectionResult(
        culprit_items=culprit_ids,
        total_steps=step_num,
        steps=steps,
        original_pass_rate=original_rate,
        regression_magnitude=regression_mag,
    )


def identify_regression_items(
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    score_field: str = "score",
) -> list[str]:
    """Find items where score dropped between run A (baseline) and run B.

    Items are matched by "id" field. Returns IDs where score_b < score_a.
    """
    scores_a = {item.get("id", ""): item.get(score_field, 0.0) for item in items_a}
    scores_b = {item.get("id", ""): item.get(score_field, 0.0) for item in items_b}

    regressed: list[str] = []
    for item_id in scores_a:
        if item_id in scores_b:
            if scores_b[item_id] < scores_a[item_id]:
                regressed.append(item_id)

    return regressed


def rank_by_regression_impact(
    items: list[dict[str, Any]],
    baseline_scores: dict[str, float],
    score_field: str = "score",
) -> list[tuple[str, float]]:
    """Rank items by score drop magnitude relative to baseline.

    Returns list of (item_id, drop_magnitude) sorted by drop descending.
    Only includes items where score decreased.
    """
    impacts: list[tuple[str, float]] = []
    for item in items:
        item_id = item.get("id", "")
        if item_id not in baseline_scores:
            continue
        current_score = item.get(score_field, 0.0)
        baseline = baseline_scores[item_id]
        drop = baseline - current_score
        if drop > 0:
            impacts.append((item_id, drop))

    impacts.sort(key=lambda x: x[1], reverse=True)
    return impacts


def build_bisection_report(result: BisectionResult) -> str:
    """Formatted summary of a bisection result."""
    return result.format_text()
