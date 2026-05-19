"""Few-shot example ordering for judge prompts.

Optimize the order of examples to improve judge prompt effectiveness:
- Difficulty-based ordering (easiest first)
- Label grouping (group by label, fails last)
- Diversity ordering (alternate labels)
- Optimal ordering via search
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections.abc import Callable


@dataclass
class OrderedExample:
    """A few-shot example with ordering metadata."""

    id: str
    text: str
    label: str = ""
    difficulty: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderedExample:
        return cls(
            id=data["id"],
            text=data["text"],
            label=data.get("label", ""),
            difficulty=data.get("difficulty", 0.0),
        )


@dataclass
class OrderingResult:
    """Result of an example ordering strategy."""

    ordered_ids: list[str] = field(default_factory=list)
    method: str = ""
    estimated_impact: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordered_ids": list(self.ordered_ids),
            "method": self.method,
            "estimated_impact": self.estimated_impact,
        }

    def format_text(self) -> str:
        lines = [
            f"Ordering method: {self.method}",
            f"Estimated impact: {self.estimated_impact:.3f}",
            f"Order ({len(self.ordered_ids)} examples):",
        ]
        for i, eid in enumerate(self.ordered_ids, 1):
            lines.append(f"  {i}. {eid}")
        return "\n".join(lines)


def order_by_difficulty(examples: list[OrderedExample]) -> OrderingResult:
    """Sort examples easiest first (low difficulty), hardest last.

    Estimated impact is based on difficulty variance - higher variance
    means ordering matters more.
    """
    if not examples:
        return OrderingResult(ordered_ids=[], method="difficulty", estimated_impact=0.0)

    sorted_examples = sorted(examples, key=lambda e: e.difficulty)
    ordered_ids = [e.id for e in sorted_examples]

    # Compute variance of difficulty values
    difficulties = [e.difficulty for e in examples]
    mean_d = sum(difficulties) / len(difficulties)
    variance = sum((d - mean_d) ** 2 for d in difficulties) / len(difficulties)

    return OrderingResult(
        ordered_ids=ordered_ids,
        method="difficulty",
        estimated_impact=min(variance, 1.0),
    )


def order_by_label_grouping(examples: list[OrderedExample]) -> OrderingResult:
    """Group by label, put fails last. Within groups, sort by difficulty."""
    if not examples:
        return OrderingResult(
            ordered_ids=[], method="label_grouping", estimated_impact=0.0
        )

    fail_group = sorted(
        [e for e in examples if e.label == "fail"], key=lambda e: e.difficulty
    )
    non_fail_group = sorted(
        [e for e in examples if e.label != "fail"], key=lambda e: e.difficulty
    )

    ordered = non_fail_group + fail_group
    ordered_ids = [e.id for e in ordered]

    # Impact: higher when there is a mix of labels
    labels = {e.label for e in examples}
    impact = min(len(labels) / max(len(examples), 1), 1.0)

    return OrderingResult(
        ordered_ids=ordered_ids,
        method="label_grouping",
        estimated_impact=impact,
    )


def order_by_diversity(examples: list[OrderedExample]) -> OrderingResult:
    """Alternate between labels for maximum diversity.

    Groups examples by label and interleaves them. E.g. pass, fail, pass, fail.
    """
    if not examples:
        return OrderingResult(ordered_ids=[], method="diversity", estimated_impact=0.0)

    # Group by label
    groups: dict[str, list[OrderedExample]] = {}
    for e in examples:
        groups.setdefault(e.label, []).append(e)

    # Sort each group by difficulty
    for label in groups:
        groups[label].sort(key=lambda e: e.difficulty)

    # Interleave: round-robin across label groups
    ordered_ids: list[str] = []
    label_keys = sorted(groups.keys())
    iterators = {k: iter(groups[k]) for k in label_keys}
    remaining = set(label_keys)

    while remaining:
        for k in label_keys:
            if k not in remaining:
                continue
            try:
                e = next(iterators[k])
                ordered_ids.append(e.id)
            except StopIteration:
                remaining.discard(k)

    # Impact: higher with more distinct labels
    impact = min(len(groups) / max(len(examples), 1), 1.0)

    return OrderingResult(
        ordered_ids=ordered_ids,
        method="diversity",
        estimated_impact=impact,
    )


def reverse_order(result: OrderingResult) -> OrderingResult:
    """Reverse an ordering to test the opposite hypothesis."""
    return OrderingResult(
        ordered_ids=list(reversed(result.ordered_ids)),
        method=f"{result.method}_reversed",
        estimated_impact=result.estimated_impact,
    )


def find_optimal_order(
    examples: list[OrderedExample],
    score_fn: Callable[[list[str]], float],
    max_permutations: int = 100,
    seed: int | None = None,
) -> OrderingResult:
    """Try random permutations, return the one that maximizes score_fn.

    score_fn takes an ordered list of IDs and returns a score (higher is better).
    """
    if not examples:
        return OrderingResult(ordered_ids=[], method="optimal", estimated_impact=0.0)

    rng = random.Random(seed)
    ids = [e.id for e in examples]

    best_ids = list(ids)
    best_score = score_fn(best_ids)

    for _ in range(max_permutations - 1):
        candidate = list(ids)
        rng.shuffle(candidate)
        score = score_fn(candidate)
        if score > best_score:
            best_score = score
            best_ids = candidate

    return OrderingResult(
        ordered_ids=best_ids,
        method="optimal",
        estimated_impact=best_score,
    )
