"""Few-shot example selection for judge prompts.

Provides strategies to pick the most useful examples: diverse label coverage,
informative boundary cases, and maximum word coverage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections.abc import Callable


@dataclass
class Example:
    """A single few-shot example."""

    id: str
    input_text: str
    output_text: str
    label: str = ""  # "pass" / "fail"
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "label": self.label,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Example:
        return cls(
            id=data["id"],
            input_text=data["input_text"],
            output_text=data["output_text"],
            label=data.get("label", ""),
            score=data.get("score", 0.0),
        )


@dataclass
class SelectionResult:
    """Result of a few-shot selection strategy."""

    selected: list[Example] = field(default_factory=list)
    method: str = ""
    coverage_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": [e.as_dict() for e in self.selected],
            "method": self.method,
            "coverage_score": self.coverage_score,
        }

    def format_text(self) -> str:
        lines = [
            f"Selection method: {self.method}",
            f"Selected: {len(self.selected)} examples",
            f"Coverage score: {self.coverage_score:.3f}",
        ]
        for ex in self.selected:
            lines.append(f"  [{ex.id}] label={ex.label} score={ex.score:.2f}")
        return "\n".join(lines)


def _word_set(example: Example) -> set:
    """Extract unique lowercase words from an example."""
    text = f"{example.input_text} {example.output_text}"
    return set(text.lower().split())


def _coverage_score(selected: list[Example]) -> float:
    """Compute word diversity as coverage score."""
    if not selected:
        return 0.0
    all_words: set = set()
    for ex in selected:
        all_words |= _word_set(ex)
    return float(len(all_words))


def select_diverse(
    examples: list[Example], k: int = 5, seed: int | None = None
) -> SelectionResult:
    """Select k examples covering different failure modes.

    Groups by label, samples proportionally. Breaks ties by score diversity -
    within each label group, sorts by score and spaces picks evenly.
    """
    if not examples or k <= 0:
        return SelectionResult(selected=[], method="diverse", coverage_score=0.0)
    if k >= len(examples):
        return SelectionResult(
            selected=list(examples),
            method="diverse",
            coverage_score=_coverage_score(examples),
        )

    rng = random.Random(seed)

    # Group by label
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(ex.label, []).append(ex)

    # Sort each group by score for diversity spacing
    for label in groups:
        groups[label].sort(key=lambda e: e.score)

    # Allocate slots proportionally
    labels = sorted(groups.keys())
    n = len(examples)
    allocation: dict[str, int] = {}
    remaining = k
    for i, label in enumerate(labels):
        if i == len(labels) - 1:
            allocation[label] = remaining
        else:
            share = max(1, round(k * len(groups[label]) / n))
            share = min(share, remaining)
            allocation[label] = share
            remaining -= share

    selected: list[Example] = []
    for label in labels:
        group = groups[label]
        count = min(allocation[label], len(group))
        if count <= 0:
            continue
        if count >= len(group):
            selected.extend(group)
        else:
            # Evenly space picks through the score-sorted group
            step = len(group) / count
            indices = [int(i * step) for i in range(count)]
            selected.extend(group[idx] for idx in indices)

    # If we got fewer than k (rounding), fill from remaining
    selected_ids = {e.id for e in selected}
    remaining_examples = [e for e in examples if e.id not in selected_ids]
    rng.shuffle(remaining_examples)
    while len(selected) < k and remaining_examples:
        selected.append(remaining_examples.pop())

    # Trim if over
    selected = selected[:k]

    return SelectionResult(
        selected=selected,
        method="diverse",
        coverage_score=_coverage_score(selected),
    )


def select_informative(
    examples: list[Example], k: int = 5
) -> SelectionResult:
    """Select examples with scores near the decision boundary.

    Prioritizes scores in the 0.4-0.6 range, then expands outward.
    """
    if not examples or k <= 0:
        return SelectionResult(selected=[], method="informative", coverage_score=0.0)
    if k >= len(examples):
        return SelectionResult(
            selected=list(examples),
            method="informative",
            coverage_score=_coverage_score(examples),
        )

    # Sort by distance from 0.5
    ranked = sorted(examples, key=lambda e: abs(e.score - 0.5))
    selected = ranked[:k]
    return SelectionResult(
        selected=selected,
        method="informative",
        coverage_score=_coverage_score(selected),
    )


def select_by_coverage(
    examples: list[Example], k: int = 5
) -> SelectionResult:
    """Maximize word diversity across selected examples.

    Greedy: pick the example that adds the most new words each step.
    """
    if not examples or k <= 0:
        return SelectionResult(selected=[], method="coverage", coverage_score=0.0)
    if k >= len(examples):
        return SelectionResult(
            selected=list(examples),
            method="coverage",
            coverage_score=_coverage_score(examples),
        )

    covered: set = set()
    selected: list[Example] = []
    remaining = list(examples)

    for _ in range(k):
        if not remaining:
            break
        best_idx = 0
        best_new = 0
        for i, ex in enumerate(remaining):
            new_words = len(_word_set(ex) - covered)
            if new_words > best_new:
                best_new = new_words
                best_idx = i
        chosen = remaining.pop(best_idx)
        covered |= _word_set(chosen)
        selected.append(chosen)

    return SelectionResult(
        selected=selected,
        method="coverage",
        coverage_score=float(len(covered)),
    )


def evaluate_leave_one_out(
    examples: list[Example],
    score_fn: Callable[[list[Example]], float],
) -> dict[str, float]:
    """For each example, compute score without it.

    Returns dict of id -> contribution (full_score - score_without).
    """
    if not examples:
        return {}

    full_score = score_fn(examples)
    result: dict[str, float] = {}
    for i, ex in enumerate(examples):
        subset = examples[:i] + examples[i + 1 :]
        score_without = score_fn(subset) if subset else 0.0
        result[ex.id] = full_score - score_without
    return result


def find_optimal_k(
    examples: list[Example],
    score_fn: Callable[[list[Example]], float],
    max_k: int = 10,
) -> int:
    """Try k=1..max_k, return k that maximizes score_fn.

    Uses the first k examples from the list for each trial.
    """
    if not examples or max_k <= 0:
        return 0

    best_k = 1
    best_score = score_fn(examples[:1])
    for k in range(2, min(max_k, len(examples)) + 1):
        s = score_fn(examples[:k])
        if s > best_score:
            best_score = s
            best_k = k
    return best_k
