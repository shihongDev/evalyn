"""Calibration difficulty weighting.

Weight alignment errors by item difficulty so hard items count more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class WeightedItem:
    """A single scored item with difficulty and weight."""

    item_id: str
    score: float
    ground_truth: bool
    difficulty: float  # 0-1
    weight: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "ground_truth": self.ground_truth,
            "difficulty": self.difficulty,
            "weight": self.weight,
        }


@dataclass
class WeightedAlignmentResult:
    """Result of weighted accuracy computation."""

    weighted_accuracy: float
    unweighted_accuracy: float
    num_items: int
    total_weight: float
    hard_item_accuracy: float
    easy_item_accuracy: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "weighted_accuracy": self.weighted_accuracy,
            "unweighted_accuracy": self.unweighted_accuracy,
            "num_items": self.num_items,
            "total_weight": self.total_weight,
            "hard_item_accuracy": self.hard_item_accuracy,
            "easy_item_accuracy": self.easy_item_accuracy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WeightedAlignmentResult:
        return cls(
            weighted_accuracy=data["weighted_accuracy"],
            unweighted_accuracy=data["unweighted_accuracy"],
            num_items=data["num_items"],
            total_weight=data["total_weight"],
            hard_item_accuracy=data["hard_item_accuracy"],
            easy_item_accuracy=data["easy_item_accuracy"],
        )

    def format_text(self) -> str:
        lines = [
            "Weighted Alignment",
            f"  Weighted accuracy:   {self.weighted_accuracy:.4f}",
            f"  Unweighted accuracy: {self.unweighted_accuracy:.4f}",
            f"  Items: {self.num_items}  Total weight: {self.total_weight:.2f}",
            f"  Hard item accuracy:  {self.hard_item_accuracy:.4f}",
            f"  Easy item accuracy:  {self.easy_item_accuracy:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def assign_difficulty_weights(
    items: List[WeightedItem],
    method: str = "inverse",
) -> List[WeightedItem]:
    """Assign weights based on difficulty.

    Methods:
      - "inverse": weight = 1 + difficulty (harder items get more weight)
      - "exponential": weight = 2 ** difficulty
      - "equal": weight = 1 for all
    """
    for item in items:
        if method == "inverse":
            item.weight = 1.0 + item.difficulty
        elif method == "exponential":
            item.weight = 2.0 ** item.difficulty
        elif method == "equal":
            item.weight = 1.0
        else:
            item.weight = 1.0
    return items


def compute_weighted_accuracy(
    items: List[WeightedItem],
    threshold: float = 0.5,
) -> WeightedAlignmentResult:
    """Weighted accuracy where each item's correctness is multiplied by weight.

    Correctness: (score >= threshold) == ground_truth.
    Also computes unweighted accuracy and hard/easy split.
    """
    if not items:
        return WeightedAlignmentResult(
            weighted_accuracy=0.0,
            unweighted_accuracy=0.0,
            num_items=0,
            total_weight=0.0,
            hard_item_accuracy=0.0,
            easy_item_accuracy=0.0,
        )

    total_weight = 0.0
    weighted_correct = 0.0
    correct_count = 0

    hard_total = 0
    hard_correct = 0
    easy_total = 0
    easy_correct = 0

    for item in items:
        predicted = item.score >= threshold
        is_correct = predicted == item.ground_truth

        total_weight += item.weight
        if is_correct:
            weighted_correct += item.weight
            correct_count += 1

        if item.difficulty > 0.5:
            hard_total += 1
            if is_correct:
                hard_correct += 1
        else:
            easy_total += 1
            if is_correct:
                easy_correct += 1

    weighted_acc = weighted_correct / total_weight if total_weight > 0 else 0.0
    unweighted_acc = correct_count / len(items)
    hard_acc = hard_correct / hard_total if hard_total > 0 else 0.0
    easy_acc = easy_correct / easy_total if easy_total > 0 else 0.0

    return WeightedAlignmentResult(
        weighted_accuracy=weighted_acc,
        unweighted_accuracy=unweighted_acc,
        num_items=len(items),
        total_weight=total_weight,
        hard_item_accuracy=hard_acc,
        easy_item_accuracy=easy_acc,
    )


def identify_hard_failures(
    items: List[WeightedItem],
    threshold: float = 0.5,
) -> List[WeightedItem]:
    """Return items that are hard (difficulty > 0.5) AND incorrectly predicted.

    These are highest priority for calibration attention.
    """
    failures: List[WeightedItem] = []
    for item in items:
        if item.difficulty <= 0.5:
            continue
        predicted = item.score >= threshold
        if predicted != item.ground_truth:
            failures.append(item)
    return failures
