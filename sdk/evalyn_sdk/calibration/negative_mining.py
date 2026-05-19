"""Calibration negative example mining.

Find the hardest examples where a calibrated prompt still fails.
Provides dataclasses and pure functions for mining, ranking, and
analyzing failure cases.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class NegativeExample:
    """A single failure case where prediction disagrees with ground truth."""

    item_id: str
    predicted_score: float
    ground_truth: bool
    error_magnitude: float
    difficulty: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "predicted_score": self.predicted_score,
            "ground_truth": self.ground_truth,
            "error_magnitude": self.error_magnitude,
            "difficulty": self.difficulty,
        }


@dataclass
class MiningResult:
    """Result of a negative mining pass."""

    negatives: list[NegativeExample]
    total_failures: int
    hardest_failures: list[NegativeExample]
    failure_rate: float
    common_patterns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "negatives": [n.as_dict() for n in self.negatives],
            "total_failures": self.total_failures,
            "hardest_failures": [n.as_dict() for n in self.hardest_failures],
            "failure_rate": self.failure_rate,
            "common_patterns": self.common_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MiningResult:
        negatives = [
            NegativeExample(**n) for n in data.get("negatives", [])
        ]
        hardest = [
            NegativeExample(**n) for n in data.get("hardest_failures", [])
        ]
        return cls(
            negatives=negatives,
            total_failures=data.get("total_failures", 0),
            hardest_failures=hardest,
            failure_rate=data.get("failure_rate", 0.0),
            common_patterns=data.get("common_patterns", []),
        )

    def format_text(self) -> str:
        lines = [
            "Negative Mining Results",
            f"  Total failures: {self.total_failures}",
            f"  Failure rate:   {self.failure_rate:.2%}",
        ]
        if self.hardest_failures:
            lines.append("  Hardest failures:")
            for neg in self.hardest_failures:
                lines.append(
                    f"    - {neg.item_id}: error={neg.error_magnitude:.3f}"
                )
        if self.common_patterns:
            lines.append("  Common patterns:")
            for pattern in self.common_patterns:
                lines.append(f"    - {pattern}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def find_negatives(
    predictions: list[tuple[str, float, bool]],
    threshold: float = 0.5,
) -> list[NegativeExample]:
    """Find items where prediction disagrees with ground truth.

    Args:
        predictions: list of (item_id, predicted_score, ground_truth) tuples.
        threshold: score threshold for binary classification.

    Returns:
        List of NegativeExample for each disagreement.
    """
    negatives: list[NegativeExample] = []
    for item_id, score, truth in predictions:
        predicted_positive = score >= threshold
        if predicted_positive != truth:
            error_mag = abs(score - (1.0 if truth else 0.0))
            negatives.append(
                NegativeExample(
                    item_id=item_id,
                    predicted_score=score,
                    ground_truth=truth,
                    error_magnitude=error_mag,
                )
            )
    return negatives


def rank_by_difficulty(
    negatives: list[NegativeExample],
) -> list[NegativeExample]:
    """Sort negatives by error_magnitude descending (most wrong first)."""
    return sorted(negatives, key=lambda n: n.error_magnitude, reverse=True)


def mine_hard_negatives(
    predictions: list[tuple[str, float, bool]],
    top_k: int = 10,
) -> MiningResult:
    """Find negatives, rank by difficulty, return top_k hardest.

    Args:
        predictions: list of (item_id, predicted_score, ground_truth) tuples.
        top_k: number of hardest failures to keep.

    Returns:
        MiningResult with ranked negatives and summary stats.
    """
    all_negatives = find_negatives(predictions)
    ranked = rank_by_difficulty(all_negatives)
    total = len(predictions)
    failure_rate = len(all_negatives) / total if total > 0 else 0.0
    hardest = ranked[:top_k]
    return MiningResult(
        negatives=all_negatives,
        total_failures=len(all_negatives),
        hardest_failures=hardest,
        failure_rate=failure_rate,
    )


def analyze_failure_patterns(
    negatives: list[NegativeExample],
    item_texts: dict[str, str] | None = None,
) -> list[str]:
    """Simple pattern analysis on failure cases.

    If item_texts is provided, find common words across failure texts.
    Returns a list of pattern description strings.
    """
    if item_texts is None:
        item_texts = {}

    patterns: list[str] = []

    if not negatives:
        return patterns

    # Count false positives vs false negatives
    fp_count = sum(1 for n in negatives if not n.ground_truth)
    fn_count = sum(1 for n in negatives if n.ground_truth)
    if fp_count > fn_count:
        patterns.append(f"More false positives ({fp_count}) than false negatives ({fn_count})")
    elif fn_count > fp_count:
        patterns.append(f"More false negatives ({fn_count}) than false positives ({fp_count})")
    else:
        patterns.append(f"Equal false positives and false negatives ({fp_count})")

    # Word frequency analysis across failure texts
    failure_ids = {n.item_id for n in negatives}
    texts = [item_texts[fid] for fid in failure_ids if fid in item_texts]
    if texts:
        word_counts: Counter[str] = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(set(words))  # unique per text
        # Words appearing in majority of failure texts
        threshold = len(texts) / 2
        common = [
            word for word, count in word_counts.most_common()
            if count > threshold and len(word) > 2
        ]
        if common:
            top_words = common[:5]
            patterns.append(f"Common words in failures: {', '.join(top_words)}")

    return patterns
