"""
Simulation difficulty grading: auto-tag generated items with estimated difficulty.

Computes linguistic complexity factors (word count, vocabulary richness,
conditional/negation density) and assigns easy/medium/hard grades.
Pure Python - no external deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DifficultyFactors:
    """Raw linguistic complexity factors extracted from text."""

    word_count: int
    avg_word_length: float
    unique_word_ratio: float
    sentence_count: int
    question_marks: int
    conditional_count: int
    negation_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "word_count": self.word_count,
            "avg_word_length": self.avg_word_length,
            "unique_word_ratio": self.unique_word_ratio,
            "sentence_count": self.sentence_count,
            "question_marks": self.question_marks,
            "conditional_count": self.conditional_count,
            "negation_count": self.negation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifficultyFactors:
        return cls(
            word_count=data["word_count"],
            avg_word_length=data["avg_word_length"],
            unique_word_ratio=data["unique_word_ratio"],
            sentence_count=data["sentence_count"],
            question_marks=data["question_marks"],
            conditional_count=data["conditional_count"],
            negation_count=data["negation_count"],
        )


@dataclass
class DifficultyGrade:
    """Difficulty assessment for a single item."""

    item_id: str
    level: str  # "easy", "medium", "hard"
    score: float  # 0.0 to 1.0
    factors: DifficultyFactors
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "level": self.level,
            "score": self.score,
            "factors": self.factors.as_dict(),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifficultyGrade:
        return cls(
            item_id=data["item_id"],
            level=data["level"],
            score=data["score"],
            factors=DifficultyFactors.from_dict(data["factors"]),
            tags=data.get("tags", []),
        )


@dataclass
class DifficultyDistribution:
    """Count of items per difficulty level."""

    easy_count: int
    medium_count: int
    hard_count: int
    total: int
    balanced: bool  # within 20% of uniform distribution

    def as_dict(self) -> dict[str, Any]:
        return {
            "easy_count": self.easy_count,
            "medium_count": self.medium_count,
            "hard_count": self.hard_count,
            "total": self.total,
            "balanced": self.balanced,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifficultyDistribution:
        return cls(
            easy_count=data["easy_count"],
            medium_count=data["medium_count"],
            hard_count=data["hard_count"],
            total=data["total"],
            balanced=data["balanced"],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONDITIONAL_PATTERN = re.compile(r"\b(if|unless|when)\b", re.IGNORECASE)
_NEGATION_PATTERN = re.compile(r"\b(not|no|never|\w+n['\u2019]t)\b", re.IGNORECASE)
_SENTENCE_TERMINATORS = re.compile(r"[.!?]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w]


# ---------------------------------------------------------------------------
# Factor extraction
# ---------------------------------------------------------------------------


def extract_factors(text: str) -> DifficultyFactors:
    """Compute all linguistic complexity factors from text."""
    words = _tokenize(text)
    word_count = len(words)

    if word_count == 0:
        return DifficultyFactors(
            word_count=0,
            avg_word_length=0.0,
            unique_word_ratio=0.0,
            sentence_count=0,
            question_marks=0,
            conditional_count=0,
            negation_count=0,
        )

    avg_word_length = sum(len(w) for w in words) / word_count
    unique_word_ratio = len(set(words)) / word_count

    # Sentence count: split on terminators, filter empty
    raw_sentences = _SENTENCE_TERMINATORS.split(text)
    sentence_count = sum(1 for s in raw_sentences if s.strip())

    question_marks = text.count("?")
    conditional_count = len(_CONDITIONAL_PATTERN.findall(text))
    negation_count = len(_NEGATION_PATTERN.findall(text))

    return DifficultyFactors(
        word_count=word_count,
        avg_word_length=round(avg_word_length, 4),
        unique_word_ratio=round(unique_word_ratio, 4),
        sentence_count=sentence_count,
        question_marks=question_marks,
        conditional_count=conditional_count,
        negation_count=negation_count,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Normalization ceilings: values at or above these map to 1.0
_NORM_CEILINGS = {
    "word_count": 200,
    "avg_word_length": 8.0,
    "unique_word_ratio": 1.0,
    "sentence_count": 15,
    "question_marks": 5,
    "conditional_count": 5,
    "negation_count": 5,
}

_WEIGHTS = {
    "word_count": 0.20,
    "avg_word_length": 0.15,
    "unique_word_ratio": 0.15,
    "sentence_count": 0.10,
    "question_marks": 0.10,
    "conditional_count": 0.15,
    "negation_count": 0.15,
}


def compute_difficulty_score(factors: DifficultyFactors) -> float:
    """Weighted difficulty score from 0.0 (easy) to 1.0 (hard)."""
    vals = {
        "word_count": factors.word_count,
        "avg_word_length": factors.avg_word_length,
        "unique_word_ratio": factors.unique_word_ratio,
        "sentence_count": factors.sentence_count,
        "question_marks": factors.question_marks,
        "conditional_count": factors.conditional_count,
        "negation_count": factors.negation_count,
    }
    score = 0.0
    for key, raw in vals.items():
        normalized = min(raw / _NORM_CEILINGS[key], 1.0) if _NORM_CEILINGS[key] else 0.0
        score += normalized * _WEIGHTS[key]
    return round(min(max(score, 0.0), 1.0), 4)


def _assign_tags(factors: DifficultyFactors) -> list[str]:
    """Generate descriptive tags based on factors."""
    tags: list[str] = []
    if factors.word_count > 150:
        tags.append("long")
    if factors.word_count < 20:
        tags.append("short")
    if factors.conditional_count >= 3:
        tags.append("conditional-heavy")
    if factors.negation_count >= 3:
        tags.append("negation-heavy")
    if factors.question_marks >= 2:
        tags.append("multi-question")
    if factors.unique_word_ratio > 0.9:
        tags.append("high-vocabulary")
    if factors.avg_word_length > 6.0:
        tags.append("complex-words")
    return tags


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade_difficulty(text: str, item_id: str = "") -> DifficultyGrade:
    """Grade a single text item's difficulty."""
    factors = extract_factors(text)
    score = compute_difficulty_score(factors)

    if score < 0.33:
        level = "easy"
    elif score < 0.66:
        level = "medium"
    else:
        level = "hard"

    tags = _assign_tags(factors)

    return DifficultyGrade(
        item_id=item_id,
        level=level,
        score=score,
        factors=factors,
        tags=tags,
    )


def grade_batch(
    items: list[dict[str, Any]], text_field: str = "input"
) -> list[DifficultyGrade]:
    """Grade a batch of items. Each item is a dict with at least text_field."""
    grades: list[DifficultyGrade] = []
    for i, item in enumerate(items):
        text = item.get(text_field, "")
        item_id = item.get("id", str(i))
        grades.append(grade_difficulty(text, item_id=item_id))
    return grades


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------


def compute_distribution(grades: list[DifficultyGrade]) -> DifficultyDistribution:
    """Compute the distribution of difficulty levels."""
    easy = sum(1 for g in grades if g.level == "easy")
    medium = sum(1 for g in grades if g.level == "medium")
    hard = sum(1 for g in grades if g.level == "hard")
    total = len(grades)

    # Balanced: each bucket within 20% of uniform (total/3)
    balanced = False
    if total > 0:
        target = total / 3.0
        threshold = target * 0.2
        balanced = all(
            abs(count - target) <= threshold for count in [easy, medium, hard]
        )

    return DifficultyDistribution(
        easy_count=easy,
        medium_count=medium,
        hard_count=hard,
        total=total,
        balanced=balanced,
    )


def rebalance_suggestions(
    distribution: DifficultyDistribution,
) -> list[str]:
    """Suggest which difficulty levels need more items."""
    if distribution.total == 0:
        return ["Add items at all difficulty levels."]

    target = distribution.total / 3.0
    suggestions: list[str] = []

    counts = {
        "easy": distribution.easy_count,
        "medium": distribution.medium_count,
        "hard": distribution.hard_count,
    }

    for level, count in counts.items():
        deficit = target - count
        if deficit > 0:
            suggestions.append(
                f"Add more {level} items (currently {count}, target ~{target:.0f})."
            )

    if not suggestions:
        suggestions.append("Distribution is already balanced.")

    return suggestions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_difficulty_report(
    grades: list[DifficultyGrade],
    distribution: DifficultyDistribution,
) -> str:
    """Human-readable difficulty report."""
    lines: list[str] = []
    lines.append("Difficulty Grading Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Distribution:")
    lines.append(f"  Easy:   {distribution.easy_count}")
    lines.append(f"  Medium: {distribution.medium_count}")
    lines.append(f"  Hard:   {distribution.hard_count}")
    lines.append(f"  Total:  {distribution.total}")
    lines.append(f"  Balanced: {'yes' if distribution.balanced else 'no'}")
    lines.append("")
    lines.append("Items:")

    for g in grades:
        tag_str = ", ".join(g.tags) if g.tags else "none"
        lines.append(
            f"  [{g.item_id}] {g.level} (score={g.score:.2f}) tags={tag_str}"
        )

    suggestions = rebalance_suggestions(distribution)
    if suggestions:
        lines.append("")
        lines.append("Suggestions:")
        for s in suggestions:
            lines.append(f"  - {s}")

    return "\n".join(lines)
