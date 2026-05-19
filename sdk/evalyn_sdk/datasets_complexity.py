"""
Dataset complexity scoring: auto-compute per-item difficulty from input
features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ComplexityFeatures:
    """Linguistic features extracted from input text."""

    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    unique_word_ratio: float = 0.0
    has_code: bool = False
    has_numbers: bool = False
    question_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_word_length": self.avg_word_length,
            "unique_word_ratio": self.unique_word_ratio,
            "has_code": self.has_code,
            "has_numbers": self.has_numbers,
            "question_count": self.question_count,
        }


@dataclass
class ComplexityScore:
    """Complexity score for a single dataset item."""

    item_id: str
    score: float  # 0-1
    difficulty: str = "medium"  # "easy", "medium", "hard"
    features: Optional[ComplexityFeatures] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "difficulty": self.difficulty,
            "features": self.features.as_dict() if self.features else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComplexityScore:
        features_data = data.get("features")
        features = ComplexityFeatures(**features_data) if features_data else None
        return cls(
            item_id=data.get("item_id", ""),
            score=data.get("score", 0.0),
            difficulty=data.get("difficulty", "medium"),
            features=features,
        )


@dataclass
class ComplexityReport:
    """Aggregated complexity report for a dataset."""

    scores: List[ComplexityScore] = field(default_factory=list)
    avg_complexity: float = 0.0
    distribution: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "avg_complexity": self.avg_complexity,
            "distribution": dict(self.distribution),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComplexityReport:
        scores = [ComplexityScore.from_dict(s) for s in data.get("scores", [])]
        return cls(
            scores=scores,
            avg_complexity=data.get("avg_complexity", 0.0),
            distribution=data.get("distribution", {}),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Complexity Report")
        lines.append("=" * 40)
        lines.append(f"  Items scored: {len(self.scores)}")
        lines.append(f"  Average complexity: {self.avg_complexity:.3f}")
        if self.distribution:
            lines.append("  Distribution:")
            for difficulty in ["easy", "medium", "hard"]:
                count = self.distribution.get(difficulty, 0)
                lines.append(f"    {difficulty}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def extract_features(text: str) -> ComplexityFeatures:
    """Extract linguistic features from text."""
    if not text or not text.strip():
        return ComplexityFeatures()

    words = text.split()
    word_count = len(words)

    # Sentence count: split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if s.strip()])
    if sentence_count == 0:
        sentence_count = 1

    # Average word length
    total_chars = sum(len(w) for w in words)
    avg_word_length = total_chars / word_count if word_count > 0 else 0.0

    # Unique word ratio
    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0.0

    # Code detection: look for common code patterns
    code_patterns = [
        r"```",
        r"def\s+\w+\(",
        r"class\s+\w+",
        r"import\s+\w+",
        r"function\s+\w+",
        r"\{.*\}",
        r"=>",
        r"&&|\|\|",
        r"==|!=|<=|>=",
    ]
    has_code = any(re.search(p, text) for p in code_patterns)

    # Number detection
    has_numbers = bool(re.search(r"\d+", text))

    # Question count
    question_count = text.count("?")

    return ComplexityFeatures(
        word_count=word_count,
        sentence_count=sentence_count,
        avg_word_length=avg_word_length,
        unique_word_ratio=unique_word_ratio,
        has_code=has_code,
        has_numbers=has_numbers,
        question_count=question_count,
    )


def compute_complexity_score(features: ComplexityFeatures) -> float:
    """
    Weighted combination of features normalized to 0-1.

    More words, longer words, code, multiple questions = higher complexity.
    """
    # Word count contribution: scales up to ~100 words, then saturates
    word_score = min(features.word_count / 100.0, 1.0)

    # Average word length: typical range 3-8 chars
    length_score = min(max(features.avg_word_length - 3.0, 0.0) / 5.0, 1.0)

    # Unique word ratio: higher ratio = more diverse vocabulary = harder
    vocab_score = features.unique_word_ratio

    # Code presence
    code_score = 1.0 if features.has_code else 0.0

    # Number presence
    number_score = 0.5 if features.has_numbers else 0.0

    # Question count: more questions = more complex
    question_score = min(features.question_count / 3.0, 1.0)

    # Sentence count contribution
    sentence_score = min(features.sentence_count / 5.0, 1.0)

    # Weighted combination
    score = (
        word_score * 0.20
        + length_score * 0.10
        + vocab_score * 0.15
        + code_score * 0.20
        + number_score * 0.10
        + question_score * 0.15
        + sentence_score * 0.10
    )

    return min(max(score, 0.0), 1.0)


def classify_difficulty(score: float) -> str:
    """Classify score into difficulty bucket."""
    if score < 0.33:
        return "easy"
    if score < 0.66:
        return "medium"
    return "hard"


def score_item(item: Dict[str, Any], input_field: str = "input") -> ComplexityScore:
    """Score one item for complexity."""
    item_id = str(item.get("id", ""))
    text = str(item.get(input_field, ""))
    features = extract_features(text)
    score = compute_complexity_score(features)
    difficulty = classify_difficulty(score)
    return ComplexityScore(
        item_id=item_id,
        score=score,
        difficulty=difficulty,
        features=features,
    )


def score_dataset(
    items: List[Dict[str, Any]],
    input_field: str = "input",
) -> ComplexityReport:
    """Score all items and compute distribution."""
    scores: List[ComplexityScore] = []
    for item in items:
        scores.append(score_item(item, input_field=input_field))

    avg_complexity = 0.0
    if scores:
        avg_complexity = sum(s.score for s in scores) / len(scores)

    distribution: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for s in scores:
        distribution[s.difficulty] = distribution.get(s.difficulty, 0) + 1

    return ComplexityReport(
        scores=scores,
        avg_complexity=avg_complexity,
        distribution=distribution,
    )


def sort_by_complexity(
    scores: List[ComplexityScore],
    ascending: bool = True,
) -> List[ComplexityScore]:
    """Sort scores by complexity score."""
    return sorted(scores, key=lambda s: s.score, reverse=not ascending)
