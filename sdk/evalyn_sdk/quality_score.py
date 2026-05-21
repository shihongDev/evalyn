"""
Simulation quality score: evaluate generated items for naturalness.

Compares generated text to a seed set using statistical heuristics -
length distribution, vocabulary overlap, repetitiveness, and coherence.
Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class QualityDimension:
    """A single quality dimension result."""

    name: str
    score: float
    weight: float = 1.0
    details: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityDimension:
        return cls(
            name=data["name"],
            score=data["score"],
            weight=data.get("weight", 1.0),
            details=data.get("details", ""),
        )


@dataclass
class ItemQualityScore:
    """Quality assessment for a single generated item."""

    item_id: str
    overall_score: float
    dimensions: list[QualityDimension] = field(default_factory=list)
    passed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "overall_score": self.overall_score,
            "dimensions": [d.as_dict() for d in self.dimensions],
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ItemQualityScore:
        return cls(
            item_id=data["item_id"],
            overall_score=data["overall_score"],
            dimensions=[QualityDimension.from_dict(d) for d in data.get("dimensions", [])],
            passed=data.get("passed", True),
        )


@dataclass
class QualityReport:
    """Aggregated quality report for a batch of generated items."""

    scores: list[ItemQualityScore] = field(default_factory=list)
    mean_score: float = 0.0
    rejection_count: int = 0
    acceptance_rate: float = 0.0
    total_items: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "mean_score": self.mean_score,
            "rejection_count": self.rejection_count,
            "acceptance_rate": self.acceptance_rate,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityReport:
        return cls(
            scores=[ItemQualityScore.from_dict(s) for s in data.get("scores", [])],
            mean_score=data.get("mean_score", 0.0),
            rejection_count=data.get("rejection_count", 0),
            acceptance_rate=data.get("acceptance_rate", 0.0),
            total_items=data.get("total_items", 0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r"[.!?]+", text)
    return [s.strip() for s in raw if s.strip()]


def _bigrams(words: list[str]) -> list[str]:
    """Generate bigram strings from word list."""
    return [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_word_validity(text: str) -> float:
    """Fraction of words that look like real words (contain vowels, reasonable length).

    Returns 0.0 for all-gibberish, 1.0 for all valid-looking words.
    """
    words = _tokenize(text)
    if not words:
        return 0.0

    vowels = set("aeiouy")
    valid = 0
    for w in words:
        has_vowel = any(c in vowels for c in w)
        reasonable_length = 1 <= len(w) <= 25
        not_all_same = len(set(w)) > 1 or len(w) == 1
        if has_vowel and reasonable_length and not_all_same:
            valid += 1

    return round(valid / len(words), 4)


def score_length_naturalness(text: str, seed_lengths: list[int]) -> float:
    """How close text length is to the seed distribution (z-score based).

    Returns a score from 0.0 to 1.0. Closer to seed mean = higher score.
    """
    if not seed_lengths:
        return 0.5  # no reference, neutral score

    text_len = len(text)
    mean = sum(seed_lengths) / len(seed_lengths)

    if len(seed_lengths) < 2:
        # Single seed: exact match = 1.0, decay with distance
        diff = abs(text_len - mean)
        return max(0.0, 1.0 - diff / max(mean, 1.0))

    variance = sum((x - mean) ** 2 for x in seed_lengths) / len(seed_lengths)
    std = math.sqrt(variance) if variance > 0 else 1.0

    z = abs(text_len - mean) / std
    # Map z-score to 0-1: z=0 -> 1.0, z=3 -> 0.0
    score = max(0.0, 1.0 - z / 3.0)
    return round(score, 4)


def score_vocabulary_naturalness(text: str, seed_vocab: set[str]) -> float:
    """Fraction of words in text that appear in seed vocabulary.

    Returns 0.0 to 1.0. Higher = more natural vocabulary.
    """
    words = _tokenize(text)
    if not words or not seed_vocab:
        return 0.0 if not words else 0.5

    overlap = sum(1 for w in words if w in seed_vocab)
    return round(overlap / len(words), 4)


def score_repetitiveness(text: str) -> float:
    """1.0 minus repeated bigram ratio. More repetitive = lower score."""
    words = _tokenize(text)
    if len(words) < 2:
        return 1.0  # too short to be repetitive

    bgs = _bigrams(words)
    if not bgs:
        return 1.0

    total = len(bgs)
    unique = len(set(bgs))
    repeated_ratio = 1.0 - (unique / total)
    return round(1.0 - repeated_ratio, 4)


def score_coherence(text: str) -> float:
    """Heuristic coherence score based on sentence structure.

    Checks:
    - Has at least one sentence
    - No sentences start with lowercase (except after sentence boundary)
    - Reasonable average sentence length (5-30 words)
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0

    score = 1.0
    penalties = 0.0

    # Check for lowercase sentence starts (skip first sentence)
    for sent in sentences[1:]:
        if sent and sent[0].islower():
            penalties += 0.15

    # Average sentence length check
    words_per_sent = [len(_tokenize(s)) for s in sentences]
    avg_sent_len = sum(words_per_sent) / len(words_per_sent) if words_per_sent else 0

    if avg_sent_len < 3:
        penalties += 0.3
    elif avg_sent_len > 50:
        penalties += 0.2

    # Sentence count relative to text length
    total_words = sum(words_per_sent)
    if total_words > 0 and len(sentences) == 1 and total_words > 40:
        # Very long single sentence
        penalties += 0.2

    score = max(0.0, score - penalties)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_quality_score(
    text: str,
    seed_texts: list[str],
    item_id: str = "",
    threshold: float = 0.3,
) -> ItemQualityScore:
    """Compute all quality dimensions and overall score for one item."""
    # Build seed statistics
    seed_lengths = [len(t) for t in seed_texts]
    seed_vocab: set[str] = set()
    for st in seed_texts:
        seed_vocab.update(_tokenize(st))

    # Compute dimensions
    dims = [
        QualityDimension(
            name="word_validity",
            score=score_word_validity(text),
            weight=2.0,
            details="fraction of words that contain vowels and have reasonable structure",
        ),
        QualityDimension(
            name="length_naturalness",
            score=score_length_naturalness(text, seed_lengths),
            weight=1.0,
            details="z-score distance from seed length distribution",
        ),
        QualityDimension(
            name="vocabulary_naturalness",
            score=score_vocabulary_naturalness(text, seed_vocab),
            weight=1.0,
            details="fraction of words found in seed vocabulary",
        ),
        QualityDimension(
            name="repetitiveness",
            score=score_repetitiveness(text),
            weight=1.0,
            details="1 - repeated bigram ratio",
        ),
        QualityDimension(
            name="coherence",
            score=score_coherence(text),
            weight=1.0,
            details="sentence structure heuristic",
        ),
    ]

    # Weighted average
    total_weight = sum(d.weight for d in dims)
    if total_weight > 0:
        overall = sum(d.score * d.weight for d in dims) / total_weight
    else:
        overall = 0.0
    overall = round(overall, 4)

    passed = overall >= threshold

    return ItemQualityScore(
        item_id=item_id,
        overall_score=overall,
        dimensions=dims,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Batch reporting
# ---------------------------------------------------------------------------


def build_quality_report(
    gen_texts: list[str],
    seed_texts: list[str],
    threshold: float = 0.3,
) -> QualityReport:
    """Build a quality report for a batch of generated texts."""
    scores: list[ItemQualityScore] = []
    for i, text in enumerate(gen_texts):
        scores.append(compute_quality_score(text, seed_texts, item_id=str(i), threshold=threshold))

    total = len(scores)
    rejection_count = sum(1 for s in scores if not s.passed)
    mean_score = sum(s.overall_score for s in scores) / total if total else 0.0
    acceptance_rate = (total - rejection_count) / total if total else 0.0

    return QualityReport(
        scores=scores,
        mean_score=round(mean_score, 4),
        rejection_count=rejection_count,
        acceptance_rate=round(acceptance_rate, 4),
        total_items=total,
    )


def filter_high_quality(
    scores: list[ItemQualityScore],
) -> list[ItemQualityScore]:
    """Return only items that passed the quality threshold."""
    return [s for s in scores if s.passed]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_quality_report(report: QualityReport) -> str:
    """Human-readable quality report."""
    lines: list[str] = []
    lines.append("Quality Score Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Total items:     {report.total_items}")
    lines.append(f"Mean score:      {report.mean_score:.4f}")
    lines.append(f"Acceptance rate: {report.acceptance_rate:.1%}")
    lines.append(f"Rejections:      {report.rejection_count}")
    lines.append("")
    lines.append("Per-item scores:")

    for s in report.scores:
        status = "PASS" if s.passed else "FAIL"
        lines.append(f"  [{s.item_id}] {s.overall_score:.4f} {status}")
        for d in s.dimensions:
            lines.append(f"    {d.name}: {d.score:.4f} (w={d.weight})")

    return "\n".join(lines)
