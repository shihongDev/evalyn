"""
Simulation diversity metrics: quantify how diverse a generated set is.

Computes vocabulary spread, pairwise distance, uniqueness, novelty,
and length diversity for collections of generated text.
Pure Python - no external deps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DiversityScore:
    """A single diversity metric result."""

    metric_name: str
    score: float
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiversityScore:
        return cls(
            metric_name=data["metric_name"],
            score=data["score"],
            details=data["details"],
        )


@dataclass
class DiversityReport:
    """Aggregated diversity report across multiple metrics."""

    scores: list[DiversityScore] = field(default_factory=list)
    overall_diversity: float = 0.0
    total_items: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "overall_diversity": self.overall_diversity,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiversityReport:
        return cls(
            scores=[DiversityScore.from_dict(s) for s in data.get("scores", [])],
            overall_diversity=data.get("overall_diversity", 0.0),
            total_items=data.get("total_items", 0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _word_set(text: str) -> set:
    """Return the set of unique lowercase words."""
    return set(_tokenize(text))


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _jaccard_distance(set_a: set, set_b: set) -> float:
    """Jaccard distance = 1 - Jaccard similarity."""
    return 1.0 - _jaccard_similarity(set_a, set_b)


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------


def compute_vocabulary_spread(texts: list[str]) -> float:
    """Type-token ratio: unique words / total words.

    Returns 0.0 for empty input.
    """
    if not texts:
        return 0.0
    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(_tokenize(t))
    if not all_tokens:
        return 0.0
    unique = set(all_tokens)
    return len(unique) / len(all_tokens)


def compute_pairwise_distance(texts: list[str], sample_size: int = 100) -> float:
    """Average pairwise Jaccard distance of word sets.

    If more than sample_size items, randomly sample pairs.
    Returns 0.0 for fewer than 2 items.
    """
    if len(texts) < 2:
        return 0.0

    word_sets = [_word_set(t) for t in texts]
    n = len(word_sets)

    # Build all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if len(all_pairs) > sample_size:
        rng = random.Random(len(texts))
        pairs = rng.sample(all_pairs, sample_size)
    else:
        pairs = all_pairs

    if not pairs:
        return 0.0

    total_dist = sum(_jaccard_distance(word_sets[i], word_sets[j]) for i, j in pairs)
    return total_dist / len(pairs)


def compute_uniqueness_ratio(gen_texts: list[str], seed_texts: list[str]) -> float:
    """Fraction of generated items whose word set has < 0.5 Jaccard overlap with any seed.

    An item is "unique" if its max Jaccard similarity to all seeds is below 0.5.
    Returns 1.0 if no seed texts are provided.
    """
    if not gen_texts:
        return 0.0
    if not seed_texts:
        return 1.0

    seed_sets = [_word_set(t) for t in seed_texts]
    unique_count = 0

    for gt in gen_texts:
        gen_set = _word_set(gt)
        max_sim = max(_jaccard_similarity(gen_set, ss) for ss in seed_sets)
        if max_sim < 0.5:
            unique_count += 1

    return unique_count / len(gen_texts)


def compute_novelty_score(
    gen_texts: list[str],
    seed_texts: list[str],
    threshold: float = 0.3,
) -> float:
    """Fraction of gen items where max Jaccard similarity to any seed < threshold.

    Returns 1.0 if no seed texts are provided.
    """
    if not gen_texts:
        return 0.0
    if not seed_texts:
        return 1.0

    seed_sets = [_word_set(t) for t in seed_texts]
    novel_count = 0

    for gt in gen_texts:
        gen_set = _word_set(gt)
        max_sim = max(_jaccard_similarity(gen_set, ss) for ss in seed_sets)
        if max_sim < threshold:
            novel_count += 1

    return novel_count / len(gen_texts)


def compute_length_diversity(texts: list[str]) -> float:
    """Coefficient of variation of text lengths.

    CV = stdev / mean. Returns 0.0 for empty or single-item input.
    """
    if len(texts) < 2:
        return 0.0

    lengths = [len(t) for t in texts]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.0

    variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
    stdev = math.sqrt(variance)
    return stdev / mean_len


# ---------------------------------------------------------------------------
# Report Building
# ---------------------------------------------------------------------------


def build_diversity_report(
    gen_texts: list[str],
    seed_texts: list[str],
) -> DiversityReport:
    """Compute all diversity metrics and return an aggregated report.

    Overall diversity is the mean of all individual metric scores.
    """
    scores: list[DiversityScore] = []

    vocab = compute_vocabulary_spread(gen_texts)
    scores.append(DiversityScore(
        metric_name="vocabulary_spread",
        score=vocab,
        details=f"Type-token ratio across {len(gen_texts)} texts",
    ))

    pairwise = compute_pairwise_distance(gen_texts)
    scores.append(DiversityScore(
        metric_name="pairwise_distance",
        score=pairwise,
        details=f"Average Jaccard distance across {len(gen_texts)} texts",
    ))

    uniqueness = compute_uniqueness_ratio(gen_texts, seed_texts)
    scores.append(DiversityScore(
        metric_name="uniqueness_ratio",
        score=uniqueness,
        details=f"Fraction unique vs {len(seed_texts)} seeds (threshold 0.5)",
    ))

    novelty = compute_novelty_score(gen_texts, seed_texts)
    scores.append(DiversityScore(
        metric_name="novelty_score",
        score=novelty,
        details=f"Fraction novel vs {len(seed_texts)} seeds (threshold 0.3)",
    ))

    length_div = compute_length_diversity(gen_texts)
    scores.append(DiversityScore(
        metric_name="length_diversity",
        score=length_div,
        details=f"CV of lengths across {len(gen_texts)} texts",
    ))

    if scores:
        overall = sum(s.score for s in scores) / len(scores)
    else:
        overall = 0.0

    return DiversityReport(
        scores=scores,
        overall_diversity=overall,
        total_items=len(gen_texts),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_diversity_report(report: DiversityReport) -> str:
    """Format a DiversityReport as human-readable text."""
    lines: list[str] = []
    lines.append(f"Diversity Report ({report.total_items} items)")
    lines.append("-" * 40)

    for s in report.scores:
        lines.append(f"  {s.metric_name}: {s.score:.4f}")
        lines.append(f"    {s.details}")

    lines.append("-" * 40)
    lines.append(f"Overall diversity: {report.overall_diversity:.4f}")
    return "\n".join(lines)


def compare_diversity(report_a: DiversityReport, report_b: DiversityReport) -> str:
    """Compare two diversity reports and return a human-readable summary."""
    lines: list[str] = []
    lines.append("Diversity Comparison")
    lines.append("=" * 50)
    lines.append(f"  Report A: {report_a.total_items} items, overall={report_a.overall_diversity:.4f}")
    lines.append(f"  Report B: {report_b.total_items} items, overall={report_b.overall_diversity:.4f}")
    lines.append("")

    # Build lookup for B
    b_lookup: dict[str, float] = {s.metric_name: s.score for s in report_b.scores}

    for sa in report_a.scores:
        sb_score = b_lookup.get(sa.metric_name)
        if sb_score is not None:
            diff = sb_score - sa.score
            direction = "higher" if diff > 0 else "lower" if diff < 0 else "same"
            lines.append(f"  {sa.metric_name}: A={sa.score:.4f}  B={sb_score:.4f}  ({direction}, delta={diff:+.4f})")
        else:
            lines.append(f"  {sa.metric_name}: A={sa.score:.4f}  B=N/A")

    overall_diff = report_b.overall_diversity - report_a.overall_diversity
    direction = "higher" if overall_diff > 0 else "lower" if overall_diff < 0 else "same"
    lines.append("")
    lines.append(f"Overall: B is {direction} by {abs(overall_diff):.4f}")
    return "\n".join(lines)
