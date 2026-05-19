"""
Seed selection optimization: choose which seed items produce the most diverse simulations.

Pure Python, no external dependencies. Uses word-set-based Jaccard distance
as a lightweight proxy for semantic diversity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Stop words (common English words to ignore in text analysis)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "are", "be",
    "this", "that", "not", "has", "had", "have", "been", "will", "can",
    "do", "does", "did", "if", "so", "no", "up", "out", "its", "all",
    "my", "we", "he", "she", "they", "you", "me", "us", "him", "her",
    "who", "what", "when", "how", "which", "where", "there", "here",
    "then", "than", "also", "just", "about", "into", "over", "after",
    "more", "some", "any", "each", "very", "too", "own", "same",
})


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SeedScore:
    """Score for a single seed based on diversity of its generated outputs."""

    seed_id: str
    diversity_score: float
    unique_outputs: int
    near_duplicate_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "diversity_score": self.diversity_score,
            "unique_outputs": self.unique_outputs,
            "near_duplicate_count": self.near_duplicate_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeedScore:
        return cls(
            seed_id=data["seed_id"],
            diversity_score=data["diversity_score"],
            unique_outputs=data["unique_outputs"],
            near_duplicate_count=data["near_duplicate_count"],
        )


@dataclass
class SelectionResult:
    """Result of greedy seed selection."""

    selected_seeds: list[str]
    dropped_seeds: list[str]
    scores: list[SeedScore]
    coverage_improvement: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_seeds": list(self.selected_seeds),
            "dropped_seeds": list(self.dropped_seeds),
            "scores": [s.as_dict() for s in self.scores],
            "coverage_improvement": self.coverage_improvement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectionResult:
        return cls(
            selected_seeds=data["selected_seeds"],
            dropped_seeds=data["dropped_seeds"],
            scores=[SeedScore.from_dict(s) for s in data["scores"]],
            coverage_improvement=data["coverage_improvement"],
        )


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on whitespace, filter short words and stop words."""
    return {
        w for w in text.lower().split()
        if len(w) >= 3 and w not in _STOP_WORDS
    }


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both are empty."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 1.0
    intersection = len(a & b)
    return intersection / union


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def compute_text_diversity(texts: list[str]) -> float:
    """Average pairwise Jaccard distance (1 - similarity) of word sets.

    Returns 0.0 for fewer than 2 texts.
    """
    if len(texts) < 2:
        return 0.0

    word_sets = [_tokenize(t) for t in texts]
    n = len(word_sets)
    total_distance = 0.0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaccard_similarity(word_sets[i], word_sets[j])
            total_distance += 1.0 - sim
            pair_count += 1

    if pair_count == 0:
        return 0.0
    return total_distance / pair_count


def score_seed(seed_id: str, generated_outputs: list[str]) -> SeedScore:
    """Score a seed by diversity of its outputs and near-duplicate detection.

    Near-duplicates are pairs with Jaccard similarity > 0.9.
    """
    diversity = compute_text_diversity(generated_outputs)

    # Count unique outputs (by exact text)
    unique_outputs = len(set(generated_outputs))

    # Count near-duplicate pairs
    word_sets = [_tokenize(t) for t in generated_outputs]
    near_dup_count = 0
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            if _jaccard_similarity(word_sets[i], word_sets[j]) > 0.9:
                near_dup_count += 1

    return SeedScore(
        seed_id=seed_id,
        diversity_score=diversity,
        unique_outputs=unique_outputs,
        near_duplicate_count=near_dup_count,
    )


def greedy_select(
    seeds: dict[str, list[str]],
    max_seeds: int,
    min_diversity: float = 0.1,
) -> SelectionResult:
    """Greedily pick seeds that maximize cumulative word coverage.

    At each step, pick the seed whose outputs add the most new unique words
    to the already-selected set. Drop seeds below min_diversity.

    Args:
        seeds: mapping of seed_id to list of generated output texts
        max_seeds: maximum number of seeds to select
        min_diversity: minimum diversity score to keep a seed

    Returns:
        SelectionResult with selected/dropped seeds, scores, and coverage improvement.
    """
    if not seeds:
        return SelectionResult(
            selected_seeds=[],
            dropped_seeds=[],
            scores=[],
            coverage_improvement=0.0,
        )

    # Pre-compute word sets for each seed (union of all output word sets)
    seed_word_sets: dict[str, set[str]] = {}
    for sid, outputs in seeds.items():
        combined: set[str] = set()
        for text in outputs:
            combined |= _tokenize(text)
        seed_word_sets[sid] = combined

    # Pre-compute scores for all seeds
    all_scores: dict[str, SeedScore] = {}
    for sid, outputs in seeds.items():
        all_scores[sid] = score_seed(sid, outputs)

    # Total vocabulary across all seeds
    total_vocab: set[str] = set()
    for ws in seed_word_sets.values():
        total_vocab |= ws

    # Greedy selection
    selected: list[str] = []
    dropped: list[str] = []
    covered_words: set[str] = set()
    remaining = set(seeds.keys())

    while len(selected) < max_seeds and remaining:
        # Pick the seed that adds the most new words
        best_sid = None
        best_new_count = -1
        for sid in remaining:
            new_count = len(seed_word_sets[sid] - covered_words)
            if new_count > best_new_count:
                best_new_count = new_count
                best_sid = sid

        if best_sid is None:
            break

        remaining.discard(best_sid)

        # Check diversity threshold
        if all_scores[best_sid].diversity_score < min_diversity:
            dropped.append(best_sid)
            continue

        selected.append(best_sid)
        covered_words |= seed_word_sets[best_sid]

    # All remaining seeds that were not selected are dropped
    for sid in remaining:
        dropped.append(sid)

    # Coverage improvement: fraction of total vocab covered by selected seeds
    if total_vocab:
        coverage_improvement = len(covered_words) / len(total_vocab)
    else:
        coverage_improvement = 0.0

    # Collect scores for selected + dropped, in selection order then drop order
    score_list = [all_scores[sid] for sid in selected + dropped]

    return SelectionResult(
        selected_seeds=selected,
        dropped_seeds=dropped,
        scores=score_list,
        coverage_improvement=coverage_improvement,
    )


def detect_redundant_seeds(
    seeds: dict[str, list[str]],
    threshold: float = 0.8,
) -> list[tuple[str, str]]:
    """Find seed pairs whose output sets are too similar.

    Computes Jaccard similarity of the combined word sets of each seed's outputs.
    Returns pairs where similarity exceeds the threshold.
    """
    # Build combined word set per seed
    seed_word_sets: dict[str, set[str]] = {}
    for sid, outputs in seeds.items():
        combined: set[str] = set()
        for text in outputs:
            combined |= _tokenize(text)
        seed_word_sets[sid] = combined

    seed_ids = sorted(seeds.keys())
    redundant_pairs: list[tuple[str, str]] = []

    for i in range(len(seed_ids)):
        for j in range(i + 1, len(seed_ids)):
            sid_a = seed_ids[i]
            sid_b = seed_ids[j]
            sim = _jaccard_similarity(seed_word_sets[sid_a], seed_word_sets[sid_b])
            if sim > threshold:
                redundant_pairs.append((sid_a, sid_b))

    return redundant_pairs


def format_selection_report(result: SelectionResult) -> str:
    """Format a human-readable seed selection report."""
    lines: list[str] = []
    lines.append("SEED SELECTION REPORT")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Selected: {len(result.selected_seeds)} seeds")
    lines.append(f"Dropped:  {len(result.dropped_seeds)} seeds")
    lines.append(f"Coverage: {result.coverage_improvement:.1%}")
    lines.append("")

    if result.selected_seeds:
        lines.append("Selected Seeds:")
        for sid in result.selected_seeds:
            score = next((s for s in result.scores if s.seed_id == sid), None)
            if score:
                lines.append(
                    f"  {sid}: diversity={score.diversity_score:.3f}, "
                    f"unique={score.unique_outputs}, "
                    f"near_dups={score.near_duplicate_count}"
                )
            else:
                lines.append(f"  {sid}")
        lines.append("")

    if result.dropped_seeds:
        lines.append("Dropped Seeds:")
        for sid in result.dropped_seeds:
            score = next((s for s in result.scores if s.seed_id == sid), None)
            if score:
                lines.append(
                    f"  {sid}: diversity={score.diversity_score:.3f}"
                )
            else:
                lines.append(f"  {sid}")
        lines.append("")

    return "\n".join(lines)
