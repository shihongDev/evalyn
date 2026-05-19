"""Similarity-based sampling for evaluation datasets.

Sample items most or least similar to a reference using word-set Jaccard
similarity as a proxy. Pure Python, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SimilarityScore:
    """Similarity score between an item and a reference."""

    item_id: str
    similarity: float  # 0 to 1
    distance: float  # 1 - similarity

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "similarity": self.similarity,
            "distance": self.distance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimilarityScore:
        return cls(
            item_id=data["item_id"],
            similarity=data["similarity"],
            distance=data["distance"],
        )


@dataclass
class SimilarityConfig:
    """Configuration for similarity-based sampling."""

    mode: str = "similar"  # "similar" or "dissimilar"
    sample_size: int = 20
    reference_id: str = ""
    reference_text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "sample_size": self.sample_size,
            "reference_id": self.reference_id,
            "reference_text": self.reference_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimilarityConfig:
        return cls(
            mode=data.get("mode", "similar"),
            sample_size=data.get("sample_size", 20),
            reference_id=data.get("reference_id", ""),
            reference_text=data.get("reference_text", ""),
        )


@dataclass
class SimilarityResult:
    """Result of a similarity-based sampling run."""

    selected_ids: List[str] = field(default_factory=list)
    scores: List[SimilarityScore] = field(default_factory=list)
    reference_id: str = ""
    mode: str = "similar"
    total_pool: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "scores": [s.as_dict() for s in self.scores],
            "reference_id": self.reference_id,
            "mode": self.mode,
            "total_pool": self.total_pool,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimilarityResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            scores=[
                SimilarityScore.from_dict(s) for s in data.get("scores", [])
            ],
            reference_id=data.get("reference_id", ""),
            mode=data.get("mode", "similar"),
            total_pool=data.get("total_pool", 0),
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Split text into a set of lowercase words."""
    return set(text.lower().split())


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute word-set Jaccard similarity between two texts.

    Returns a float in [0, 1]. Returns 0.0 if both texts are empty.
    """
    set_a = _tokenize(text_a)
    set_b = _tokenize(text_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def rank_by_similarity(
    reference_text: str, items: Dict[str, str]
) -> List[SimilarityScore]:
    """Rank all items by similarity to the reference text.

    Returns a list of SimilarityScore sorted by similarity descending.
    """
    scores: List[SimilarityScore] = []
    for item_id, text in items.items():
        sim = compute_jaccard_similarity(reference_text, text)
        scores.append(
            SimilarityScore(
                item_id=item_id,
                similarity=sim,
                distance=1.0 - sim,
            )
        )
    scores.sort(key=lambda s: (-s.similarity, s.item_id))
    return scores


def select_most_similar(scores: List[SimilarityScore], n: int) -> List[str]:
    """Select top-N most similar items (highest similarity first)."""
    sorted_scores = sorted(scores, key=lambda s: (-s.similarity, s.item_id))
    return [s.item_id for s in sorted_scores[:n]]


def select_most_dissimilar(scores: List[SimilarityScore], n: int) -> List[str]:
    """Select top-N most dissimilar items (lowest similarity first)."""
    sorted_scores = sorted(scores, key=lambda s: (s.similarity, s.item_id))
    return [s.item_id for s in sorted_scores[:n]]


def run_similarity_sampling(
    items: Dict[str, str], config: SimilarityConfig
) -> SimilarityResult:
    """Full similarity sampling pipeline.

    Uses config.reference_text to rank items, then selects based on mode.
    If config.reference_id is provided and found in items, uses its text
    as the reference (unless reference_text is already set).
    """
    ref_text = config.reference_text
    ref_id = config.reference_id

    # Resolve reference text from reference_id if needed
    if not ref_text and ref_id and ref_id in items:
        ref_text = items[ref_id]

    # Exclude reference item from candidate pool
    candidates = {k: v for k, v in items.items() if k != ref_id}

    scores = rank_by_similarity(ref_text, candidates)

    if config.mode == "dissimilar":
        selected = select_most_dissimilar(scores, config.sample_size)
    else:
        selected = select_most_similar(scores, config.sample_size)

    return SimilarityResult(
        selected_ids=selected,
        scores=scores,
        reference_id=ref_id,
        mode=config.mode,
        total_pool=len(candidates),
    )


def find_nearest_neighbors(
    reference_text: str, items: Dict[str, str], k: int = 5
) -> List[SimilarityScore]:
    """Convenience function for k-nearest-neighbor lookup.

    Returns the k items most similar to reference_text.
    """
    scores = rank_by_similarity(reference_text, items)
    return scores[:k]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_similarity_report(result: SimilarityResult) -> str:
    """Format a similarity result as a human-readable report."""
    lines: List[str] = []
    lines.append(
        f"Similarity Sampling Report (mode={result.mode}, "
        f"pool={result.total_pool})"
    )
    if result.reference_id:
        lines.append(f"Reference: {result.reference_id}")
    lines.append(f"Selected: {len(result.selected_ids)} items")
    lines.append("-" * 56)
    header = f"{'Item ID':<30} {'Similarity':>12} {'Distance':>12}"
    lines.append(header)
    lines.append("-" * 56)
    # Show scores for selected items
    selected_set = set(result.selected_ids)
    shown = [s for s in result.scores if s.item_id in selected_set]
    if result.mode == "dissimilar":
        shown.sort(key=lambda s: (s.similarity, s.item_id))
    else:
        shown.sort(key=lambda s: (-s.similarity, s.item_id))
    for s in shown:
        row = f"{s.item_id:<30} {s.similarity:>12.4f} {s.distance:>12.4f}"
        lines.append(row)
    lines.append("-" * 56)
    return "\n".join(lines)
