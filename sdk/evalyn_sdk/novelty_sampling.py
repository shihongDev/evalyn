"""Novelty-based sampling for evaluation datasets.

Prioritize items most unlike the existing labeled set. Uses word-set
Jaccard distance as a proxy for novelty. Pure Python, no external
dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NoveltyScore:
    """Novelty score for an unlabeled item relative to a labeled set."""

    item_id: str
    novelty: float  # 0 to 1, higher = more novel
    nearest_labeled_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "novelty": self.novelty,
            "nearest_labeled_id": self.nearest_labeled_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NoveltyScore:
        return cls(
            item_id=data["item_id"],
            novelty=data["novelty"],
            nearest_labeled_id=data.get("nearest_labeled_id", ""),
        )


@dataclass
class NoveltyConfig:
    """Configuration for novelty-based sampling."""

    sample_size: int = 50
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NoveltyConfig:
        return cls(
            sample_size=data.get("sample_size", 50),
            seed=data.get("seed"),
        )


@dataclass
class NoveltyResult:
    """Result of a novelty-based sampling run."""

    selected_ids: List[str] = field(default_factory=list)
    scores: List[NoveltyScore] = field(default_factory=list)
    mean_novelty: float = 0.0
    total_unlabeled: int = 0
    total_labeled: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "scores": [s.as_dict() for s in self.scores],
            "mean_novelty": self.mean_novelty,
            "total_unlabeled": self.total_unlabeled,
            "total_labeled": self.total_labeled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NoveltyResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            scores=[
                NoveltyScore.from_dict(s) for s in data.get("scores", [])
            ],
            mean_novelty=data.get("mean_novelty", 0.0),
            total_unlabeled=data.get("total_unlabeled", 0),
            total_labeled=data.get("total_labeled", 0),
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Split text into a set of lowercase words."""
    return set(text.lower().split())


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not set_a and not set_b:
        return 0.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def compute_novelty(
    text: str, labeled_texts: list[str]
) -> tuple[float, int]:
    """Compute novelty of text relative to a set of labeled texts.

    Novelty = 1 - max(Jaccard similarity to any labeled item).
    Returns (novelty, index of nearest labeled item).
    Empty labeled set returns (1.0, -1).
    """
    if not labeled_texts:
        return (1.0, -1)

    text_tokens = _tokenize(text)
    max_sim = 0.0
    nearest_idx = 0

    for i, labeled_text in enumerate(labeled_texts):
        labeled_tokens = _tokenize(labeled_text)
        sim = _jaccard_similarity(text_tokens, labeled_tokens)
        if sim > max_sim:
            max_sim = sim
            nearest_idx = i

    return (1.0 - max_sim, nearest_idx)


def score_all_items(
    unlabeled: dict[str, str], labeled: dict[str, str]
) -> list[NoveltyScore]:
    """Score all unlabeled items by novelty relative to the labeled set.

    Returns a list of NoveltyScore sorted by novelty descending.
    """
    labeled_ids = list(labeled.keys())
    labeled_texts = list(labeled.values())

    scores: list[NoveltyScore] = []
    for item_id, text in unlabeled.items():
        novelty, nearest_idx = compute_novelty(text, labeled_texts)
        nearest_id = labeled_ids[nearest_idx] if nearest_idx >= 0 else ""
        scores.append(
            NoveltyScore(
                item_id=item_id,
                novelty=novelty,
                nearest_labeled_id=nearest_id,
            )
        )

    scores.sort(key=lambda s: (-s.novelty, s.item_id))
    return scores


def select_most_novel(scores: list[NoveltyScore], n: int) -> list[str]:
    """Select top-N items by novelty (highest novelty first)."""
    sorted_scores = sorted(scores, key=lambda s: (-s.novelty, s.item_id))
    return [s.item_id for s in sorted_scores[:n]]


def run_novelty_sampling(
    unlabeled: dict[str, str],
    labeled: dict[str, str],
    config: NoveltyConfig,
) -> NoveltyResult:
    """Full novelty sampling pipeline.

    Scores all unlabeled items, selects the most novel ones, and returns
    a NoveltyResult with statistics.
    """
    scores = score_all_items(unlabeled, labeled)
    selected = select_most_novel(scores, config.sample_size)

    selected_scores = [s for s in scores if s.item_id in set(selected)]
    mean_novelty = 0.0
    if selected_scores:
        mean_novelty = sum(s.novelty for s in selected_scores) / len(
            selected_scores
        )

    return NoveltyResult(
        selected_ids=selected,
        scores=scores,
        mean_novelty=mean_novelty,
        total_unlabeled=len(unlabeled),
        total_labeled=len(labeled),
    )


def iterative_novelty_sampling(
    unlabeled: dict[str, str],
    labeled: dict[str, str],
    n_rounds: int = 3,
    per_round: int = 10,
) -> list[str]:
    """Greedy iterative novelty sampling.

    Each round: score unlabeled items, select most novel, move them to
    the labeled set, and repeat. Returns all selected IDs across rounds.
    """
    remaining = dict(unlabeled)
    current_labeled = dict(labeled)
    all_selected: list[str] = []

    for _ in range(n_rounds):
        if not remaining:
            break
        scores = score_all_items(remaining, current_labeled)
        n = min(per_round, len(remaining))
        chosen = select_most_novel(scores, n)
        all_selected.extend(chosen)
        for item_id in chosen:
            text = remaining.pop(item_id)
            current_labeled[item_id] = text

    return all_selected


def format_novelty_report(result: NoveltyResult) -> str:
    """Format a novelty result as a human-readable report."""
    lines: list[str] = []
    lines.append(
        f"Novelty Sampling Report "
        f"(unlabeled={result.total_unlabeled}, "
        f"labeled={result.total_labeled})"
    )
    lines.append(f"Selected: {len(result.selected_ids)} items")
    lines.append(f"Mean novelty: {result.mean_novelty:.4f}")
    lines.append("-" * 60)
    header = f"{'Item ID':<25} {'Novelty':>10} {'Nearest Labeled':<25}"
    lines.append(header)
    lines.append("-" * 60)

    selected_set = set(result.selected_ids)
    shown = [s for s in result.scores if s.item_id in selected_set]
    shown.sort(key=lambda s: (-s.novelty, s.item_id))
    for s in shown:
        row = (
            f"{s.item_id:<25} {s.novelty:>10.4f} "
            f"{s.nearest_labeled_id:<25}"
        )
        lines.append(row)

    lines.append("-" * 60)
    return "\n".join(lines)
