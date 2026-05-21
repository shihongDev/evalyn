"""Coverage-aware sampling for maximizing input feature space coverage.

Greedy set-cover approach using word-set features as a lightweight proxy
for embedding-based diversity. Pure Python, no external dependencies.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoverageConfig:
    """Configuration for coverage-aware sampling."""

    sample_size: int = 100
    diversity_weight: float = 1.0
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "diversity_weight": self.diversity_weight,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageConfig:
        return cls(
            sample_size=data.get("sample_size", 100),
            diversity_weight=data.get("diversity_weight", 1.0),
            seed=data.get("seed"),
        )


@dataclass
class CoverageScore:
    """Score for a single item's marginal coverage contribution."""

    item_id: str
    marginal_coverage: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "marginal_coverage": self.marginal_coverage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageScore:
        return cls(
            item_id=data["item_id"],
            marginal_coverage=data["marginal_coverage"],
        )


@dataclass
class CoverageResult:
    """Result of a coverage-aware sampling run."""

    selected_ids: list[str] = field(default_factory=list)
    coverage_rate: float = 0.0
    total_pool: int = 0
    vocabulary_covered: int = 0
    vocabulary_total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "coverage_rate": self.coverage_rate,
            "total_pool": self.total_pool,
            "vocabulary_covered": self.vocabulary_covered,
            "vocabulary_total": self.vocabulary_total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            coverage_rate=data.get("coverage_rate", 0.0),
            total_pool=data.get("total_pool", 0),
            vocabulary_covered=data.get("vocabulary_covered", 0),
            vocabulary_total=data.get("vocabulary_total", 0),
        )


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def compute_item_features(text: str) -> set[str]:
    """Extract lowercased word tokens as a feature set.

    Words shorter than 2 characters are excluded.
    """
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if len(w) >= 2}


def compute_coverage_rate(selected_features: set[str], total_features: set[str]) -> float:
    """Fraction of total vocabulary covered by selected features."""
    if not total_features:
        return 1.0
    return len(selected_features & total_features) / len(total_features)


def score_marginal_coverage(
    item_features: set[str],
    already_covered: set[str],
    total_features: set[str],
) -> float:
    """Fraction of currently-uncovered features that this item would cover."""
    uncovered = total_features - already_covered
    if not uncovered:
        return 0.0
    new_coverage = item_features & uncovered
    return len(new_coverage) / len(uncovered)


def greedy_max_coverage(
    items: dict[str, str],
    sample_size: int,
    seed: int | None = None,
) -> CoverageResult:
    """Greedily select items that maximize cumulative vocabulary coverage.

    At each step, pick the unselected item that adds the most new words.
    Ties are broken using the seeded RNG (shuffle candidates, pick first).
    """
    if not items:
        return CoverageResult(
            selected_ids=[],
            coverage_rate=1.0,
            total_pool=0,
            vocabulary_covered=0,
            vocabulary_total=0,
        )

    rng = random.Random(seed)

    # Pre-compute features for every item
    item_features: dict[str, set[str]] = {}
    total_vocab: set[str] = set()
    for item_id, text in items.items():
        feats = compute_item_features(text)
        item_features[item_id] = feats
        total_vocab |= feats

    actual_size = min(sample_size, len(items))
    selected: list[str] = []
    covered: set[str] = set()
    remaining = set(items.keys())

    for _ in range(actual_size):
        if not remaining:
            break

        # Score all remaining items by how many new words they add
        best_count = -1
        candidates: list[str] = []
        for item_id in remaining:
            new_words = item_features[item_id] - covered
            count = len(new_words)
            if count > best_count:
                best_count = count
                candidates = [item_id]
            elif count == best_count:
                candidates.append(item_id)

        # Tiebreak: shuffle candidates with seeded RNG and pick first
        if len(candidates) > 1:
            rng.shuffle(candidates)
        pick = candidates[0]

        selected.append(pick)
        covered |= item_features[pick]
        remaining.discard(pick)

    rate = compute_coverage_rate(covered, total_vocab)

    return CoverageResult(
        selected_ids=selected,
        coverage_rate=rate,
        total_pool=len(items),
        vocabulary_covered=len(covered & total_vocab),
        vocabulary_total=len(total_vocab),
    )


def run_coverage_sampling(items: dict[str, str], config: CoverageConfig) -> CoverageResult:
    """Full coverage-aware sampling pipeline.

    Uses greedy_max_coverage with the given configuration.
    """
    return greedy_max_coverage(
        items,
        sample_size=config.sample_size,
        seed=config.seed,
    )


def format_coverage_sampling_report(result: CoverageResult) -> str:
    """Format a human-readable coverage sampling report."""
    lines: list[str] = []
    lines.append("Coverage Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Selected samples: {len(result.selected_ids)}")
    lines.append(f"Coverage rate: {result.coverage_rate:.2%}")
    lines.append(
        f"Vocabulary: {result.vocabulary_covered} / {result.vocabulary_total} words covered"
    )
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
