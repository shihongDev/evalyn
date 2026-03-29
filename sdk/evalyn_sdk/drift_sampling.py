"""Embedding drift sampling for prioritizing items whose representations shifted.

Uses word-set Jaccard distance as a lightweight proxy for embedding distance
between dataset versions. Pure Python, no external dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DriftScore:
    """Drift magnitude for a single item between dataset versions."""

    item_id: str
    drift_magnitude: float
    old_version: str = ""
    new_version: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "drift_magnitude": self.drift_magnitude,
            "old_version": self.old_version,
            "new_version": self.new_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DriftScore:
        return cls(
            item_id=data["item_id"],
            drift_magnitude=data["drift_magnitude"],
            old_version=data.get("old_version", ""),
            new_version=data.get("new_version", ""),
        )


@dataclass
class DriftConfig:
    """Configuration for drift-based sampling."""

    sample_size: int = 50
    min_drift: float = 0.1
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "min_drift": self.min_drift,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DriftConfig:
        return cls(
            sample_size=data.get("sample_size", 50),
            min_drift=data.get("min_drift", 0.1),
            seed=data.get("seed"),
        )


@dataclass
class DriftResult:
    """Result of a drift sampling run."""

    selected_ids: List[str] = field(default_factory=list)
    drift_scores: List[DriftScore] = field(default_factory=list)
    mean_drift: float = 0.0
    max_drift: float = 0.0
    total_pool: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "drift_scores": [s.as_dict() for s in self.drift_scores],
            "mean_drift": self.mean_drift,
            "max_drift": self.max_drift,
            "total_pool": self.total_pool,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DriftResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            drift_scores=[
                DriftScore.from_dict(s) for s in data.get("drift_scores", [])
            ],
            mean_drift=data.get("mean_drift", 0.0),
            max_drift=data.get("max_drift", 0.0),
            total_pool=data.get("total_pool", 0),
        )


def compute_text_drift(old_text: str, new_text: str) -> float:
    """Compute drift between two texts as 1 - Jaccard similarity of word sets.

    Returns 0.0 for identical texts, 1.0 for completely different texts.
    Empty inputs: two empty strings yield 0.0; one empty and one non-empty
    yields 1.0.
    """
    old_words = set(old_text.lower().split())
    new_words = set(new_text.lower().split())
    if not old_words and not new_words:
        return 0.0
    union = old_words | new_words
    if not union:
        return 0.0
    intersection = old_words & new_words
    return 1.0 - len(intersection) / len(union)


def compute_dataset_drift(
    old_items: Dict[str, str], new_items: Dict[str, str]
) -> List[DriftScore]:
    """Compute drift for all items present in both old and new versions.

    Items only in one version are ignored (use identify_new_items /
    identify_removed_items for those).
    """
    common_ids = sorted(set(old_items.keys()) & set(new_items.keys()))
    scores: List[DriftScore] = []
    for item_id in common_ids:
        magnitude = compute_text_drift(old_items[item_id], new_items[item_id])
        scores.append(
            DriftScore(
                item_id=item_id,
                drift_magnitude=magnitude,
                old_version=old_items[item_id],
                new_version=new_items[item_id],
            )
        )
    return scores


def identify_new_items(
    old_items: Dict[str, str], new_items: Dict[str, str]
) -> List[str]:
    """Return sorted list of item IDs present in new but not in old."""
    return sorted(set(new_items.keys()) - set(old_items.keys()))


def identify_removed_items(
    old_items: Dict[str, str], new_items: Dict[str, str]
) -> List[str]:
    """Return sorted list of item IDs present in old but not in new."""
    return sorted(set(old_items.keys()) - set(new_items.keys()))


def sample_by_drift(
    drift_scores: List[DriftScore], config: DriftConfig
) -> List[str]:
    """Select top-N items by drift magnitude above min_drift threshold.

    Filters to items with drift >= min_drift, sorts descending by magnitude,
    then takes up to sample_size items. When items tie on magnitude, a seeded
    RNG breaks the tie.
    """
    filtered = [s for s in drift_scores if s.drift_magnitude >= config.min_drift]
    rng = random.Random(config.seed)
    # Shuffle first so ties are broken randomly with seed
    rng.shuffle(filtered)
    # Stable sort descending by magnitude preserves shuffled order for ties
    filtered.sort(key=lambda s: s.drift_magnitude, reverse=True)
    selected = filtered[: config.sample_size]
    return [s.item_id for s in selected]


def run_drift_sampling(
    old_items: Dict[str, str],
    new_items: Dict[str, str],
    config: DriftConfig,
) -> DriftResult:
    """Full drift sampling pipeline.

    1. Compute drift scores for all common items
    2. Sample by drift magnitude
    3. Compute summary statistics
    """
    drift_scores = compute_dataset_drift(old_items, new_items)
    selected_ids = sample_by_drift(drift_scores, config)

    if drift_scores:
        magnitudes = [s.drift_magnitude for s in drift_scores]
        mean_drift = sum(magnitudes) / len(magnitudes)
        max_drift = max(magnitudes)
    else:
        mean_drift = 0.0
        max_drift = 0.0

    return DriftResult(
        selected_ids=selected_ids,
        drift_scores=drift_scores,
        mean_drift=mean_drift,
        max_drift=max_drift,
        total_pool=len(drift_scores),
    )


def format_drift_report(result: DriftResult) -> str:
    """Format a human-readable drift sampling report."""
    lines: List[str] = []
    lines.append("Drift Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total common items: {result.total_pool}")
    lines.append(f"Selected samples: {len(result.selected_ids)}")
    lines.append(f"Mean drift: {result.mean_drift:.4f}")
    lines.append(f"Max drift: {result.max_drift:.4f}")
    lines.append("")
    lines.append("Top drifted items:")
    # Show drift scores for selected items
    selected_set = set(result.selected_ids)
    shown = [s for s in result.drift_scores if s.item_id in selected_set]
    shown.sort(key=lambda s: s.drift_magnitude, reverse=True)
    for score in shown:
        lines.append(f"  {score.item_id}: {score.drift_magnitude:.4f}")
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
