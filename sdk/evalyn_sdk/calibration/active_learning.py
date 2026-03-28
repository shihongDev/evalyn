"""Smart sample selection for annotation via active learning strategies.

Provides pure functions for selecting the most informative items
to annotate next, using uncertainty, disagreement, and diversity heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ScoredItem:
    """An item with a score, confidence, and optional heuristic score."""

    item_id: str
    score: float
    confidence: float = 0.0
    heuristic_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "item_id": self.item_id,
            "score": self.score,
            "confidence": self.confidence,
        }
        if self.heuristic_score is not None:
            d["heuristic_score"] = self.heuristic_score
        return d


@dataclass
class SelectionResult:
    """Result of an active learning selection pass."""

    selected_ids: List[str]
    method: str
    total_candidates: int
    selected_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "method": self.method,
            "total_candidates": self.total_candidates,
            "selected_count": self.selected_count,
        }

    def format_text(self) -> str:
        lines = [
            f"Active Learning Selection ({self.method})",
            f"  Candidates: {self.total_candidates}",
            f"  Selected:   {self.selected_count}",
        ]
        if self.selected_ids:
            lines.append("  Items:")
            for item_id in self.selected_ids:
                lines.append(f"    - {item_id}")
        return "\n".join(lines)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning sample selection."""

    method: str = "uncertainty"  # "uncertainty" / "disagreement" / "diversity" / "combined"
    batch_size: int = 10
    diversity_weight: float = 0.3

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "batch_size": self.batch_size,
            "diversity_weight": self.diversity_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActiveLearningConfig:
        return cls(
            method=data.get("method", "uncertainty"),
            batch_size=data.get("batch_size", 10),
            diversity_weight=data.get("diversity_weight", 0.3),
        )


# ---------------------------------------------------------------------------
# Selection Functions
# ---------------------------------------------------------------------------


def select_by_uncertainty(
    items: List[ScoredItem], batch_size: int = 10
) -> SelectionResult:
    """Select items with lowest confidence scores.

    Sort ascending by confidence, take top batch_size.
    """
    if not items:
        return SelectionResult(
            selected_ids=[],
            method="uncertainty",
            total_candidates=0,
            selected_count=0,
        )

    sorted_items = sorted(items, key=lambda x: x.confidence)
    selected = sorted_items[:batch_size]
    return SelectionResult(
        selected_ids=[item.item_id for item in selected],
        method="uncertainty",
        total_candidates=len(items),
        selected_count=len(selected),
    )


def select_by_disagreement(
    items: List[ScoredItem], batch_size: int = 10
) -> SelectionResult:
    """Select items where abs(score - heuristic_score) is largest.

    Items without heuristic_score are skipped.
    """
    eligible = [item for item in items if item.heuristic_score is not None]

    if not eligible:
        return SelectionResult(
            selected_ids=[],
            method="disagreement",
            total_candidates=len(items),
            selected_count=0,
        )

    sorted_items = sorted(
        eligible,
        key=lambda x: abs(x.score - x.heuristic_score),  # type: ignore[operator]
        reverse=True,
    )
    selected = sorted_items[:batch_size]
    return SelectionResult(
        selected_ids=[item.item_id for item in selected],
        method="disagreement",
        total_candidates=len(items),
        selected_count=len(selected),
    )


def select_by_diversity(
    items: List[ScoredItem], batch_size: int = 10
) -> SelectionResult:
    """Select items spread across the score range.

    Divide [0,1] into batch_size bins, pick one item per bin
    (the one closest to bin center).
    """
    if not items:
        return SelectionResult(
            selected_ids=[],
            method="diversity",
            total_candidates=0,
            selected_count=0,
        )

    actual_bins = min(batch_size, len(items))
    bin_width = 1.0 / actual_bins if actual_bins > 0 else 1.0
    selected_ids: List[str] = []
    used: set[str] = set()

    for i in range(actual_bins):
        bin_center = bin_width * (i + 0.5)
        best_item: Optional[ScoredItem] = None
        best_dist = float("inf")
        for item in items:
            if item.item_id in used:
                continue
            dist = abs(item.score - bin_center)
            if dist < best_dist:
                best_dist = dist
                best_item = item
        if best_item is not None:
            selected_ids.append(best_item.item_id)
            used.add(best_item.item_id)

    return SelectionResult(
        selected_ids=selected_ids,
        method="diversity",
        total_candidates=len(items),
        selected_count=len(selected_ids),
    )


def select_for_annotation(
    items: List[ScoredItem], config: ActiveLearningConfig
) -> SelectionResult:
    """Route to appropriate selection method based on config."""
    if config.method == "uncertainty":
        return select_by_uncertainty(items, config.batch_size)
    elif config.method == "disagreement":
        return select_by_disagreement(items, config.batch_size)
    elif config.method == "diversity":
        return select_by_diversity(items, config.batch_size)
    elif config.method == "combined":
        # Combined: merge uncertainty and diversity results
        unc = select_by_uncertainty(items, config.batch_size)
        div = select_by_diversity(items, config.batch_size)

        # Weighted merge: take diversity_weight fraction from diversity, rest from uncertainty
        n_diversity = max(1, int(config.batch_size * config.diversity_weight))
        n_uncertainty = config.batch_size - n_diversity

        combined_ids: List[str] = []
        seen: set[str] = set()

        for item_id in div.selected_ids[:n_diversity]:
            if item_id not in seen:
                combined_ids.append(item_id)
                seen.add(item_id)

        for item_id in unc.selected_ids:
            if len(combined_ids) >= config.batch_size:
                break
            if item_id not in seen:
                combined_ids.append(item_id)
                seen.add(item_id)

        return SelectionResult(
            selected_ids=combined_ids,
            method="combined",
            total_candidates=len(items),
            selected_count=len(combined_ids),
        )
    else:
        raise ValueError(f"Unknown selection method: {config.method}")
