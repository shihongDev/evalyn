"""Human-AI hybrid scoring.

Route uncertain items to human annotators during evaluation.
Merge human and AI scores using configurable strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HybridItem:
    """A single item with AI and optional human scores."""

    item_id: str
    metric_id: str
    ai_score: float
    ai_confidence: float
    needs_human: bool = False
    human_score: float | None = None
    final_score: float = 0.0
    source: str = "ai"

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "ai_score": self.ai_score,
            "ai_confidence": self.ai_confidence,
            "needs_human": self.needs_human,
            "human_score": self.human_score,
            "final_score": self.final_score,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridItem:
        return cls(
            item_id=data["item_id"],
            metric_id=data["metric_id"],
            ai_score=data["ai_score"],
            ai_confidence=data["ai_confidence"],
            needs_human=data.get("needs_human", False),
            human_score=data.get("human_score"),
            final_score=data.get("final_score", 0.0),
            source=data.get("source", "ai"),
        )


@dataclass
class HybridConfig:
    """Configuration for hybrid scoring behavior."""

    confidence_threshold: float = 0.5
    merge_strategy: str = "human_override"
    human_weight: float = 0.7

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "merge_strategy": self.merge_strategy,
            "human_weight": self.human_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridConfig:
        return cls(
            confidence_threshold=data.get("confidence_threshold", 0.5),
            merge_strategy=data.get("merge_strategy", "human_override"),
            human_weight=data.get("human_weight", 0.7),
        )


@dataclass
class HybridReport:
    """Aggregate report of hybrid scoring results."""

    items: list[HybridItem] = field(default_factory=list)
    ai_only_count: int = 0
    human_reviewed_count: int = 0
    merged_count: int = 0
    total_items: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "ai_only_count": self.ai_only_count,
            "human_reviewed_count": self.human_reviewed_count,
            "merged_count": self.merged_count,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridReport:
        return cls(
            items=[HybridItem.from_dict(i) for i in data.get("items", [])],
            ai_only_count=data.get("ai_only_count", 0),
            human_reviewed_count=data.get("human_reviewed_count", 0),
            merged_count=data.get("merged_count", 0),
            total_items=data.get("total_items", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Hybrid Scoring Report",
            f"  Total items: {self.total_items}",
            f"  AI only: {self.ai_only_count}",
            f"  Human reviewed: {self.human_reviewed_count}",
            f"  Merged: {self.merged_count}",
        ]
        for item in self.items:
            flag = " [human]" if item.source != "ai" else ""
            lines.append(
                f"  {item.item_id}/{item.metric_id}: {item.final_score:.4f} ({item.source}){flag}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def identify_for_human_review(items: list[HybridItem], config: HybridConfig) -> list[HybridItem]:
    """Mark items needing human review when AI confidence is below threshold."""
    result: list[HybridItem] = []
    for item in items:
        needs = item.ai_confidence < config.confidence_threshold
        result.append(
            HybridItem(
                item_id=item.item_id,
                metric_id=item.metric_id,
                ai_score=item.ai_score,
                ai_confidence=item.ai_confidence,
                needs_human=needs,
                human_score=item.human_score,
                final_score=item.final_score,
                source=item.source,
            )
        )
    return result


def merge_human_score(item: HybridItem, human_score: float, config: HybridConfig) -> HybridItem:
    """Merge a human score into an item based on the configured strategy."""
    strategy = config.merge_strategy

    if strategy == "human_override":
        final = human_score
        source = "human"
    elif strategy == "weighted_average":
        w = config.human_weight
        final = w * human_score + (1 - w) * item.ai_score
        source = "merged"
    elif strategy == "human_only":
        final = human_score
        source = "human"
    elif strategy == "ai_only":
        final = item.ai_score
        source = "ai"
    else:
        final = human_score
        source = "human"

    return HybridItem(
        item_id=item.item_id,
        metric_id=item.metric_id,
        ai_score=item.ai_score,
        ai_confidence=item.ai_confidence,
        needs_human=item.needs_human,
        human_score=human_score,
        final_score=final,
        source=source,
    )


def compute_final_scores(items: list[HybridItem], config: HybridConfig) -> list[HybridItem]:
    """Finalize scores for all items.

    Items with a human score are merged using the config strategy.
    Items without a human score keep their AI score.
    """
    result: list[HybridItem] = []
    for item in items:
        if item.human_score is not None:
            result.append(merge_human_score(item, item.human_score, config))
        else:
            result.append(
                HybridItem(
                    item_id=item.item_id,
                    metric_id=item.metric_id,
                    ai_score=item.ai_score,
                    ai_confidence=item.ai_confidence,
                    needs_human=item.needs_human,
                    human_score=item.human_score,
                    final_score=item.ai_score,
                    source="ai",
                )
            )
    return result


def build_hybrid_report(items: list[HybridItem]) -> HybridReport:
    """Build an aggregate report from scored items."""
    ai_only = sum(1 for i in items if i.source == "ai")
    human_reviewed = sum(1 for i in items if i.source == "human")
    merged = sum(1 for i in items if i.source == "merged")
    return HybridReport(
        items=list(items),
        ai_only_count=ai_only,
        human_reviewed_count=human_reviewed,
        merged_count=merged,
        total_items=len(items),
    )


def estimate_human_workload(
    items: list[HybridItem], avg_seconds_per_item: float = 30.0
) -> dict[str, Any]:
    """Estimate human review workload for items flagged as needing review."""
    needing = [i for i in items if i.needs_human]
    count = len(needing)
    total_seconds = count * avg_seconds_per_item
    return {
        "items_needing_review": count,
        "avg_seconds_per_item": avg_seconds_per_item,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60.0,
        "total_hours": total_seconds / 3600.0,
    }
