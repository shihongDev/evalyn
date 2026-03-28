"""Confidence-based re-evaluation: re-evaluate uncertain items with a stronger model.

Identifies items with low confidence scores and creates a plan to re-evaluate
them using a more capable model, then merges results back into the original set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class UncertainItem:
    """An item flagged as uncertain based on its confidence score."""

    item_id: str
    metric_id: str
    original_score: float
    confidence: float
    original_model: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "original_score": self.original_score,
            "confidence": self.confidence,
            "original_model": self.original_model,
        }


@dataclass
class ReEvalPlan:
    """Plan for re-evaluating uncertain items with a stronger model."""

    uncertain_items: List[UncertainItem] = field(default_factory=list)
    target_model: str = ""
    estimated_calls: int = 0
    confidence_threshold: float = 0.5

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uncertain_items": [item.as_dict() for item in self.uncertain_items],
            "target_model": self.target_model,
            "estimated_calls": self.estimated_calls,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReEvalPlan:
        items = [
            UncertainItem(**item) for item in data.get("uncertain_items", [])
        ]
        return cls(
            uncertain_items=items,
            target_model=data.get("target_model", ""),
            estimated_calls=data.get("estimated_calls", 0),
            confidence_threshold=data.get("confidence_threshold", 0.5),
        )

    def format_text(self) -> str:
        lines = [
            "Re-Evaluation Plan",
            f"  Target model:          {self.target_model}",
            f"  Confidence threshold:  {self.confidence_threshold}",
            f"  Uncertain items:       {len(self.uncertain_items)}",
            f"  Estimated API calls:   {self.estimated_calls}",
        ]
        return "\n".join(lines)


@dataclass
class ReEvalResult:
    """Result of re-evaluating a single item."""

    item_id: str
    metric_id: str
    original_score: float
    new_score: float
    new_model: str
    improved: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "original_score": self.original_score,
            "new_score": self.new_score,
            "new_model": self.new_model,
            "improved": self.improved,
        }


@dataclass
class ReEvalReport:
    """Summary report of a re-evaluation batch."""

    results: List[ReEvalResult] = field(default_factory=list)
    total_re_evaluated: int = 0
    improved_count: int = 0
    unchanged_count: int = 0
    degraded_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_re_evaluated": self.total_re_evaluated,
            "improved_count": self.improved_count,
            "unchanged_count": self.unchanged_count,
            "degraded_count": self.degraded_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReEvalReport:
        results = [
            ReEvalResult(**r) for r in data.get("results", [])
        ]
        return cls(
            results=results,
            total_re_evaluated=data.get("total_re_evaluated", 0),
            improved_count=data.get("improved_count", 0),
            unchanged_count=data.get("unchanged_count", 0),
            degraded_count=data.get("degraded_count", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Re-Evaluation Report",
            f"  Total re-evaluated:  {self.total_re_evaluated}",
            f"  Improved:            {self.improved_count}",
            f"  Unchanged:           {self.unchanged_count}",
            f"  Degraded:            {self.degraded_count}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def identify_uncertain_items(
    scores: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
) -> List[UncertainItem]:
    """Find items where confidence is below the threshold.

    Each score dict must have: item_id, metric_id, score, confidence.
    Optionally: model.
    """
    uncertain: List[UncertainItem] = []
    for s in scores:
        confidence = s.get("confidence", 1.0)
        if confidence < confidence_threshold:
            uncertain.append(
                UncertainItem(
                    item_id=s["item_id"],
                    metric_id=s["metric_id"],
                    original_score=s["score"],
                    confidence=confidence,
                    original_model=s.get("model", ""),
                )
            )
    return uncertain


def create_reeval_plan(
    uncertain: List[UncertainItem],
    target_model: str = "gemini-2.5-pro",
) -> ReEvalPlan:
    """Build a re-evaluation plan from a list of uncertain items."""
    return ReEvalPlan(
        uncertain_items=list(uncertain),
        target_model=target_model,
        estimated_calls=len(uncertain),
        confidence_threshold=uncertain[0].confidence if uncertain else 0.5,
    )


def merge_reeval_results(
    original_scores: List[Dict[str, Any]],
    reeval_results: List[ReEvalResult],
) -> List[Dict[str, Any]]:
    """Merge re-evaluated scores back into the original score list.

    For re-evaluated items, replaces the original score with the new score.
    Non-re-evaluated items are returned unchanged.
    """
    lookup: Dict[tuple[str, str], ReEvalResult] = {
        (r.item_id, r.metric_id): r for r in reeval_results
    }
    merged: List[Dict[str, Any]] = []
    for s in original_scores:
        key = (s["item_id"], s["metric_id"])
        if key in lookup:
            result = lookup[key]
            updated = dict(s)
            updated["score"] = result.new_score
            updated["model"] = result.new_model
            merged.append(updated)
        else:
            merged.append(dict(s))
    return merged


def build_reeval_report(
    results: List[ReEvalResult],
    threshold: float = 0.01,
) -> ReEvalReport:
    """Build a summary report from re-evaluation results.

    An item is considered improved if new_score > original_score + threshold,
    degraded if new_score < original_score - threshold, otherwise unchanged.
    """
    improved = 0
    degraded = 0
    unchanged = 0
    for r in results:
        delta = r.new_score - r.original_score
        if delta > threshold:
            improved += 1
        elif delta < -threshold:
            degraded += 1
        else:
            unchanged += 1
    return ReEvalReport(
        results=list(results),
        total_re_evaluated=len(results),
        improved_count=improved,
        unchanged_count=unchanged,
        degraded_count=degraded,
    )
