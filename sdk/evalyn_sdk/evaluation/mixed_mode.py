"""Mixed-mode evaluation: batch API for large runs, real-time for small.

Automatically decides whether to use batch or real-time evaluation
based on item count, cost thresholds, and user preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ModeDecision:
    """Result of the batch-vs-realtime decision."""

    mode: str  # "batch" or "realtime"
    reason: str
    item_count: int = 0
    estimated_cost_usd: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "item_count": self.item_count,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass
class MixedModeConfig:
    """Configuration for mixed-mode evaluation."""

    batch_threshold: int = 50  # use batch above this count
    cost_threshold_usd: float = 1.0
    prefer_batch: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_threshold": self.batch_threshold,
            "cost_threshold_usd": self.cost_threshold_usd,
            "prefer_batch": self.prefer_batch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MixedModeConfig:
        return cls(
            batch_threshold=data.get("batch_threshold", 50),
            cost_threshold_usd=data.get("cost_threshold_usd", 1.0),
            prefer_batch=data.get("prefer_batch", True),
        )


@dataclass
class MixedModeReport:
    """Summary report for a mixed-mode evaluation run."""

    mode_used: str
    items_batch: int = 0
    items_realtime: int = 0
    total_items: int = 0
    estimated_savings_pct: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode_used": self.mode_used,
            "items_batch": self.items_batch,
            "items_realtime": self.items_realtime,
            "total_items": self.total_items,
            "estimated_savings_pct": round(self.estimated_savings_pct, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MixedModeReport:
        return cls(
            mode_used=data["mode_used"],
            items_batch=data.get("items_batch", 0),
            items_realtime=data.get("items_realtime", 0),
            total_items=data.get("total_items", 0),
            estimated_savings_pct=data.get("estimated_savings_pct", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            "Mixed-Mode Report",
            f"  Mode:       {self.mode_used}",
            f"  Batch:      {self.items_batch}",
            f"  Realtime:   {self.items_realtime}",
            f"  Total:      {self.total_items}",
            f"  Savings:    {self.estimated_savings_pct:.1%}",
        ]
        return "\n".join(lines)


def decide_mode(item_count: int, config: MixedModeConfig) -> ModeDecision:
    """Decide whether to use batch or real-time evaluation.

    Items above the batch_threshold go to batch; otherwise real-time.
    When prefer_batch is False, real-time is used unless the count
    exceeds the threshold.
    """
    estimated_cost = item_count * 0.01  # rough per-item estimate

    if item_count > config.batch_threshold:
        return ModeDecision(
            mode="batch",
            reason=f"Item count {item_count} exceeds threshold {config.batch_threshold}",
            item_count=item_count,
            estimated_cost_usd=estimated_cost,
        )

    if estimated_cost > config.cost_threshold_usd and config.prefer_batch:
        return ModeDecision(
            mode="batch",
            reason=f"Estimated cost ${estimated_cost:.2f} exceeds threshold ${config.cost_threshold_usd:.2f}",
            item_count=item_count,
            estimated_cost_usd=estimated_cost,
        )

    return ModeDecision(
        mode="realtime",
        reason=f"Item count {item_count} within realtime threshold {config.batch_threshold}",
        item_count=item_count,
        estimated_cost_usd=estimated_cost,
    )


def split_for_mixed_mode(
    items: List[Any], config: MixedModeConfig
) -> Tuple[List[Any], List[Any]]:
    """Split items into (batch_items, realtime_items).

    If the total count exceeds the threshold, all items go to batch.
    Otherwise all items go to realtime.
    """
    if len(items) > config.batch_threshold:
        return (list(items), [])
    return ([], list(items))


def estimate_cost_savings(
    batch_count: int, realtime_count: int, cost_per_item: float = 0.01
) -> float:
    """Estimate percentage savings from using batch mode.

    Batch is typically 50% cheaper than real-time. Returns savings as
    a fraction (0.0 to 1.0).
    """
    total = batch_count + realtime_count
    if total == 0:
        return 0.0
    realtime_cost = total * cost_per_item
    actual_cost = (batch_count * cost_per_item * 0.5) + (
        realtime_count * cost_per_item
    )
    if realtime_cost == 0:
        return 0.0
    return (realtime_cost - actual_cost) / realtime_cost


def build_mixed_mode_report(
    batch_count: int, realtime_count: int, config: MixedModeConfig
) -> MixedModeReport:
    """Build a summary report for a mixed-mode run."""
    total = batch_count + realtime_count
    savings = estimate_cost_savings(batch_count, realtime_count)

    if batch_count > 0 and realtime_count > 0:
        mode_used = "mixed"
    elif batch_count > 0:
        mode_used = "batch"
    else:
        mode_used = "realtime"

    return MixedModeReport(
        mode_used=mode_used,
        items_batch=batch_count,
        items_realtime=realtime_count,
        total_items=total,
        estimated_savings_pct=savings,
    )
