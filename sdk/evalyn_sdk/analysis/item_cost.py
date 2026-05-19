"""Item-level cost attribution: track exact cost per dataset item.

Aggregates input/output tokens across all metrics for each item,
computes per-item cost, and identifies the most expensive items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ItemCost:
    """Cost data for a single dataset item."""

    item_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    metric_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "metric_count": self.metric_count,
        }


@dataclass
class ItemCostReport:
    """Cost attribution report across all items."""

    items: list[ItemCost] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    @property
    def avg_cost_per_item(self) -> float:
        return self.total_cost_usd / len(self.items) if self.items else 0.0

    @property
    def most_expensive(self) -> ItemCost | None:
        return max(self.items, key=lambda x: x.cost_usd) if self.items else None

    @property
    def cheapest(self) -> ItemCost | None:
        return min(self.items, key=lambda x: x.cost_usd) if self.items else None

    def top_n_expensive(self, n: int = 5) -> list[ItemCost]:
        return sorted(self.items, key=lambda x: x.cost_usd, reverse=True)[:n]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "item_count": len(self.items),
            "avg_cost_per_item": round(self.avg_cost_per_item, 6),
            "most_expensive_item": self.most_expensive.item_id if self.most_expensive else None,
            "items": [i.as_dict() for i in self.items],
        }

    def format_text(self) -> str:
        lines = [
            "Item-Level Cost Attribution",
            f"  Total cost:       ${self.total_cost_usd:.4f}",
            f"  Items:            {len(self.items)}",
            f"  Avg cost/item:    ${self.avg_cost_per_item:.6f}",
        ]
        top = self.top_n_expensive(5)
        if top:
            lines.append("")
            lines.append("  Most expensive items:")
            for ic in top:
                lines.append(f"    {ic.item_id}: ${ic.cost_usd:.6f} ({ic.total_tokens} tokens)")
        return "\n".join(lines)


def compute_item_costs(
    metric_results: list,
    model: str = "gemini-2.5-flash-lite",
) -> ItemCostReport:
    """Compute per-item cost from evaluation results.

    Aggregates tokens across all metrics for each item, then computes
    cost using the pricing table.

    Args:
        metric_results: List of MetricResult objects.
        model: Default model for cost calculation.

    Returns:
        ItemCostReport with per-item breakdown.
    """
    from ..trace.instrumentation.providers._shared import calculate_cost

    # Aggregate by item
    by_item: dict[str, ItemCost] = {}

    for mr in metric_results:
        iid = mr.item_id
        if iid not in by_item:
            by_item[iid] = ItemCost(item_id=iid)

        ic = by_item[iid]
        ic.input_tokens += mr.input_tokens or 0
        ic.output_tokens += mr.output_tokens or 0
        ic.metric_count += 1

    # Compute costs
    for ic in by_item.values():
        ic.total_tokens = ic.input_tokens + ic.output_tokens
        actual_model = model
        # Try to find actual model from results
        for mr in metric_results:
            if mr.item_id == ic.item_id and mr.model:
                actual_model = mr.model
                break
        if ic.total_tokens > 0:
            ic.cost_usd = calculate_cost(actual_model, ic.input_tokens, ic.output_tokens)

    items = sorted(by_item.values(), key=lambda x: x.cost_usd, reverse=True)
    total_cost = sum(ic.cost_usd for ic in items)
    total_tokens = sum(ic.total_tokens for ic in items)

    return ItemCostReport(
        items=items,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
    )
