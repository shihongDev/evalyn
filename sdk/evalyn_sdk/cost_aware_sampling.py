"""Cost-aware sampling for budget-constrained evaluation runs.

Prefer shorter/cheaper items when budget is limited. Supports greedy
budget-filling and fractional knapsack selection strategies.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ItemCost:
    """Estimated cost for a single evaluation item."""

    item_id: str
    estimated_tokens: int
    estimated_cost: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "estimated_tokens": self.estimated_tokens,
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ItemCost:
        return cls(
            item_id=data["item_id"],
            estimated_tokens=data["estimated_tokens"],
            estimated_cost=data["estimated_cost"],
        )


@dataclass
class CostBudget:
    """Budget constraints for cost-aware sampling."""

    max_tokens: int = 0  # 0 means unlimited
    max_cost: float = 0.0  # 0 means unlimited
    cost_per_1k_tokens: float = 0.075

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostBudget:
        return cls(
            max_tokens=data.get("max_tokens", 0),
            max_cost=data.get("max_cost", 0.0),
            cost_per_1k_tokens=data.get("cost_per_1k_tokens", 0.075),
        )


@dataclass
class CostAwareResult:
    """Result of a cost-aware sampling run."""

    selected_ids: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    items_selected: int = 0
    budget_utilization: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "items_selected": self.items_selected,
            "budget_utilization": self.budget_utilization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostAwareResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            items_selected=data.get("items_selected", 0),
            budget_utilization=data.get("budget_utilization", 0.0),
        )


def estimate_item_tokens(text: str) -> int:
    """Approximate token count: ceil(len(text) / 4) for input + estimated output.

    Output estimate scales with input: min 50 tokens, max 300, proportional.
    """
    input_tokens = max(1, math.ceil(len(text) / 4))
    output_estimate = min(300, max(50, input_tokens))
    return input_tokens + output_estimate


def compute_item_costs(items: dict[str, str], cost_per_1k: float = 0.075) -> list[ItemCost]:
    """Compute estimated token counts and costs for all items.

    Items are returned sorted by item_id for deterministic ordering.
    """
    costs: list[ItemCost] = []
    for item_id in sorted(items.keys()):
        tokens = estimate_item_tokens(items[item_id])
        cost = tokens * cost_per_1k / 1000.0
        costs.append(
            ItemCost(
                item_id=item_id,
                estimated_tokens=tokens,
                estimated_cost=cost,
            )
        )
    return costs


def _fits_budget(tokens: int, cost: float, budget: CostBudget) -> bool:
    """Check whether adding tokens/cost stays within budget limits."""
    if budget.max_tokens > 0 and tokens > budget.max_tokens:
        return False
    if budget.max_cost > 0 and cost > budget.max_cost:
        return False
    return True


def greedy_budget_selection(costs: list[ItemCost], budget: CostBudget) -> list[str]:
    """Sort by cost ascending and greedily select items until budget exhausted.

    Maximizes item count within budget constraints. When both token and cost
    budgets are zero, returns empty (zero budget means zero capacity).
    Use negative values or very large values for unlimited.
    """
    if budget.max_tokens == 0 and budget.max_cost == 0.0:
        return []
    sorted_costs = sorted(costs, key=lambda c: c.estimated_cost)
    selected: list[str] = []
    running_tokens = 0
    running_cost = 0.0

    for item in sorted_costs:
        new_tokens = running_tokens + item.estimated_tokens
        new_cost = running_cost + item.estimated_cost
        if _fits_budget(new_tokens, new_cost, budget):
            selected.append(item.item_id)
            running_tokens = new_tokens
            running_cost = new_cost

    return selected


def knapsack_selection(
    costs: list[ItemCost],
    budget: CostBudget,
    value_fn: Callable[[ItemCost], float] | None = None,
) -> list[str]:
    """Greedy fractional knapsack: sort by value/cost ratio, select greedily.

    Default value function assigns 1.0 per item (maximizing item count per
    dollar). Custom value functions can weight items by importance, difficulty,
    or other criteria. Items with zero cost are selected first (infinite ratio).
    """
    _value_fn = value_fn if value_fn is not None else lambda item: 1.0

    def ratio_key(item: ItemCost) -> float:
        val = _value_fn(item)
        if item.estimated_cost <= 0:
            return float("inf")
        return val / item.estimated_cost

    sorted_costs = sorted(costs, key=ratio_key, reverse=True)
    selected: list[str] = []
    running_tokens = 0
    running_cost = 0.0

    for item in sorted_costs:
        new_tokens = running_tokens + item.estimated_tokens
        new_cost = running_cost + item.estimated_cost
        if _fits_budget(new_tokens, new_cost, budget):
            selected.append(item.item_id)
            running_tokens = new_tokens
            running_cost = new_cost

    return selected


def _compute_budget_utilization(total_tokens: int, total_cost: float, budget: CostBudget) -> float:
    """Compute budget utilization as fraction of the binding constraint used.

    If both budgets are unlimited, returns 0.0. If both are set, returns
    the higher utilization of the two.
    """
    token_util = 0.0
    cost_util = 0.0
    if budget.max_tokens > 0:
        token_util = total_tokens / budget.max_tokens
    if budget.max_cost > 0:
        cost_util = total_cost / budget.max_cost
    if budget.max_tokens == 0 and budget.max_cost == 0:
        return 0.0
    return max(token_util, cost_util)


def run_cost_aware_sampling(items: dict[str, str], budget: CostBudget) -> CostAwareResult:
    """Full cost-aware sampling pipeline.

    1. Compute item costs
    2. Greedy budget selection
    3. Summarize results with budget utilization
    """
    costs = compute_item_costs(items, cost_per_1k=budget.cost_per_1k_tokens)
    selected_ids = greedy_budget_selection(costs, budget)

    cost_map = {c.item_id: c for c in costs}
    total_tokens = sum(cost_map[sid].estimated_tokens for sid in selected_ids)
    total_cost = sum(cost_map[sid].estimated_cost for sid in selected_ids)
    utilization = _compute_budget_utilization(total_tokens, total_cost, budget)

    return CostAwareResult(
        selected_ids=selected_ids,
        total_tokens=total_tokens,
        total_cost=total_cost,
        items_selected=len(selected_ids),
        budget_utilization=utilization,
    )


def format_cost_report(result: CostAwareResult) -> str:
    """Format a human-readable cost sampling report with budget utilization."""
    lines: list[str] = []
    lines.append("Cost-Aware Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Items selected: {result.items_selected}")
    lines.append(f"Total tokens: {result.total_tokens}")
    lines.append(f"Total cost: ${result.total_cost:.4f}")
    lines.append(f"Budget utilization: {result.budget_utilization:.2%}")
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
