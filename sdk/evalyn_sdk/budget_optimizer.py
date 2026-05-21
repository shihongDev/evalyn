"""
Simulation budget optimizer: allocate token budget across simulation modes.

Given a total token budget, optimizes the mix of simulation modes to maximize
diversity under the budget constraint. Pure Python - no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SimulationMode:
    """A simulation mode with token cost and diversity weight."""

    mode_id: str
    name: str
    description: str
    estimated_tokens_per_item: int
    diversity_weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "name": self.name,
            "description": self.description,
            "estimated_tokens_per_item": self.estimated_tokens_per_item,
            "diversity_weight": self.diversity_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationMode:
        return cls(
            mode_id=data["mode_id"],
            name=data["name"],
            description=data["description"],
            estimated_tokens_per_item=data["estimated_tokens_per_item"],
            diversity_weight=data.get("diversity_weight", 1.0),
        )


@dataclass
class BudgetAllocation:
    """Token allocation for a single simulation mode."""

    mode_id: str
    item_count: int
    estimated_tokens: int
    actual_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "item_count": self.item_count,
            "estimated_tokens": self.estimated_tokens,
            "actual_tokens": self.actual_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetAllocation:
        return cls(
            mode_id=data["mode_id"],
            item_count=data["item_count"],
            estimated_tokens=data["estimated_tokens"],
            actual_tokens=data.get("actual_tokens", 0),
        )


@dataclass
class BudgetPlan:
    """Complete budget allocation plan across all modes."""

    allocations: list[BudgetAllocation]
    total_budget: int
    estimated_total_tokens: int
    estimated_total_items: int
    diversity_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocations": [a.as_dict() for a in self.allocations],
            "total_budget": self.total_budget,
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_total_items": self.estimated_total_items,
            "diversity_score": self.diversity_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetPlan:
        return cls(
            allocations=[BudgetAllocation.from_dict(a) for a in data["allocations"]],
            total_budget=data["total_budget"],
            estimated_total_tokens=data["estimated_total_tokens"],
            estimated_total_items=data["estimated_total_items"],
            diversity_score=data["diversity_score"],
        )


@dataclass
class BudgetReport:
    """Post-execution report comparing planned vs actual token usage."""

    plan: BudgetPlan
    actual_total_tokens: int
    budget_utilization: float
    items_generated: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "actual_total_tokens": self.actual_total_tokens,
            "budget_utilization": self.budget_utilization,
            "items_generated": self.items_generated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetReport:
        return cls(
            plan=BudgetPlan.from_dict(data["plan"]),
            actual_total_tokens=data["actual_total_tokens"],
            budget_utilization=data["budget_utilization"],
            items_generated=data["items_generated"],
        )


# ---------------------------------------------------------------------------
# Default Modes
# ---------------------------------------------------------------------------

DEFAULT_MODES: list[SimulationMode] = [
    SimulationMode(
        mode_id="similar",
        name="Similar",
        description="Generate inputs similar to existing examples",
        estimated_tokens_per_item=200,
        diversity_weight=0.5,
    ),
    SimulationMode(
        mode_id="diverse",
        name="Diverse",
        description="Generate maximally diverse inputs",
        estimated_tokens_per_item=300,
        diversity_weight=1.5,
    ),
    SimulationMode(
        mode_id="adversarial",
        name="Adversarial",
        description="Generate adversarial challenge inputs",
        estimated_tokens_per_item=250,
        diversity_weight=2.0,
    ),
    SimulationMode(
        mode_id="edge_case",
        name="Edge Case",
        description="Generate edge case and boundary inputs",
        estimated_tokens_per_item=150,
        diversity_weight=1.0,
    ),
    SimulationMode(
        mode_id="domain_transfer",
        name="Domain Transfer",
        description="Generate inputs from adjacent domains",
        estimated_tokens_per_item=350,
        diversity_weight=1.2,
    ),
]


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def estimate_mode_cost(mode: SimulationMode, count: int) -> int:
    """Estimate total tokens for running a mode a given number of times."""
    return mode.estimated_tokens_per_item * count


def compute_plan_diversity(plan: BudgetPlan, modes: list[SimulationMode]) -> float:
    """Compute weighted diversity score for a budget plan.

    Diversity = sum(diversity_weight * item_count / total_items) for each mode.
    Returns 0.0 if no items allocated.
    """
    total_items = sum(a.item_count for a in plan.allocations)
    if total_items == 0:
        return 0.0

    mode_weights = {m.mode_id: m.diversity_weight for m in modes}
    score = 0.0
    for alloc in plan.allocations:
        weight = mode_weights.get(alloc.mode_id, 1.0)
        score += weight * alloc.item_count / total_items
    return score


def optimize_budget(
    budget_tokens: int,
    modes: list[SimulationMode] | None = None,
    min_per_mode: int = 1,
) -> BudgetPlan:
    """Optimize simulation mode allocation within a token budget.

    Strategy:
    1. Ensure min_per_mode items for each mode (if budget allows).
    2. Allocate remaining budget proportional to diversity_weight / tokens_per_item.
    3. Greedy: assign one item at a time to the mode with the highest ratio until
       the budget is exhausted.
    """
    if modes is None:
        modes = list(DEFAULT_MODES)

    if not modes:
        return BudgetPlan(
            allocations=[],
            total_budget=budget_tokens,
            estimated_total_tokens=0,
            estimated_total_items=0,
            diversity_score=0.0,
        )

    # Track counts per mode
    counts: dict[str, int] = {m.mode_id: 0 for m in modes}
    tokens_used = 0

    # Phase 1: allocate minimum per mode
    for mode in modes:
        cost = estimate_mode_cost(mode, min_per_mode)
        if tokens_used + cost <= budget_tokens:
            counts[mode.mode_id] = min_per_mode
            tokens_used += cost

    # Phase 2: greedy allocation of remaining budget
    # Compute priority ratio for each mode
    ratios = {m.mode_id: m.diversity_weight / m.estimated_tokens_per_item for m in modes}
    # Sort modes by ratio descending for greedy allocation
    sorted_modes = sorted(modes, key=lambda m: ratios[m.mode_id], reverse=True)

    changed = True
    while changed:
        changed = False
        for mode in sorted_modes:
            if tokens_used + mode.estimated_tokens_per_item <= budget_tokens:
                counts[mode.mode_id] += 1
                tokens_used += mode.estimated_tokens_per_item
                changed = True

    # Build allocations
    allocations = []
    for mode in modes:
        count = counts[mode.mode_id]
        allocations.append(
            BudgetAllocation(
                mode_id=mode.mode_id,
                item_count=count,
                estimated_tokens=estimate_mode_cost(mode, count),
            )
        )

    total_items = sum(a.item_count for a in allocations)

    plan = BudgetPlan(
        allocations=allocations,
        total_budget=budget_tokens,
        estimated_total_tokens=tokens_used,
        estimated_total_items=total_items,
        diversity_score=0.0,
    )
    plan.diversity_score = compute_plan_diversity(plan, modes)
    return plan


def update_budget_report(plan: BudgetPlan, actual_tokens: dict[str, int]) -> BudgetReport:
    """Update a budget plan with actual token usage to create a report."""
    # Create new allocations with actual tokens (do not mutate the input plan)
    updated_allocations = [
        BudgetAllocation(
            mode_id=alloc.mode_id,
            item_count=alloc.item_count,
            estimated_tokens=alloc.estimated_tokens,
            actual_tokens=actual_tokens.get(alloc.mode_id, 0),
        )
        for alloc in plan.allocations
    ]
    updated_plan = BudgetPlan(
        allocations=updated_allocations,
        total_budget=plan.total_budget,
        estimated_total_tokens=plan.estimated_total_tokens,
        estimated_total_items=plan.estimated_total_items,
        diversity_score=plan.diversity_score,
    )

    actual_total = sum(actual_tokens.values())
    utilization = actual_total / plan.total_budget if plan.total_budget > 0 else 0.0
    items_generated = plan.estimated_total_items

    return BudgetReport(
        plan=updated_plan,
        actual_total_tokens=actual_total,
        budget_utilization=utilization,
        items_generated=items_generated,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_budget_plan(plan: BudgetPlan) -> str:
    """Format a budget plan as human-readable text."""
    lines = [
        "Budget Plan",
        "-" * 40,
        f"Total budget: {plan.total_budget:,} tokens",
        f"Estimated usage: {plan.estimated_total_tokens:,} tokens",
        f"Estimated items: {plan.estimated_total_items}",
        f"Diversity score: {plan.diversity_score:.3f}",
        "",
        "Per-mode breakdown:",
    ]
    for alloc in plan.allocations:
        lines.append(
            f"  {alloc.mode_id}: {alloc.item_count} items ({alloc.estimated_tokens:,} tokens)"
        )
    return "\n".join(lines)


def format_budget_report(report: BudgetReport) -> str:
    """Format a budget report comparing planned vs actual usage."""
    lines = [
        "Budget Report",
        "-" * 40,
        f"Total budget: {report.plan.total_budget:,} tokens",
        f"Estimated tokens: {report.plan.estimated_total_tokens:,}",
        f"Actual tokens: {report.actual_total_tokens:,}",
        f"Budget utilization: {report.budget_utilization:.1%}",
        f"Items generated: {report.items_generated}",
        "",
        "Per-mode comparison:",
    ]
    for alloc in report.plan.allocations:
        diff = alloc.actual_tokens - alloc.estimated_tokens
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"  {alloc.mode_id}: planned {alloc.estimated_tokens:,} / "
            f"actual {alloc.actual_tokens:,} ({sign}{diff:,})"
        )
    return "\n".join(lines)
