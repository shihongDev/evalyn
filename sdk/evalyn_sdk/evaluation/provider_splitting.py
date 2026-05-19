"""Multi-provider batch splitting.

Split a single evaluation batch across multiple providers using
round-robin, cost-optimized, or capacity-weighted strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderAllocation:
    """Allocation of items to a single provider."""

    provider: str
    model: str
    item_count: int
    estimated_cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "item_count": self.item_count,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass
class SplitConfig:
    """Configuration for splitting items across providers."""

    providers: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = "round_robin"

    def as_dict(self) -> dict[str, Any]:
        return {
            "providers": self.providers,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitConfig:
        return cls(
            providers=data.get("providers", []),
            strategy=data.get("strategy", "round_robin"),
        )


@dataclass
class SplitPlan:
    """Plan describing how items are split across providers."""

    allocations: list[ProviderAllocation] = field(default_factory=list)
    total_items: int = 0
    total_estimated_cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocations": [a.as_dict() for a in self.allocations],
            "total_items": self.total_items,
            "total_estimated_cost": self.total_estimated_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitPlan:
        allocations = [
            ProviderAllocation(
                provider=a["provider"],
                model=a["model"],
                item_count=a["item_count"],
                estimated_cost_usd=a.get("estimated_cost_usd", 0.0),
            )
            for a in data.get("allocations", [])
        ]
        return cls(
            allocations=allocations,
            total_items=data.get("total_items", 0),
            total_estimated_cost=data.get("total_estimated_cost", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            f"Split Plan: {self.total_items} items across {len(self.allocations)} providers",
            f"  Total estimated cost: ${self.total_estimated_cost:.4f}",
        ]
        for alloc in self.allocations:
            lines.append(
                f"  {alloc.provider}/{alloc.model}: "
                f"{alloc.item_count} items (${alloc.estimated_cost_usd:.4f})"
            )
        return "\n".join(lines)


@dataclass
class SplitResult:
    """Result of splitting: which item IDs go to which provider."""

    provider: str
    model: str
    item_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "item_ids": self.item_ids,
        }


def split_round_robin(
    item_ids: list[str], providers: list[dict[str, Any]]
) -> list[SplitResult]:
    """Distribute items evenly across providers in round-robin order."""
    if not providers:
        return []

    results = [
        SplitResult(provider=p["provider"], model=p["model"])
        for p in providers
    ]

    for i, item_id in enumerate(item_ids):
        idx = i % len(providers)
        results[idx].item_ids.append(item_id)

    return results


def split_cost_optimized(
    item_ids: list[str], providers: list[dict[str, Any]]
) -> list[SplitResult]:
    """Assign items to cheapest provider first until capacity reached, then next cheapest."""
    if not providers:
        return []

    sorted_providers = sorted(providers, key=lambda p: p.get("cost_per_item", 0.0))

    results = {
        (p["provider"], p["model"]): SplitResult(provider=p["provider"], model=p["model"])
        for p in sorted_providers
    }
    remaining_capacity = {
        (p["provider"], p["model"]): p.get("capacity", len(item_ids))
        for p in sorted_providers
    }

    for item_id in item_ids:
        assigned = False
        for p in sorted_providers:
            key = (p["provider"], p["model"])
            if remaining_capacity[key] > 0:
                results[key].item_ids.append(item_id)
                remaining_capacity[key] -= 1
                assigned = True
                break
        if not assigned:
            # Overflow to last provider
            key = (sorted_providers[-1]["provider"], sorted_providers[-1]["model"])
            results[key].item_ids.append(item_id)

    return [r for r in results.values() if r.item_ids]


def split_capacity_weighted(
    item_ids: list[str], providers: list[dict[str, Any]]
) -> list[SplitResult]:
    """Distribute proportional to each provider's capacity."""
    if not providers:
        return []

    total_capacity = sum(p.get("capacity", 0) for p in providers)
    if total_capacity <= 0:
        return split_round_robin(item_ids, providers)

    results = [
        SplitResult(provider=p["provider"], model=p["model"])
        for p in providers
    ]

    n = len(item_ids)
    assigned = 0
    for i, p in enumerate(providers):
        cap = p.get("capacity", 0)
        if i == len(providers) - 1:
            # Last provider gets the remainder
            count = n - assigned
        else:
            count = int(n * cap / total_capacity)
        results[i].item_ids = list(item_ids[assigned : assigned + count])
        assigned += count

    return [r for r in results if r.item_ids]


def create_split_plan(
    item_ids: list[str], config: SplitConfig
) -> SplitPlan:
    """Route to the appropriate splitting strategy and build a plan."""
    if config.strategy == "cost_optimized":
        split_results = split_cost_optimized(item_ids, config.providers)
    elif config.strategy == "capacity_weighted":
        split_results = split_capacity_weighted(item_ids, config.providers)
    else:
        split_results = split_round_robin(item_ids, config.providers)

    provider_cost_map = {
        (p["provider"], p["model"]): p.get("cost_per_item", 0.0)
        for p in config.providers
    }

    allocations = []
    total_cost = 0.0
    for sr in split_results:
        cost_per = provider_cost_map.get((sr.provider, sr.model), 0.0)
        item_cost = len(sr.item_ids) * cost_per
        total_cost += item_cost
        allocations.append(
            ProviderAllocation(
                provider=sr.provider,
                model=sr.model,
                item_count=len(sr.item_ids),
                estimated_cost_usd=item_cost,
            )
        )

    return SplitPlan(
        allocations=allocations,
        total_items=len(item_ids),
        total_estimated_cost=total_cost,
    )


def estimate_split_cost(plan: SplitPlan) -> float:
    """Return total estimated cost from a split plan."""
    return sum(a.estimated_cost_usd for a in plan.allocations)
