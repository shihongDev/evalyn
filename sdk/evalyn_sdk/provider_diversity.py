"""Provider diversity: manage multiple LLM providers for simulation variety.

Provides a provider pool with weighted round-robin selection, item allocation
proportional to provider weights, result merging, and quality comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    provider_id: str
    name: str
    model: str = ""
    weight: float = 1.0
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "model": self.model,
            "weight": self.weight,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConfig:
        return cls(
            provider_id=data["provider_id"],
            name=data.get("name", data["provider_id"]),
            model=data.get("model", ""),
            weight=data.get("weight", 1.0),
            enabled=data.get("enabled", True),
        )


@dataclass
class ProviderResult:
    """Result from a single provider's simulation run."""

    items: list[dict[str, Any]]
    provider_id: str
    item_count: int
    quality_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": list(self.items),
            "provider_id": self.provider_id,
            "item_count": self.item_count,
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderResult:
        return cls(
            items=data.get("items", []),
            provider_id=data["provider_id"],
            item_count=data.get("item_count", 0),
            quality_score=data.get("quality_score", 0.0),
        )


@dataclass
class DiversityResult:
    """Merged result from multiple providers."""

    all_items: list[dict[str, Any]]
    per_provider: list[ProviderResult]
    total_items: int
    providers_used: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_items": list(self.all_items),
            "per_provider": [r.as_dict() for r in self.per_provider],
            "total_items": self.total_items,
            "providers_used": self.providers_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiversityResult:
        return cls(
            all_items=data.get("all_items", []),
            per_provider=[ProviderResult.from_dict(r) for r in data.get("per_provider", [])],
            total_items=data.get("total_items", 0),
            providers_used=data.get("providers_used", 0),
        )


# ---------------------------------------------------------------------------
# Provider Pool
# ---------------------------------------------------------------------------


class ProviderPool:
    """Weighted round-robin pool of LLM providers."""

    def __init__(self, providers: list[ProviderConfig]) -> None:
        self._providers = list(providers)
        self._index = 0

    def next_provider(self) -> ProviderConfig:
        """Return the next enabled provider using weighted round-robin.

        Providers with higher weight appear more often in the rotation.
        Disabled providers are skipped. Raises ValueError if no providers
        are enabled.
        """
        enabled = self.list_enabled()
        if not enabled:
            raise ValueError("No enabled providers in pool")

        # Build weighted list
        weighted: list[ProviderConfig] = []
        for p in enabled:
            count = max(1, round(p.weight))
            weighted.extend([p] * count)

        idx = self._index % len(weighted)
        self._index += 1
        return weighted[idx]

    def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """Look up a provider by ID. Returns None if not found."""
        for p in self._providers:
            if p.provider_id == provider_id:
                return p
        return None

    def list_enabled(self) -> list[ProviderConfig]:
        """Return all enabled providers."""
        return [p for p in self._providers if p.enabled]

    def allocate_items(self, total: int) -> dict[str, int]:
        """Allocate items to enabled providers proportional to weight.

        Returns a dict mapping provider_id to number of items.
        All items are accounted for (remainders distributed round-robin).
        """
        enabled = self.list_enabled()
        if not enabled:
            return {}

        total_weight = sum(p.weight for p in enabled)
        if total_weight <= 0:
            # Equal split when all weights are zero
            per = total // len(enabled)
            allocation = {p.provider_id: per for p in enabled}
            remainder = total - per * len(enabled)
            for i in range(remainder):
                allocation[enabled[i].provider_id] += 1
            return allocation

        # Proportional allocation with floor
        allocation: dict[str, int] = {}
        assigned = 0
        for p in enabled:
            share = (p.weight / total_weight) * total
            count = math.floor(share)
            allocation[p.provider_id] = count
            assigned += count

        # Distribute remainder by largest fractional part
        remainder = total - assigned
        if remainder > 0:
            fractional = []
            for p in enabled:
                share = (p.weight / total_weight) * total
                frac = share - math.floor(share)
                fractional.append((frac, p.provider_id))
            fractional.sort(key=lambda x: x[0], reverse=True)
            for i in range(remainder):
                allocation[fractional[i][1]] += 1

        return allocation


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def merge_provider_results(results: list[ProviderResult]) -> DiversityResult:
    """Merge results from multiple providers into a single DiversityResult.

    Each item gets a '_provider_id' metadata tag indicating its source.
    """
    all_items: list[dict[str, Any]] = []
    for r in results:
        for item in r.items:
            tagged = dict(item)
            if "metadata" not in tagged:
                tagged["metadata"] = {}
            if isinstance(tagged["metadata"], dict):
                tagged["metadata"]["_provider_id"] = r.provider_id
            all_items.append(tagged)

    return DiversityResult(
        all_items=all_items,
        per_provider=list(results),
        total_items=len(all_items),
        providers_used=len({r.provider_id for r in results}),
    )


def compare_provider_quality(results: list[ProviderResult]) -> dict[str, float]:
    """Build a provider_id to quality_score mapping from results."""
    return {r.provider_id: r.quality_score for r in results}


def format_provider_report(result: DiversityResult) -> str:
    """Format a human-readable per-provider breakdown report."""
    lines: list[str] = []
    lines.append("Provider Diversity Report")
    lines.append("=" * 40)
    lines.append(f"Total items: {result.total_items}")
    lines.append(f"Providers used: {result.providers_used}")
    lines.append("")

    for pr in result.per_provider:
        pct = (pr.item_count / result.total_items * 100) if result.total_items else 0.0
        lines.append(f"  {pr.provider_id}")
        lines.append(f"    Items: {pr.item_count} ({pct:.1f}%)")
        lines.append(f"    Quality: {pr.quality_score:.2f}")

    return "\n".join(lines)
