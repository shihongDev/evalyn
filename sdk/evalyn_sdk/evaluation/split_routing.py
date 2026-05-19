"""Evaluation split-model routing.

Routes objective metrics to local compute (no API cost) and subjective
metrics to LLM API providers. Reports cost savings from split routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingPlan:
    """Plan for splitting metrics between local and API evaluation."""

    local_metrics: list = field(default_factory=list)   # objective - run locally
    api_metrics: list = field(default_factory=list)      # subjective - send to API
    local_count: int = 0
    api_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_metrics": [m.spec.id for m in self.local_metrics],
            "api_metrics": [m.spec.id for m in self.api_metrics],
            "local_count": self.local_count,
            "api_count": self.api_count,
            "total": self.local_count + self.api_count,
        }

    def format_text(self) -> str:
        lines = [
            "Split-Model Routing Plan:",
            f"  Local (objective):    {self.local_count} metrics (free)",
            f"  API (subjective):     {self.api_count} metrics (LLM cost)",
        ]
        if self.local_metrics:
            lines.append(f"  Local: {', '.join(m.spec.id for m in self.local_metrics)}")
        if self.api_metrics:
            lines.append(f"  API:   {', '.join(m.spec.id for m in self.api_metrics)}")
        return "\n".join(lines)


@dataclass
class RoutingResult:
    """Result of split-model evaluation."""

    local_results: list = field(default_factory=list)
    api_results: list = field(default_factory=list)
    local_time_ms: float = 0.0
    api_time_ms: float = 0.0
    api_cost_usd: float = 0.0

    @property
    def all_results(self) -> list:
        return self.local_results + self.api_results

    @property
    def total_results(self) -> int:
        return len(self.local_results) + len(self.api_results)

    def savings_report(self, full_api_cost_estimate: float = 0.0) -> dict[str, Any]:
        """Report cost savings from split routing vs all-API evaluation.

        Args:
            full_api_cost_estimate: Estimated cost if all metrics used API.

        Returns:
            Dict with savings data.
        """
        if full_api_cost_estimate > 0:
            saved = full_api_cost_estimate - self.api_cost_usd
            pct = saved / full_api_cost_estimate
        else:
            saved = 0.0
            pct = 0.0

        return {
            "actual_api_cost": round(self.api_cost_usd, 6),
            "estimated_full_cost": round(full_api_cost_estimate, 6),
            "saved_usd": round(saved, 6),
            "savings_pct": round(pct, 4),
            "local_results_count": len(self.local_results),
            "api_results_count": len(self.api_results),
        }


def plan_split_routing(metrics: list) -> RoutingPlan:
    """Create a routing plan that splits metrics by type.

    Objective metrics run locally (no API calls).
    Subjective metrics run via LLM API.

    Args:
        metrics: List of Metric objects.

    Returns:
        RoutingPlan with local and API metric lists.
    """
    local = []
    api = []

    for m in metrics:
        if m.spec.type == "objective":
            local.append(m)
        else:
            api.append(m)

    return RoutingPlan(
        local_metrics=local,
        api_metrics=api,
        local_count=len(local),
        api_count=len(api),
    )


def estimate_routing_savings(
    plan: RoutingPlan,
    item_count: int,
    cost_per_subjective_eval: float = 0.001,
) -> dict[str, Any]:
    """Estimate cost savings from split routing.

    Args:
        plan: The routing plan.
        item_count: Number of items to evaluate.
        cost_per_subjective_eval: Average cost per subjective metric per item.

    Returns:
        Dict with estimated savings.
    """
    total_metrics = plan.local_count + plan.api_count
    if total_metrics == 0:
        return {"savings_pct": 0.0, "saved_usd": 0.0}

    # Full API cost: all metrics as subjective
    full_cost = total_metrics * item_count * cost_per_subjective_eval

    # Split cost: only subjective metrics hit API
    split_cost = plan.api_count * item_count * cost_per_subjective_eval

    saved = full_cost - split_cost
    pct = saved / full_cost if full_cost > 0 else 0.0

    return {
        "full_api_cost_estimate": round(full_cost, 6),
        "split_routing_cost": round(split_cost, 6),
        "saved_usd": round(saved, 6),
        "savings_pct": round(pct, 4),
        "local_fraction": plan.local_count / total_metrics if total_metrics > 0 else 0.0,
    }
