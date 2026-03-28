"""Analysis what-if simulator: model metric improvements interactively.

Answer questions like "what if helpfulness improved 20%?" or "what's the
minimum improvement needed to reach 90% overall pass rate?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WhatIfScenario:
    """A what-if scenario with projected outcomes."""

    description: str
    changes: Dict[str, float]  # metric_id -> improvement (absolute, e.g. 0.1 = +10%)
    projected_pass_rate: float
    current_pass_rate: float
    delta: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "changes": {k: round(v, 4) for k, v in self.changes.items()},
            "current_pass_rate": round(self.current_pass_rate, 4),
            "projected_pass_rate": round(self.projected_pass_rate, 4),
            "delta": round(self.delta, 4),
        }

    def format_text(self) -> str:
        lines = [f"What-If: {self.description}"]
        for mid, change in self.changes.items():
            lines.append(f"  {mid}: {change*100:+.1f}%")
        lines.append(f"  Current:   {self.current_pass_rate:.1%}")
        lines.append(f"  Projected: {self.projected_pass_rate:.1%}")
        lines.append(f"  Delta:     {self.delta*100:+.1f}%")
        return "\n".join(lines)


def simulate_improvement(
    current_rates: Dict[str, float],
    improvements: Dict[str, float],
    description: str = "",
) -> WhatIfScenario:
    """Simulate the effect of improving specific metrics.

    Args:
        current_rates: Dict of metric_id -> current pass rate (0-1).
        improvements: Dict of metric_id -> absolute improvement (e.g. 0.1 = +10%).
        description: Human-readable description of the scenario.

    Returns:
        WhatIfScenario with projected overall pass rate.
    """
    projected_rates = dict(current_rates)
    for mid, improvement in improvements.items():
        if mid in projected_rates:
            projected_rates[mid] = min(1.0, max(0.0, projected_rates[mid] + improvement))

    current_overall = _compute_overall(current_rates)
    projected_overall = _compute_overall(projected_rates)

    return WhatIfScenario(
        description=description or f"Improve {', '.join(improvements.keys())}",
        changes=improvements,
        projected_pass_rate=projected_overall,
        current_pass_rate=current_overall,
        delta=projected_overall - current_overall,
    )


def find_minimum_improvement(
    current_rates: Dict[str, float],
    target_pass_rate: float,
    metric_id: str,
    step: float = 0.01,
    max_improvement: float = 1.0,
) -> Optional[WhatIfScenario]:
    """Find the minimum improvement on one metric to reach a target pass rate.

    Args:
        current_rates: Current per-metric pass rates.
        target_pass_rate: Target overall pass rate.
        metric_id: Metric to improve.
        step: Search step size.
        max_improvement: Maximum improvement to consider.

    Returns:
        WhatIfScenario if achievable, None if not.
    """
    if metric_id not in current_rates:
        return None

    current_overall = _compute_overall(current_rates)
    if current_overall >= target_pass_rate:
        return WhatIfScenario(
            description=f"Already at target ({target_pass_rate:.0%})",
            changes={},
            projected_pass_rate=current_overall,
            current_pass_rate=current_overall,
            delta=0.0,
        )

    improvement = 0.0
    while improvement <= max_improvement:
        improvement += step
        scenario = simulate_improvement(
            current_rates, {metric_id: improvement},
            description=f"Improve {metric_id} by {improvement*100:.0f}%",
        )
        if scenario.projected_pass_rate >= target_pass_rate:
            return scenario

    return None  # not achievable


def find_best_metric_to_improve(
    current_rates: Dict[str, float],
    improvement: float = 0.1,
) -> Optional[WhatIfScenario]:
    """Find which metric would give the biggest overall improvement.

    Simulates improving each metric by the same amount and returns
    the one with the highest projected overall pass rate.

    Args:
        current_rates: Current per-metric pass rates.
        improvement: Improvement to simulate (default 10%).

    Returns:
        Best WhatIfScenario, or None if no metrics.
    """
    if not current_rates:
        return None

    best = None
    for mid in current_rates:
        scenario = simulate_improvement(
            current_rates, {mid: improvement},
            description=f"Improve {mid} by {improvement*100:.0f}%",
        )
        if best is None or scenario.projected_pass_rate > best.projected_pass_rate:
            best = scenario

    return best


def _compute_overall(rates: Dict[str, float]) -> float:
    """Compute overall pass rate as average of per-metric rates."""
    if not rates:
        return 0.0
    return sum(rates.values()) / len(rates)
