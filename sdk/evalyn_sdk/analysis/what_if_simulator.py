"""Analysis what-if simulator: model metric improvements interactively.

Answer questions like "what if helpfulness improved 20%?" or "what's the
minimum improvement needed to reach 90% overall pass rate?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class WhatIfScenario:
    """A single what-if scenario describing a metric improvement."""

    metric_id: str
    improvement_pct: float
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "improvement_pct": self.improvement_pct,
            "description": self.description,
        }


@dataclass
class WhatIfResult:
    """Result of simulating a single what-if scenario."""

    scenario: WhatIfScenario
    original_pass_rate: float
    projected_pass_rate: float
    delta: float = 0.0
    feasibility: str = "possible"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.as_dict(),
            "original_pass_rate": round(self.original_pass_rate, 4),
            "projected_pass_rate": round(self.projected_pass_rate, 4),
            "delta": round(self.delta, 4),
            "feasibility": self.feasibility,
        }

    def format_text(self) -> str:
        desc = self.scenario.description or self.scenario.metric_id
        lines = [
            f"Scenario: {desc}",
            f"  Metric: {self.scenario.metric_id} +{self.scenario.improvement_pct:.1f}%",
            f"  Original:  {self.original_pass_rate:.1%}",
            f"  Projected: {self.projected_pass_rate:.1%}",
            f"  Delta:     {self.delta * 100:+.1f}%",
            f"  Feasibility: {self.feasibility}",
        ]
        return "\n".join(lines)


@dataclass
class WhatIfReport:
    """Report containing multiple what-if simulation results."""

    results: list[WhatIfResult] = field(default_factory=list)
    best_scenario: str | None = None
    total_scenarios: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "best_scenario": self.best_scenario,
            "total_scenarios": self.total_scenarios,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WhatIfReport:
        results = []
        for r in data.get("results", []):
            scenario_data = r.get("scenario", {})
            scenario = WhatIfScenario(
                metric_id=scenario_data.get("metric_id", ""),
                improvement_pct=scenario_data.get("improvement_pct", 0.0),
                description=scenario_data.get("description", ""),
            )
            results.append(
                WhatIfResult(
                    scenario=scenario,
                    original_pass_rate=r.get("original_pass_rate", 0.0),
                    projected_pass_rate=r.get("projected_pass_rate", 0.0),
                    delta=r.get("delta", 0.0),
                    feasibility=r.get("feasibility", "possible"),
                )
            )
        return cls(
            results=results,
            best_scenario=data.get("best_scenario"),
            total_scenarios=data.get("total_scenarios", 0),
        )

    def format_text(self) -> str:
        lines = [f"What-If Report ({self.total_scenarios} scenarios)"]
        lines.append("-" * 40)
        for r in self.results:
            lines.append(r.format_text())
            lines.append("")
        if self.best_scenario:
            lines.append(f"Best scenario: {self.best_scenario}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _compute_weighted_rate(
    rates: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute weighted average pass rate.

    If weights are empty, uses equal weighting.
    """
    if not rates:
        return 0.0
    if not weights:
        return sum(rates.values()) / len(rates)

    total_weight = 0.0
    weighted_sum = 0.0
    for mid, rate in rates.items():
        w = weights.get(mid, 1.0)
        weighted_sum += rate * w
        total_weight += w

    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight


def classify_feasibility(improvement_pct: float) -> str:
    """Classify the feasibility of an improvement percentage.

    Returns:
        "easy" if < 10%, "possible" if < 30%, "hard" if < 50%, "unlikely" if >= 50%.
    """
    if improvement_pct < 10.0:
        return "easy"
    if improvement_pct < 30.0:
        return "possible"
    if improvement_pct < 50.0:
        return "hard"
    return "unlikely"


def simulate_improvement(
    metric_id: str,
    current_rates: dict[str, float],
    improvement_pct: float,
    weights: dict[str, float] | None = None,
) -> WhatIfResult:
    """Project new overall pass rate if a metric improved by improvement_pct.

    Args:
        metric_id: Which metric to improve.
        current_rates: Dict of metric_id to current pass rate (0-1).
        improvement_pct: Percentage improvement (e.g. 20 means +20%).
        weights: Optional weights per metric for computing overall rate.

    Returns:
        WhatIfResult with projected rates and feasibility.
    """
    if weights is None:
        weights = {}
    original_overall = _compute_weighted_rate(current_rates, weights)

    projected_rates = dict(current_rates)
    if metric_id in projected_rates:
        current_rate = projected_rates[metric_id]
        # improvement_pct is percentage points to add (e.g. 20 -> +0.20)
        new_rate = min(1.0, current_rate + improvement_pct / 100.0)
        projected_rates[metric_id] = new_rate

    projected_overall = _compute_weighted_rate(projected_rates, weights)

    scenario = WhatIfScenario(
        metric_id=metric_id,
        improvement_pct=improvement_pct,
        description=f"Improve {metric_id} by {improvement_pct:.1f}%",
    )

    return WhatIfResult(
        scenario=scenario,
        original_pass_rate=original_overall,
        projected_pass_rate=projected_overall,
        delta=projected_overall - original_overall,
        feasibility=classify_feasibility(improvement_pct),
    )


def simulate_multi_improvement(
    scenarios: list[WhatIfScenario],
    current_rates: dict[str, float],
    weights: dict[str, float] | None = None,
) -> WhatIfReport:
    """Simulate multiple improvements simultaneously.

    Args:
        scenarios: List of WhatIfScenario describing improvements.
        current_rates: Current per-metric pass rates.
        weights: Optional weights per metric.

    Returns:
        WhatIfReport with all results and best scenario identified.
    """
    if weights is None:
        weights = {}
    results: list[WhatIfResult] = []

    for scenario in scenarios:
        result = simulate_improvement(
            metric_id=scenario.metric_id,
            current_rates=current_rates,
            improvement_pct=scenario.improvement_pct,
            weights=weights,
        )
        # Preserve the scenario description if provided
        if scenario.description:
            result.scenario.description = scenario.description
        results.append(result)

    best_scenario_name: str | None = None
    if results:
        best = max(results, key=lambda r: r.delta)
        best_scenario_name = best.scenario.description or best.scenario.metric_id

    return WhatIfReport(
        results=results,
        best_scenario=best_scenario_name,
        total_scenarios=len(scenarios),
    )


def find_minimum_improvement(
    metric_id: str,
    current_rates: dict[str, float],
    target_pass_rate: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Binary search for minimum improvement pct needed to reach target pass rate.

    Args:
        metric_id: Which metric to improve.
        current_rates: Current per-metric pass rates.
        target_pass_rate: Target overall pass rate (0-1).
        weights: Optional weights per metric.

    Returns:
        Minimum improvement percentage needed. Returns 0.0 if already at target.
        Returns 100.0 if not achievable even with 100% improvement.
    """
    if weights is None:
        weights = {}
    original_overall = _compute_weighted_rate(current_rates, weights)
    if original_overall >= target_pass_rate:
        return 0.0

    if metric_id not in current_rates:
        return 100.0

    # Check if 100% improvement is enough
    result_at_max = simulate_improvement(metric_id, current_rates, 100.0, weights)
    if result_at_max.projected_pass_rate < target_pass_rate:
        return 100.0

    # Binary search between 0 and 100
    low = 0.0
    high = 100.0
    for _ in range(50):  # 50 iterations gives ~1e-15 precision
        mid = (low + high) / 2.0
        result = simulate_improvement(metric_id, current_rates, mid, weights)
        if result.projected_pass_rate >= target_pass_rate:
            high = mid
        else:
            low = mid

    return round(high, 2)


def render_what_if_chart(report: WhatIfReport, width: int = 60) -> str:
    """Render an ASCII bar chart of what-if scenarios.

    Args:
        report: WhatIfReport to visualize.
        width: Chart width in characters.

    Returns:
        ASCII chart string.
    """
    if not report.results:
        return "No scenarios to display."

    lines: list[str] = []
    lines.append("What-If Scenario Comparison")
    lines.append("=" * width)

    # Find max projected rate for scaling
    max_rate = max(r.projected_pass_rate for r in report.results)
    if max_rate == 0:
        max_rate = 1.0

    # Find max label length for alignment
    labels = []
    for r in report.results:
        label = f"{r.scenario.metric_id} +{r.scenario.improvement_pct:.0f}%"
        labels.append(label)

    max_label_len = max(len(lbl) for lbl in labels)
    bar_width = width - max_label_len - 12  # leave room for label + percentage

    for i, r in enumerate(report.results):
        label = labels[i].ljust(max_label_len)
        bar_len = int(r.projected_pass_rate / max_rate * bar_width) if max_rate > 0 else 0
        bar_len = max(0, bar_len)
        bar = "#" * bar_len
        pct = f" {r.projected_pass_rate:.1%}"
        delta_str = f" ({r.delta * 100:+.1f}%)"
        lines.append(f"  {label} |{bar}{pct}{delta_str}")

    return "\n".join(lines)


def find_best_roi_metric(
    current_rates: dict[str, float],
    weights: dict[str, float] | None = None,
) -> str:
    """Find which metric gives the most lift per percentage point of improvement.

    Simulates a 10% improvement on each metric and returns the one
    with the highest delta in overall pass rate.

    Args:
        current_rates: Current per-metric pass rates.
        weights: Optional weights per metric.

    Returns:
        metric_id of the best ROI metric. Returns empty string if no metrics.
    """
    if not current_rates:
        return ""

    best_metric = ""
    best_delta = -1.0

    for metric_id in current_rates:
        result = simulate_improvement(metric_id, current_rates, 10.0, weights)
        if result.delta > best_delta:
            best_delta = result.delta
            best_metric = metric_id

    return best_metric
