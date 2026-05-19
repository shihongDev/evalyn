"""
Cost tracking dashboard: visualize LLM API costs over time.

Provides dataclasses and pure functions for aggregating daily costs,
trend detection, ASCII charting, spike detection, and projection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CostEntry:
    """A single cost record for one API call or batch."""

    date: str
    model: str
    cost_usd: float
    tokens: int = 0
    calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "tokens": self.tokens,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostEntry:
        return cls(
            date=data["date"],
            model=data["model"],
            cost_usd=data.get("cost_usd", 0.0),
            tokens=data.get("tokens", 0),
            calls=data.get("calls", 0),
        )


@dataclass
class DailyCost:
    """Aggregated cost for a single day."""

    date: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_calls: int = 0
    by_model: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "by_model": dict(self.by_model),
        }


@dataclass
class CostDashboard:
    """Full cost dashboard with trend analysis."""

    daily_costs: list[DailyCost] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    avg_daily_cost: float = 0.0
    trend: str = "stable"
    most_expensive_model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "daily_costs": [d.as_dict() for d in self.daily_costs],
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "avg_daily_cost": self.avg_daily_cost,
            "trend": self.trend,
            "most_expensive_model": self.most_expensive_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostDashboard:
        return cls(
            daily_costs=[],  # daily_costs are derived, not round-tripped
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_tokens=data.get("total_tokens", 0),
            avg_daily_cost=data.get("avg_daily_cost", 0.0),
            trend=data.get("trend", "stable"),
            most_expensive_model=data.get("most_expensive_model", ""),
        )

    def format_text(self) -> str:
        lines = ["COST DASHBOARD"]
        lines.append("-" * 40)
        lines.append(f"Total cost: ${self.total_cost_usd:.4f}")
        lines.append(f"Total tokens: {self.total_tokens}")
        lines.append(f"Avg daily cost: ${self.avg_daily_cost:.4f}")
        lines.append(f"Trend: {self.trend}")
        if self.most_expensive_model:
            lines.append(f"Most expensive model: {self.most_expensive_model}")

        if self.daily_costs:
            lines.append("")
            lines.append("Daily breakdown:")
            for d in self.daily_costs:
                models_str = ", ".join(
                    f"{m}=${c:.4f}" for m, c in sorted(d.by_model.items())
                )
                lines.append(
                    f"  {d.date}: ${d.total_cost_usd:.4f} "
                    f"({d.total_tokens} tokens, {d.total_calls} calls)"
                )
                if models_str:
                    lines.append(f"    {models_str}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def aggregate_daily_costs(entries: list[CostEntry]) -> list[DailyCost]:
    """Group entries by date, sum costs/tokens/calls."""
    if not entries:
        return []

    by_date: dict[str, DailyCost] = {}
    for e in entries:
        if e.date not in by_date:
            by_date[e.date] = DailyCost(date=e.date)
        d = by_date[e.date]
        d.total_cost_usd += e.cost_usd
        d.total_tokens += e.tokens
        d.total_calls += e.calls
        d.by_model[e.model] = d.by_model.get(e.model, 0.0) + e.cost_usd

    # Sort by date
    return sorted(by_date.values(), key=lambda d: d.date)


def detect_cost_trend(daily_costs: list[DailyCost]) -> str:
    """Detect trend: compare first 3 vs last 3 days average.

    Returns "increasing" if last 3 > first 3 avg,
    "decreasing" if opposite, else "stable".
    Requires at least 3 days for a meaningful comparison.
    """
    if len(daily_costs) < 3:
        return "stable"

    first_3 = daily_costs[:3]
    last_3 = daily_costs[-3:]

    first_avg = sum(d.total_cost_usd for d in first_3) / len(first_3)
    last_avg = sum(d.total_cost_usd for d in last_3) / len(last_3)

    if first_avg == 0 and last_avg == 0:
        return "stable"

    if last_avg > first_avg:
        return "increasing"
    elif last_avg < first_avg:
        return "decreasing"
    return "stable"


def build_cost_dashboard(entries: list[CostEntry]) -> CostDashboard:
    """Full cost analysis with trend detection."""
    if not entries:
        return CostDashboard()

    daily = aggregate_daily_costs(entries)
    total_cost = sum(d.total_cost_usd for d in daily)
    total_tokens = sum(d.total_tokens for d in daily)
    avg_daily = total_cost / len(daily) if daily else 0.0
    trend = detect_cost_trend(daily)

    # Find most expensive model
    model_totals: dict[str, float] = defaultdict(float)
    for e in entries:
        model_totals[e.model] += e.cost_usd

    most_expensive = ""
    if model_totals:
        most_expensive = max(model_totals, key=lambda m: model_totals[m])

    return CostDashboard(
        daily_costs=daily,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        avg_daily_cost=avg_daily,
        trend=trend,
        most_expensive_model=most_expensive,
    )


def render_cost_chart_ascii(dashboard: CostDashboard, width: int = 60) -> str:
    """ASCII bar chart of daily costs."""
    if not dashboard.daily_costs:
        return "No data"

    max_cost = max(d.total_cost_usd for d in dashboard.daily_costs)
    if max_cost == 0:
        max_cost = 1.0  # avoid division by zero

    lines = ["Daily Cost Chart", "=" * width]
    for d in dashboard.daily_costs:
        bar_len = int((d.total_cost_usd / max_cost) * (width - 20))
        bar_len = max(bar_len, 0)
        bar = "#" * bar_len
        lines.append(f"{d.date} | {bar} ${d.total_cost_usd:.4f}")

    return "\n".join(lines)


def find_cost_spikes(
    daily_costs: list[DailyCost], threshold_multiplier: float = 2.0
) -> list[str]:
    """Return dates where cost > threshold_multiplier * average."""
    if not daily_costs:
        return []

    avg = sum(d.total_cost_usd for d in daily_costs) / len(daily_costs)
    threshold = avg * threshold_multiplier

    return [d.date for d in daily_costs if d.total_cost_usd > threshold]


def project_monthly_cost(dashboard: CostDashboard) -> float:
    """Extrapolate monthly cost from avg_daily_cost * 30."""
    return dashboard.avg_daily_cost * 30
