"""Subagent cost allocation for multi-agent traces.

Track cost per subagent, compare efficiency, and detect cost anomalies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models import Span


@dataclass
class SubagentCost:
    """Cost summary for a single subagent."""

    agent_id: str
    agent_name: str = ""
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    call_count: int = 0
    avg_cost_per_call: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "avg_cost_per_call": self.avg_cost_per_call,
        }


@dataclass
class CostAllocationReport:
    """Cost allocation across all subagents."""

    agents: list[SubagentCost] = field(default_factory=list)
    total_cost_usd: float = 0.0
    most_expensive_agent: str = ""
    cheapest_agent: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.as_dict() for a in self.agents],
            "total_cost_usd": self.total_cost_usd,
            "most_expensive_agent": self.most_expensive_agent,
            "cheapest_agent": self.cheapest_agent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostAllocationReport:
        agents = [
            SubagentCost(
                agent_id=a["agent_id"],
                agent_name=a.get("agent_name", ""),
                total_cost_usd=a.get("total_cost_usd", 0.0),
                total_tokens=a.get("total_tokens", 0),
                call_count=a.get("call_count", 0),
                avg_cost_per_call=a.get("avg_cost_per_call", 0.0),
            )
            for a in data.get("agents", [])
        ]
        return cls(
            agents=agents,
            total_cost_usd=data.get("total_cost_usd", 0.0),
            most_expensive_agent=data.get("most_expensive_agent", ""),
            cheapest_agent=data.get("cheapest_agent", ""),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Subagent Cost Allocation")
        lines.append("-" * 50)
        for a in sorted(self.agents, key=lambda x: x.total_cost_usd, reverse=True):
            name = a.agent_name or a.agent_id
            lines.append(
                f"  {name:<20} ${a.total_cost_usd:.4f}  "
                f"{a.total_tokens:>7} tok  "
                f"{a.call_count:>3} calls  "
                f"avg ${a.avg_cost_per_call:.4f}"
            )
        lines.append("-" * 50)
        lines.append(f"  Total: ${self.total_cost_usd:.4f}")
        if self.most_expensive_agent:
            lines.append(f"  Most expensive: {self.most_expensive_agent}")
        if self.cheapest_agent:
            lines.append(f"  Cheapest: {self.cheapest_agent}")
        return "\n".join(lines)


def extract_agent_id(span: Span) -> str:
    """Extract agent ID from span attributes or fall back to span name."""
    attrs = span.attributes
    if "agent_id" in attrs:
        return str(attrs["agent_id"])
    if "agent_name" in attrs:
        return str(attrs["agent_name"])
    return span.name


def allocate_costs(spans: list[Span]) -> CostAllocationReport:
    """Group spans by agent and compute per-agent cost/token aggregation."""
    if not spans:
        return CostAllocationReport()

    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cost": 0.0, "tokens": 0, "count": 0, "name": ""}
    )

    for s in spans:
        aid = extract_agent_id(s)
        d = buckets[aid]
        d["cost"] += s.attributes.get("cost", 0)
        d["tokens"] += s.attributes.get("tokens", 0)
        d["count"] += 1
        # Prefer agent_name attribute for display name
        if not d["name"] and "agent_name" in s.attributes:
            d["name"] = str(s.attributes["agent_name"])

    agent_costs: list[SubagentCost] = []
    for aid, d in buckets.items():
        avg = d["cost"] / d["count"] if d["count"] > 0 else 0.0
        agent_costs.append(
            SubagentCost(
                agent_id=aid,
                agent_name=d["name"],
                total_cost_usd=d["cost"],
                total_tokens=d["tokens"],
                call_count=d["count"],
                avg_cost_per_call=avg,
            )
        )

    total_cost = sum(a.total_cost_usd for a in agent_costs)

    most_expensive = ""
    cheapest = ""
    if agent_costs:
        most_expensive = max(agent_costs, key=lambda a: a.total_cost_usd).agent_id
        cheapest = min(agent_costs, key=lambda a: a.total_cost_usd).agent_id

    return CostAllocationReport(
        agents=agent_costs,
        total_cost_usd=total_cost,
        most_expensive_agent=most_expensive,
        cheapest_agent=cheapest,
    )


def compare_agent_costs(report: CostAllocationReport) -> dict[str, Any]:
    """Compare agents by cost ratio, token ratio, and efficiency ranking."""
    if not report.agents:
        return {"cost_ratio": {}, "token_ratio": {}, "efficiency_ranking": []}

    total_cost = report.total_cost_usd
    total_tokens = sum(a.total_tokens for a in report.agents)

    cost_ratio: dict[str, float] = {}
    token_ratio: dict[str, float] = {}

    for a in report.agents:
        cost_ratio[a.agent_id] = (
            a.total_cost_usd / total_cost if total_cost > 0 else 0.0
        )
        token_ratio[a.agent_id] = (
            a.total_tokens / total_tokens if total_tokens > 0 else 0.0
        )

    # Efficiency: lower avg cost per call is better
    efficiency_ranking = [
        a.agent_id
        for a in sorted(report.agents, key=lambda a: a.avg_cost_per_call)
    ]

    return {
        "cost_ratio": cost_ratio,
        "token_ratio": token_ratio,
        "efficiency_ranking": efficiency_ranking,
    }


def find_cost_anomalies(
    report: CostAllocationReport, threshold_multiplier: float = 2.0
) -> list[str]:
    """Find agents whose avg cost per call exceeds threshold * overall average."""
    if not report.agents:
        return []

    total_calls = sum(a.call_count for a in report.agents)
    if total_calls == 0:
        return []

    overall_avg = report.total_cost_usd / total_calls

    anomalies: list[str] = []
    for a in report.agents:
        if a.avg_cost_per_call > threshold_multiplier * overall_avg:
            anomalies.append(a.agent_id)

    return anomalies
