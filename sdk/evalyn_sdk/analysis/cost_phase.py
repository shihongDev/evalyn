"""Trace cost breakdown by phase.

Classifies spans into phases and computes per-phase cost/token aggregation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from ..models import Span

PHASE_MAPPING: dict[str, str] = {
    "llm_call": "reasoning",
    "tool_call": "tool_use",
    "retrieval": "retrieval",
    "scorer": "evaluation",
    "node": "orchestration",
    "agent": "orchestration",
    "custom": "other",
}


@dataclass
class PhaseCost:
    """Aggregated cost for a single phase."""

    phase: str
    span_count: int
    total_cost_usd: float
    total_tokens: int
    total_duration_ms: float
    percentage_cost: float  # 0-100

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "span_count": self.span_count,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "percentage_cost": self.percentage_cost,
        }


@dataclass
class CostBreakdown:
    """Full cost breakdown across all phases."""

    phases: list[PhaseCost]
    total_cost_usd: float
    total_tokens: int
    dominant_phase: str  # phase with highest cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "phases": [p.as_dict() for p in self.phases],
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "dominant_phase": self.dominant_phase,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostBreakdown:
        phases = [
            PhaseCost(
                phase=p["phase"],
                span_count=p["span_count"],
                total_cost_usd=p["total_cost_usd"],
                total_tokens=p["total_tokens"],
                total_duration_ms=p["total_duration_ms"],
                percentage_cost=p["percentage_cost"],
            )
            for p in data.get("phases", [])
        ]
        return cls(
            phases=phases,
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_tokens=data.get("total_tokens", 0),
            dominant_phase=data.get("dominant_phase", "other"),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Cost Breakdown by Phase")
        lines.append("-" * 40)
        for p in sorted(self.phases, key=lambda x: x.total_cost_usd, reverse=True):
            lines.append(
                f"  {p.phase:<15} ${p.total_cost_usd:.4f}  "
                f"{p.total_tokens:>7} tok  "
                f"{p.span_count:>3} spans  "
                f"{p.percentage_cost:5.1f}%"
            )
        lines.append("-" * 40)
        lines.append(
            f"  Total: ${self.total_cost_usd:.4f}  {self.total_tokens} tok"
        )
        lines.append(f"  Dominant phase: {self.dominant_phase}")
        return "\n".join(lines)


def classify_phase(span_type: str) -> str:
    """Map a span type to a phase name. Unknown types map to 'other'."""
    return PHASE_MAPPING.get(span_type, "other")


def compute_cost_breakdown(spans: list[Span]) -> CostBreakdown:
    """Group spans by phase and compute per-phase cost/token aggregation."""
    phase_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cost": 0.0, "tokens": 0, "duration": 0.0, "count": 0}
    )

    for s in spans:
        phase = classify_phase(s.span_type)
        d = phase_data[phase]
        d["cost"] += s.attributes.get("cost", 0)
        d["tokens"] += s.attributes.get("tokens", 0)
        d["duration"] += s.duration_ms or 0.0
        d["count"] += 1

    total_cost = sum(d["cost"] for d in phase_data.values())
    total_tokens = sum(d["tokens"] for d in phase_data.values())

    phase_costs: list[PhaseCost] = []
    for phase, d in phase_data.items():
        pct = (d["cost"] / total_cost * 100) if total_cost > 0 else 0.0
        phase_costs.append(
            PhaseCost(
                phase=phase,
                span_count=d["count"],
                total_cost_usd=d["cost"],
                total_tokens=d["tokens"],
                total_duration_ms=d["duration"],
                percentage_cost=pct,
            )
        )

    if phase_costs:
        dominant = max(phase_costs, key=lambda p: p.total_cost_usd).phase
    else:
        dominant = "other"

    return CostBreakdown(
        phases=phase_costs,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        dominant_phase=dominant,
    )
