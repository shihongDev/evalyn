"""Calibration cost tracking.

Track the LLM cost of the calibration process itself - tokens, calls,
and USD across alignment, optimization, and validation phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class PhaseCost:
    """Cost for a single calibration phase."""

    phase: str  # "alignment" / "optimization" / "validation"
    tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "calls": self.calls,
        }


@dataclass
class CalibrationCostRecord:
    """Full cost record for one calibration run."""

    metric_id: str
    optimizer: str
    phases: List[PhaseCost] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    timestamp: Optional[datetime] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "optimizer": self.optimizer,
            "phases": [p.as_dict() for p in self.phases],
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_calls": self.total_calls,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationCostRecord:
        phases = [
            PhaseCost(
                phase=p["phase"],
                tokens=p.get("tokens", 0),
                cost_usd=p.get("cost_usd", 0.0),
                calls=p.get("calls", 0),
            )
            for p in data.get("phases", [])
        ]
        ts_raw = data.get("timestamp")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else None
        return cls(
            metric_id=data["metric_id"],
            optimizer=data["optimizer"],
            phases=phases,
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_calls=data.get("total_calls", 0),
            timestamp=ts,
        )

    def format_text(self) -> str:
        lines = [
            f"Calibration Cost: {self.metric_id} ({self.optimizer})",
            f"  Total tokens: {self.total_tokens}",
            f"  Total cost: ${self.total_cost_usd:.4f}",
            f"  Total calls: {self.total_calls}",
        ]
        for p in self.phases:
            lines.append(
                f"  {p.phase}: {p.tokens} tokens, ${p.cost_usd:.4f}, {p.calls} calls"
            )
        return "\n".join(lines)


class CalibrationCostTracker:
    """Tracks cost across calibration phases."""

    def __init__(self, metric_id: str, optimizer: str) -> None:
        self.metric_id = metric_id
        self.optimizer = optimizer
        self._phases: Dict[str, PhaseCost] = {}

    def record(self, phase: str, tokens: int, cost_usd: float) -> None:
        """Add cost to a phase."""
        if phase not in self._phases:
            self._phases[phase] = PhaseCost(phase=phase)
        pc = self._phases[phase]
        pc.tokens += tokens
        pc.cost_usd += cost_usd
        pc.calls += 1

    def get_phase_cost(self, phase: str) -> PhaseCost:
        """Get cost for one phase."""
        if phase in self._phases:
            return self._phases[phase]
        return PhaseCost(phase=phase)

    def get_record(self) -> CalibrationCostRecord:
        """Get full record with computed totals."""
        phases = list(self._phases.values())
        total_tokens = sum(p.tokens for p in phases)
        total_cost = sum(p.cost_usd for p in phases)
        total_calls = sum(p.calls for p in phases)
        return CalibrationCostRecord(
            metric_id=self.metric_id,
            optimizer=self.optimizer,
            phases=phases,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            total_calls=total_calls,
            timestamp=datetime.now(timezone.utc),
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._phases.clear()


def compare_calibration_costs(
    records: List[CalibrationCostRecord],
) -> Dict[str, Any]:
    """Compare costs across records.

    Returns dict with cheapest, most_expensive metric_id, and average cost.
    """
    if not records:
        return {"cheapest": None, "most_expensive": None, "average_cost": 0.0}

    cheapest = min(records, key=lambda r: r.total_cost_usd)
    most_expensive = max(records, key=lambda r: r.total_cost_usd)
    avg = sum(r.total_cost_usd for r in records) / len(records)
    return {
        "cheapest": cheapest.metric_id,
        "most_expensive": most_expensive.metric_id,
        "average_cost": avg,
    }


def estimate_calibration_cost(
    num_items: int,
    num_iterations: int = 10,
    avg_tokens_per_call: int = 500,
    cost_per_1k_tokens: float = 0.001,
) -> float:
    """Estimate total cost before running calibration."""
    total_calls = num_items * num_iterations
    total_tokens = total_calls * avg_tokens_per_call
    return total_tokens / 1000 * cost_per_1k_tokens
