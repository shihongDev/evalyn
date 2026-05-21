from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BudgetConfig:
    """Configuration for a cost budget."""

    max_cost_usd: float
    warning_threshold: float = 0.8  # fraction of max to warn at
    hard_stop: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_cost_usd": self.max_cost_usd,
            "warning_threshold": self.warning_threshold,
            "hard_stop": self.hard_stop,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetConfig:
        return cls(
            max_cost_usd=data["max_cost_usd"],
            warning_threshold=data.get("warning_threshold", 0.8),
            hard_stop=data.get("hard_stop", True),
        )


@dataclass
class BudgetStatus:
    """Current status of a cost budget."""

    current_cost_usd: float
    max_cost_usd: float
    remaining_usd: float
    utilization: float  # 0-1
    status: str  # "ok", "warning", "exceeded"
    alerts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_cost_usd": self.current_cost_usd,
            "max_cost_usd": self.max_cost_usd,
            "remaining_usd": self.remaining_usd,
            "utilization": self.utilization,
            "status": self.status,
            "alerts": list(self.alerts),
        }

    def format_text(self) -> str:
        lines = [
            f"Budget: ${self.current_cost_usd:.4f} / ${self.max_cost_usd:.4f} ({self.utilization:.0%} used)",
            f"Remaining: ${self.remaining_usd:.4f}",
            f"Status: {self.status}",
        ]
        for alert in self.alerts:
            lines.append(f"  ALERT: {alert}")
        return "\n".join(lines)


class BudgetTracker:
    """Mutable tracker that accumulates cost against a budget."""

    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self._total_cost: float = 0.0
        self._run_costs: dict[str, float] = {}

    def add_cost(self, amount: float, run_id: str = "") -> None:
        """Add cost, optionally attributed to a run."""
        self._total_cost += amount
        if run_id:
            self._run_costs[run_id] = self._run_costs.get(run_id, 0.0) + amount

    def get_status(self) -> BudgetStatus:
        """Compute current budget status."""
        max_cost = self.config.max_cost_usd
        remaining = max(0.0, max_cost - self._total_cost)
        utilization = self._total_cost / max_cost if max_cost > 0 else 0.0

        alerts: list[str] = []
        if self._total_cost >= max_cost:
            status = "exceeded"
            alerts.append(f"Budget exceeded: ${self._total_cost:.4f} >= ${max_cost:.4f}")
        elif utilization >= self.config.warning_threshold:
            status = "warning"
            alerts.append(
                f"Budget warning: {utilization:.0%} used (threshold: {self.config.warning_threshold:.0%})"
            )
        else:
            status = "ok"

        return BudgetStatus(
            current_cost_usd=self._total_cost,
            max_cost_usd=max_cost,
            remaining_usd=remaining,
            utilization=utilization,
            status=status,
            alerts=alerts,
        )

    def check(self) -> BudgetStatus:
        """Alias for get_status."""
        return self.get_status()

    def should_stop(self) -> bool:
        """True if hard_stop is enabled and budget is exceeded."""
        return self.config.hard_stop and self._total_cost >= self.config.max_cost_usd

    def reset(self) -> None:
        """Reset all accumulated costs to zero."""
        self._total_cost = 0.0
        self._run_costs = {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "total_cost": self._total_cost,
            "run_costs": dict(self._run_costs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetTracker:
        config = BudgetConfig.from_dict(data["config"])
        tracker = cls(config)
        tracker._total_cost = data.get("total_cost", 0.0)
        tracker._run_costs = dict(data.get("run_costs", {}))
        return tracker


def create_budget(max_cost_usd: float, warning_threshold: float = 0.8) -> BudgetTracker:
    """Factory to create a BudgetTracker with the given limits."""
    config = BudgetConfig(max_cost_usd=max_cost_usd, warning_threshold=warning_threshold)
    return BudgetTracker(config)
