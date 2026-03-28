"""Calibration time budget.

Stop optimization after N minutes regardless of convergence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class TimeBudgetConfig:
    """Configuration for time-budgeted calibration."""

    max_minutes: float = 30.0
    warning_at_pct: float = 0.8

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_minutes": self.max_minutes,
            "warning_at_pct": self.warning_at_pct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeBudgetConfig:
        return cls(
            max_minutes=data.get("max_minutes", 30.0),
            warning_at_pct=data.get("warning_at_pct", 0.8),
        )


@dataclass
class TimeBudgetStatus:
    """Current status of the time budget."""

    elapsed_minutes: float
    remaining_minutes: float
    utilization_pct: float
    status: str  # "ok" / "warning" / "expired"
    should_stop: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_minutes": self.elapsed_minutes,
            "remaining_minutes": self.remaining_minutes,
            "utilization_pct": self.utilization_pct,
            "status": self.status,
            "should_stop": self.should_stop,
        }

    def format_text(self) -> str:
        lines = [
            f"Time Budget: {self.status}",
            f"  Elapsed: {self.elapsed_minutes:.2f} min",
            f"  Remaining: {self.remaining_minutes:.2f} min",
            f"  Utilization: {self.utilization_pct:.1%}",
        ]
        if self.should_stop:
            lines.append("  Action: STOP - budget expired")
        return "\n".join(lines)


class TimeBudgetTracker:
    """Tracks elapsed time against a calibration time budget."""

    def __init__(self, config: TimeBudgetConfig) -> None:
        self._config = config
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Record start time."""
        self._start_time = time.monotonic()

    def elapsed_minutes(self) -> float:
        """Time since start in minutes."""
        if self._start_time is None:
            return 0.0
        return (time.monotonic() - self._start_time) / 60.0

    def remaining_minutes(self) -> float:
        """Time left in minutes."""
        remaining = self._config.max_minutes - self.elapsed_minutes()
        return max(remaining, 0.0)

    def should_stop(self) -> bool:
        """True if the budget is expired."""
        return self.elapsed_minutes() >= self._config.max_minutes

    def check(self) -> TimeBudgetStatus:
        """Check current status."""
        elapsed = self.elapsed_minutes()
        remaining = self.remaining_minutes()
        max_min = self._config.max_minutes
        utilization = elapsed / max_min if max_min > 0 else 1.0

        if elapsed >= max_min:
            status = "expired"
        elif utilization >= self._config.warning_at_pct:
            status = "warning"
        else:
            status = "ok"

        return TimeBudgetStatus(
            elapsed_minutes=elapsed,
            remaining_minutes=remaining,
            utilization_pct=utilization,
            status=status,
            should_stop=elapsed >= max_min,
        )

    def reset(self) -> None:
        """Clear start time."""
        self._start_time = None
