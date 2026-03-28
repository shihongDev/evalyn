"""
Storage auto-vacuum scheduling: trigger vacuum based on database growth.

Monitors database size over time and determines when a vacuum operation
should be performed, based on configurable growth thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class VacuumScheduleConfig:
    """Configuration for automatic vacuum scheduling."""

    growth_threshold_mb: float = 50.0
    check_interval_minutes: float = 60.0
    max_vacuum_time_seconds: float = 300.0
    enabled: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "growth_threshold_mb": self.growth_threshold_mb,
            "check_interval_minutes": self.check_interval_minutes,
            "max_vacuum_time_seconds": self.max_vacuum_time_seconds,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VacuumScheduleConfig:
        return cls(
            growth_threshold_mb=data.get("growth_threshold_mb", 50.0),
            check_interval_minutes=data.get("check_interval_minutes", 60.0),
            max_vacuum_time_seconds=data.get("max_vacuum_time_seconds", 300.0),
            enabled=data.get("enabled", True),
        )


@dataclass
class VacuumStatus:
    """Current vacuum scheduling status."""

    last_vacuum: str = ""
    last_check: str = ""
    current_size_mb: float = 0.0
    size_at_last_vacuum_mb: float = 0.0
    growth_since_vacuum_mb: float = 0.0
    needs_vacuum: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "last_vacuum": self.last_vacuum,
            "last_check": self.last_check,
            "current_size_mb": self.current_size_mb,
            "size_at_last_vacuum_mb": self.size_at_last_vacuum_mb,
            "growth_since_vacuum_mb": self.growth_since_vacuum_mb,
            "needs_vacuum": self.needs_vacuum,
        }

    def format_text(self) -> str:
        lines = [
            "Vacuum Status",
            "-" * 40,
            f"  Current size: {self.current_size_mb:.1f} MB",
            f"  Size at last vacuum: {self.size_at_last_vacuum_mb:.1f} MB",
            f"  Growth since vacuum: {self.growth_since_vacuum_mb:.1f} MB",
            f"  Needs vacuum: {'yes' if self.needs_vacuum else 'no'}",
            f"  Last vacuum: {self.last_vacuum or 'never'}",
            f"  Last check: {self.last_check or 'never'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class VacuumScheduler:
    """Schedule automatic vacuum based on database growth."""

    def __init__(self, config: VacuumScheduleConfig) -> None:
        self._config = config
        self._status = VacuumStatus()

    def check(self, current_size_mb: float) -> VacuumStatus:
        """Update status and determine if vacuum is needed."""
        now = datetime.now(timezone.utc).isoformat()
        self._status.current_size_mb = current_size_mb
        self._status.last_check = now

        # On the very first check, record the baseline size
        if self._status.size_at_last_vacuum_mb == 0.0 and not self._status.last_vacuum:
            self._status.size_at_last_vacuum_mb = current_size_mb

        growth = current_size_mb - self._status.size_at_last_vacuum_mb
        self._status.growth_since_vacuum_mb = max(growth, 0.0)

        if self._config.enabled and growth >= self._config.growth_threshold_mb:
            self._status.needs_vacuum = True
        else:
            self._status.needs_vacuum = False

        return self._status

    def record_vacuum(self, size_after_mb: float) -> None:
        """Record that a vacuum was performed."""
        now = datetime.now(timezone.utc).isoformat()
        self._status.last_vacuum = now
        self._status.size_at_last_vacuum_mb = size_after_mb
        self._status.current_size_mb = size_after_mb
        self._status.growth_since_vacuum_mb = 0.0
        self._status.needs_vacuum = False

    def get_status(self) -> VacuumStatus:
        """Return current status."""
        return self._status

    def should_vacuum(self) -> bool:
        """Quick check whether vacuum is needed."""
        return self._status.needs_vacuum

    def time_since_last_vacuum_minutes(self) -> Optional[float]:
        """Minutes since last vacuum, or None if never vacuumed."""
        if not self._status.last_vacuum:
            return None
        last = datetime.fromisoformat(self._status.last_vacuum)
        now = datetime.now(timezone.utc)
        delta = now - last
        return delta.total_seconds() / 60.0

    def reset(self) -> None:
        """Clear all status."""
        self._status = VacuumStatus()


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_vacuum_scheduler(
    growth_threshold_mb: float = 50.0,
) -> VacuumScheduler:
    """Factory: create a VacuumScheduler with the given growth threshold."""
    config = VacuumScheduleConfig(growth_threshold_mb=growth_threshold_mb)
    return VacuumScheduler(config)


def estimate_vacuum_time(db_size_mb: float) -> float:
    """Estimate vacuum duration in seconds - roughly 1 second per 100 MB."""
    if db_size_mb <= 0.0:
        return 0.0
    return db_size_mb / 100.0


def recommend_vacuum_schedule(
    db_size_mb: float,
    daily_growth_mb: float,
) -> VacuumScheduleConfig:
    """Recommend a vacuum schedule based on DB size and daily growth rate.

    Heuristics:
    - growth_threshold_mb: aim to vacuum roughly once per day -
      set threshold to daily_growth_mb (minimum 10 MB).
    - check_interval_minutes: check every 30 minutes for large/fast-growing
      databases, every 120 minutes for small/slow ones.
    - max_vacuum_time_seconds: estimate based on current DB size with 2x margin.
    """
    # Growth threshold: at least 10 MB, otherwise match daily growth
    growth_threshold = max(daily_growth_mb, 10.0)

    # Check interval: more frequent for larger or faster-growing DBs
    if db_size_mb > 1000.0 or daily_growth_mb > 100.0:
        check_interval = 30.0
    elif db_size_mb > 200.0 or daily_growth_mb > 20.0:
        check_interval = 60.0
    else:
        check_interval = 120.0

    # Max vacuum time: estimate with 2x safety margin
    estimated = estimate_vacuum_time(db_size_mb)
    max_vacuum_time = max(estimated * 2.0, 30.0)

    return VacuumScheduleConfig(
        growth_threshold_mb=growth_threshold,
        check_interval_minutes=check_interval,
        max_vacuum_time_seconds=max_vacuum_time,
        enabled=True,
    )
