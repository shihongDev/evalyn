"""Span collector statistics.

Tracks collected, orphaned, and lost spans per session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CollectorStats:
    """Snapshot of span collection statistics."""

    collected: int = 0
    expected: int = 0
    orphaned: int = 0
    lost: int = 0
    duplicates: int = 0
    warning_threshold: float = 0.05

    @property
    def loss_rate(self) -> float:
        if self.expected > 0:
            return self.lost / self.expected
        return 0.0

    @property
    def has_warning(self) -> bool:
        return self.loss_rate > self.warning_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "collected": self.collected,
            "expected": self.expected,
            "orphaned": self.orphaned,
            "lost": self.lost,
            "duplicates": self.duplicates,
            "warning_threshold": self.warning_threshold,
            "loss_rate": self.loss_rate,
            "has_warning": self.has_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectorStats:
        return cls(
            collected=data.get("collected", 0),
            expected=data.get("expected", 0),
            orphaned=data.get("orphaned", 0),
            lost=data.get("lost", 0),
            duplicates=data.get("duplicates", 0),
            warning_threshold=data.get("warning_threshold", 0.05),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Collector Stats")
        lines.append("-" * 30)
        lines.append(f"  Collected:  {self.collected}")
        lines.append(f"  Expected:   {self.expected}")
        lines.append(f"  Orphaned:   {self.orphaned}")
        lines.append(f"  Lost:       {self.lost}")
        lines.append(f"  Duplicates: {self.duplicates}")
        lines.append(f"  Loss rate:  {self.loss_rate:.2%}")
        if self.has_warning:
            lines.append("  WARNING: loss rate exceeds threshold")
        return "\n".join(lines)


class SpanCollector:
    """Mutable tracker for span collection health."""

    def __init__(self) -> None:
        self._stats = CollectorStats()
        self._seen_ids: set[str] = set()

    def record_collected(self, span_id: str) -> None:
        """Record a collected span. Detects duplicates."""
        if span_id in self._seen_ids:
            self._stats.duplicates += 1
            return
        self._seen_ids.add(span_id)
        self._stats.collected += 1

    def record_expected(self, count: int = 1) -> None:
        """Increment the expected span count."""
        self._stats.expected += count

    def record_orphaned(self, count: int = 1) -> None:
        """Increment the orphaned span count."""
        self._stats.orphaned += count

    def get_stats(self) -> CollectorStats:
        """Return computed stats (lost = expected - collected - orphaned)."""
        lost = self._stats.expected - self._stats.collected - self._stats.orphaned
        return CollectorStats(
            collected=self._stats.collected,
            expected=self._stats.expected,
            orphaned=self._stats.orphaned,
            lost=max(lost, 0),
            duplicates=self._stats.duplicates,
            warning_threshold=self._stats.warning_threshold,
        )

    def reset(self) -> None:
        """Clear all tracking state."""
        self._stats = CollectorStats()
        self._seen_ids = set()

    def check_health(self) -> list[str]:
        """Return list of warning messages. Empty if healthy."""
        stats = self.get_stats()
        warnings: list[str] = []
        if stats.has_warning:
            warnings.append(
                f"High loss rate: {stats.loss_rate:.2%} "
                f"({stats.lost} of {stats.expected} spans lost)"
            )
        if stats.duplicates > 0:
            warnings.append(
                f"Duplicate spans detected: {stats.duplicates}"
            )
        if stats.orphaned > 0:
            warnings.append(
                f"Orphaned spans: {stats.orphaned}"
            )
        return warnings
