"""Evaluation throttle control: dynamic concurrency adjustment.

Adjusts the number of concurrent workers based on API response times.
Reduces workers when latency is high (provider overloaded), increases
when latency is low (headroom available).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ThrottleConfig:
    """Configuration for dynamic throttle control."""

    min_workers: int = 1
    max_workers: int = 8
    initial_workers: int = 2
    target_latency_ms: float = 2000.0   # target per-call latency
    high_latency_ms: float = 5000.0     # reduce workers above this
    low_latency_ms: float = 1000.0      # increase workers below this
    window_size: int = 10               # rolling window of latencies
    adjust_interval: int = 5            # adjust every N calls


class ThrottleController:
    """Dynamically adjusts concurrency based on API response times.

    Usage:
        throttle = ThrottleController(ThrottleConfig(max_workers=8))
        while items:
            workers = throttle.current_workers
            # ... run evaluation with `workers` concurrent threads
            throttle.record_latency(elapsed_ms)
    """

    def __init__(self, config: Optional[ThrottleConfig] = None):
        self.config = config or ThrottleConfig()
        self._current_workers = self.config.initial_workers
        self._latencies: deque = deque(maxlen=self.config.window_size)
        self._call_count = 0
        self._adjustments: List[Dict[str, Any]] = []

    @property
    def current_workers(self) -> int:
        return self._current_workers

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def adjustment_history(self) -> List[Dict[str, Any]]:
        return list(self._adjustments)

    def record_latency(self, latency_ms: float) -> None:
        """Record a call latency and potentially adjust workers.

        Args:
            latency_ms: Time taken for the API call in milliseconds.
        """
        self._latencies.append(latency_ms)
        self._call_count += 1

        if self._call_count % self.config.adjust_interval == 0:
            self._maybe_adjust()

    def _maybe_adjust(self) -> None:
        """Adjust worker count based on recent latencies."""
        if len(self._latencies) < 3:
            return

        avg = self.avg_latency_ms
        old_workers = self._current_workers

        if avg > self.config.high_latency_ms:
            # Provider is slow - reduce workers
            self._current_workers = max(
                self.config.min_workers,
                self._current_workers - 1,
            )
        elif avg < self.config.low_latency_ms:
            # Provider has headroom - increase workers
            self._current_workers = min(
                self.config.max_workers,
                self._current_workers + 1,
            )

        if self._current_workers != old_workers:
            self._adjustments.append({
                "call_count": self._call_count,
                "avg_latency_ms": round(avg, 1),
                "old_workers": old_workers,
                "new_workers": self._current_workers,
                "reason": "high_latency" if avg > self.config.high_latency_ms else "low_latency",
            })

    def summary(self) -> Dict[str, Any]:
        return {
            "current_workers": self._current_workers,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_calls": self._call_count,
            "adjustments": len(self._adjustments),
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
        }
