"""Evaluation resource monitoring: track memory and CPU during evaluation.

Provides lightweight resource tracking without hard dependency on psutil.
Falls back gracefully when psutil is not installed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ResourceSnapshot:
    """A point-in-time resource measurement."""

    timestamp: float       # time.time()
    memory_mb: float       # RSS in megabytes
    memory_pct: float      # memory percentage (0-100)
    items_evaluated: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "memory_mb": round(self.memory_mb, 1),
            "memory_pct": round(self.memory_pct, 1),
            "items_evaluated": self.items_evaluated,
        }


@dataclass
class ResourceReport:
    """Resource usage report for an evaluation run."""

    peak_memory_mb: float = 0.0
    peak_memory_pct: float = 0.0
    start_memory_mb: float = 0.0
    end_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    snapshots: list[ResourceSnapshot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "peak_memory_pct": round(self.peak_memory_pct, 1),
            "start_memory_mb": round(self.start_memory_mb, 1),
            "end_memory_mb": round(self.end_memory_mb, 1),
            "memory_growth_mb": round(self.memory_growth_mb, 1),
            "snapshot_count": len(self.snapshots),
            "warnings": list(self.warnings),
        }

    def format_text(self) -> str:
        lines = [
            "Resource Usage:",
            f"  Peak memory:    {self.peak_memory_mb:.1f} MB ({self.peak_memory_pct:.1f}%)",
            f"  Memory growth:  {self.memory_growth_mb:+.1f} MB",
            f"  Start:          {self.start_memory_mb:.1f} MB",
            f"  End:            {self.end_memory_mb:.1f} MB",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


class ResourceMonitor:
    """Tracks memory usage during evaluation.

    Uses psutil if available, falls back to /proc/self/status on Linux,
    or returns zeros if neither is available.

    Usage:
        monitor = ResourceMonitor(warning_threshold_mb=500)
        monitor.start()
        for item in items:
            monitor.checkpoint(items_evaluated)
        report = monitor.finish()
    """

    def __init__(
        self,
        warning_threshold_mb: float = 1000.0,
        enabled: bool = True,
    ):
        self.warning_threshold_mb = warning_threshold_mb
        self.enabled = enabled
        self._snapshots: list[ResourceSnapshot] = []
        self._warnings: list[str] = []
        self._start_memory: float = 0.0

    def start(self) -> None:
        """Record starting resource usage."""
        if not self.enabled:
            return
        self._start_memory = _get_memory_mb()
        self._snapshots = [ResourceSnapshot(
            timestamp=time.time(),
            memory_mb=self._start_memory,
            memory_pct=_get_memory_pct(),
            items_evaluated=0,
        )]

    def checkpoint(self, items_evaluated: int = 0) -> None:
        """Record a resource checkpoint.

        Args:
            items_evaluated: Number of items completed so far.
        """
        if not self.enabled:
            return

        mem_mb = _get_memory_mb()
        mem_pct = _get_memory_pct()

        self._snapshots.append(ResourceSnapshot(
            timestamp=time.time(),
            memory_mb=mem_mb,
            memory_pct=mem_pct,
            items_evaluated=items_evaluated,
        ))

        # Check threshold
        if mem_mb > self.warning_threshold_mb:
            warning = f"Memory {mem_mb:.0f}MB exceeds threshold {self.warning_threshold_mb:.0f}MB at {items_evaluated} items"
            if not self._warnings or self._warnings[-1] != warning:
                self._warnings.append(warning)

    def finish(self) -> ResourceReport:
        """Complete monitoring and return report."""
        if not self.enabled or not self._snapshots:
            return ResourceReport()

        self.checkpoint(self._snapshots[-1].items_evaluated)

        peak_mb = max(s.memory_mb for s in self._snapshots)
        peak_pct = max(s.memory_pct for s in self._snapshots)
        end_mb = self._snapshots[-1].memory_mb

        return ResourceReport(
            peak_memory_mb=peak_mb,
            peak_memory_pct=peak_pct,
            start_memory_mb=self._start_memory,
            end_memory_mb=end_mb,
            memory_growth_mb=end_mb - self._start_memory,
            snapshots=list(self._snapshots),
            warnings=list(self._warnings),
        )


def _get_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    # Fallback: read /proc/self/status on Linux
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB to MB
    except (FileNotFoundError, ValueError, IndexError):
        pass

    return 0.0


def _get_memory_pct() -> float:
    """Get current memory usage percentage."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_percent()
    except ImportError:
        return 0.0
