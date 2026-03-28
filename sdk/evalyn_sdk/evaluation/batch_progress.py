"""Batch progress polling: live progress updates while batch job is processing.

Provides a ProgressTracker that computes completion percentage, ETA,
and renders ASCII progress bars for batch evaluation jobs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProgressUpdate:
    """A single progress snapshot for a batch job."""

    job_id: str
    completed: int
    total: int
    pct: float
    elapsed_ms: float = 0.0
    eta_ms: float = 0.0
    status: str = "running"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "completed": self.completed,
            "total": self.total,
            "pct": self.pct,
            "elapsed_ms": self.elapsed_ms,
            "eta_ms": self.eta_ms,
            "status": self.status,
        }

    def format_text(self) -> str:
        lines = [
            f"Job {self.job_id}: {self.completed}/{self.total} ({self.pct:.1f}%)",
            f"  Status: {self.status}",
            f"  Elapsed: {self.elapsed_ms:.0f}ms",
        ]
        if self.eta_ms > 0:
            lines.append(f"  ETA: {self.eta_ms:.0f}ms")
        return "\n".join(lines)


@dataclass
class ProgressConfig:
    """Configuration for progress polling."""

    poll_interval_ms: float = 1000.0
    show_eta: bool = True
    show_bar: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "poll_interval_ms": self.poll_interval_ms,
            "show_eta": self.show_eta,
            "show_bar": self.show_bar,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProgressConfig:
        return cls(
            poll_interval_ms=data.get("poll_interval_ms", 1000.0),
            show_eta=data.get("show_eta", True),
            show_bar=data.get("show_bar", True),
        )


class ProgressTracker:
    """Tracks batch job progress with ETA estimation and ASCII rendering."""

    def __init__(
        self,
        job_id: str,
        total: int,
        config: Optional[ProgressConfig] = None,
    ):
        self._job_id = job_id
        self._total = total
        self._config = config or ProgressConfig()
        self._completed = 0
        self._first_update_time: Optional[float] = None
        self._last_update_time: Optional[float] = None
        self._start_time = time.monotonic()

    def update(self, completed: int) -> ProgressUpdate:
        """Record progress and compute ETA."""
        now = time.monotonic()
        if self._first_update_time is None:
            self._first_update_time = now
        self._last_update_time = now
        self._completed = completed

        elapsed = self.elapsed_ms()
        eta = self.estimate_eta_ms()
        pct = self._compute_pct()
        status = "complete" if self.is_complete() else "running"

        return ProgressUpdate(
            job_id=self._job_id,
            completed=self._completed,
            total=self._total,
            pct=pct,
            elapsed_ms=elapsed,
            eta_ms=eta,
            status=status,
        )

    def get_current(self) -> ProgressUpdate:
        """Get latest progress update."""
        elapsed = self.elapsed_ms()
        eta = self.estimate_eta_ms()
        pct = self._compute_pct()
        status = "complete" if self.is_complete() else "running"

        return ProgressUpdate(
            job_id=self._job_id,
            completed=self._completed,
            total=self._total,
            pct=pct,
            elapsed_ms=elapsed,
            eta_ms=eta,
            status=status,
        )

    def is_complete(self) -> bool:
        """Check if completed >= total."""
        return self._completed >= self._total

    def render_bar(self, width: int = 40) -> str:
        """Render ASCII progress bar like [=========>          ] 45%."""
        if self._total <= 0:
            return f"[{'?' * width}] 0%"

        pct = self._compute_pct()
        filled = int(width * self._completed / self._total)
        filled = min(filled, width)

        if filled < width and filled > 0:
            bar = "=" * (filled - 1) + ">" + " " * (width - filled)
        elif filled >= width:
            bar = "=" * width
        else:
            bar = " " * width

        return f"[{bar}] {pct:.0f}%"

    def render_status(self) -> str:
        """Full status line with bar, ETA, and counts."""
        bar = self.render_bar()
        elapsed = self.elapsed_ms()
        parts = [
            bar,
            f"{self._completed}/{self._total}",
            f"elapsed={elapsed:.0f}ms",
        ]
        if self._config.show_eta:
            eta = self.estimate_eta_ms()
            if eta > 0:
                parts.append(f"eta={eta:.0f}ms")
        return "  ".join(parts)

    def elapsed_ms(self) -> float:
        """Time in ms since first update (or since construction if no updates)."""
        ref = self._first_update_time or self._start_time
        return (time.monotonic() - ref) * 1000.0

    def estimate_eta_ms(self) -> float:
        """Estimate remaining time based on current rate."""
        if self._completed <= 0 or self._total <= 0:
            return 0.0
        if self.is_complete():
            return 0.0

        elapsed = self.elapsed_ms()
        if elapsed <= 0:
            return 0.0

        rate = self._completed / elapsed  # items per ms
        remaining = self._total - self._completed
        return remaining / rate

    def _compute_pct(self) -> float:
        if self._total <= 0:
            return 0.0
        return min(100.0, (self._completed / self._total) * 100.0)
