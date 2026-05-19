"""Calibration rollback.

Revert to a previous calibration if the new one degrades alignment.
Maintains a versioned snapshot history per metric, allowing comparison
and rollback when alignment scores drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class CalibrationSnapshot:
    """A single versioned calibration snapshot."""

    metric_id: str
    prompt: str
    alignment_score: float
    timestamp: datetime
    version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "prompt": self.prompt,
            "alignment_score": self.alignment_score,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationSnapshot:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            metric_id=data["metric_id"],
            prompt=data["prompt"],
            alignment_score=data["alignment_score"],
            timestamp=ts,
            version=data["version"],
        )


@dataclass
class RollbackDecision:
    """Result of a rollback check."""

    metric_id: str
    should_rollback: bool
    current_score: float
    previous_score: float
    score_delta: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "should_rollback": self.should_rollback,
            "current_score": self.current_score,
            "previous_score": self.previous_score,
            "score_delta": self.score_delta,
            "reason": self.reason,
        }


class CalibrationHistory:
    """Maintains version history per metric."""

    def __init__(self) -> None:
        self._history: dict[str, list[CalibrationSnapshot]] = {}

    def add_snapshot(
        self, metric_id: str, prompt: str, alignment_score: float
    ) -> CalibrationSnapshot:
        """Add a new snapshot with auto-incremented version."""
        if metric_id not in self._history:
            self._history[metric_id] = []
        existing = self._history[metric_id]
        version = existing[-1].version + 1 if existing else 1
        snapshot = CalibrationSnapshot(
            metric_id=metric_id,
            prompt=prompt,
            alignment_score=alignment_score,
            timestamp=datetime.now(timezone.utc),
            version=version,
        )
        existing.append(snapshot)
        return snapshot

    def get_history(self, metric_id: str) -> list[CalibrationSnapshot]:
        """Get all snapshots for a metric, ordered by version."""
        return list(self._history.get(metric_id, []))

    def get_latest(self, metric_id: str) -> CalibrationSnapshot | None:
        """Get most recent snapshot."""
        history = self._history.get(metric_id, [])
        return history[-1] if history else None

    def get_previous(self, metric_id: str) -> CalibrationSnapshot | None:
        """Get second-to-last snapshot."""
        history = self._history.get(metric_id, [])
        return history[-2] if len(history) >= 2 else None

    def rollback(self, metric_id: str) -> CalibrationSnapshot | None:
        """Remove latest snapshot, return the new latest.

        Returns None if no history remains after removal.
        """
        history = self._history.get(metric_id, [])
        if not history:
            return None
        history.pop()
        return history[-1] if history else None

    def check_rollback(
        self, metric_id: str, degradation_threshold: float = 0.05
    ) -> RollbackDecision:
        """Compare latest vs previous. Recommend rollback if score dropped > threshold."""
        latest = self.get_latest(metric_id)
        previous = self.get_previous(metric_id)

        if latest is None or previous is None:
            return RollbackDecision(
                metric_id=metric_id,
                should_rollback=False,
                current_score=latest.alignment_score if latest else 0.0,
                previous_score=previous.alignment_score if previous else 0.0,
                score_delta=0.0,
                reason="Not enough history to compare",
            )

        delta = latest.alignment_score - previous.alignment_score
        should_rollback = delta < -degradation_threshold

        if should_rollback:
            reason = (
                f"Score dropped by {abs(delta):.4f} "
                f"(threshold: {degradation_threshold})"
            )
        else:
            reason = "Score is stable or improved"

        return RollbackDecision(
            metric_id=metric_id,
            should_rollback=should_rollback,
            current_score=latest.alignment_score,
            previous_score=previous.alignment_score,
            score_delta=delta,
            reason=reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            metric_id: [s.as_dict() for s in snapshots]
            for metric_id, snapshots in self._history.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationHistory:
        history = cls()
        for metric_id, snapshot_list in data.items():
            history._history[metric_id] = [
                CalibrationSnapshot.from_dict(s) for s in snapshot_list
            ]
        return history
