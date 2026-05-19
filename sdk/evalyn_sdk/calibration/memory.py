"""Calibration memory.

Remember what approaches failed in past calibration runs.
Tracks attempts per metric so the engine can avoid repeating
failed optimizers and prefer successful ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CalibrationAttempt:
    """A single calibration attempt record."""

    metric_id: str
    optimizer: str
    prompt_summary: str
    alignment_score: float
    success: bool
    timestamp: datetime
    failure_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "optimizer": self.optimizer,
            "prompt_summary": self.prompt_summary,
            "alignment_score": self.alignment_score,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationAttempt:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            metric_id=data["metric_id"],
            optimizer=data["optimizer"],
            prompt_summary=data["prompt_summary"],
            alignment_score=data["alignment_score"],
            success=data["success"],
            timestamp=ts,
            failure_reason=data.get("failure_reason", ""),
        )


@dataclass
class CalibrationMemoryStore:
    """Stores calibration attempts and provides query helpers."""

    attempts: list[CalibrationAttempt] = field(default_factory=list)

    def record_attempt(self, attempt: CalibrationAttempt) -> None:
        """Add an attempt to the store."""
        self.attempts.append(attempt)

    def get_failed_approaches(self, metric_id: str) -> list[CalibrationAttempt]:
        """Get failed attempts for a metric."""
        return [
            a for a in self.attempts if a.metric_id == metric_id and not a.success
        ]

    def get_successful_approaches(self, metric_id: str) -> list[CalibrationAttempt]:
        """Get successful attempts for a metric."""
        return [
            a for a in self.attempts if a.metric_id == metric_id and a.success
        ]

    def get_best_approach(self, metric_id: str) -> CalibrationAttempt | None:
        """Return the highest-scoring successful attempt for a metric."""
        successes = self.get_successful_approaches(metric_id)
        if not successes:
            return None
        return max(successes, key=lambda a: a.alignment_score)

    def has_been_tried(self, metric_id: str, optimizer: str) -> bool:
        """Check if a given optimizer was tried for a metric."""
        return any(
            a.metric_id == metric_id and a.optimizer == optimizer
            for a in self.attempts
        )

    def suggest_next_optimizer(
        self, metric_id: str, all_optimizers: list[str]
    ) -> str | None:
        """Return the first untried optimizer for a metric, or None."""
        for opt in all_optimizers:
            if not self.has_been_tried(metric_id, opt):
                return opt
        return None

    @property
    def count(self) -> int:
        """Number of recorded attempts."""
        return len(self.attempts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": [a.as_dict() for a in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationMemoryStore:
        return cls(
            attempts=[
                CalibrationAttempt.from_dict(a) for a in data.get("attempts", [])
            ],
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, data: str) -> CalibrationMemoryStore:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(data))
