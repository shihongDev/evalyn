from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CheckpointData:
    """Snapshot of optimizer state at a given step."""

    metric_id: str
    optimizer: str
    step: int
    best_score: float = 0.0
    best_prompt: str = ""
    current_prompt: str = ""
    scores_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "optimizer": self.optimizer,
            "step": self.step,
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
            "current_prompt": self.current_prompt,
            "scores_history": list(self.scores_history),
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        return cls(
            metric_id=data["metric_id"],
            optimizer=data["optimizer"],
            step=data["step"],
            best_score=data.get("best_score", 0.0),
            best_prompt=data.get("best_prompt", ""),
            current_prompt=data.get("current_prompt", ""),
            scores_history=data.get("scores_history", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint behavior."""

    enabled: bool = True
    save_every_n_steps: int = 5
    keep_last_n: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "save_every_n_steps": self.save_every_n_steps,
            "keep_last_n": self.keep_last_n,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointConfig:
        return cls(
            enabled=data.get("enabled", True),
            save_every_n_steps=data.get("save_every_n_steps", 5),
            keep_last_n=data.get("keep_last_n", 3),
        )


class CheckpointManager:
    """Manages saving and loading calibration checkpoints."""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self._checkpoints: dict[str, list[CheckpointData]] = {}

    def save(self, data: CheckpointData) -> None:
        """Save a checkpoint. Trims to keep_last_n per metric."""
        if not data.timestamp:
            data.timestamp = datetime.now(timezone.utc).isoformat()
        lst = self._checkpoints.setdefault(data.metric_id, [])
        lst.append(data)
        # Trim to keep_last_n
        if len(lst) > self.config.keep_last_n:
            self._checkpoints[data.metric_id] = lst[-self.config.keep_last_n :]

    def load_latest(self, metric_id: str) -> CheckpointData | None:
        """Return the most recently saved checkpoint for a metric."""
        lst = self._checkpoints.get(metric_id, [])
        if not lst:
            return None
        return lst[-1]

    def load_best(self, metric_id: str) -> CheckpointData | None:
        """Return the checkpoint with the highest best_score for a metric."""
        lst = self._checkpoints.get(metric_id, [])
        if not lst:
            return None
        return max(lst, key=lambda c: c.best_score)

    def list_checkpoints(self, metric_id: str) -> list[CheckpointData]:
        """Return all checkpoints for a metric."""
        return list(self._checkpoints.get(metric_id, []))

    def should_checkpoint(self, step: int) -> bool:
        """Return True when step is a checkpoint boundary."""
        return step % self.config.save_every_n_steps == 0

    def clear(self, metric_id: str) -> None:
        """Remove all checkpoints for a metric."""
        self._checkpoints.pop(metric_id, None)

    def clear_all(self) -> None:
        """Remove all checkpoints."""
        self._checkpoints.clear()

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "checkpoints": {
                mid: [c.as_dict() for c in clist] for mid, clist in self._checkpoints.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManager:
        config = CheckpointConfig.from_dict(data.get("config", {}))
        mgr = cls(config)
        for mid, clist in data.get("checkpoints", {}).items():
            mgr._checkpoints[mid] = [CheckpointData.from_dict(c) for c in clist]
        return mgr

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, data: str) -> CheckpointManager:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(data))
