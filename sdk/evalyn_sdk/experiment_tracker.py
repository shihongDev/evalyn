"""
Experiment tracker integration: push eval results to experiment trackers.

Provides an abstraction layer for logging metrics, parameters, and artifacts
to external experiment trackers (wandb, mlflow, neptune) or a built-in
console tracker for testing and preview.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class TrackerConfig:
    """Configuration for an experiment tracker backend."""

    tracker_type: str  # "wandb" / "mlflow" / "neptune" / "console"
    project_name: str = "evalyn"
    run_name: str = ""
    tags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tracker_type": self.tracker_type,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackerConfig:
        return cls(
            tracker_type=data["tracker_type"],
            project_name=data.get("project_name", "evalyn"),
            run_name=data.get("run_name", ""),
            tags=list(data.get("tags", [])),
        )


@dataclass
class TrackerEvent:
    """A single logged event (metric, param, or artifact)."""

    event_type: str  # "metric" / "param" / "artifact"
    key: str
    value: Any
    step: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "key": self.key,
            "value": self.value,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackerEvent:
        return cls(
            event_type=data["event_type"],
            key=data["key"],
            value=data["value"],
            step=data.get("step", 0),
        )


@dataclass
class TrackerLog:
    """Accumulated log of all events for a tracker session."""

    events: List[TrackerEvent] = field(default_factory=list)
    config: TrackerConfig = field(default_factory=lambda: TrackerConfig(tracker_type="console"))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "events": [e.as_dict() for e in self.events],
            "config": self.config.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackerLog:
        return cls(
            events=[TrackerEvent.from_dict(e) for e in data.get("events", [])],
            config=TrackerConfig.from_dict(data["config"]),
        )


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------


class ExperimentTracker(ABC):
    """Abstract base for experiment tracker backends."""

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int = 0) -> TrackerEvent:
        """Log a numeric metric."""

    @abstractmethod
    def log_param(self, key: str, value: Any) -> TrackerEvent:
        """Log a parameter (any serializable value)."""

    @abstractmethod
    def log_artifact(self, key: str, path: str) -> TrackerEvent:
        """Log an artifact by file path."""

    @abstractmethod
    def get_log(self) -> TrackerLog:
        """Return the accumulated event log."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tracker backend name."""


# ---------------------------------------------------------------------------
# Console Tracker
# ---------------------------------------------------------------------------


class ConsoleTracker(ExperimentTracker):
    """In-memory tracker that formats events as text. For testing and preview."""

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._events: List[TrackerEvent] = []

    def log_metric(self, key: str, value: float, step: int = 0) -> TrackerEvent:
        event = TrackerEvent(event_type="metric", key=key, value=value, step=step)
        self._events.append(event)
        return event

    def log_param(self, key: str, value: Any) -> TrackerEvent:
        event = TrackerEvent(event_type="param", key=key, value=value, step=0)
        self._events.append(event)
        return event

    def log_artifact(self, key: str, path: str) -> TrackerEvent:
        event = TrackerEvent(event_type="artifact", key=key, value=path, step=0)
        self._events.append(event)
        return event

    def get_log(self) -> TrackerLog:
        return TrackerLog(events=list(self._events), config=self._config)

    @property
    def name(self) -> str:
        return "console"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_tracker(config: TrackerConfig) -> ExperimentTracker:
    """Create a tracker from config. Only 'console' is built-in."""
    if config.tracker_type == "console":
        return ConsoleTracker(config)
    raise ValueError(
        f"Unsupported tracker type: {config.tracker_type!r}. "
        f"Only 'console' is built-in. Install a provider package for "
        f"'{config.tracker_type}' support."
    )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def log_eval_results(
    tracker: ExperimentTracker,
    results: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
) -> List[TrackerEvent]:
    """Log all metrics and optional params in one call."""
    events: List[TrackerEvent] = []
    if params:
        for key, value in sorted(params.items()):
            events.append(tracker.log_param(key, value))
    for key, value in sorted(results.items()):
        events.append(tracker.log_metric(key, value))
    return events


def format_tracker_log(log: TrackerLog) -> str:
    """Format a tracker log as human-readable text."""
    lines: List[str] = []
    lines.append(f"Tracker: {log.config.tracker_type}")
    lines.append(f"Project: {log.config.project_name}")
    if log.config.run_name:
        lines.append(f"Run: {log.config.run_name}")
    if log.config.tags:
        lines.append(f"Tags: {', '.join(log.config.tags)}")
    lines.append("")
    lines.append(f"Events ({len(log.events)}):")
    for event in log.events:
        if event.event_type == "metric":
            lines.append(f"  [metric] {event.key} = {event.value} (step {event.step})")
        elif event.event_type == "param":
            lines.append(f"  [param]  {event.key} = {event.value}")
        elif event.event_type == "artifact":
            lines.append(f"  [artifact] {event.key} -> {event.value}")
        else:
            lines.append(f"  [{event.event_type}] {event.key} = {event.value}")
    return "\n".join(lines)
