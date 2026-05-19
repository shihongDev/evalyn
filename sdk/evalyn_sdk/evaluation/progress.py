"""Evaluation progress API: structured events for external monitoring.

Emits JSON events to a file or callback for integration with CI dashboards,
Slack bots, and custom monitoring tools.

Events: run_started, item_started, item_complete, metric_scored, run_complete, error
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections.abc import Callable


@dataclass
class ProgressEvent:
    """A structured progress event."""

    event_type: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event_type,
            "timestamp": self.timestamp,
            **self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), default=str)


class ProgressTracker:
    """Tracks and emits evaluation progress events.

    Usage:
        tracker = ProgressTracker(file_path=".evalyn/progress.jsonl")
        tracker.run_started("run-1", 100, 5)
        for item in items:
            tracker.item_started(item.id)
            for metric in metrics:
                tracker.metric_scored(metric.id, item.id, result)
            tracker.item_complete(item.id, results)
        tracker.run_complete(run)
    """

    def __init__(
        self,
        file_path: str | None = None,
        callback: Callable[[ProgressEvent], None] | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._file_path = Path(file_path) if file_path else None
        self._callback = callback
        self._events: list[ProgressEvent] = []
        self._start_time: float | None = None
        self._items_complete = 0
        self._total_items = 0

    @property
    def events(self) -> list[ProgressEvent]:
        return list(self._events)

    @property
    def items_complete(self) -> int:
        return self._items_complete

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def progress_fraction(self) -> float:
        if self._total_items == 0:
            return 0.0
        return self._items_complete / self._total_items

    def run_started(self, run_id: str, item_count: int, metric_count: int) -> None:
        self._start_time = time.time()
        self._total_items = item_count
        self._emit("run_started", {
            "run_id": run_id,
            "item_count": item_count,
            "metric_count": metric_count,
        })

    def item_started(self, item_id: str) -> None:
        self._emit("item_started", {"item_id": item_id})

    def item_complete(self, item_id: str, pass_count: int = 0, fail_count: int = 0) -> None:
        self._items_complete += 1
        self._emit("item_complete", {
            "item_id": item_id,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "progress": round(self.progress_fraction, 4),
        })

    def metric_scored(self, metric_id: str, item_id: str, score: float, passed: bool) -> None:
        self._emit("metric_scored", {
            "metric_id": metric_id,
            "item_id": item_id,
            "score": score,
            "passed": passed,
        })

    def run_complete(self, run_id: str, overall_pass_rate: float = 0.0, total_cost: float = 0.0) -> None:
        self._emit("run_complete", {
            "run_id": run_id,
            "overall_pass_rate": round(overall_pass_rate, 4),
            "total_cost_usd": round(total_cost, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "items_evaluated": self._items_complete,
        })

    def error(self, metric_id: str, item_id: str, error_message: str) -> None:
        self._emit("error", {
            "metric_id": metric_id,
            "item_id": item_id,
            "error": error_message,
        })

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return

        event = ProgressEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        self._events.append(event)

        # Write to file
        if self._file_path:
            try:
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._file_path, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except OSError:
                pass

        # Fire callback
        if self._callback:
            try:
                self._callback(event)
            except Exception:
                pass

    def summary(self) -> dict[str, Any]:
        return {
            "total_events": len(self._events),
            "items_complete": self._items_complete,
            "total_items": self._total_items,
            "progress": round(self.progress_fraction, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }
