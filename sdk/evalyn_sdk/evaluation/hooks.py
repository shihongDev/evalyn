"""Evaluation event callback hooks.

Register functions that fire on evaluation lifecycle events:
on_item_complete, on_metric_complete, on_run_complete.

Usage:
    hooks = EvalHooks()
    hooks.on_item_complete(lambda item_id, results: print(f"Done: {item_id}"))
    hooks.on_run_complete(lambda run: save_to_dashboard(run))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from collections.abc import Callable

logger = logging.getLogger(__name__)


class EvalHooks:
    """Collection of evaluation lifecycle callbacks.

    All callbacks are optional. Exceptions in callbacks are caught
    and logged (never crash the evaluation).
    """

    def __init__(self):
        self._on_item_complete: list[Callable] = []
        self._on_metric_complete: list[Callable] = []
        self._on_run_complete: list[Callable] = []
        self._on_run_start: list[Callable] = []
        self._on_error: list[Callable] = []

    def on_item_complete(self, callback: Callable) -> None:
        """Register callback for when an item finishes all metrics.

        Callback signature: (item_id: str, results: List[MetricResult]) -> None
        """
        self._on_item_complete.append(callback)

    def on_metric_complete(self, callback: Callable) -> None:
        """Register callback for when a single metric completes on an item.

        Callback signature: (metric_id: str, item_id: str, result: MetricResult) -> None
        """
        self._on_metric_complete.append(callback)

    def on_run_complete(self, callback: Callable) -> None:
        """Register callback for when the full evaluation run finishes.

        Callback signature: (run: EvalRun) -> None
        """
        self._on_run_complete.append(callback)

    def on_run_start(self, callback: Callable) -> None:
        """Register callback for when an evaluation run begins.

        Callback signature: (run_id: str, item_count: int, metric_count: int) -> None
        """
        self._on_run_start.append(callback)

    def on_error(self, callback: Callable) -> None:
        """Register callback for evaluation errors.

        Callback signature: (metric_id: str, item_id: str, error: Exception) -> None
        """
        self._on_error.append(callback)

    def fire_item_complete(self, item_id: str, results: list) -> None:
        """Fire all on_item_complete callbacks."""
        for cb in self._on_item_complete:
            try:
                cb(item_id, results)
            except Exception as e:
                logger.warning("Hook on_item_complete raised: %s", e)

    def fire_metric_complete(self, metric_id: str, item_id: str, result: Any) -> None:
        """Fire all on_metric_complete callbacks."""
        for cb in self._on_metric_complete:
            try:
                cb(metric_id, item_id, result)
            except Exception as e:
                logger.warning("Hook on_metric_complete raised: %s", e)

    def fire_run_complete(self, run: Any) -> None:
        """Fire all on_run_complete callbacks."""
        for cb in self._on_run_complete:
            try:
                cb(run)
            except Exception as e:
                logger.warning("Hook on_run_complete raised: %s", e)

    def fire_run_start(self, run_id: str, item_count: int, metric_count: int) -> None:
        """Fire all on_run_start callbacks."""
        for cb in self._on_run_start:
            try:
                cb(run_id, item_count, metric_count)
            except Exception as e:
                logger.warning("Hook on_run_start raised: %s", e)

    def fire_error(self, metric_id: str, item_id: str, error: Exception) -> None:
        """Fire all on_error callbacks."""
        for cb in self._on_error:
            try:
                cb(metric_id, item_id, error)
            except Exception as e:
                logger.warning("Hook on_error raised: %s", e)

    @property
    def has_hooks(self) -> bool:
        """Check if any hooks are registered."""
        return bool(
            self._on_item_complete
            or self._on_metric_complete
            or self._on_run_complete
            or self._on_run_start
            or self._on_error
        )

    def summary(self) -> dict[str, int]:
        """Return count of registered hooks by type."""
        return {
            "on_item_complete": len(self._on_item_complete),
            "on_metric_complete": len(self._on_metric_complete),
            "on_run_complete": len(self._on_run_complete),
            "on_run_start": len(self._on_run_start),
            "on_error": len(self._on_error),
        }
