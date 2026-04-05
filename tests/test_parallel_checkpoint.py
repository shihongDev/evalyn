"""Tests for ParallelStrategy checkpoint correctness on partial completion.

Verifies that items with only partial metric results are NOT marked as
completed, preventing silent data loss on resume after KeyboardInterrupt.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from evalyn_sdk.evaluation.execution import ParallelStrategy
from evalyn_sdk.models import DatasetItem, FunctionCall, Metric, MetricResult, MetricSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(item_id: str) -> DatasetItem:
    return DatasetItem(id=item_id, input={"q": item_id}, output="ok")


def _make_call(item_id: str) -> FunctionCall:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return FunctionCall(
        id=f"call-{item_id}",
        function_name="test_fn",
        inputs={"q": item_id},
        output="ok",
        error=None,
        started_at=now,
        ended_at=now,
        duration_ms=0.0,
        session_id=None,
    )


def _make_metric(metric_id: str) -> Metric:
    spec = MetricSpec(id=metric_id, name=metric_id, type="objective")

    def handler(call, item):
        return MetricResult(
            metric_id=metric_id,
            item_id=item.id,
            call_id=call.id,
            score=1.0,
            passed=True,
        )

    return Metric(spec, handler)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_metrics_complete_marks_item_completed():
    """When all metrics finish for an item, it should be marked completed."""
    items = [(_make_item("a"), _make_call("a"))]
    metrics = [_make_metric("m1"), _make_metric("m2")]

    def evaluate_fn(metric, call, item):
        return MetricResult(
            metric_id=metric.spec.id,
            item_id=item.id,
            call_id=call.id,
            score=1.0,
            passed=True,
        )

    completed = set()
    strategy = ParallelStrategy(
        evaluate_fn=evaluate_fn,
        max_workers=1,
    )
    results = strategy.execute(items, metrics, None, "run-1", completed)

    assert len(results) == 2
    assert "a" in completed


def test_partial_metrics_does_not_mark_completed():
    """If only some metrics complete (simulating interrupt), item must NOT be marked completed."""
    items = [(_make_item("a"), _make_call("a"))]
    metrics = [_make_metric("m1"), _make_metric("m2"), _make_metric("m3")]

    call_count = 0
    barrier = threading.Event()

    def evaluate_fn(metric, call, item):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            # Simulate interrupt: block so that not all metrics complete
            barrier.wait(timeout=0.01)
            raise KeyboardInterrupt("simulated")
        return MetricResult(
            metric_id=metric.spec.id,
            item_id=item.id,
            call_id=call.id,
            score=1.0,
            passed=True,
        )

    completed = set()
    checkpointed: Dict[str, object] = {}

    def checkpoint_fn(results, completed_items, run_id):
        checkpointed["results"] = list(results)
        checkpointed["completed"] = set(completed_items)
        return True

    strategy = ParallelStrategy(
        evaluate_fn=evaluate_fn,
        checkpoint_fn=checkpoint_fn,
        max_workers=1,
    )

    with pytest.raises(KeyboardInterrupt):
        strategy.execute(items, metrics, None, "run-1", completed)

    # The item should NOT be in completed since not all 3 metrics finished
    assert "a" not in completed, (
        "Item 'a' was marked completed despite only partial metric results - "
        "this would cause data loss on resume"
    )


def test_mixed_complete_and_partial_items():
    """With multiple items, only fully-evaluated items should be marked completed."""
    items = [
        (_make_item("a"), _make_call("a")),
        (_make_item("b"), _make_call("b")),
    ]
    metrics = [_make_metric("m1"), _make_metric("m2")]

    # Track which evaluations to allow to complete
    # Item "a" gets both metrics; item "b" gets only one
    allowed = {"a-m1", "a-m2", "b-m1"}
    call_count = 0

    def evaluate_fn(metric, call, item):
        nonlocal call_count
        key = f"{item.id}-{metric.spec.id}"
        call_count += 1
        if key not in allowed:
            raise KeyboardInterrupt("simulated")
        return MetricResult(
            metric_id=metric.spec.id,
            item_id=item.id,
            call_id=call.id,
            score=1.0,
            passed=True,
        )

    completed = set()
    strategy = ParallelStrategy(
        evaluate_fn=evaluate_fn,
        max_workers=1,
    )

    with pytest.raises(KeyboardInterrupt):
        strategy.execute(items, metrics, None, "run-1", completed)

    # Item "a" had all metrics complete, item "b" did not
    assert "a" in completed
    assert "b" not in completed
