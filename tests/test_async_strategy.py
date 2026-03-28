"""Tests for async evaluation strategy.

All tests use synthetic data - no storage or ML deps required.
"""

from __future__ import annotations

import time

import pytest

from evalyn_sdk.evaluation.async_strategy import (
    AsyncConfig,
    AsyncReport,
    AsyncResult,
    build_async_report,
    create_async_config,
    run_parallel,
    run_sequential,
    select_strategy,
)


# ---------------------------------------------------------------------------
# AsyncConfig
# ---------------------------------------------------------------------------


class TestAsyncConfig:
    def test_defaults(self):
        cfg = AsyncConfig()
        assert cfg.max_concurrency == 10
        assert cfg.timeout_seconds == 30.0
        assert cfg.strategy == "async"

    def test_as_dict(self):
        cfg = AsyncConfig(max_concurrency=4, timeout_seconds=10.0, strategy="parallel")
        d = cfg.as_dict()
        assert d["max_concurrency"] == 4
        assert d["timeout_seconds"] == 10.0
        assert d["strategy"] == "parallel"

    def test_from_dict(self):
        d = {"max_concurrency": 8, "timeout_seconds": 15.0, "strategy": "sequential"}
        cfg = AsyncConfig.from_dict(d)
        assert cfg.max_concurrency == 8
        assert cfg.timeout_seconds == 15.0
        assert cfg.strategy == "sequential"

    def test_from_dict_defaults(self):
        cfg = AsyncConfig.from_dict({})
        assert cfg.max_concurrency == 10
        assert cfg.timeout_seconds == 30.0
        assert cfg.strategy == "async"

    def test_roundtrip(self):
        cfg = AsyncConfig(max_concurrency=3, timeout_seconds=5.0, strategy="parallel")
        restored = AsyncConfig.from_dict(cfg.as_dict())
        assert restored.max_concurrency == 3
        assert restored.timeout_seconds == 5.0
        assert restored.strategy == "parallel"


# ---------------------------------------------------------------------------
# AsyncResult
# ---------------------------------------------------------------------------


class TestAsyncResult:
    def test_success_as_dict(self):
        r = AsyncResult(
            item_id="i1", metric_id="m1", success=True,
            result={"score": 0.9}, duration_ms=12.345,
        )
        d = r.as_dict()
        assert d["item_id"] == "i1"
        assert d["success"] is True
        assert d["result"] == {"score": 0.9}
        assert d["duration_ms"] == 12.35

    def test_failure_as_dict(self):
        r = AsyncResult(
            item_id="i2", metric_id="m2", success=False,
            error="timeout", duration_ms=1000.0,
        )
        d = r.as_dict()
        assert d["success"] is False
        assert d["error"] == "timeout"
        assert d["result"] is None

    def test_default_fields(self):
        r = AsyncResult(item_id="x", metric_id="y", success=True)
        assert r.result is None
        assert r.error == ""
        assert r.duration_ms == 0.0


# ---------------------------------------------------------------------------
# AsyncReport
# ---------------------------------------------------------------------------


class TestAsyncReport:
    def test_format_text_contains_header(self):
        report = AsyncReport(total_items=5, succeeded=3, failed=2)
        text = report.format_text()
        assert "Async Strategy Report" in text
        assert "Total items:" in text
        assert "Succeeded:" in text
        assert "Failed:" in text

    def test_format_text_values(self):
        report = AsyncReport(
            total_items=10, succeeded=8, failed=2,
            total_duration_ms=500.0, avg_concurrency=3.5,
        )
        text = report.format_text()
        assert "10" in text
        assert "8" in text
        assert "500.0ms" in text
        assert "3.50" in text

    def test_as_dict(self):
        r = AsyncResult(item_id="a", metric_id="m", success=True, duration_ms=10.0)
        report = AsyncReport(
            results=[r], total_items=1, succeeded=1, failed=0,
            total_duration_ms=10.0, avg_concurrency=1.0,
        )
        d = report.as_dict()
        assert len(d["results"]) == 1
        assert d["total_items"] == 1
        assert d["succeeded"] == 1

    def test_from_dict(self):
        d = {
            "results": [
                {"item_id": "a", "metric_id": "m", "success": True, "duration_ms": 5.0},
            ],
            "total_items": 1,
            "succeeded": 1,
            "failed": 0,
            "total_duration_ms": 5.0,
            "avg_concurrency": 1.0,
        }
        report = AsyncReport.from_dict(d)
        assert len(report.results) == 1
        assert report.results[0].item_id == "a"
        assert report.total_items == 1

    def test_roundtrip(self):
        r = AsyncResult(item_id="b", metric_id="m2", success=False, error="boom")
        report = build_async_report([r])
        restored = AsyncReport.from_dict(report.as_dict())
        assert len(restored.results) == 1
        assert restored.results[0].error == "boom"
        assert restored.failed == 1


# ---------------------------------------------------------------------------
# create_async_config
# ---------------------------------------------------------------------------


class TestCreateAsyncConfig:
    def test_factory_defaults(self):
        cfg = create_async_config()
        assert cfg.max_concurrency == 10
        assert cfg.timeout_seconds == 30.0

    def test_factory_custom(self):
        cfg = create_async_config(max_concurrency=2, timeout=5.0)
        assert cfg.max_concurrency == 2
        assert cfg.timeout_seconds == 5.0


# ---------------------------------------------------------------------------
# run_sequential
# ---------------------------------------------------------------------------


class TestRunSequential:
    def test_executes_all_tasks(self):
        results_collected = []
        def make_fn(val):
            def fn():
                results_collected.append(val)
                return {"v": val}
            return fn

        tasks = [
            ("i1", "m1", make_fn(1)),
            ("i2", "m1", make_fn(2)),
            ("i3", "m1", make_fn(3)),
        ]
        report = run_sequential(tasks, AsyncConfig())
        assert report.total_items == 3
        assert report.succeeded == 3
        assert report.failed == 0
        assert results_collected == [1, 2, 3]

    def test_captures_errors(self):
        def fail():
            raise ValueError("bad input")

        tasks = [
            ("i1", "m1", lambda: {"ok": True}),
            ("i2", "m1", fail),
        ]
        report = run_sequential(tasks, AsyncConfig())
        assert report.total_items == 2
        assert report.succeeded == 1
        assert report.failed == 1
        failed = [r for r in report.results if not r.success]
        assert "bad input" in failed[0].error

    def test_empty_tasks(self):
        report = run_sequential([], AsyncConfig())
        assert report.total_items == 0
        assert report.succeeded == 0

    def test_single_task(self):
        report = run_sequential([("i1", "m1", lambda: 42)], AsyncConfig())
        assert report.total_items == 1
        assert report.succeeded == 1

    def test_all_fail(self):
        def boom():
            raise RuntimeError("fail")
        tasks = [("i1", "m1", boom), ("i2", "m2", boom), ("i3", "m3", boom)]
        report = run_sequential(tasks, AsyncConfig())
        assert report.total_items == 3
        assert report.succeeded == 0
        assert report.failed == 3

    def test_concurrency_is_one(self):
        report = run_sequential([("i1", "m1", lambda: 1)], AsyncConfig())
        assert report.avg_concurrency == 1.0

    def test_duration_recorded(self):
        def slow():
            time.sleep(0.01)
            return {"done": True}
        report = run_sequential([("i1", "m1", slow)], AsyncConfig())
        assert report.total_duration_ms > 0
        assert report.results[0].duration_ms > 0


# ---------------------------------------------------------------------------
# run_parallel
# ---------------------------------------------------------------------------


class TestRunParallel:
    def test_concurrent_execution(self):
        """All tasks should complete."""
        tasks = [
            ("i1", "m1", lambda: {"v": 1}),
            ("i2", "m1", lambda: {"v": 2}),
            ("i3", "m1", lambda: {"v": 3}),
        ]
        report = run_parallel(tasks, AsyncConfig(max_concurrency=3))
        assert report.total_items == 3
        assert report.succeeded == 3

    def test_respects_max_concurrency(self):
        """With max_concurrency=1, acts like sequential."""
        order = []
        def make_fn(val):
            def fn():
                order.append(val)
                return {"v": val}
            return fn
        tasks = [("i1", "m1", make_fn(1)), ("i2", "m1", make_fn(2))]
        report = run_parallel(tasks, AsyncConfig(max_concurrency=1))
        assert report.total_items == 2
        assert report.succeeded == 2

    def test_captures_errors(self):
        def fail():
            raise ValueError("parallel fail")
        tasks = [
            ("i1", "m1", lambda: {"ok": True}),
            ("i2", "m1", fail),
        ]
        report = run_parallel(tasks, AsyncConfig(max_concurrency=2))
        assert report.succeeded == 1
        assert report.failed == 1

    def test_empty_tasks(self):
        report = run_parallel([], AsyncConfig())
        assert report.total_items == 0

    def test_single_task(self):
        report = run_parallel([("i1", "m1", lambda: 99)], AsyncConfig())
        assert report.total_items == 1
        assert report.succeeded == 1

    def test_all_fail(self):
        def boom():
            raise RuntimeError("nope")
        tasks = [("i1", "m1", boom), ("i2", "m2", boom)]
        report = run_parallel(tasks, AsyncConfig(max_concurrency=2))
        assert report.failed == 2
        assert report.succeeded == 0


# ---------------------------------------------------------------------------
# select_strategy
# ---------------------------------------------------------------------------


class TestSelectStrategy:
    def test_small_count_sequential(self):
        assert select_strategy(1, AsyncConfig()) == "sequential"
        assert select_strategy(4, AsyncConfig()) == "sequential"

    def test_large_count_parallel(self):
        assert select_strategy(5, AsyncConfig()) == "parallel"
        assert select_strategy(100, AsyncConfig()) == "parallel"

    def test_zero_items(self):
        assert select_strategy(0, AsyncConfig()) == "sequential"

    def test_boundary(self):
        assert select_strategy(4, AsyncConfig()) == "sequential"
        assert select_strategy(5, AsyncConfig()) == "parallel"


# ---------------------------------------------------------------------------
# build_async_report
# ---------------------------------------------------------------------------


class TestBuildAsyncReport:
    def test_aggregation(self):
        r1 = AsyncResult(item_id="a", metric_id="m", success=True, duration_ms=10.0)
        r2 = AsyncResult(item_id="b", metric_id="m", success=False, error="e", duration_ms=5.0)
        report = build_async_report([r1, r2])
        assert report.total_items == 2
        assert report.succeeded == 1
        assert report.failed == 1

    def test_empty(self):
        report = build_async_report([])
        assert report.total_items == 0
        assert report.succeeded == 0
        assert report.failed == 0

    def test_all_succeed(self):
        results = [
            AsyncResult(item_id=f"i{i}", metric_id="m", success=True, duration_ms=1.0)
            for i in range(5)
        ]
        report = build_async_report(results)
        assert report.succeeded == 5
        assert report.failed == 0

    def test_override_duration(self):
        r = AsyncResult(item_id="a", metric_id="m", success=True, duration_ms=10.0)
        report = build_async_report([r], total_duration_override=99.0)
        assert report.total_duration_ms == 99.0

    def test_override_concurrency(self):
        r = AsyncResult(item_id="a", metric_id="m", success=True, duration_ms=10.0)
        report = build_async_report([r], concurrency=4.0)
        assert report.avg_concurrency == 4.0
