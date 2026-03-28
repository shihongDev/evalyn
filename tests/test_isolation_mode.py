"""Tests for evaluation.isolation_mode module."""

from __future__ import annotations

import time

import pytest

from evalyn_sdk.evaluation.isolation_mode import (
    IsolatedResult,
    IsolationConfig,
    IsolationReport,
    run_all_isolated,
    run_isolated,
    should_use_isolation,
)


# -- IsolationConfig ---------------------------------------------------------


class TestIsolationConfig:
    def test_defaults(self):
        cfg = IsolationConfig()
        assert cfg.enabled is True
        assert cfg.timeout_seconds == 30.0
        assert cfg.capture_stderr is True

    def test_as_dict(self):
        cfg = IsolationConfig(enabled=False, timeout_seconds=10.0, capture_stderr=False)
        d = cfg.as_dict()
        assert d["enabled"] is False
        assert d["timeout_seconds"] == 10.0
        assert d["capture_stderr"] is False

    def test_from_dict(self):
        cfg = IsolationConfig.from_dict(
            {"enabled": False, "timeout_seconds": 5.0, "capture_stderr": False}
        )
        assert cfg.enabled is False
        assert cfg.timeout_seconds == 5.0
        assert cfg.capture_stderr is False

    def test_from_dict_defaults(self):
        cfg = IsolationConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.timeout_seconds == 30.0
        assert cfg.capture_stderr is True

    def test_roundtrip(self):
        cfg = IsolationConfig(enabled=False, timeout_seconds=15.0, capture_stderr=False)
        restored = IsolationConfig.from_dict(cfg.as_dict())
        assert restored.enabled == cfg.enabled
        assert restored.timeout_seconds == cfg.timeout_seconds
        assert restored.capture_stderr == cfg.capture_stderr


# -- IsolatedResult -----------------------------------------------------------


class TestIsolatedResult:
    def test_as_dict_success(self):
        r = IsolatedResult(metric_id="m1", success=True, score=0.95, duration_ms=12.3)
        d = r.as_dict()
        assert d["metric_id"] == "m1"
        assert d["success"] is True
        assert d["score"] == 0.95
        assert d["duration_ms"] == 12.3
        assert d["error"] == ""
        assert d["timed_out"] is False

    def test_as_dict_failure(self):
        r = IsolatedResult(
            metric_id="m2", success=False, error="boom", stderr_output="trace",
            duration_ms=5.0, timed_out=False,
        )
        d = r.as_dict()
        assert d["success"] is False
        assert d["error"] == "boom"
        assert d["stderr_output"] == "trace"

    def test_as_dict_timed_out(self):
        r = IsolatedResult(metric_id="m3", success=False, timed_out=True)
        d = r.as_dict()
        assert d["timed_out"] is True

    def test_defaults(self):
        r = IsolatedResult(metric_id="m4", success=True)
        assert r.score is None
        assert r.error == ""
        assert r.stderr_output == ""
        assert r.duration_ms == 0.0
        assert r.timed_out is False


# -- IsolationReport ----------------------------------------------------------


class TestIsolationReport:
    def test_as_dict(self):
        rpt = IsolationReport(
            results=[IsolatedResult(metric_id="m1", success=True, score=1.0)],
            total_metrics=1,
            succeeded=1,
            failed=0,
            timed_out=0,
        )
        d = rpt.as_dict()
        assert d["total_metrics"] == 1
        assert d["succeeded"] == 1
        assert len(d["results"]) == 1
        assert d["results"][0]["metric_id"] == "m1"

    def test_from_dict(self):
        data = {
            "results": [
                {"metric_id": "m1", "success": True, "score": 0.8, "duration_ms": 10.0},
                {"metric_id": "m2", "success": False, "error": "err"},
            ],
            "total_metrics": 2,
            "succeeded": 1,
            "failed": 1,
            "timed_out": 0,
        }
        rpt = IsolationReport.from_dict(data)
        assert rpt.total_metrics == 2
        assert rpt.succeeded == 1
        assert rpt.failed == 1
        assert len(rpt.results) == 2
        assert rpt.results[0].score == 0.8
        assert rpt.results[1].error == "err"

    def test_roundtrip(self):
        rpt = IsolationReport(
            results=[
                IsolatedResult(metric_id="m1", success=True, score=0.5, duration_ms=7.0),
                IsolatedResult(metric_id="m2", success=False, error="bad"),
            ],
            total_metrics=2,
            succeeded=1,
            failed=1,
            timed_out=0,
        )
        restored = IsolationReport.from_dict(rpt.as_dict())
        assert restored.total_metrics == rpt.total_metrics
        assert restored.succeeded == rpt.succeeded
        assert restored.failed == rpt.failed
        assert len(restored.results) == 2
        assert restored.results[0].metric_id == "m1"
        assert restored.results[1].error == "bad"

    def test_from_dict_empty(self):
        rpt = IsolationReport.from_dict({})
        assert rpt.total_metrics == 0
        assert rpt.results == []

    def test_format_text_success(self):
        rpt = IsolationReport(
            results=[IsolatedResult(metric_id="m1", success=True, score=0.9, duration_ms=5.0)],
            total_metrics=1,
            succeeded=1,
            failed=0,
            timed_out=0,
        )
        text = rpt.format_text()
        assert "1 metrics" in text
        assert "Succeeded: 1" in text
        assert "[OK]" in text
        assert "m1" in text
        assert "score=0.9" in text

    def test_format_text_failure(self):
        rpt = IsolationReport(
            results=[
                IsolatedResult(metric_id="m1", success=False, error="crash", duration_ms=2.0),
            ],
            total_metrics=1,
            succeeded=0,
            failed=1,
            timed_out=0,
        )
        text = rpt.format_text()
        assert "[FAIL]" in text
        assert "crash" in text

    def test_format_text_timeout(self):
        rpt = IsolationReport(
            results=[
                IsolatedResult(metric_id="m1", success=False, timed_out=True, duration_ms=30000.0),
            ],
            total_metrics=1,
            succeeded=0,
            failed=1,
            timed_out=1,
        )
        text = rpt.format_text()
        assert "[TIMEOUT]" in text
        assert "Timed out: 1" in text

    def test_format_text_empty(self):
        rpt = IsolationReport()
        text = rpt.format_text()
        assert "0 metrics" in text


# -- run_isolated -------------------------------------------------------------


class TestRunIsolated:
    def test_success(self):
        def ok_fn():
            return {"score": 0.85}

        cfg = IsolationConfig()
        result = run_isolated("metric_a", ok_fn, cfg)
        assert result.success is True
        assert result.score == 0.85
        assert result.error == ""
        assert result.metric_id == "metric_a"

    def test_exception_caught(self):
        def bad_fn():
            raise RuntimeError("kaboom")

        cfg = IsolationConfig()
        result = run_isolated("metric_b", bad_fn, cfg)
        assert result.success is False
        assert "kaboom" in result.error

    def test_captures_error_message(self):
        def bad_fn():
            raise ValueError("invalid input")

        cfg = IsolationConfig(capture_stderr=True)
        result = run_isolated("metric_c", bad_fn, cfg)
        assert result.success is False
        assert "invalid input" in result.error
        assert "ValueError" in result.stderr_output

    def test_no_capture_stderr(self):
        def bad_fn():
            raise RuntimeError("oops")

        cfg = IsolationConfig(capture_stderr=False)
        result = run_isolated("metric_d", bad_fn, cfg)
        assert result.success is False
        assert result.stderr_output == ""

    def test_duration_recorded(self):
        def slow_fn():
            time.sleep(0.02)
            return {"score": 1.0}

        cfg = IsolationConfig()
        result = run_isolated("metric_e", slow_fn, cfg)
        assert result.success is True
        assert result.duration_ms >= 15.0  # at least ~20ms

    def test_return_no_score(self):
        def no_score_fn():
            return {"status": "done"}

        cfg = IsolationConfig()
        result = run_isolated("metric_f", no_score_fn, cfg)
        assert result.success is True
        assert result.score is None


# -- run_all_isolated ---------------------------------------------------------


class TestRunAllIsolated:
    def test_mixed_results(self):
        def ok():
            return {"score": 1.0}

        def fail():
            raise RuntimeError("fail")

        metrics = [("m1", ok), ("m2", fail), ("m3", ok)]
        report = run_all_isolated(metrics)
        assert report.total_metrics == 3
        assert report.succeeded == 2
        assert report.failed == 1

    def test_empty_list(self):
        report = run_all_isolated([])
        assert report.total_metrics == 0
        assert report.succeeded == 0
        assert report.failed == 0
        assert report.results == []

    def test_all_succeed(self):
        def ok():
            return {"score": 0.5}

        metrics = [("m1", ok), ("m2", ok)]
        report = run_all_isolated(metrics)
        assert report.succeeded == 2
        assert report.failed == 0

    def test_all_fail(self):
        def fail():
            raise RuntimeError("bad")

        metrics = [("m1", fail), ("m2", fail)]
        report = run_all_isolated(metrics)
        assert report.succeeded == 0
        assert report.failed == 2

    def test_custom_config(self):
        def ok():
            return {"score": 0.7}

        cfg = IsolationConfig(capture_stderr=False)
        report = run_all_isolated([("m1", ok)], config=cfg)
        assert report.succeeded == 1


# -- should_use_isolation -----------------------------------------------------


class TestShouldUseIsolation:
    def test_low_count_no_untrusted(self):
        assert should_use_isolation(3) is False

    def test_high_count(self):
        assert should_use_isolation(6) is True

    def test_exactly_five(self):
        assert should_use_isolation(5) is False

    def test_untrusted_overrides(self):
        assert should_use_isolation(1, has_untrusted_metrics=True) is True

    def test_high_count_and_untrusted(self):
        assert should_use_isolation(10, has_untrusted_metrics=True) is True

    def test_zero_metrics(self):
        assert should_use_isolation(0) is False
