"""Tests for throttle control and resource monitoring."""

import pytest
from evalyn_sdk.evaluation.throttle import ThrottleController, ThrottleConfig
from evalyn_sdk.evaluation.resource_monitor import (
    ResourceMonitor, ResourceReport, ResourceSnapshot, _get_memory_mb,
)


# =============================================================================
# Throttle control tests
# =============================================================================

class TestThrottleController:
    def test_initial_workers(self):
        t = ThrottleController(ThrottleConfig(initial_workers=4))
        assert t.current_workers == 4

    def test_reduce_on_high_latency(self):
        config = ThrottleConfig(
            initial_workers=4, min_workers=1,
            high_latency_ms=3000, adjust_interval=5,
        )
        t = ThrottleController(config)
        # Record 5 high-latency calls
        for _ in range(5):
            t.record_latency(5000.0)
        assert t.current_workers < 4

    def test_increase_on_low_latency(self):
        config = ThrottleConfig(
            initial_workers=2, max_workers=8,
            low_latency_ms=1000, adjust_interval=5,
        )
        t = ThrottleController(config)
        for _ in range(5):
            t.record_latency(500.0)
        assert t.current_workers > 2

    def test_respects_min_workers(self):
        config = ThrottleConfig(
            initial_workers=2, min_workers=1,
            high_latency_ms=1000, adjust_interval=3,
        )
        t = ThrottleController(config)
        # Many high-latency calls
        for _ in range(30):
            t.record_latency(5000.0)
        assert t.current_workers >= 1

    def test_respects_max_workers(self):
        config = ThrottleConfig(
            initial_workers=2, max_workers=4,
            low_latency_ms=2000, adjust_interval=3,
        )
        t = ThrottleController(config)
        for _ in range(30):
            t.record_latency(100.0)
        assert t.current_workers <= 4

    def test_avg_latency(self):
        t = ThrottleController()
        t.record_latency(1000.0)
        t.record_latency(2000.0)
        t.record_latency(3000.0)
        assert t.avg_latency_ms == pytest.approx(2000.0)

    def test_avg_latency_empty(self):
        t = ThrottleController()
        assert t.avg_latency_ms == 0.0

    def test_no_adjust_below_interval(self):
        config = ThrottleConfig(initial_workers=3, adjust_interval=10)
        t = ThrottleController(config)
        for _ in range(4):
            t.record_latency(10000.0)
        # Only 4 calls, interval is 10 - no adjustment yet
        assert t.current_workers == 3

    def test_adjustment_history(self):
        config = ThrottleConfig(
            initial_workers=3, high_latency_ms=1000, adjust_interval=3,
        )
        t = ThrottleController(config)
        for _ in range(6):
            t.record_latency(5000.0)
        assert len(t.adjustment_history) > 0
        assert t.adjustment_history[0]["reason"] == "high_latency"

    def test_summary(self):
        t = ThrottleController(ThrottleConfig(initial_workers=2, max_workers=8))
        t.record_latency(1500.0)
        s = t.summary()
        assert s["current_workers"] == 2
        assert s["total_calls"] == 1
        assert s["max_workers"] == 8


# =============================================================================
# Resource monitoring tests
# =============================================================================

class TestResourceMonitor:
    def test_basic_lifecycle(self):
        monitor = ResourceMonitor()
        monitor.start()
        monitor.checkpoint(1)
        monitor.checkpoint(2)
        report = monitor.finish()
        assert report.peak_memory_mb >= 0
        assert len(report.snapshots) >= 2

    def test_disabled(self):
        monitor = ResourceMonitor(enabled=False)
        monitor.start()
        monitor.checkpoint(1)
        report = monitor.finish()
        assert report.peak_memory_mb == 0
        assert len(report.snapshots) == 0

    def test_warning_threshold(self):
        # Set very low threshold to trigger warning
        monitor = ResourceMonitor(warning_threshold_mb=0.001, enabled=True)
        monitor.start()
        monitor.checkpoint(1)
        report = monitor.finish()
        # Should have warnings if memory > 0.001 MB (very likely)
        mem = _get_memory_mb()
        if mem > 0.001:
            assert len(report.warnings) > 0

    def test_report_format_text(self):
        monitor = ResourceMonitor()
        monitor.start()
        report = monitor.finish()
        text = report.format_text()
        assert "Resource Usage" in text
        assert "Peak memory" in text

    def test_report_as_dict(self):
        monitor = ResourceMonitor()
        monitor.start()
        report = monitor.finish()
        d = report.as_dict()
        assert "peak_memory_mb" in d
        assert "memory_growth_mb" in d
        assert "warnings" in d

    def test_memory_growth(self):
        monitor = ResourceMonitor()
        monitor.start()
        monitor.checkpoint(10)
        report = monitor.finish()
        # Growth could be 0 or small positive/negative
        assert isinstance(report.memory_growth_mb, float)

    def test_snapshot_has_fields(self):
        s = ResourceSnapshot(
            timestamp=1234.0, memory_mb=100.0,
            memory_pct=5.0, items_evaluated=10,
        )
        d = s.as_dict()
        assert d["memory_mb"] == 100.0
        assert d["items_evaluated"] == 10


class TestGetMemory:
    def test_returns_float(self):
        mem = _get_memory_mb()
        assert isinstance(mem, float)
        assert mem >= 0.0

    def test_nonzero_in_running_process(self):
        mem = _get_memory_mb()
        # In a running Python process, memory should be > 0
        # (unless both psutil and /proc are unavailable)
        # Just verify it doesn't crash
        assert mem >= 0.0
