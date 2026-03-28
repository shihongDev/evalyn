"""Tests for calibration time budget."""

from __future__ import annotations

import sys
import time
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.time_budget import (
    TimeBudgetConfig,
    TimeBudgetStatus,
    TimeBudgetTracker,
)


# -------------------------------------------------------------------
# TimeBudgetConfig
# -------------------------------------------------------------------

class TestTimeBudgetConfig:
    def test_defaults(self):
        cfg = TimeBudgetConfig()
        assert cfg.max_minutes == 30.0
        assert cfg.warning_at_pct == 0.8

    def test_custom_values(self):
        cfg = TimeBudgetConfig(max_minutes=10.0, warning_at_pct=0.9)
        assert cfg.max_minutes == 10.0
        assert cfg.warning_at_pct == 0.9

    def test_as_dict(self):
        cfg = TimeBudgetConfig(max_minutes=5.0, warning_at_pct=0.7)
        d = cfg.as_dict()
        assert d["max_minutes"] == 5.0
        assert d["warning_at_pct"] == 0.7

    def test_from_dict(self):
        cfg = TimeBudgetConfig.from_dict({"max_minutes": 15.0, "warning_at_pct": 0.6})
        assert cfg.max_minutes == 15.0
        assert cfg.warning_at_pct == 0.6

    def test_from_dict_defaults(self):
        cfg = TimeBudgetConfig.from_dict({})
        assert cfg.max_minutes == 30.0
        assert cfg.warning_at_pct == 0.8

    def test_as_dict_roundtrip(self):
        original = TimeBudgetConfig(max_minutes=12.5, warning_at_pct=0.75)
        restored = TimeBudgetConfig.from_dict(original.as_dict())
        assert restored.max_minutes == original.max_minutes
        assert restored.warning_at_pct == original.warning_at_pct


# -------------------------------------------------------------------
# TimeBudgetStatus
# -------------------------------------------------------------------

class TestTimeBudgetStatus:
    def test_as_dict(self):
        s = TimeBudgetStatus(
            elapsed_minutes=5.0,
            remaining_minutes=25.0,
            utilization_pct=0.167,
            status="ok",
            should_stop=False,
        )
        d = s.as_dict()
        assert d["elapsed_minutes"] == 5.0
        assert d["remaining_minutes"] == 25.0
        assert d["status"] == "ok"
        assert d["should_stop"] is False

    def test_format_text_ok(self):
        s = TimeBudgetStatus(
            elapsed_minutes=5.0,
            remaining_minutes=25.0,
            utilization_pct=0.167,
            status="ok",
            should_stop=False,
        )
        text = s.format_text()
        assert "ok" in text
        assert "5.00" in text
        assert "STOP" not in text

    def test_format_text_expired(self):
        s = TimeBudgetStatus(
            elapsed_minutes=30.0,
            remaining_minutes=0.0,
            utilization_pct=1.0,
            status="expired",
            should_stop=True,
        )
        text = s.format_text()
        assert "expired" in text
        assert "STOP" in text

    def test_format_text_warning(self):
        s = TimeBudgetStatus(
            elapsed_minutes=25.0,
            remaining_minutes=5.0,
            utilization_pct=0.833,
            status="warning",
            should_stop=False,
        )
        text = s.format_text()
        assert "warning" in text


# -------------------------------------------------------------------
# TimeBudgetTracker - start/check
# -------------------------------------------------------------------

class TestTimeBudgetTrackerBasic:
    def test_start_and_check(self):
        cfg = TimeBudgetConfig(max_minutes=30.0)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        status = tracker.check()
        assert status.status == "ok"
        assert status.should_stop is False
        assert status.elapsed_minutes >= 0.0

    def test_should_stop_false_within_budget(self):
        cfg = TimeBudgetConfig(max_minutes=30.0)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        assert tracker.should_stop() is False

    def test_elapsed_zero_before_start(self):
        cfg = TimeBudgetConfig(max_minutes=30.0)
        tracker = TimeBudgetTracker(cfg)
        assert tracker.elapsed_minutes() == 0.0

    def test_remaining_equals_max_before_start(self):
        cfg = TimeBudgetConfig(max_minutes=10.0)
        tracker = TimeBudgetTracker(cfg)
        assert tracker.remaining_minutes() == 10.0

    def test_remaining_decreases_after_start(self):
        cfg = TimeBudgetConfig(max_minutes=10.0)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        time.sleep(0.05)
        assert tracker.remaining_minutes() < 10.0


# -------------------------------------------------------------------
# TimeBudgetTracker - expiry
# -------------------------------------------------------------------

class TestTimeBudgetTrackerExpiry:
    def test_should_stop_true_when_expired(self):
        # 0.001 minutes = 0.06 seconds
        cfg = TimeBudgetConfig(max_minutes=0.001)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        time.sleep(0.1)
        assert tracker.should_stop() is True

    def test_check_expired_status(self):
        cfg = TimeBudgetConfig(max_minutes=0.001)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        time.sleep(0.1)
        status = tracker.check()
        assert status.status == "expired"
        assert status.should_stop is True

    def test_remaining_zero_when_expired(self):
        cfg = TimeBudgetConfig(max_minutes=0.001)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        time.sleep(0.1)
        assert tracker.remaining_minutes() == 0.0

    def test_zero_budget_expires_immediately(self):
        cfg = TimeBudgetConfig(max_minutes=0.0)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        assert tracker.should_stop() is True
        status = tracker.check()
        assert status.status == "expired"


# -------------------------------------------------------------------
# TimeBudgetTracker - warning
# -------------------------------------------------------------------

class TestTimeBudgetTrackerWarning:
    def test_warning_status_near_limit(self):
        # Budget: 0.002 min = 0.12s, warning at 80% = 0.096s
        cfg = TimeBudgetConfig(max_minutes=0.002, warning_at_pct=0.8)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        time.sleep(0.1)
        status = tracker.check()
        # Should be warning or expired depending on timing
        assert status.status in ("warning", "expired")


# -------------------------------------------------------------------
# TimeBudgetTracker - reset
# -------------------------------------------------------------------

class TestTimeBudgetTrackerReset:
    def test_reset_clears_start(self):
        cfg = TimeBudgetConfig(max_minutes=30.0)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        time.sleep(0.01)
        tracker.reset()
        assert tracker.elapsed_minutes() == 0.0

    def test_reset_then_restart(self):
        cfg = TimeBudgetConfig(max_minutes=30.0)
        tracker = TimeBudgetTracker(cfg)
        tracker.start()
        time.sleep(0.01)
        tracker.reset()
        tracker.start()
        assert tracker.elapsed_minutes() < 0.01
