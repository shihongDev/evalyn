"""
Tests for storage auto-vacuum scheduling.

Run with: uv run pytest tests/test_auto_vacuum.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from evalyn_sdk.storage.auto_vacuum import (
    VacuumScheduleConfig,
    VacuumScheduler,
    VacuumStatus,
    create_vacuum_scheduler,
    estimate_vacuum_time,
    recommend_vacuum_schedule,
)


# ---------------------------------------------------------------------------
# VacuumScheduleConfig
# ---------------------------------------------------------------------------


class TestVacuumScheduleConfig:
    def test_defaults(self):
        cfg = VacuumScheduleConfig()
        assert cfg.growth_threshold_mb == 50.0
        assert cfg.check_interval_minutes == 60.0
        assert cfg.max_vacuum_time_seconds == 300.0
        assert cfg.enabled is True

    def test_as_dict(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=25.0, enabled=False)
        d = cfg.as_dict()
        assert d["growth_threshold_mb"] == 25.0
        assert d["enabled"] is False

    def test_from_dict(self):
        d = {"growth_threshold_mb": 100.0, "check_interval_minutes": 30.0}
        cfg = VacuumScheduleConfig.from_dict(d)
        assert cfg.growth_threshold_mb == 100.0
        assert cfg.check_interval_minutes == 30.0
        assert cfg.enabled is True  # default

    def test_from_dict_empty(self):
        cfg = VacuumScheduleConfig.from_dict({})
        assert cfg.growth_threshold_mb == 50.0
        assert cfg.enabled is True

    def test_roundtrip(self):
        cfg = VacuumScheduleConfig(
            growth_threshold_mb=75.0,
            check_interval_minutes=15.0,
            max_vacuum_time_seconds=600.0,
            enabled=False,
        )
        restored = VacuumScheduleConfig.from_dict(cfg.as_dict())
        assert restored.growth_threshold_mb == cfg.growth_threshold_mb
        assert restored.check_interval_minutes == cfg.check_interval_minutes
        assert restored.max_vacuum_time_seconds == cfg.max_vacuum_time_seconds
        assert restored.enabled == cfg.enabled


# ---------------------------------------------------------------------------
# VacuumStatus
# ---------------------------------------------------------------------------


class TestVacuumStatus:
    def test_defaults(self):
        s = VacuumStatus()
        assert s.current_size_mb == 0.0
        assert s.needs_vacuum is False
        assert s.last_vacuum == ""

    def test_as_dict(self):
        s = VacuumStatus(current_size_mb=120.0, needs_vacuum=True)
        d = s.as_dict()
        assert d["current_size_mb"] == 120.0
        assert d["needs_vacuum"] is True

    def test_format_text_never_vacuumed(self):
        s = VacuumStatus(current_size_mb=50.0)
        text = s.format_text()
        assert "50.0 MB" in text
        assert "never" in text
        assert "Vacuum Status" in text

    def test_format_text_with_vacuum(self):
        s = VacuumStatus(
            current_size_mb=200.0,
            size_at_last_vacuum_mb=150.0,
            growth_since_vacuum_mb=50.0,
            needs_vacuum=True,
            last_vacuum="2026-01-01T00:00:00+00:00",
            last_check="2026-01-01T01:00:00+00:00",
        )
        text = s.format_text()
        assert "200.0 MB" in text
        assert "yes" in text
        assert "2026-01-01" in text


# ---------------------------------------------------------------------------
# VacuumScheduler
# ---------------------------------------------------------------------------


class TestVacuumScheduler:
    def test_check_triggers_vacuum_above_threshold(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=10.0)
        sched = VacuumScheduler(cfg)
        # First check sets baseline at 100 MB
        sched.check(100.0)
        assert sched.should_vacuum() is False
        # Growth of 15 MB exceeds 10 MB threshold
        status = sched.check(115.0)
        assert status.needs_vacuum is True
        assert sched.should_vacuum() is True

    def test_check_below_threshold(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=50.0)
        sched = VacuumScheduler(cfg)
        sched.check(100.0)
        status = sched.check(120.0)
        assert status.needs_vacuum is False
        assert status.growth_since_vacuum_mb == 20.0

    def test_check_exact_threshold(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=10.0)
        sched = VacuumScheduler(cfg)
        sched.check(100.0)
        status = sched.check(110.0)
        assert status.needs_vacuum is True

    def test_check_disabled(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=1.0, enabled=False)
        sched = VacuumScheduler(cfg)
        sched.check(100.0)
        status = sched.check(200.0)
        assert status.needs_vacuum is False

    def test_record_vacuum_resets_growth(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=10.0)
        sched = VacuumScheduler(cfg)
        sched.check(100.0)
        sched.check(115.0)
        assert sched.should_vacuum() is True
        sched.record_vacuum(110.0)
        assert sched.should_vacuum() is False
        status = sched.get_status()
        assert status.size_at_last_vacuum_mb == 110.0
        assert status.growth_since_vacuum_mb == 0.0
        assert status.last_vacuum != ""

    def test_should_vacuum_false_initially(self):
        cfg = VacuumScheduleConfig()
        sched = VacuumScheduler(cfg)
        assert sched.should_vacuum() is False

    def test_time_since_last_vacuum_none_initially(self):
        cfg = VacuumScheduleConfig()
        sched = VacuumScheduler(cfg)
        assert sched.time_since_last_vacuum_minutes() is None

    def test_time_since_last_vacuum_after_record(self):
        cfg = VacuumScheduleConfig()
        sched = VacuumScheduler(cfg)
        sched.record_vacuum(100.0)
        minutes = sched.time_since_last_vacuum_minutes()
        assert minutes is not None
        # Should be very close to zero (just recorded)
        assert minutes < 1.0

    def test_get_status_returns_current(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=5.0)
        sched = VacuumScheduler(cfg)
        sched.check(50.0)
        status = sched.get_status()
        assert status.current_size_mb == 50.0
        assert status.last_check != ""

    def test_reset_clears_status(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=5.0)
        sched = VacuumScheduler(cfg)
        sched.check(100.0)
        sched.check(110.0)
        sched.record_vacuum(105.0)
        sched.reset()
        status = sched.get_status()
        assert status.current_size_mb == 0.0
        assert status.last_vacuum == ""
        assert status.last_check == ""
        assert status.needs_vacuum is False

    def test_check_updates_last_check(self):
        cfg = VacuumScheduleConfig()
        sched = VacuumScheduler(cfg)
        sched.check(10.0)
        status = sched.get_status()
        assert status.last_check != ""

    def test_growth_never_negative(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=10.0)
        sched = VacuumScheduler(cfg)
        sched.check(100.0)
        # Size decreased (e.g. manual cleanup)
        status = sched.check(90.0)
        assert status.growth_since_vacuum_mb == 0.0

    def test_zero_growth_no_vacuum(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=0.0)
        sched = VacuumScheduler(cfg)
        # First check sets baseline
        status = sched.check(100.0)
        # Zero growth, threshold is 0 - growth (0) >= threshold (0) is True
        assert status.needs_vacuum is True

    def test_first_check_sets_baseline(self):
        cfg = VacuumScheduleConfig(growth_threshold_mb=50.0)
        sched = VacuumScheduler(cfg)
        sched.check(200.0)
        status = sched.get_status()
        assert status.size_at_last_vacuum_mb == 200.0
        assert status.growth_since_vacuum_mb == 0.0


# ---------------------------------------------------------------------------
# Factory & Functions
# ---------------------------------------------------------------------------


class TestFactoryAndFunctions:
    def test_create_vacuum_scheduler_default(self):
        sched = create_vacuum_scheduler()
        sched.check(100.0)
        assert sched.should_vacuum() is False

    def test_create_vacuum_scheduler_custom_threshold(self):
        sched = create_vacuum_scheduler(growth_threshold_mb=5.0)
        sched.check(100.0)
        sched.check(106.0)
        assert sched.should_vacuum() is True

    def test_estimate_vacuum_time_zero(self):
        assert estimate_vacuum_time(0.0) == 0.0

    def test_estimate_vacuum_time_negative(self):
        assert estimate_vacuum_time(-10.0) == 0.0

    def test_estimate_vacuum_time_100mb(self):
        assert estimate_vacuum_time(100.0) == pytest.approx(1.0)

    def test_estimate_vacuum_time_500mb(self):
        assert estimate_vacuum_time(500.0) == pytest.approx(5.0)

    def test_recommend_small_db(self):
        cfg = recommend_vacuum_schedule(db_size_mb=50.0, daily_growth_mb=5.0)
        assert cfg.enabled is True
        assert cfg.check_interval_minutes == 120.0
        # growth threshold at least 10 MB
        assert cfg.growth_threshold_mb == 10.0

    def test_recommend_medium_db(self):
        cfg = recommend_vacuum_schedule(db_size_mb=500.0, daily_growth_mb=30.0)
        assert cfg.check_interval_minutes == 60.0
        assert cfg.growth_threshold_mb == 30.0

    def test_recommend_large_db(self):
        cfg = recommend_vacuum_schedule(db_size_mb=2000.0, daily_growth_mb=200.0)
        assert cfg.check_interval_minutes == 30.0
        assert cfg.growth_threshold_mb == 200.0

    def test_recommend_zero_growth(self):
        cfg = recommend_vacuum_schedule(db_size_mb=100.0, daily_growth_mb=0.0)
        assert cfg.growth_threshold_mb == 10.0  # minimum
        assert cfg.enabled is True
