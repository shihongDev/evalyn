"""Tests for evaluation.batch_progress module."""

from __future__ import annotations

import time

import pytest

from evalyn_sdk.evaluation.batch_progress import (
    ProgressConfig,
    ProgressTracker,
    ProgressUpdate,
)


# -- ProgressUpdate ----------------------------------------------------------


class TestProgressUpdate:
    def test_as_dict(self):
        u = ProgressUpdate(
            job_id="j1", completed=5, total=10, pct=50.0,
            elapsed_ms=1000.0, eta_ms=1000.0, status="running",
        )
        d = u.as_dict()
        assert d["job_id"] == "j1"
        assert d["completed"] == 5
        assert d["total"] == 10
        assert d["pct"] == 50.0
        assert d["elapsed_ms"] == 1000.0
        assert d["eta_ms"] == 1000.0
        assert d["status"] == "running"

    def test_as_dict_defaults(self):
        u = ProgressUpdate(job_id="j2", completed=0, total=5, pct=0.0)
        d = u.as_dict()
        assert d["elapsed_ms"] == 0.0
        assert d["eta_ms"] == 0.0
        assert d["status"] == "running"

    def test_format_text_basic(self):
        u = ProgressUpdate(
            job_id="j1", completed=3, total=10, pct=30.0,
            elapsed_ms=500.0, status="running",
        )
        text = u.format_text()
        assert "j1" in text
        assert "3/10" in text
        assert "30.0%" in text
        assert "500ms" in text

    def test_format_text_with_eta(self):
        u = ProgressUpdate(
            job_id="j1", completed=5, total=10, pct=50.0,
            elapsed_ms=1000.0, eta_ms=1000.0,
        )
        text = u.format_text()
        assert "ETA" in text
        assert "1000ms" in text

    def test_format_text_no_eta(self):
        u = ProgressUpdate(
            job_id="j1", completed=5, total=10, pct=50.0,
            elapsed_ms=1000.0, eta_ms=0.0,
        )
        text = u.format_text()
        assert "ETA" not in text


# -- ProgressConfig ----------------------------------------------------------


class TestProgressConfig:
    def test_defaults(self):
        cfg = ProgressConfig()
        assert cfg.poll_interval_ms == 1000.0
        assert cfg.show_eta is True
        assert cfg.show_bar is True

    def test_as_dict(self):
        cfg = ProgressConfig(poll_interval_ms=500.0, show_eta=False, show_bar=False)
        d = cfg.as_dict()
        assert d["poll_interval_ms"] == 500.0
        assert d["show_eta"] is False
        assert d["show_bar"] is False

    def test_from_dict(self):
        d = {"poll_interval_ms": 2000.0, "show_eta": False, "show_bar": True}
        cfg = ProgressConfig.from_dict(d)
        assert cfg.poll_interval_ms == 2000.0
        assert cfg.show_eta is False
        assert cfg.show_bar is True

    def test_from_dict_defaults(self):
        cfg = ProgressConfig.from_dict({})
        assert cfg.poll_interval_ms == 1000.0
        assert cfg.show_eta is True
        assert cfg.show_bar is True

    def test_roundtrip(self):
        cfg = ProgressConfig(poll_interval_ms=750.0, show_eta=False, show_bar=True)
        restored = ProgressConfig.from_dict(cfg.as_dict())
        assert restored.poll_interval_ms == cfg.poll_interval_ms
        assert restored.show_eta == cfg.show_eta
        assert restored.show_bar == cfg.show_bar


# -- ProgressTracker ---------------------------------------------------------


class TestProgressTracker:
    def test_initial_state(self):
        t = ProgressTracker("j1", 10)
        current = t.get_current()
        assert current.completed == 0
        assert current.total == 10
        assert current.pct == 0.0
        assert current.status == "running"

    def test_update_increments(self):
        t = ProgressTracker("j1", 10)
        u1 = t.update(3)
        assert u1.completed == 3
        assert u1.pct == pytest.approx(30.0)
        u2 = t.update(7)
        assert u2.completed == 7
        assert u2.pct == pytest.approx(70.0)

    def test_get_current_returns_latest(self):
        t = ProgressTracker("j1", 10)
        t.update(5)
        current = t.get_current()
        assert current.completed == 5
        assert current.pct == pytest.approx(50.0)

    def test_is_complete_false(self):
        t = ProgressTracker("j1", 10)
        t.update(5)
        assert t.is_complete() is False

    def test_is_complete_true_at_total(self):
        t = ProgressTracker("j1", 10)
        t.update(10)
        assert t.is_complete() is True

    def test_is_complete_true_beyond_total(self):
        t = ProgressTracker("j1", 10)
        t.update(15)
        assert t.is_complete() is True

    def test_update_status_complete(self):
        t = ProgressTracker("j1", 5)
        u = t.update(5)
        assert u.status == "complete"

    def test_update_status_running(self):
        t = ProgressTracker("j1", 5)
        u = t.update(3)
        assert u.status == "running"

    def test_render_bar_zero(self):
        t = ProgressTracker("j1", 10)
        bar = t.render_bar()
        assert "[" in bar
        assert "]" in bar
        assert "0%" in bar

    def test_render_bar_partial(self):
        t = ProgressTracker("j1", 10)
        t.update(5)
        bar = t.render_bar(width=20)
        assert ">" in bar
        assert "50%" in bar

    def test_render_bar_full(self):
        t = ProgressTracker("j1", 10)
        t.update(10)
        bar = t.render_bar(width=20)
        assert ">" not in bar
        assert "=" * 20 in bar
        assert "100%" in bar

    def test_render_bar_proportional(self):
        t = ProgressTracker("j1", 4)
        t.update(1)
        bar = t.render_bar(width=40)
        # 25% of 40 = 10 filled chars
        assert "25%" in bar

    def test_render_status_includes_percentage(self):
        t = ProgressTracker("j1", 10)
        t.update(3)
        status = t.render_status()
        assert "30%" in status
        assert "3/10" in status
        assert "elapsed=" in status

    def test_render_status_with_eta(self):
        t = ProgressTracker("j1", 100)
        t.update(1)
        # Force some elapsed time
        time.sleep(0.01)
        t.update(50)
        status = t.render_status()
        assert "eta=" in status

    def test_elapsed_ms_increases(self):
        t = ProgressTracker("j1", 10)
        t.update(1)
        time.sleep(0.01)
        elapsed = t.elapsed_ms()
        assert elapsed > 0

    def test_estimate_eta_ms_zero_at_start(self):
        t = ProgressTracker("j1", 10)
        assert t.estimate_eta_ms() == 0.0

    def test_estimate_eta_ms_decreases_with_progress(self):
        t = ProgressTracker("j1", 100)
        t.update(1)
        time.sleep(0.01)
        eta1 = t.estimate_eta_ms()
        t.update(50)
        eta2 = t.estimate_eta_ms()
        # ETA should decrease as we complete more items (fewer remaining)
        assert eta2 < eta1

    def test_estimate_eta_ms_zero_when_complete(self):
        t = ProgressTracker("j1", 10)
        t.update(10)
        assert t.estimate_eta_ms() == 0.0

    def test_custom_config(self):
        cfg = ProgressConfig(poll_interval_ms=500.0, show_eta=False)
        t = ProgressTracker("j1", 10, config=cfg)
        t.update(5)
        time.sleep(0.01)
        t.update(6)
        status = t.render_status()
        # show_eta is False, so no eta in status
        assert "eta=" not in status

    def test_zero_total(self):
        t = ProgressTracker("j1", 0)
        current = t.get_current()
        assert current.pct == 0.0
        assert t.is_complete() is True  # 0 >= 0
        bar = t.render_bar()
        assert "0%" in bar

    def test_single_item(self):
        t = ProgressTracker("j1", 1)
        assert t.is_complete() is False
        t.update(1)
        assert t.is_complete() is True
        assert t.get_current().pct == pytest.approx(100.0)

    def test_render_bar_zero_total(self):
        t = ProgressTracker("j1", 0)
        bar = t.render_bar(width=20)
        assert "?" in bar
