"""Tests for mixed-mode evaluation."""

import pytest
from evalyn_sdk.evaluation.mixed_mode import (
    MixedModeConfig,
    MixedModeReport,
    ModeDecision,
    build_mixed_mode_report,
    decide_mode,
    estimate_cost_savings,
    split_for_mixed_mode,
)


# -- ModeDecision ------------------------------------------------------------


class TestModeDecision:
    def test_as_dict(self):
        d = ModeDecision(mode="batch", reason="large", item_count=100).as_dict()
        assert d["mode"] == "batch"
        assert d["reason"] == "large"
        assert d["item_count"] == 100
        assert d["estimated_cost_usd"] == 0.0

    def test_defaults(self):
        md = ModeDecision(mode="realtime", reason="small")
        assert md.item_count == 0
        assert md.estimated_cost_usd == 0.0


# -- MixedModeConfig ---------------------------------------------------------


class TestMixedModeConfig:
    def test_defaults(self):
        cfg = MixedModeConfig()
        assert cfg.batch_threshold == 50
        assert cfg.cost_threshold_usd == 1.0
        assert cfg.prefer_batch is True

    def test_as_dict(self):
        cfg = MixedModeConfig(batch_threshold=100)
        d = cfg.as_dict()
        assert d["batch_threshold"] == 100

    def test_from_dict(self):
        cfg = MixedModeConfig.from_dict(
            {"batch_threshold": 200, "cost_threshold_usd": 5.0, "prefer_batch": False}
        )
        assert cfg.batch_threshold == 200
        assert cfg.cost_threshold_usd == 5.0
        assert cfg.prefer_batch is False

    def test_from_dict_defaults(self):
        cfg = MixedModeConfig.from_dict({})
        assert cfg.batch_threshold == 50
        assert cfg.cost_threshold_usd == 1.0
        assert cfg.prefer_batch is True

    def test_roundtrip(self):
        original = MixedModeConfig(
            batch_threshold=75, cost_threshold_usd=2.5, prefer_batch=False
        )
        restored = MixedModeConfig.from_dict(original.as_dict())
        assert restored.batch_threshold == original.batch_threshold
        assert restored.cost_threshold_usd == original.cost_threshold_usd
        assert restored.prefer_batch == original.prefer_batch


# -- MixedModeReport ---------------------------------------------------------


class TestMixedModeReport:
    def test_as_dict(self):
        r = MixedModeReport(
            mode_used="batch", items_batch=100, total_items=100
        )
        d = r.as_dict()
        assert d["mode_used"] == "batch"
        assert d["items_batch"] == 100

    def test_from_dict(self):
        r = MixedModeReport.from_dict(
            {
                "mode_used": "mixed",
                "items_batch": 80,
                "items_realtime": 20,
                "total_items": 100,
                "estimated_savings_pct": 0.4,
            }
        )
        assert r.mode_used == "mixed"
        assert r.items_batch == 80
        assert r.items_realtime == 20

    def test_roundtrip(self):
        original = MixedModeReport(
            mode_used="realtime",
            items_batch=0,
            items_realtime=30,
            total_items=30,
            estimated_savings_pct=0.0,
        )
        restored = MixedModeReport.from_dict(original.as_dict())
        assert restored.mode_used == original.mode_used
        assert restored.items_realtime == original.items_realtime

    def test_format_text(self):
        r = MixedModeReport(
            mode_used="batch",
            items_batch=50,
            items_realtime=0,
            total_items=50,
            estimated_savings_pct=0.5,
        )
        text = r.format_text()
        assert "Mixed-Mode Report" in text
        assert "batch" in text
        assert "50" in text
        assert "50.0%" in text

    def test_format_text_zero_savings(self):
        r = MixedModeReport(
            mode_used="realtime",
            items_batch=0,
            items_realtime=10,
            total_items=10,
            estimated_savings_pct=0.0,
        )
        text = r.format_text()
        assert "0.0%" in text


# -- decide_mode --------------------------------------------------------------


class TestDecideMode:
    def test_batch_for_large_count(self):
        cfg = MixedModeConfig(batch_threshold=50)
        decision = decide_mode(100, cfg)
        assert decision.mode == "batch"
        assert decision.item_count == 100

    def test_realtime_for_small_count(self):
        cfg = MixedModeConfig(batch_threshold=50)
        decision = decide_mode(10, cfg)
        assert decision.mode == "realtime"
        assert decision.item_count == 10

    def test_boundary_at_threshold(self):
        cfg = MixedModeConfig(batch_threshold=50)
        decision = decide_mode(50, cfg)
        assert decision.mode == "realtime"

    def test_just_above_threshold(self):
        cfg = MixedModeConfig(batch_threshold=50)
        decision = decide_mode(51, cfg)
        assert decision.mode == "batch"

    def test_zero_items(self):
        cfg = MixedModeConfig(batch_threshold=50)
        decision = decide_mode(0, cfg)
        assert decision.mode == "realtime"
        assert decision.estimated_cost_usd == 0.0

    def test_cost_threshold_triggers_batch(self):
        cfg = MixedModeConfig(
            batch_threshold=500, cost_threshold_usd=0.5, prefer_batch=True
        )
        # 100 items * 0.01 = $1.00, exceeds $0.50 threshold
        decision = decide_mode(100, cfg)
        assert decision.mode == "batch"

    def test_cost_threshold_no_batch_when_not_preferred(self):
        cfg = MixedModeConfig(
            batch_threshold=500, cost_threshold_usd=0.5, prefer_batch=False
        )
        decision = decide_mode(100, cfg)
        assert decision.mode == "realtime"

    def test_reason_populated(self):
        cfg = MixedModeConfig(batch_threshold=50)
        decision = decide_mode(100, cfg)
        assert len(decision.reason) > 0


# -- split_for_mixed_mode -----------------------------------------------------


class TestSplitForMixedMode:
    def test_above_threshold_all_batch(self):
        cfg = MixedModeConfig(batch_threshold=3)
        items = [1, 2, 3, 4, 5]
        batch, realtime = split_for_mixed_mode(items, cfg)
        assert batch == [1, 2, 3, 4, 5]
        assert realtime == []

    def test_below_threshold_all_realtime(self):
        cfg = MixedModeConfig(batch_threshold=10)
        items = [1, 2, 3]
        batch, realtime = split_for_mixed_mode(items, cfg)
        assert batch == []
        assert realtime == [1, 2, 3]

    def test_at_threshold_realtime(self):
        cfg = MixedModeConfig(batch_threshold=3)
        items = [1, 2, 3]
        batch, realtime = split_for_mixed_mode(items, cfg)
        assert batch == []
        assert realtime == [1, 2, 3]

    def test_empty_list(self):
        cfg = MixedModeConfig(batch_threshold=5)
        batch, realtime = split_for_mixed_mode([], cfg)
        assert batch == []
        assert realtime == []

    def test_does_not_mutate_original(self):
        cfg = MixedModeConfig(batch_threshold=2)
        items = [1, 2, 3]
        batch, realtime = split_for_mixed_mode(items, cfg)
        batch.append(99)
        assert 99 not in items


# -- estimate_cost_savings ----------------------------------------------------


class TestEstimateCostSavings:
    def test_all_batch(self):
        savings = estimate_cost_savings(100, 0)
        assert savings == pytest.approx(0.5)

    def test_all_realtime(self):
        savings = estimate_cost_savings(0, 100)
        assert savings == pytest.approx(0.0)

    def test_mixed(self):
        # 50 batch + 50 realtime
        # realtime cost: 100 * 0.01 = 1.0
        # actual cost: 50*0.005 + 50*0.01 = 0.25 + 0.50 = 0.75
        # savings: (1.0 - 0.75) / 1.0 = 0.25
        savings = estimate_cost_savings(50, 50)
        assert savings == pytest.approx(0.25)

    def test_zero_items(self):
        savings = estimate_cost_savings(0, 0)
        assert savings == 0.0

    def test_custom_cost_per_item(self):
        savings = estimate_cost_savings(100, 0, cost_per_item=0.05)
        assert savings == pytest.approx(0.5)


# -- build_mixed_mode_report --------------------------------------------------


class TestBuildMixedModeReport:
    def test_batch_only(self):
        cfg = MixedModeConfig()
        report = build_mixed_mode_report(100, 0, cfg)
        assert report.mode_used == "batch"
        assert report.items_batch == 100
        assert report.items_realtime == 0
        assert report.total_items == 100
        assert report.estimated_savings_pct == pytest.approx(0.5)

    def test_realtime_only(self):
        cfg = MixedModeConfig()
        report = build_mixed_mode_report(0, 30, cfg)
        assert report.mode_used == "realtime"
        assert report.items_realtime == 30
        assert report.estimated_savings_pct == pytest.approx(0.0)

    def test_mixed(self):
        cfg = MixedModeConfig()
        report = build_mixed_mode_report(60, 40, cfg)
        assert report.mode_used == "mixed"
        assert report.total_items == 100

    def test_zero_items(self):
        cfg = MixedModeConfig()
        report = build_mixed_mode_report(0, 0, cfg)
        assert report.mode_used == "realtime"
        assert report.total_items == 0
        assert report.estimated_savings_pct == 0.0
