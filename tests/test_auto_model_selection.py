"""Tests for auto_model_selection module."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.auto_model_selection import (
    AutoSelectionConfig,
    DEFAULT_TIERS,
    ModelTier,
    SelectionReport,
    SelectionResult,
    estimate_complexity,
    estimate_cost_savings,
    get_default_config,
    select_model,
    select_models_batch,
)


# ---------------------------------------------------------------------------
# estimate_complexity
# ---------------------------------------------------------------------------


class TestEstimateComplexity:
    def test_short_text_low_complexity(self):
        c = estimate_complexity("hi")
        assert c < 0.3

    def test_empty_text_zero(self):
        c = estimate_complexity("")
        assert c == 0.0

    def test_long_text_high_complexity(self):
        c = estimate_complexity("word " * 1000)
        assert c > 0.7

    def test_medium_text_moderate(self):
        c = estimate_complexity("a " * 100)
        assert 0.2 < c < 0.9

    def test_with_low_score_increases_complexity(self):
        base = estimate_complexity("hello world")
        with_low = estimate_complexity("hello world", avg_score=0.1)
        assert with_low > base

    def test_with_high_score_decreases_complexity(self):
        long_text = "word " * 500
        base = estimate_complexity(long_text)
        with_high = estimate_complexity(long_text, avg_score=1.0)
        assert with_high <= base

    def test_score_none_ignored(self):
        c1 = estimate_complexity("test text")
        c2 = estimate_complexity("test text", avg_score=None)
        assert c1 == c2

    def test_returns_float_in_range(self):
        c = estimate_complexity("any text", avg_score=0.5)
        assert 0.0 <= c <= 1.0

    def test_score_clamped_below_zero(self):
        c = estimate_complexity("text", avg_score=-0.5)
        assert 0.0 <= c <= 1.0

    def test_score_clamped_above_one(self):
        c = estimate_complexity("text", avg_score=1.5)
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# select_model
# ---------------------------------------------------------------------------


class TestSelectModel:
    def test_fast_tier_for_low_complexity(self):
        config = get_default_config()
        tier = select_model(0.2, config)
        assert tier.name == "fast"

    def test_balanced_tier_for_medium_complexity(self):
        config = get_default_config()
        tier = select_model(0.6, config)
        assert tier.name == "balanced"

    def test_smart_tier_for_high_complexity(self):
        config = get_default_config()
        tier = select_model(0.9, config)
        assert tier.name == "smart"

    def test_zero_complexity_uses_fast(self):
        config = get_default_config()
        tier = select_model(0.0, config)
        assert tier.name == "fast"

    def test_exact_threshold_boundary(self):
        config = get_default_config()
        # complexity=0.5 should fit in fast tier (max_complexity=0.5)
        tier = select_model(0.5, config)
        assert tier.name == "fast"

    def test_above_all_tiers_uses_highest(self):
        config = get_default_config()
        tier = select_model(1.0, config)
        assert tier.name == "smart"

    def test_no_tiers_returns_default(self):
        config = AutoSelectionConfig(tiers=[], default_tier="fallback")
        tier = select_model(0.5, config)
        assert tier.name == "fallback"


# ---------------------------------------------------------------------------
# select_models_batch
# ---------------------------------------------------------------------------


class TestSelectModelsBatch:
    def test_distribution_mixed(self):
        items = [
            {"id": "short", "text": "hi"},
            {"id": "long", "text": "word " * 1000},
        ]
        report = select_models_batch(items)
        assert len(report.selections) == 2
        assert sum(report.tier_distribution.values()) == 2

    def test_empty_items(self):
        report = select_models_batch([])
        assert report.selections == []
        assert report.tier_distribution == {}

    def test_all_short_items_use_fast(self):
        items = [{"id": f"i{i}", "text": "ok"} for i in range(5)]
        report = select_models_batch(items)
        assert report.tier_distribution.get("fast", 0) == 5

    def test_custom_config(self):
        config = AutoSelectionConfig(
            tiers=[ModelTier("only", "model-x", "test", 0.001, 1.0)],
        )
        items = [{"id": "1", "text": "anything"}]
        report = select_models_batch(items, config=config)
        assert report.selections[0].selected_tier == "only"

    def test_items_with_avg_score(self):
        items = [
            {"id": "1", "text": "hello", "avg_score": 0.1},
        ]
        report = select_models_batch(items)
        assert len(report.selections) == 1
        assert report.selections[0].complexity > 0

    def test_report_has_cost_savings(self):
        items = [{"id": "1", "text": "short"}]
        report = select_models_batch(items)
        assert isinstance(report.estimated_cost_savings_pct, float)


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------


class TestGetDefaultConfig:
    def test_has_tiers(self):
        config = get_default_config()
        assert len(config.tiers) == 3

    def test_tier_names(self):
        config = get_default_config()
        names = [t.name for t in config.tiers]
        assert "fast" in names
        assert "balanced" in names
        assert "smart" in names

    def test_tiers_sorted_by_max_complexity(self):
        config = get_default_config()
        complexities = [t.max_complexity for t in config.tiers]
        assert complexities == sorted(complexities)


# ---------------------------------------------------------------------------
# estimate_cost_savings
# ---------------------------------------------------------------------------


class TestEstimateCostSavings:
    def test_all_fast_saves_money(self):
        selections = [
            SelectionResult("i1", "gemini-2.5-flash", "fast", 0.1),
            SelectionResult("i2", "gemini-2.5-flash", "fast", 0.2),
        ]
        report = SelectionReport(
            selections=selections,
            tier_distribution={"fast": 2},
        )
        savings = estimate_cost_savings(report)
        assert savings > 0

    def test_all_smart_no_savings(self):
        selections = [
            SelectionResult("i1", "gemini-2.5-pro", "smart", 0.9),
        ]
        report = SelectionReport(
            selections=selections,
            tier_distribution={"smart": 1},
        )
        savings = estimate_cost_savings(report)
        assert savings == 0.0

    def test_empty_report(self):
        report = SelectionReport()
        savings = estimate_cost_savings(report)
        assert savings == 0.0


# ---------------------------------------------------------------------------
# SelectionReport format_text
# ---------------------------------------------------------------------------


class TestSelectionReportFormatText:
    def test_format_text_contains_header(self):
        report = SelectionReport(
            selections=[
                SelectionResult("i1", "m", "fast", 0.1),
            ],
            tier_distribution={"fast": 1},
            estimated_cost_savings_pct=50.0,
        )
        text = report.format_text()
        assert "Model Selection Report" in text
        assert "fast" in text
        assert "50.0%" in text

    def test_format_text_empty(self):
        text = SelectionReport().format_text()
        assert "Total items: 0" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_model_tier_roundtrip(self):
        tier = ModelTier("fast", "gemini-2.5-flash", "gemini", 0.00015, 0.5)
        restored = ModelTier.from_dict(tier.as_dict())
        assert restored.name == tier.name
        assert restored.model == tier.model
        assert restored.cost_per_1k == tier.cost_per_1k
        assert restored.max_complexity == tier.max_complexity

    def test_auto_selection_config_roundtrip(self):
        config = get_default_config()
        d = config.as_dict()
        restored = AutoSelectionConfig.from_dict(d)
        assert len(restored.tiers) == len(config.tiers)
        assert restored.complexity_threshold == config.complexity_threshold

    def test_selection_report_roundtrip(self):
        report = SelectionReport(
            selections=[
                SelectionResult("i1", "m1", "fast", 0.3, "low"),
            ],
            tier_distribution={"fast": 1},
            estimated_cost_savings_pct=42.0,
        )
        d = report.as_dict()
        restored = SelectionReport.from_dict(d)
        assert len(restored.selections) == 1
        assert restored.selections[0].item_id == "i1"
        assert restored.estimated_cost_savings_pct == 42.0

    def test_selection_result_as_dict(self):
        r = SelectionResult("x", "model", "tier", 0.5, "reason")
        d = r.as_dict()
        assert d["item_id"] == "x"
        assert d["reason"] == "reason"
