"""Tests for cost_estimation module."""

from __future__ import annotations

import math

import pytest

from evalyn_sdk.cost_estimation import (
    DEFAULT_COST_MODELS,
    CostEstimate,
    CostModel,
    CostReport,
    build_cost_report,
    compare_model_costs,
    estimate_simulation_cost,
    estimate_tokens,
    format_cost_report,
)


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_defaults(self):
        cm = CostModel(model_name="test", input_cost_per_1k=1.0, output_cost_per_1k=2.0)
        assert cm.currency == "USD"

    def test_custom_currency(self):
        cm = CostModel("m", 1.0, 2.0, currency="EUR")
        assert cm.currency == "EUR"

    def test_as_dict(self):
        cm = CostModel("x", 0.5, 1.5, "JPY")
        d = cm.as_dict()
        assert d["model_name"] == "x"
        assert d["input_cost_per_1k"] == 0.5
        assert d["output_cost_per_1k"] == 1.5
        assert d["currency"] == "JPY"

    def test_from_dict(self):
        d = {"model_name": "z", "input_cost_per_1k": 3.0, "output_cost_per_1k": 9.0}
        cm = CostModel.from_dict(d)
        assert cm.model_name == "z"
        assert cm.currency == "USD"

    def test_roundtrip(self):
        cm = CostModel("rt", 0.1, 0.2, "GBP")
        assert CostModel.from_dict(cm.as_dict()).as_dict() == cm.as_dict()


# ---------------------------------------------------------------------------
# CostEstimate
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_basic(self):
        ce = CostEstimate(
            total_tokens=500, total_cost=1.23,
            input_tokens=200, output_tokens=300,
            breakdown=[],
        )
        assert ce.currency == "USD"
        assert ce.total_tokens == 500

    def test_as_dict(self):
        ce = CostEstimate(100, 0.5, 40, 60, [{"mode": "a"}], "EUR")
        d = ce.as_dict()
        assert d["total_tokens"] == 100
        assert d["currency"] == "EUR"
        assert len(d["breakdown"]) == 1

    def test_from_dict(self):
        d = {
            "total_tokens": 800,
            "total_cost": 2.5,
            "input_tokens": 300,
            "output_tokens": 500,
            "breakdown": [],
        }
        ce = CostEstimate.from_dict(d)
        assert ce.total_tokens == 800
        assert ce.currency == "USD"

    def test_from_dict_defaults(self):
        ce = CostEstimate.from_dict({})
        assert ce.total_tokens == 0
        assert ce.total_cost == 0.0

    def test_roundtrip(self):
        ce = CostEstimate(1000, 5.5, 400, 600, [{"mode": "x", "cost": 5.5}], "USD")
        assert CostEstimate.from_dict(ce.as_dict()).as_dict() == ce.as_dict()


# ---------------------------------------------------------------------------
# CostReport
# ---------------------------------------------------------------------------


class TestCostReport:
    def test_basic(self):
        cm = CostModel("m", 1.0, 2.0)
        ce = CostEstimate(500, 1.0, 200, 300, [])
        cr = CostReport(estimate=ce, model=cm, item_count=10)
        assert cr.is_dry_run is True

    def test_as_dict(self):
        cm = CostModel("m", 1.0, 2.0)
        ce = CostEstimate(500, 1.0, 200, 300, [])
        cr = CostReport(ce, cm, 10, False)
        d = cr.as_dict()
        assert d["is_dry_run"] is False
        assert d["item_count"] == 10
        assert "estimate" in d
        assert "model" in d

    def test_from_dict(self):
        d = {
            "estimate": {
                "total_tokens": 100,
                "total_cost": 0.5,
                "input_tokens": 40,
                "output_tokens": 60,
                "breakdown": [],
            },
            "model": {
                "model_name": "test",
                "input_cost_per_1k": 1.0,
                "output_cost_per_1k": 2.0,
            },
            "item_count": 5,
            "is_dry_run": False,
        }
        cr = CostReport.from_dict(d)
        assert cr.item_count == 5
        assert cr.model.model_name == "test"
        assert cr.is_dry_run is False

    def test_roundtrip(self):
        cm = CostModel("rt", 0.5, 1.0)
        ce = CostEstimate(200, 0.3, 80, 120, [])
        cr = CostReport(ce, cm, 20, True)
        assert CostReport.from_dict(cr.as_dict()).as_dict() == cr.as_dict()


# ---------------------------------------------------------------------------
# DEFAULT_COST_MODELS
# ---------------------------------------------------------------------------


class TestDefaultCostModels:
    def test_contains_expected_models(self):
        expected = {"gemini-2.0-flash", "gpt-4o", "gpt-4o-mini", "claude-sonnet"}
        assert expected == set(DEFAULT_COST_MODELS.keys())

    def test_gemini_pricing(self):
        m = DEFAULT_COST_MODELS["gemini-2.0-flash"]
        assert m.input_cost_per_1k == 0.075
        assert m.output_cost_per_1k == 0.30

    def test_gpt4o_pricing(self):
        m = DEFAULT_COST_MODELS["gpt-4o"]
        assert m.input_cost_per_1k == 2.50
        assert m.output_cost_per_1k == 10.00

    def test_gpt4o_mini_pricing(self):
        m = DEFAULT_COST_MODELS["gpt-4o-mini"]
        assert m.input_cost_per_1k == 0.15
        assert m.output_cost_per_1k == 0.60

    def test_claude_pricing(self):
        m = DEFAULT_COST_MODELS["claude-sonnet"]
        assert m.input_cost_per_1k == 3.00
        assert m.output_cost_per_1k == 15.00

    def test_all_usd(self):
        for m in DEFAULT_COST_MODELS.values():
            assert m.currency == "USD"


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short(self):
        # "hi" has length 2 -> ceil(2/4) = 1
        assert estimate_tokens("hi") == 1

    def test_exact_multiple(self):
        assert estimate_tokens("abcd") == 1

    def test_not_exact(self):
        assert estimate_tokens("abcde") == 2

    def test_longer(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_ceil_behavior(self):
        assert estimate_tokens("abc") == 1  # ceil(3/4) = 1


# ---------------------------------------------------------------------------
# estimate_simulation_cost
# ---------------------------------------------------------------------------


class TestEstimateSimulationCost:
    def test_defaults(self):
        est = estimate_simulation_cost(item_count=10)
        # 10 * 200 = 2000 input, 10 * 300 = 3000 output
        assert est.input_tokens == 2000
        assert est.output_tokens == 3000
        assert est.total_tokens == 5000
        # Cost: (2000/1000)*0.075 + (3000/1000)*0.30 = 0.15 + 0.90 = 1.05
        assert abs(est.total_cost - 1.05) < 0.001

    def test_custom_model(self):
        model = CostModel("custom", 1.0, 2.0)
        est = estimate_simulation_cost(1, 1000, 1000, model=model)
        # (1000/1000)*1.0 + (1000/1000)*2.0 = 3.0
        assert abs(est.total_cost - 3.0) < 0.001

    def test_zero_items(self):
        est = estimate_simulation_cost(0)
        assert est.total_tokens == 0
        assert est.total_cost == 0.0

    def test_with_modes(self):
        est = estimate_simulation_cost(
            item_count=999,  # should be ignored
            modes={"generate": 5, "evaluate": 3},
        )
        assert est.input_tokens == 8 * 200
        assert est.output_tokens == 8 * 300

    def test_modes_breakdown(self):
        est = estimate_simulation_cost(
            item_count=0,
            modes={"a": 10, "b": 20},
        )
        assert len(est.breakdown) == 2
        modes_in_breakdown = {e["mode"] for e in est.breakdown}
        assert modes_in_breakdown == {"a", "b"}

    def test_modes_cost_sums(self):
        est = estimate_simulation_cost(
            item_count=0,
            modes={"x": 10, "y": 10},
        )
        breakdown_sum = sum(e["cost"] for e in est.breakdown)
        assert abs(breakdown_sum - est.total_cost) < 0.001

    def test_no_modes_no_breakdown(self):
        est = estimate_simulation_cost(item_count=5)
        assert est.breakdown == []


# ---------------------------------------------------------------------------
# build_cost_report
# ---------------------------------------------------------------------------


class TestBuildCostReport:
    def test_default_model(self):
        report = build_cost_report(item_count=10)
        assert report.model.model_name == "gemini-2.0-flash"
        assert report.is_dry_run is True
        assert report.item_count == 10

    def test_specific_model(self):
        report = build_cost_report(item_count=5, model_name="gpt-4o")
        assert report.model.model_name == "gpt-4o"

    def test_unknown_model_falls_back(self):
        report = build_cost_report(item_count=1, model_name="nonexistent-model")
        assert report.model.model_name == "gemini-2.0-flash"

    def test_with_modes(self):
        report = build_cost_report(item_count=0, modes={"a": 5, "b": 5})
        assert report.item_count == 10

    def test_estimate_populated(self):
        report = build_cost_report(item_count=10)
        assert report.estimate.total_tokens > 0
        assert report.estimate.total_cost > 0


# ---------------------------------------------------------------------------
# format_cost_report
# ---------------------------------------------------------------------------


class TestFormatCostReport:
    def test_contains_header(self):
        report = build_cost_report(item_count=10)
        text = format_cost_report(report)
        assert "Cost Estimation Report" in text

    def test_contains_model(self):
        report = build_cost_report(item_count=10, model_name="gpt-4o")
        text = format_cost_report(report)
        assert "gpt-4o" in text

    def test_contains_tokens(self):
        report = build_cost_report(item_count=10)
        text = format_cost_report(report)
        assert "Input tokens" in text
        assert "Output tokens" in text

    def test_contains_mode_breakdown(self):
        report = build_cost_report(item_count=0, modes={"gen": 5, "eval": 3})
        text = format_cost_report(report)
        assert "gen" in text
        assert "eval" in text
        assert "Per-mode breakdown" in text

    def test_no_breakdown_when_no_modes(self):
        report = build_cost_report(item_count=10)
        text = format_cost_report(report)
        assert "Per-mode breakdown" not in text


# ---------------------------------------------------------------------------
# compare_model_costs
# ---------------------------------------------------------------------------


class TestCompareModelCosts:
    def test_default_all_models(self):
        text = compare_model_costs(item_count=10)
        for name in DEFAULT_COST_MODELS:
            assert name in text

    def test_specific_models(self):
        text = compare_model_costs(item_count=10, models=["gpt-4o", "gpt-4o-mini"])
        assert "gpt-4o" in text
        assert "gpt-4o-mini" in text
        assert "claude-sonnet" not in text

    def test_contains_header(self):
        text = compare_model_costs(item_count=5)
        assert "Model Cost Comparison" in text

    def test_contains_item_count(self):
        text = compare_model_costs(item_count=42)
        assert "Items: 42" in text

    def test_unknown_model_skipped(self):
        text = compare_model_costs(item_count=1, models=["nonexistent"])
        # Should just have header, no model rows
        assert "nonexistent" not in text.split("\n")[3:]  # not in data rows
