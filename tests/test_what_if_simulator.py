"""Tests for the what-if simulator."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.what_if_simulator import (
    WhatIfReport,
    WhatIfResult,
    WhatIfScenario,
    classify_feasibility,
    find_best_roi_metric,
    find_minimum_improvement,
    render_what_if_chart,
    simulate_improvement,
    simulate_multi_improvement,
)


# ---------------------------------------------------------------------------
# classify_feasibility
# ---------------------------------------------------------------------------


class TestClassifyFeasibility:
    def test_easy(self):
        assert classify_feasibility(5.0) == "easy"
        assert classify_feasibility(0.0) == "easy"
        assert classify_feasibility(9.9) == "easy"

    def test_possible(self):
        assert classify_feasibility(10.0) == "possible"
        assert classify_feasibility(20.0) == "possible"
        assert classify_feasibility(29.9) == "possible"

    def test_hard(self):
        assert classify_feasibility(30.0) == "hard"
        assert classify_feasibility(40.0) == "hard"
        assert classify_feasibility(49.9) == "hard"

    def test_unlikely(self):
        assert classify_feasibility(50.0) == "unlikely"
        assert classify_feasibility(75.0) == "unlikely"
        assert classify_feasibility(100.0) == "unlikely"


# ---------------------------------------------------------------------------
# simulate_improvement
# ---------------------------------------------------------------------------


class TestSimulateImprovement:
    def test_basic(self):
        rates = {"metric_a": 0.6, "metric_b": 0.8}
        result = simulate_improvement("metric_a", rates, 20.0)
        assert result.scenario.metric_id == "metric_a"
        assert result.scenario.improvement_pct == 20.0
        assert result.original_pass_rate == 0.7  # (0.6 + 0.8) / 2
        assert abs(result.projected_pass_rate - 0.8) < 1e-9  # (0.8 + 0.8) / 2
        assert result.delta > 0

    def test_100_pct_improvement_capped_at_1(self):
        rates = {"metric_a": 0.5}
        result = simulate_improvement("metric_a", rates, 100.0)
        assert result.projected_pass_rate <= 1.0

    def test_with_weights(self):
        rates = {"metric_a": 0.5, "metric_b": 0.5}
        weights = {"metric_a": 3.0, "metric_b": 1.0}
        result = simulate_improvement("metric_a", rates, 20.0, weights)
        # weighted original: (0.5*3 + 0.5*1) / 4 = 0.5
        # weighted projected: (0.7*3 + 0.5*1) / 4 = 0.65
        assert abs(result.original_pass_rate - 0.5) < 1e-9
        assert abs(result.projected_pass_rate - 0.65) < 1e-9

    def test_unknown_metric(self):
        rates = {"metric_a": 0.5}
        result = simulate_improvement("nonexistent", rates, 20.0)
        # No change - metric not in rates
        assert result.delta == 0.0

    def test_feasibility_set(self):
        rates = {"metric_a": 0.5}
        result = simulate_improvement("metric_a", rates, 5.0)
        assert result.feasibility == "easy"

    def test_zero_improvement(self):
        rates = {"metric_a": 0.8}
        result = simulate_improvement("metric_a", rates, 0.0)
        assert result.delta == 0.0

    def test_single_metric(self):
        rates = {"only": 0.6}
        result = simulate_improvement("only", rates, 30.0)
        assert abs(result.projected_pass_rate - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# simulate_multi_improvement
# ---------------------------------------------------------------------------


class TestSimulateMultiImprovement:
    def test_basic(self):
        rates = {"a": 0.5, "b": 0.6, "c": 0.7}
        scenarios = [
            WhatIfScenario("a", 20.0, "Improve A"),
            WhatIfScenario("b", 10.0, "Improve B"),
        ]
        report = simulate_multi_improvement(scenarios, rates)
        assert report.total_scenarios == 2
        assert len(report.results) == 2
        assert report.best_scenario is not None

    def test_best_scenario_identified(self):
        rates = {"a": 0.3, "b": 0.9}
        scenarios = [
            WhatIfScenario("a", 30.0),
            WhatIfScenario("b", 30.0),
        ]
        report = simulate_multi_improvement(scenarios, rates)
        # Improving "a" by 30% from 0.3 -> 0.6 is bigger lift than "b" capped at 1.0
        # a: overall goes from 0.6 to 0.75 (delta 0.15)
        # b: overall goes from 0.6 to 0.65 (b goes from 0.9 to 1.0, delta 0.05)
        assert "a" in report.best_scenario

    def test_empty_scenarios(self):
        report = simulate_multi_improvement([], {"a": 0.5})
        assert report.total_scenarios == 0
        assert report.results == []
        assert report.best_scenario is None

    def test_preserves_description(self):
        rates = {"x": 0.5}
        scenarios = [WhatIfScenario("x", 10.0, "Custom description")]
        report = simulate_multi_improvement(scenarios, rates)
        assert report.results[0].scenario.description == "Custom description"


# ---------------------------------------------------------------------------
# find_minimum_improvement
# ---------------------------------------------------------------------------


class TestFindMinimumImprovement:
    def test_basic(self):
        rates = {"a": 0.5, "b": 0.8}
        # overall = 0.65. Want 0.75. Need a to be 0.7. That's +20% improvement.
        result = find_minimum_improvement("a", rates, 0.75)
        assert 19.0 <= result <= 21.0  # approximate due to binary search

    def test_already_at_target(self):
        rates = {"a": 0.9, "b": 0.95}
        result = find_minimum_improvement("a", rates, 0.5)
        assert result == 0.0

    def test_unreachable_target(self):
        rates = {"a": 0.0, "b": 0.0}
        # Even with a at 1.0, overall is 0.5 < 0.99
        result = find_minimum_improvement("a", rates, 0.99)
        assert result == 100.0

    def test_unknown_metric(self):
        rates = {"a": 0.5}
        result = find_minimum_improvement("nonexistent", rates, 0.9)
        assert result == 100.0

    def test_single_metric_exact(self):
        rates = {"only": 0.5}
        # Target 0.7 means metric needs to go from 0.5 to 0.7 = +20%
        result = find_minimum_improvement("only", rates, 0.7)
        assert 19.0 <= result <= 21.0


# ---------------------------------------------------------------------------
# render_what_if_chart
# ---------------------------------------------------------------------------


class TestRenderWhatIfChart:
    def test_output_has_header(self):
        report = WhatIfReport(
            results=[
                WhatIfResult(
                    scenario=WhatIfScenario("a", 10.0),
                    original_pass_rate=0.5,
                    projected_pass_rate=0.6,
                    delta=0.1,
                )
            ],
            total_scenarios=1,
        )
        chart = render_what_if_chart(report)
        assert "What-If" in chart
        assert "a" in chart

    def test_empty_report(self):
        report = WhatIfReport()
        chart = render_what_if_chart(report)
        assert "No scenarios" in chart

    def test_multiple_scenarios(self):
        results = [
            WhatIfResult(
                scenario=WhatIfScenario("m1", 10.0),
                original_pass_rate=0.5,
                projected_pass_rate=0.6,
                delta=0.1,
            ),
            WhatIfResult(
                scenario=WhatIfScenario("m2", 20.0),
                original_pass_rate=0.5,
                projected_pass_rate=0.7,
                delta=0.2,
            ),
        ]
        report = WhatIfReport(results=results, total_scenarios=2)
        chart = render_what_if_chart(report, width=50)
        assert "m1" in chart
        assert "m2" in chart
        assert "#" in chart

    def test_custom_width(self):
        report = WhatIfReport(
            results=[
                WhatIfResult(
                    scenario=WhatIfScenario("a", 10.0),
                    original_pass_rate=0.5,
                    projected_pass_rate=0.6,
                    delta=0.1,
                )
            ],
            total_scenarios=1,
        )
        chart = render_what_if_chart(report, width=40)
        # Should not crash and should produce output
        assert len(chart) > 0


# ---------------------------------------------------------------------------
# find_best_roi_metric
# ---------------------------------------------------------------------------


class TestFindBestRoiMetric:
    def test_basic(self):
        rates = {"low": 0.3, "high": 0.9}
        best = find_best_roi_metric(rates)
        # Improving "low" by 10% gives bigger overall lift
        assert best == "low"

    def test_empty_rates(self):
        assert find_best_roi_metric({}) == ""

    def test_single_metric(self):
        rates = {"only": 0.5}
        assert find_best_roi_metric(rates) == "only"

    def test_with_weights(self):
        rates = {"a": 0.3, "b": 0.3}
        weights = {"a": 10.0, "b": 1.0}
        best = find_best_roi_metric(rates, weights)
        # "a" has much higher weight, so improving it yields more
        assert best == "a"

    def test_all_perfect(self):
        rates = {"a": 1.0, "b": 1.0}
        # All at 1.0, no improvement possible, but function should still return
        best = find_best_roi_metric(rates)
        assert best in ("a", "b")


# ---------------------------------------------------------------------------
# WhatIfReport format_text
# ---------------------------------------------------------------------------


class TestWhatIfReportFormatText:
    def test_format(self):
        report = WhatIfReport(
            results=[
                WhatIfResult(
                    scenario=WhatIfScenario("a", 10.0, "Improve A"),
                    original_pass_rate=0.5,
                    projected_pass_rate=0.6,
                    delta=0.1,
                    feasibility="easy",
                )
            ],
            best_scenario="Improve A",
            total_scenarios=1,
        )
        text = report.format_text()
        assert "1 scenarios" in text
        assert "Improve A" in text

    def test_empty_report(self):
        report = WhatIfReport()
        text = report.format_text()
        assert "0 scenarios" in text


# ---------------------------------------------------------------------------
# WhatIfResult format_text
# ---------------------------------------------------------------------------


class TestWhatIfResultFormatText:
    def test_format(self):
        result = WhatIfResult(
            scenario=WhatIfScenario("helpfulness", 15.0, "Boost help"),
            original_pass_rate=0.6,
            projected_pass_rate=0.75,
            delta=0.15,
            feasibility="possible",
        )
        text = result.format_text()
        assert "helpfulness" in text
        assert "possible" in text
        assert "Projected" in text

    def test_format_no_description(self):
        result = WhatIfResult(
            scenario=WhatIfScenario("x", 5.0),
            original_pass_rate=0.5,
            projected_pass_rate=0.55,
            delta=0.05,
        )
        text = result.format_text()
        assert "x" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_scenario_as_dict(self):
        s = WhatIfScenario("m1", 25.0, "Test scenario")
        d = s.as_dict()
        assert d["metric_id"] == "m1"
        assert d["improvement_pct"] == 25.0
        assert d["description"] == "Test scenario"

    def test_result_as_dict(self):
        s = WhatIfScenario("m1", 10.0)
        r = WhatIfResult(
            scenario=s,
            original_pass_rate=0.5,
            projected_pass_rate=0.6,
            delta=0.1,
            feasibility="easy",
        )
        d = r.as_dict()
        assert d["original_pass_rate"] == 0.5
        assert d["projected_pass_rate"] == 0.6
        assert d["delta"] == 0.1
        assert d["feasibility"] == "easy"
        assert d["scenario"]["metric_id"] == "m1"

    def test_report_roundtrip(self):
        s = WhatIfScenario("m1", 10.0, "Test")
        r = WhatIfResult(
            scenario=s,
            original_pass_rate=0.5,
            projected_pass_rate=0.6,
            delta=0.1,
            feasibility="easy",
        )
        report = WhatIfReport(
            results=[r],
            best_scenario="Test",
            total_scenarios=1,
        )
        d = report.as_dict()
        restored = WhatIfReport.from_dict(d)
        assert restored.total_scenarios == 1
        assert restored.best_scenario == "Test"
        assert len(restored.results) == 1
        assert restored.results[0].scenario.metric_id == "m1"
        assert restored.results[0].original_pass_rate == 0.5

    def test_report_from_dict_defaults(self):
        report = WhatIfReport.from_dict({})
        assert report.results == []
        assert report.best_scenario is None
        assert report.total_scenarios == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_perfect_rates(self):
        rates = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = simulate_improvement("a", rates, 10.0)
        assert result.projected_pass_rate == 1.0
        assert result.delta == 0.0

    def test_all_zero_rates(self):
        rates = {"a": 0.0, "b": 0.0}
        result = simulate_improvement("a", rates, 50.0)
        assert result.original_pass_rate == 0.0
        assert result.projected_pass_rate == 0.25  # (0.5 + 0.0) / 2

    def test_single_metric_system(self):
        rates = {"only": 0.4}
        result = simulate_improvement("only", rates, 30.0)
        assert abs(result.projected_pass_rate - 0.7) < 1e-9

    def test_very_small_improvement(self):
        rates = {"a": 0.5}
        result = simulate_improvement("a", rates, 0.1)
        assert result.delta > 0
        assert result.feasibility == "easy"

    def test_minimum_improvement_with_weights(self):
        rates = {"a": 0.5, "b": 0.5}
        weights = {"a": 1.0, "b": 1.0}
        result = find_minimum_improvement("a", rates, 0.6, weights)
        assert result > 0
        assert result <= 100.0
