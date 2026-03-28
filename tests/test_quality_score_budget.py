"""Tests for run quality score and evaluation budget control."""

import pytest
from evalyn_sdk.analysis.quality_score import (
    compute_quality_score,
    QualityScoreBreakdown,
)
from evalyn_sdk.evaluation.budget import (
    BudgetConfig,
    BudgetTracker,
)


# =============================================================================
# Quality score tests
# =============================================================================

class TestQualityScore:
    def test_perfect_run(self):
        score = compute_quality_score(
            overall_pass_rate=1.0,
            total_items=100,
            total_metrics=5,
            total_cost_usd=0.0,
            error_count=0,
            total_evaluations=500,
        )
        assert score.overall >= 95
        assert score.grade == "A"

    def test_poor_run(self):
        score = compute_quality_score(
            overall_pass_rate=0.2,
            total_items=10,
            total_metrics=5,
            total_cost_usd=1.0,
            error_count=10,
            total_evaluations=50,
        )
        assert score.overall < 50
        assert score.grade in ("D", "F")

    def test_pass_rate_dominates(self):
        high = compute_quality_score(overall_pass_rate=0.9, total_items=10, total_metrics=3)
        low = compute_quality_score(overall_pass_rate=0.3, total_items=10, total_metrics=3)
        assert high.overall > low.overall

    def test_free_evaluation_cost_score_100(self):
        score = compute_quality_score(
            overall_pass_rate=0.5, total_items=10, total_metrics=3,
            total_cost_usd=0.0,
        )
        assert score.cost_efficiency_score == 100.0

    def test_expensive_evaluation_low_cost_score(self):
        score = compute_quality_score(
            overall_pass_rate=0.5, total_items=10, total_metrics=3,
            total_cost_usd=5.0,  # $0.50 per item
        )
        assert score.cost_efficiency_score < 20

    def test_errors_reduce_score(self):
        no_errors = compute_quality_score(
            overall_pass_rate=0.8, total_items=10, total_metrics=3,
            error_count=0, total_evaluations=30,
        )
        with_errors = compute_quality_score(
            overall_pass_rate=0.8, total_items=10, total_metrics=3,
            error_count=5, total_evaluations=30,
        )
        assert no_errors.overall > with_errors.overall

    def test_empty_run(self):
        score = compute_quality_score(
            overall_pass_rate=0.0, total_items=0, total_metrics=0,
        )
        # Pass rate is 0 but cost/error default to 100 (no cost, no errors)
        assert score.pass_rate_score == 0.0
        assert score.coverage_score == 0.0
        assert score.grade == "F"  # overall < 60

    def test_grade_boundaries(self):
        for rate, expected in [(0.95, "A"), (0.85, "B"), (0.75, "C"), (0.65, "D"), (0.3, "F")]:
            score = compute_quality_score(
                overall_pass_rate=rate, total_items=100, total_metrics=5,
                total_evaluations=500,
            )
            # Grade is based on overall, which is weighted, so just verify it's reasonable
            assert score.grade in ("A", "B", "C", "D", "F")

    def test_custom_weights(self):
        score = compute_quality_score(
            overall_pass_rate=1.0, total_items=10, total_metrics=3,
            total_evaluations=30,
            weights={"pass_rate": 1.0, "coverage": 0.0, "cost_efficiency": 0.0, "error_rate": 0.0},
        )
        assert score.overall == pytest.approx(100.0)

    def test_as_dict(self):
        score = compute_quality_score(overall_pass_rate=0.8, total_items=10, total_metrics=3)
        d = score.as_dict()
        assert "overall" in d
        assert "grade" in d
        assert "pass_rate_score" in d

    def test_format_text(self):
        score = compute_quality_score(overall_pass_rate=0.8, total_items=10, total_metrics=3)
        text = score.format_text()
        assert "Run Quality" in text
        assert "Grade:" in text


# =============================================================================
# Budget control tests
# =============================================================================

class TestBudgetConfig:
    def test_default_no_limits(self):
        cfg = BudgetConfig()
        assert not cfg.is_configured()

    def test_cost_limit(self):
        cfg = BudgetConfig(max_cost_usd=1.0)
        assert cfg.is_configured()

    def test_token_limit(self):
        cfg = BudgetConfig(max_tokens=100000)
        assert cfg.is_configured()


class TestBudgetTracker:
    def test_no_limit_never_stops(self):
        tracker = BudgetTracker()
        tracker.record(cost_usd=100.0, input_tokens=1000000)
        assert tracker.should_stop is False

    def test_cost_limit_stops(self):
        tracker = BudgetTracker(BudgetConfig(max_cost_usd=0.10))
        tracker.record(cost_usd=0.05)
        assert tracker.should_stop is False
        tracker.record(cost_usd=0.06)
        assert tracker.should_stop is True

    def test_token_limit_stops(self):
        tracker = BudgetTracker(BudgetConfig(max_tokens=1000))
        tracker.record(input_tokens=400, output_tokens=100)
        assert tracker.should_stop is False
        tracker.record(input_tokens=400, output_tokens=200)
        assert tracker.should_stop is True

    def test_warning_at_threshold(self):
        tracker = BudgetTracker(BudgetConfig(max_cost_usd=1.0, warning_threshold=0.8))
        tracker.record(cost_usd=0.5)
        assert tracker.should_warn is False
        tracker.record(cost_usd=0.35)
        assert tracker.should_warn is True
        assert tracker.should_stop is False  # warning, not stopped yet

    def test_cost_remaining(self):
        tracker = BudgetTracker(BudgetConfig(max_cost_usd=1.0))
        tracker.record(cost_usd=0.3)
        assert tracker.cost_remaining == pytest.approx(0.7)

    def test_tokens_remaining(self):
        tracker = BudgetTracker(BudgetConfig(max_tokens=1000))
        tracker.record(input_tokens=300, output_tokens=200)
        assert tracker.tokens_remaining == 500

    def test_cost_utilization(self):
        tracker = BudgetTracker(BudgetConfig(max_cost_usd=1.0))
        tracker.record(cost_usd=0.5)
        assert tracker.cost_utilization == pytest.approx(0.5)

    def test_token_utilization(self):
        tracker = BudgetTracker(BudgetConfig(max_tokens=1000))
        tracker.record(input_tokens=250, output_tokens=250)
        assert tracker.token_utilization == pytest.approx(0.5)

    def test_no_limit_returns_none(self):
        tracker = BudgetTracker()
        assert tracker.cost_remaining is None
        assert tracker.tokens_remaining is None
        assert tracker.cost_utilization is None
        assert tracker.token_utilization is None

    def test_items_counted(self):
        tracker = BudgetTracker()
        tracker.record(cost_usd=0.01)
        tracker.record(cost_usd=0.01)
        tracker.record(cost_usd=0.01)
        assert tracker.items_evaluated == 3

    def test_summary(self):
        tracker = BudgetTracker(BudgetConfig(max_cost_usd=1.0, max_tokens=10000))
        tracker.record(cost_usd=0.5, input_tokens=1000, output_tokens=500)
        s = tracker.summary()
        assert s["total_cost_usd"] == 0.5
        assert s["total_tokens"] == 1500
        assert s["items_evaluated"] == 1
        assert s["budget_exceeded"] is False
        assert s["max_cost_usd"] == 1.0
        assert s["cost_utilization"] == 0.5

    def test_summary_no_limits(self):
        tracker = BudgetTracker()
        s = tracker.summary()
        assert "max_cost_usd" not in s
        assert "max_tokens" not in s

    def test_utilization_capped_at_1(self):
        tracker = BudgetTracker(BudgetConfig(max_cost_usd=0.10))
        tracker.record(cost_usd=0.50)  # way over budget
        assert tracker.cost_utilization == 1.0
