"""Tests for the metric budget analysis module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.metric_budget import (
    MetricCostProfile,
    BudgetRecommendation,
    MetricBudgetReport,
    compute_signal_strength,
    detect_redundant_metrics,
    build_cost_profile,
    generate_budget_recommendations,
    build_budget_report,
    estimate_savings,
)


# ---------------------------------------------------------------------------
# compute_signal_strength
# ---------------------------------------------------------------------------


class TestComputeSignalStrength:
    def test_positive_correlation(self):
        """Perfectly correlated scores should return ~1.0."""
        metric = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        overall = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        r = compute_signal_strength(metric, overall)
        assert r > 0.99

    def test_negative_correlation(self):
        """Inversely correlated scores should return ~-1.0."""
        metric = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        overall = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        r = compute_signal_strength(metric, overall)
        assert r < -0.99

    def test_no_correlation(self):
        """Uncorrelated data should yield near-zero correlation."""
        metric = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        overall = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        r = compute_signal_strength(metric, overall)
        assert r < -0.5  # actually -1 for this alternating pattern

    def test_constant_metric_returns_zero(self):
        """Constant metric scores have zero std, should return 0.0."""
        metric = [0.5, 0.5, 0.5, 0.5]
        overall = [0.1, 0.4, 0.7, 1.0]
        r = compute_signal_strength(metric, overall)
        assert r == 0.0

    def test_constant_overall_returns_zero(self):
        """Constant overall scores have zero std, should return 0.0."""
        metric = [0.1, 0.4, 0.7, 1.0]
        overall = [0.5, 0.5, 0.5, 0.5]
        r = compute_signal_strength(metric, overall)
        assert r == 0.0

    def test_mismatched_lengths_returns_zero(self):
        r = compute_signal_strength([0.1, 0.2], [0.1, 0.2, 0.3])
        assert r == 0.0

    def test_single_item_returns_zero(self):
        r = compute_signal_strength([0.5], [0.5])
        assert r == 0.0

    def test_empty_returns_zero(self):
        r = compute_signal_strength([], [])
        assert r == 0.0


# ---------------------------------------------------------------------------
# detect_redundant_metrics
# ---------------------------------------------------------------------------


class TestDetectRedundantMetrics:
    def test_finds_redundant_pair(self):
        scores = {
            "a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        pairs = detect_redundant_metrics(scores, threshold=0.9)
        assert len(pairs) == 1
        assert ("a", "b") in pairs

    def test_no_redundant_pair(self):
        scores = {
            "a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "b": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
        pairs = detect_redundant_metrics(scores, threshold=0.9)
        assert len(pairs) == 0

    def test_threshold_sensitivity(self):
        """A moderately correlated pair should be found with lower threshold."""
        scores = {
            "a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "b": [0.15, 0.25, 0.28, 0.42, 0.48],
        }
        # High threshold: not redundant
        pairs_high = detect_redundant_metrics(scores, threshold=0.99)
        assert len(pairs_high) == 0
        # Low threshold: redundant
        pairs_low = detect_redundant_metrics(scores, threshold=0.8)
        assert len(pairs_low) == 1

    def test_multiple_metrics(self):
        """Three metrics where a and b are redundant, c is independent."""
        scores = {
            "a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "c": [0.8, 0.1, 0.5, 0.3, 0.9, 0.2, 0.6, 0.4],
        }
        pairs = detect_redundant_metrics(scores, threshold=0.9)
        assert ("a", "b") in pairs
        assert not any("c" in p for p in pairs)

    def test_empty_scores(self):
        pairs = detect_redundant_metrics({})
        assert pairs == []

    def test_single_metric(self):
        pairs = detect_redundant_metrics({"a": [0.1, 0.2, 0.3]})
        assert pairs == []


# ---------------------------------------------------------------------------
# build_cost_profile
# ---------------------------------------------------------------------------


class TestBuildCostProfile:
    def test_basic(self):
        profile = build_cost_profile(
            "test_metric",
            costs=[0.01, 0.02, 0.03],
            tokens=[100, 200, 300],
            signal=0.85,
        )
        assert profile.metric_id == "test_metric"
        assert abs(profile.avg_cost_per_item - 0.02) < 1e-9
        assert profile.avg_tokens_per_item == 200
        assert profile.signal_strength == 0.85

    def test_empty_costs(self):
        profile = build_cost_profile("empty", costs=[], tokens=[])
        assert profile.avg_cost_per_item == 0.0
        assert profile.avg_tokens_per_item == 0

    def test_default_signal(self):
        profile = build_cost_profile("m", costs=[1.0], tokens=[100])
        assert profile.signal_strength == 0.0


# ---------------------------------------------------------------------------
# generate_budget_recommendations
# ---------------------------------------------------------------------------


class TestGenerateBudgetRecommendations:
    def test_drop_low_signal(self):
        profiles = [
            MetricCostProfile(metric_id="weak", avg_cost_per_item=0.01, signal_strength=0.1),
            MetricCostProfile(metric_id="strong", avg_cost_per_item=0.01, signal_strength=0.9),
        ]
        recs = generate_budget_recommendations(profiles)
        rec_map = {r.metric_id: r for r in recs}
        assert rec_map["weak"].action == "drop"
        assert rec_map["strong"].action == "keep"

    def test_keep_high_signal(self):
        profiles = [
            MetricCostProfile(metric_id="good", avg_cost_per_item=0.05, signal_strength=0.8),
        ]
        recs = generate_budget_recommendations(profiles)
        assert recs[0].action == "keep"

    def test_replace_redundant(self):
        profiles = [
            MetricCostProfile(metric_id="a", avg_cost_per_item=0.01, signal_strength=0.9),
            MetricCostProfile(metric_id="b", avg_cost_per_item=0.01, signal_strength=0.5),
        ]
        recs = generate_budget_recommendations(profiles, redundant_pairs=[("a", "b")])
        rec_map = {r.metric_id: r for r in recs}
        assert rec_map["b"].action == "replace"
        assert rec_map["a"].action == "keep"

    def test_replace_redundant_tie_breaks_by_cost(self):
        """Equal signal: the more expensive one gets replaced."""
        profiles = [
            MetricCostProfile(metric_id="a", avg_cost_per_item=0.05, signal_strength=0.7),
            MetricCostProfile(metric_id="b", avg_cost_per_item=0.01, signal_strength=0.7),
        ]
        recs = generate_budget_recommendations(profiles, redundant_pairs=[("a", "b")])
        rec_map = {r.metric_id: r for r in recs}
        assert rec_map["a"].action == "replace"
        assert rec_map["b"].action == "keep"

    def test_empty_profiles(self):
        recs = generate_budget_recommendations([])
        assert recs == []


# ---------------------------------------------------------------------------
# build_budget_report
# ---------------------------------------------------------------------------


class TestBuildBudgetReport:
    def test_basic_report(self):
        profiles = [
            MetricCostProfile(metric_id="a", avg_cost_per_item=0.02, signal_strength=0.9),
            MetricCostProfile(metric_id="b", avg_cost_per_item=0.01, signal_strength=0.1),
        ]
        report = build_budget_report(profiles)
        assert report.total_cost_usd > 0
        assert len(report.recommendations) == 2
        assert report.potential_savings_usd > 0

    def test_no_savings_when_all_keep(self):
        profiles = [
            MetricCostProfile(metric_id="a", avg_cost_per_item=0.05, signal_strength=0.9),
            MetricCostProfile(metric_id="b", avg_cost_per_item=0.05, signal_strength=0.8),
        ]
        report = build_budget_report(profiles)
        assert report.potential_savings_usd == 0.0
        assert report.potential_savings_pct == 0.0

    def test_redundant_pairs_marked(self):
        profiles = [
            MetricCostProfile(metric_id="x", avg_cost_per_item=0.01, signal_strength=0.9),
            MetricCostProfile(metric_id="y", avg_cost_per_item=0.01, signal_strength=0.5),
        ]
        report = build_budget_report(profiles, redundant_pairs=[("x", "y")])
        for p in report.profiles:
            assert p.is_redundant is True


# ---------------------------------------------------------------------------
# estimate_savings
# ---------------------------------------------------------------------------


class TestEstimateSavings:
    def test_savings_for_dropped_metrics(self):
        profiles = [
            MetricCostProfile(metric_id="keep_me", avg_cost_per_item=0.05, signal_strength=0.9),
            MetricCostProfile(metric_id="drop_me", avg_cost_per_item=0.03, signal_strength=0.1),
        ]
        report = build_budget_report(profiles)
        savings = estimate_savings(report)
        assert savings["keep_me"] == 0.0
        assert savings["drop_me"] == 0.03

    def test_no_savings_all_keep(self):
        profiles = [
            MetricCostProfile(metric_id="a", avg_cost_per_item=0.1, signal_strength=0.9),
        ]
        report = build_budget_report(profiles)
        savings = estimate_savings(report)
        assert savings["a"] == 0.0


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_format_text_contains_sections(self):
        profiles = [
            MetricCostProfile(metric_id="m1", avg_cost_per_item=0.01, signal_strength=0.9),
        ]
        report = build_budget_report(profiles)
        text = report.format_text()
        assert "METRIC BUDGET REPORT" in text
        assert "Profiles:" in text
        assert "Recommendations:" in text
        assert "m1" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_metric_cost_profile_as_dict(self):
        p = MetricCostProfile(
            metric_id="m", avg_cost_per_item=0.01, avg_tokens_per_item=100,
            signal_strength=0.8, is_redundant=True,
        )
        d = p.as_dict()
        assert d["metric_id"] == "m"
        assert d["is_redundant"] is True

    def test_budget_recommendation_as_dict(self):
        r = BudgetRecommendation(
            metric_id="m", action="drop", reason="low signal",
            estimated_savings_pct=25.0,
        )
        d = r.as_dict()
        assert d["action"] == "drop"
        assert d["estimated_savings_pct"] == 25.0

    def test_report_roundtrip(self):
        profiles = [
            MetricCostProfile(metric_id="a", avg_cost_per_item=0.02, signal_strength=0.9),
            MetricCostProfile(metric_id="b", avg_cost_per_item=0.01, signal_strength=0.1),
        ]
        report = build_budget_report(profiles)
        d = report.as_dict()
        restored = MetricBudgetReport.from_dict(d)
        assert len(restored.profiles) == len(report.profiles)
        assert len(restored.recommendations) == len(report.recommendations)
        assert abs(restored.total_cost_usd - report.total_cost_usd) < 1e-9
        assert abs(restored.potential_savings_usd - report.potential_savings_usd) < 1e-9
        assert abs(restored.potential_savings_pct - report.potential_savings_pct) < 1e-9

    def test_from_dict_defaults(self):
        report = MetricBudgetReport.from_dict({})
        assert report.profiles == []
        assert report.recommendations == []
        assert report.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_metric_budget(self):
        profiles = [
            MetricCostProfile(metric_id="only", avg_cost_per_item=0.05, signal_strength=0.5),
        ]
        report = build_budget_report(profiles)
        assert len(report.recommendations) == 1

    def test_empty_profiles_budget(self):
        report = build_budget_report([])
        assert report.total_cost_usd == 0.0
        assert report.potential_savings_usd == 0.0
        assert report.recommendations == []

    def test_all_zero_cost(self):
        profiles = [
            MetricCostProfile(metric_id="free", avg_cost_per_item=0.0, signal_strength=0.1),
        ]
        report = build_budget_report(profiles)
        assert report.potential_savings_pct == 0.0
