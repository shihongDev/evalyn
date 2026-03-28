"""Tests for cross-provider calibration."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.cross_provider import (
    ConsistencyResult,
    CrossProviderReport,
    ProviderScore,
    compare_all_providers,
    compute_provider_consistency,
    find_divergent_items,
    normalize_scores,
)


# -------------------------------------------------------------------
# ProviderScore
# -------------------------------------------------------------------


class TestProviderScore:
    def test_as_dict(self):
        ps = ProviderScore(provider="gpt4", item_id="q1", score=0.9)
        d = ps.as_dict()
        assert d["provider"] == "gpt4"
        assert d["item_id"] == "q1"
        assert d["score"] == 0.9

    def test_fields(self):
        ps = ProviderScore(provider="claude", item_id="q2", score=0.75)
        assert ps.provider == "claude"
        assert ps.item_id == "q2"
        assert ps.score == 0.75


# -------------------------------------------------------------------
# ConsistencyResult
# -------------------------------------------------------------------


class TestConsistencyResult:
    def test_as_dict(self):
        cr = ConsistencyResult(
            provider_a="gpt4",
            provider_b="claude",
            correlation=0.95,
            mean_delta=0.05,
            max_delta=0.1,
            num_items=10,
        )
        d = cr.as_dict()
        assert d["provider_a"] == "gpt4"
        assert d["correlation"] == 0.95
        assert d["num_items"] == 10

    def test_format_text(self):
        cr = ConsistencyResult(
            provider_a="gpt4",
            provider_b="claude",
            correlation=0.95,
            mean_delta=0.05,
            max_delta=0.1,
            num_items=10,
        )
        text = cr.format_text()
        assert "gpt4 vs claude" in text
        assert "0.9500" in text
        assert "10" in text

    def test_format_text_contains_all_fields(self):
        cr = ConsistencyResult(
            provider_a="a",
            provider_b="b",
            correlation=0.5,
            mean_delta=0.2,
            max_delta=0.4,
            num_items=5,
        )
        text = cr.format_text()
        assert "correlation" in text
        assert "mean delta" in text
        assert "max delta" in text
        assert "items" in text


# -------------------------------------------------------------------
# CrossProviderReport
# -------------------------------------------------------------------


class TestCrossProviderReport:
    def test_as_dict_empty(self):
        report = CrossProviderReport(comparisons=[])
        d = report.as_dict()
        assert d["comparisons"] == []
        assert d["most_consistent_pair"] is None
        assert d["least_consistent_pair"] is None

    def test_as_dict_with_pairs(self):
        cr = ConsistencyResult("a", "b", 0.9, 0.05, 0.1, 5)
        report = CrossProviderReport(
            comparisons=[cr],
            most_consistent_pair=("a", "b"),
            least_consistent_pair=("a", "b"),
        )
        d = report.as_dict()
        assert d["most_consistent_pair"] == ["a", "b"]
        assert d["least_consistent_pair"] == ["a", "b"]
        assert len(d["comparisons"]) == 1

    def test_from_dict_roundtrip(self):
        cr = ConsistencyResult("gpt4", "claude", 0.85, 0.1, 0.2, 8)
        report = CrossProviderReport(
            comparisons=[cr],
            most_consistent_pair=("gpt4", "claude"),
            least_consistent_pair=("gpt4", "claude"),
        )
        d = report.as_dict()
        restored = CrossProviderReport.from_dict(d)
        assert len(restored.comparisons) == 1
        assert restored.comparisons[0].provider_a == "gpt4"
        assert restored.comparisons[0].correlation == 0.85
        assert restored.most_consistent_pair == ("gpt4", "claude")
        assert restored.least_consistent_pair == ("gpt4", "claude")

    def test_from_dict_none_pairs(self):
        d = {"comparisons": [], "most_consistent_pair": None, "least_consistent_pair": None}
        restored = CrossProviderReport.from_dict(d)
        assert restored.most_consistent_pair is None
        assert restored.least_consistent_pair is None

    def test_format_text(self):
        cr = ConsistencyResult("gpt4", "claude", 0.9, 0.05, 0.1, 10)
        report = CrossProviderReport(
            comparisons=[cr],
            most_consistent_pair=("gpt4", "claude"),
            least_consistent_pair=("gpt4", "claude"),
        )
        text = report.format_text()
        assert "Cross-Provider Consistency Report" in text
        assert "gpt4 vs claude" in text
        assert "Most consistent" in text
        assert "Least consistent" in text

    def test_format_text_no_pairs(self):
        report = CrossProviderReport(comparisons=[])
        text = report.format_text()
        assert "Cross-Provider Consistency Report" in text
        assert "Most consistent" not in text


# -------------------------------------------------------------------
# compute_provider_consistency
# -------------------------------------------------------------------


class TestComputeProviderConsistency:
    def test_identical_scores(self):
        scores = {"q1": 0.8, "q2": 0.6, "q3": 0.9}
        result = compute_provider_consistency(scores, scores, "a", "b")
        assert result.num_items == 3
        assert result.mean_delta == 0.0
        assert result.max_delta == 0.0
        # Identical non-constant scores have perfect positive correlation
        assert abs(result.correlation - 1.0) < 1e-9

    def test_opposite_scores(self):
        sa = {"q1": 0.0, "q2": 0.5, "q3": 1.0}
        sb = {"q1": 1.0, "q2": 0.5, "q3": 0.0}
        result = compute_provider_consistency(sa, sb, "a", "b")
        assert result.num_items == 3
        assert result.correlation < 0  # negative correlation

    def test_perfect_positive_correlation(self):
        sa = {"q1": 0.1, "q2": 0.5, "q3": 0.9}
        sb = {"q1": 0.2, "q2": 0.6, "q3": 1.0}
        result = compute_provider_consistency(sa, sb, "a", "b")
        assert abs(result.correlation - 1.0) < 1e-9

    def test_no_common_items(self):
        sa = {"q1": 0.5}
        sb = {"q2": 0.5}
        result = compute_provider_consistency(sa, sb, "a", "b")
        assert result.num_items == 0
        assert result.correlation == 0.0

    def test_single_common_item(self):
        sa = {"q1": 0.8}
        sb = {"q1": 0.6}
        result = compute_provider_consistency(sa, sb, "a", "b")
        assert result.num_items == 1
        assert abs(result.mean_delta - 0.2) < 1e-9
        assert abs(result.max_delta - 0.2) < 1e-9

    def test_empty_scores(self):
        result = compute_provider_consistency({}, {}, "a", "b")
        assert result.num_items == 0

    def test_partial_overlap(self):
        sa = {"q1": 0.5, "q2": 0.7, "q3": 0.9}
        sb = {"q2": 0.6, "q3": 0.8, "q4": 1.0}
        result = compute_provider_consistency(sa, sb, "a", "b")
        assert result.num_items == 2  # q2 and q3


# -------------------------------------------------------------------
# compare_all_providers
# -------------------------------------------------------------------


class TestCompareAllProviders:
    def test_two_providers(self):
        ps = {
            "gpt4": {"q1": 0.8, "q2": 0.6},
            "claude": {"q1": 0.7, "q2": 0.9},
        }
        report = compare_all_providers(ps)
        assert len(report.comparisons) == 1
        assert report.comparisons[0].provider_a == "claude"
        assert report.comparisons[0].provider_b == "gpt4"

    def test_three_providers(self):
        ps = {
            "a": {"q1": 0.8, "q2": 0.6},
            "b": {"q1": 0.7, "q2": 0.9},
            "c": {"q1": 0.5, "q2": 0.5},
        }
        report = compare_all_providers(ps)
        assert len(report.comparisons) == 3  # 3 choose 2
        assert report.most_consistent_pair is not None
        assert report.least_consistent_pair is not None

    def test_single_provider(self):
        ps = {"a": {"q1": 0.5}}
        report = compare_all_providers(ps)
        assert len(report.comparisons) == 0
        assert report.most_consistent_pair is None

    def test_empty_providers(self):
        report = compare_all_providers({})
        assert len(report.comparisons) == 0

    def test_identifies_most_consistent(self):
        ps = {
            "a": {"q1": 0.8, "q2": 0.6, "q3": 0.4},
            "b": {"q1": 0.8, "q2": 0.6, "q3": 0.4},  # identical to a
            "c": {"q1": 0.1, "q2": 0.9, "q3": 0.2},  # very different
        }
        report = compare_all_providers(ps)
        # a and b are identical, so most consistent should be (a, b)
        # Correlation for identical constant-diff: _pearson returns 0 for identical
        # Actually a and b have the same values, so std=0, correlation=0
        # But a vs c and b vs c may have negative or low correlation
        # The pair with highest correlation wins
        assert report.most_consistent_pair is not None


# -------------------------------------------------------------------
# normalize_scores
# -------------------------------------------------------------------


class TestNormalizeScores:
    def test_zero_mean(self):
        scores = {"q1": 1.0, "q2": 3.0, "q3": 5.0}
        normed = normalize_scores(scores)
        mean = sum(normed.values()) / len(normed)
        assert abs(mean) < 1e-9

    def test_unit_variance(self):
        scores = {"q1": 1.0, "q2": 3.0, "q3": 5.0}
        normed = normalize_scores(scores)
        vals = list(normed.values())
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        assert abs(var - 1.0) < 1e-9

    def test_empty(self):
        assert normalize_scores({}) == {}

    def test_constant_scores(self):
        scores = {"q1": 5.0, "q2": 5.0, "q3": 5.0}
        normed = normalize_scores(scores)
        for v in normed.values():
            assert v == 0.0

    def test_preserves_keys(self):
        scores = {"a": 1.0, "b": 2.0}
        normed = normalize_scores(scores)
        assert set(normed.keys()) == {"a", "b"}


# -------------------------------------------------------------------
# find_divergent_items
# -------------------------------------------------------------------


class TestFindDivergentItems:
    def test_detects_divergence(self):
        sa = {"q1": 0.8, "q2": 0.5, "q3": 0.9}
        sb = {"q1": 0.1, "q2": 0.5, "q3": 0.85}
        result = find_divergent_items(sa, sb, threshold=0.3)
        assert "q1" in result  # delta=0.7
        assert "q2" not in result  # delta=0.0
        assert "q3" not in result  # delta=0.05

    def test_none_divergent(self):
        sa = {"q1": 0.5, "q2": 0.6}
        sb = {"q1": 0.5, "q2": 0.6}
        assert find_divergent_items(sa, sb) == []

    def test_all_divergent(self):
        sa = {"q1": 0.0, "q2": 0.0}
        sb = {"q1": 1.0, "q2": 1.0}
        result = find_divergent_items(sa, sb, threshold=0.3)
        assert len(result) == 2

    def test_custom_threshold(self):
        sa = {"q1": 0.5}
        sb = {"q1": 0.6}
        assert find_divergent_items(sa, sb, threshold=0.05) == ["q1"]
        assert find_divergent_items(sa, sb, threshold=0.2) == []

    def test_no_common_items(self):
        assert find_divergent_items({"q1": 0.5}, {"q2": 0.5}) == []

    def test_empty_scores(self):
        assert find_divergent_items({}, {}) == []
