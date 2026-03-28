"""Tests for the judge debiasing module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.evaluation.judge_debiasing import (
    BiasMetrics,
    DebiasedScore,
    DebiasReport,
    detect_position_bias,
    detect_length_bias,
    correct_length_bias,
    correct_position_bias,
    analyze_judge_bias,
)


# ---------------------------------------------------------------------------
# detect_position_bias
# ---------------------------------------------------------------------------


class TestDetectPositionBias:
    def test_zero_when_equal(self):
        first = [0.5, 0.6, 0.7]
        second = [0.5, 0.6, 0.7]
        assert abs(detect_position_bias(first, second)) < 1e-9

    def test_positive_when_first_favored(self):
        first = [0.8, 0.9, 0.7]
        second = [0.3, 0.4, 0.2]
        bias = detect_position_bias(first, second)
        assert bias > 0.0

    def test_negative_when_second_favored(self):
        first = [0.2, 0.3, 0.1]
        second = [0.8, 0.9, 0.7]
        bias = detect_position_bias(first, second)
        assert bias < 0.0

    def test_empty_lists(self):
        assert detect_position_bias([], []) == 0.0

    def test_single_element(self):
        bias = detect_position_bias([1.0], [0.5])
        assert abs(bias - 0.5) < 1e-9

    def test_unequal_lengths_uses_minimum(self):
        first = [0.8, 0.9]
        second = [0.3, 0.4, 0.5]
        bias = detect_position_bias(first, second)
        # Should use min(2,3) = 2 pairs
        expected = ((0.8 - 0.3) + (0.9 - 0.4)) / 2
        assert abs(bias - expected) < 1e-9


# ---------------------------------------------------------------------------
# detect_length_bias
# ---------------------------------------------------------------------------


class TestDetectLengthBias:
    def test_zero_for_uncorrelated(self):
        # Constant scores regardless of length
        lengths = [10, 20, 30, 40, 50]
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        assert abs(detect_length_bias(lengths, scores)) < 1e-9

    def test_positive_for_correlated(self):
        # Longer outputs get higher scores
        lengths = [10, 20, 30, 40, 50]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        r = detect_length_bias(lengths, scores)
        assert abs(r - 1.0) < 1e-9  # Perfect positive correlation

    def test_negative_correlation(self):
        lengths = [10, 20, 30, 40, 50]
        scores = [0.5, 0.4, 0.3, 0.2, 0.1]
        r = detect_length_bias(lengths, scores)
        assert abs(r - (-1.0)) < 1e-9

    def test_insufficient_data_empty(self):
        assert detect_length_bias([], []) == 0.0

    def test_insufficient_data_single(self):
        assert detect_length_bias([100], [0.5]) == 0.0

    def test_constant_lengths(self):
        lengths = [100, 100, 100]
        scores = [0.1, 0.5, 0.9]
        assert abs(detect_length_bias(lengths, scores)) < 1e-9


# ---------------------------------------------------------------------------
# correct_length_bias
# ---------------------------------------------------------------------------


class TestCorrectLengthBias:
    def test_removes_correlation(self):
        lengths = [10, 20, 30, 40, 50]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        debiased = correct_length_bias(scores, lengths)
        assert len(debiased) == 5
        # After correction, all debiased scores should be roughly the same
        vals = [d.debiased_score for d in debiased]
        assert max(vals) - min(vals) < 1e-9

    def test_correction_field(self):
        lengths = [10, 20, 30, 40, 50]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        debiased = correct_length_bias(scores, lengths)
        for d in debiased:
            assert abs(d.debiased_score - d.original_score - d.correction) < 1e-9

    def test_method_label(self):
        debiased = correct_length_bias([0.5], [100])
        assert debiased[0].method == "length_regression"

    def test_empty_input(self):
        assert correct_length_bias([], []) == []

    def test_no_length_variance(self):
        debiased = correct_length_bias([0.5, 0.7], [100, 100])
        assert len(debiased) == 2
        for d in debiased:
            assert d.correction == 0.0


# ---------------------------------------------------------------------------
# correct_position_bias
# ---------------------------------------------------------------------------


class TestCorrectPositionBias:
    def test_averages_scores(self):
        first = [0.8, 0.6]
        second = [0.4, 0.2]
        result = correct_position_bias(first, second)
        assert abs(result[0] - 0.6) < 1e-9
        assert abs(result[1] - 0.4) < 1e-9

    def test_empty_input(self):
        assert correct_position_bias([], []) == []

    def test_identical_scores(self):
        first = [0.5, 0.5]
        second = [0.5, 0.5]
        result = correct_position_bias(first, second)
        assert all(abs(r - 0.5) < 1e-9 for r in result)


# ---------------------------------------------------------------------------
# BiasMetrics
# ---------------------------------------------------------------------------


class TestBiasMetrics:
    def test_no_significant_bias(self):
        m = BiasMetrics(position_bias=0.05, length_bias=0.03, verbosity_bias=0.01)
        assert not m.has_significant_bias()

    def test_significant_position_bias(self):
        m = BiasMetrics(position_bias=0.2)
        assert m.has_significant_bias()

    def test_significant_length_bias(self):
        m = BiasMetrics(length_bias=0.15)
        assert m.has_significant_bias()

    def test_custom_threshold(self):
        m = BiasMetrics(position_bias=0.05)
        assert not m.has_significant_bias(threshold=0.1)
        assert m.has_significant_bias(threshold=0.01)

    def test_as_dict_from_dict_roundtrip(self):
        m = BiasMetrics(position_bias=0.12, length_bias=0.34, verbosity_bias=0.05)
        m2 = BiasMetrics.from_dict(m.as_dict())
        assert abs(m2.position_bias - 0.12) < 0.001
        assert abs(m2.length_bias - 0.34) < 0.001


# ---------------------------------------------------------------------------
# DebiasedScore
# ---------------------------------------------------------------------------


class TestDebiasedScore:
    def test_fields(self):
        s = DebiasedScore(
            original_score=0.8, debiased_score=0.65, correction=-0.15, method="length"
        )
        assert s.original_score == 0.8
        assert s.method == "length"

    def test_as_dict(self):
        s = DebiasedScore(0.8, 0.65, -0.15, "length")
        d = s.as_dict()
        assert d["original_score"] == 0.8
        assert d["method"] == "length"


# ---------------------------------------------------------------------------
# DebiasReport
# ---------------------------------------------------------------------------


class TestDebiasReport:
    def test_as_dict_from_dict_roundtrip(self):
        report = DebiasReport(
            metrics=BiasMetrics(0.1, 0.2, 0.05),
            scores=[DebiasedScore(0.8, 0.7, -0.1, "combined")],
            method="combined",
        )
        d = report.as_dict()
        r2 = DebiasReport.from_dict(d)
        assert r2.method == "combined"
        assert len(r2.scores) == 1
        assert abs(r2.metrics.position_bias - 0.1) < 0.001

    def test_format_text(self):
        report = DebiasReport(
            metrics=BiasMetrics(0.15, 0.25, 0.10),
            scores=[],
            method="combined",
        )
        text = report.format_text()
        assert "Judge Debiasing Report" in text
        assert "Position bias" in text


# ---------------------------------------------------------------------------
# analyze_judge_bias
# ---------------------------------------------------------------------------


class TestAnalyzeJudgeBias:
    def test_full_report(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        lengths = [10, 20, 30, 40, 50]
        report = analyze_judge_bias(scores, lengths)
        assert report.method == "combined"
        assert len(report.scores) == 5
        assert abs(report.metrics.length_bias - 1.0) < 1e-6

    def test_with_positions(self):
        scores = [0.9, 0.8, 0.3, 0.2]
        lengths = [100, 100, 100, 100]
        positions = [0, 0, 1, 1]
        report = analyze_judge_bias(scores, lengths, positions)
        assert report.metrics.position_bias > 0.0

    def test_empty_input(self):
        report = analyze_judge_bias([], [])
        assert report.method == "combined"
        assert len(report.scores) == 0

    def test_no_position_bias_without_positions(self):
        scores = [0.5, 0.5, 0.5]
        lengths = [100, 100, 100]
        report = analyze_judge_bias(scores, lengths)
        assert report.metrics.position_bias == 0.0

    def test_debiased_scores_have_corrections(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        lengths = [10, 20, 30, 40, 50]
        report = analyze_judge_bias(scores, lengths)
        for s in report.scores:
            assert abs(s.debiased_score - s.original_score - s.correction) < 1e-9
