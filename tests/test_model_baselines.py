"""Tests for the local model performance baselines module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.model_baselines import (
    AgreementResult,
    BaselineReport,
    JudgeScore,
    build_baseline_report,
    compute_agreement,
    compute_correlation,
    format_baseline_report,
    rank_metrics_by_agreement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(
    model: str,
    metric: str,
    scores: list[float],
) -> list[JudgeScore]:
    """Create a list of JudgeScore for sequential item IDs."""
    return [
        JudgeScore(
            item_id=f"item_{i}",
            metric_id=metric,
            judge_model=model,
            score=s,
            latency_ms=float(i),
        )
        for i, s in enumerate(scores)
    ]


# ---------------------------------------------------------------------------
# JudgeScore dataclass
# ---------------------------------------------------------------------------


class TestJudgeScore:
    def test_as_dict(self):
        s = JudgeScore("i1", "m1", "local-7b", 0.8, 120.0)
        d = s.as_dict()
        assert d["item_id"] == "i1"
        assert d["score"] == 0.8
        assert d["latency_ms"] == 120.0

    def test_from_dict(self):
        d = {"item_id": "i2", "metric_id": "m2", "judge_model": "api", "score": 0.9}
        s = JudgeScore.from_dict(d)
        assert s.judge_model == "api"
        assert s.latency_ms == 0.0  # default

    def test_roundtrip(self):
        original = JudgeScore("i1", "m1", "model", 0.5, 50.0)
        restored = JudgeScore.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.score == original.score
        assert restored.latency_ms == original.latency_ms

    def test_default_latency(self):
        s = JudgeScore("i1", "m1", "model", 0.5)
        assert s.latency_ms == 0.0


# ---------------------------------------------------------------------------
# AgreementResult dataclass
# ---------------------------------------------------------------------------


class TestAgreementResult:
    def test_as_dict(self):
        a = AgreementResult("m1", "local", "api", 0.9, 0.95, 0.01, 100)
        d = a.as_dict()
        assert d["agreement_rate"] == 0.9
        assert d["n_items"] == 100

    def test_from_dict(self):
        d = {
            "metric_id": "m1", "local_model": "loc", "api_model": "api",
            "agreement_rate": 0.85, "correlation": 0.9,
            "mean_score_diff": -0.02, "n_items": 50,
        }
        a = AgreementResult.from_dict(d)
        assert a.correlation == 0.9

    def test_from_dict_defaults(self):
        a = AgreementResult.from_dict({"metric_id": "x", "local_model": "l", "api_model": "a"})
        assert a.agreement_rate == 0.0
        assert a.n_items == 0

    def test_roundtrip(self):
        original = AgreementResult("m1", "loc", "api", 0.85, 0.92, -0.03, 42)
        restored = AgreementResult.from_dict(original.as_dict())
        assert restored.metric_id == original.metric_id
        assert restored.agreement_rate == original.agreement_rate


# ---------------------------------------------------------------------------
# BaselineReport dataclass
# ---------------------------------------------------------------------------


class TestBaselineReport:
    def test_as_dict(self):
        report = BaselineReport(
            agreements=[], overall_agreement=0.0,
            recommended_local_metrics=[], not_recommended_metrics=[],
            generated_at="2026-01-01T00:00:00",
        )
        d = report.as_dict()
        assert d["overall_agreement"] == 0.0
        assert d["generated_at"] == "2026-01-01T00:00:00"

    def test_from_dict_with_nested(self):
        d = {
            "agreements": [
                {"metric_id": "m1", "local_model": "l", "api_model": "a",
                 "agreement_rate": 0.9, "correlation": 0.8,
                 "mean_score_diff": 0.01, "n_items": 10},
            ],
            "overall_agreement": 0.9,
            "recommended_local_metrics": ["m1"],
            "not_recommended_metrics": [],
            "generated_at": "2026-01-01",
        }
        report = BaselineReport.from_dict(d)
        assert len(report.agreements) == 1
        assert report.recommended_local_metrics == ["m1"]

    def test_roundtrip(self):
        a = AgreementResult("m1", "loc", "api", 0.9, 0.95, 0.01, 20)
        original = BaselineReport(
            agreements=[a], overall_agreement=0.9,
            recommended_local_metrics=["m1"], not_recommended_metrics=[],
            generated_at="2026-03-28T12:00:00+00:00",
        )
        restored = BaselineReport.from_dict(original.as_dict())
        assert restored.overall_agreement == 0.9
        assert restored.agreements[0].metric_id == "m1"


# ---------------------------------------------------------------------------
# compute_correlation
# ---------------------------------------------------------------------------


class TestComputeCorrelation:
    def test_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(compute_correlation(xs, ys) - 1.0) < 1e-9

    def test_perfect_negative(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(compute_correlation(xs, ys) - (-1.0)) < 1e-9

    def test_no_correlation(self):
        # Orthogonal pattern: increasing xs, symmetric ys around mean
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [3.0, 1.0, 5.0, 1.0, 3.0]
        r = compute_correlation(xs, ys)
        assert abs(r) < 0.01

    def test_zero_variance_x(self):
        xs = [5.0, 5.0, 5.0]
        ys = [1.0, 2.0, 3.0]
        assert compute_correlation(xs, ys) == 0.0

    def test_zero_variance_y(self):
        xs = [1.0, 2.0, 3.0]
        ys = [5.0, 5.0, 5.0]
        assert compute_correlation(xs, ys) == 0.0

    def test_empty_lists(self):
        assert compute_correlation([], []) == 0.0

    def test_single_element(self):
        assert compute_correlation([1.0], [2.0]) == 0.0

    def test_length_mismatch(self):
        assert compute_correlation([1.0, 2.0], [3.0]) == 0.0

    def test_moderate_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.1, 1.9, 3.2, 3.8, 5.1]
        r = compute_correlation(xs, ys)
        assert 0.9 < r < 1.0

    def test_known_value(self):
        # xs = [1,2,3], ys = [2,4,5]
        # mean_x = 2, mean_y = 3.666...
        # cov = (1-2)*(2-3.666) + (2-2)*(4-3.666) + (3-2)*(5-3.666)
        #     = 1.666 + 0 + 1.333 = 3.0
        # var_x = 1+0+1 = 2, var_y = 2.777+0.111+1.777 = 4.666
        # r = 3 / sqrt(2*4.666) = 3 / sqrt(9.333) = 3 / 3.0550... = 0.9819...
        r = compute_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 5.0])
        assert abs(r - 0.9819805) < 1e-4


# ---------------------------------------------------------------------------
# compute_agreement
# ---------------------------------------------------------------------------


class TestComputeAgreement:
    def test_perfect_agreement(self):
        local = _make_scores("local", "m1", [0.5, 0.8, 0.3])
        api = _make_scores("api", "m1", [0.5, 0.8, 0.3])
        result = compute_agreement(local, api)
        assert result.agreement_rate == 1.0
        assert result.mean_score_diff == 0.0
        assert result.n_items == 3

    def test_no_agreement(self):
        local = _make_scores("local", "m1", [0.0, 0.0, 0.0])
        api = _make_scores("api", "m1", [1.0, 1.0, 1.0])
        result = compute_agreement(local, api, threshold=0.1)
        assert result.agreement_rate == 0.0

    def test_partial_agreement(self):
        local = _make_scores("local", "m1", [0.5, 0.5, 0.5, 0.5])
        api = _make_scores("api", "m1", [0.5, 0.5, 1.0, 1.0])
        result = compute_agreement(local, api, threshold=0.1)
        assert result.agreement_rate == 0.5

    def test_threshold_boundary(self):
        local = _make_scores("local", "m1", [0.5])
        api = _make_scores("api", "m1", [1.0])
        # diff = 0.5, threshold = 0.5 -> within
        result = compute_agreement(local, api, threshold=0.5)
        assert result.agreement_rate == 1.0

    def test_threshold_just_outside(self):
        local = _make_scores("local", "m1", [0.5])
        api = _make_scores("api", "m1", [1.01])
        result = compute_agreement(local, api, threshold=0.5)
        assert result.agreement_rate == 0.0

    def test_mean_score_diff(self):
        local = _make_scores("local", "m1", [0.6, 0.8])
        api = _make_scores("api", "m1", [0.5, 0.7])
        result = compute_agreement(local, api)
        assert abs(result.mean_score_diff - 0.1) < 1e-9

    def test_correlation_in_result(self):
        local = _make_scores("local", "m1", [0.1, 0.5, 0.9])
        api = _make_scores("api", "m1", [0.2, 0.6, 1.0])
        result = compute_agreement(local, api)
        assert result.correlation > 0.99

    def test_no_matching_items(self):
        local = [JudgeScore("a", "m1", "local", 0.5)]
        api = [JudgeScore("b", "m1", "api", 0.5)]
        result = compute_agreement(local, api)
        assert result.n_items == 0
        assert result.agreement_rate == 0.0

    def test_model_names_captured(self):
        local = _make_scores("my-local", "m1", [0.5])
        api = _make_scores("gpt-4", "m1", [0.6])
        result = compute_agreement(local, api)
        assert result.local_model == "my-local"
        assert result.api_model == "gpt-4"

    def test_metric_id_captured(self):
        local = _make_scores("loc", "helpfulness", [0.5])
        api = _make_scores("api", "helpfulness", [0.6])
        result = compute_agreement(local, api)
        assert result.metric_id == "helpfulness"

    def test_empty_inputs(self):
        result = compute_agreement([], [])
        assert result.n_items == 0


# ---------------------------------------------------------------------------
# build_baseline_report
# ---------------------------------------------------------------------------


class TestBuildBaselineReport:
    def test_single_metric_recommended(self):
        local = _make_scores("loc", "m1", [0.5, 0.5, 0.5])
        api = _make_scores("api", "m1", [0.5, 0.5, 0.5])
        report = build_baseline_report(local, api, agreement_threshold=0.8)
        assert "m1" in report.recommended_local_metrics
        assert report.not_recommended_metrics == []
        assert report.overall_agreement == 1.0

    def test_single_metric_not_recommended(self):
        local = _make_scores("loc", "m1", [0.0, 0.0, 0.0])
        api = _make_scores("api", "m1", [1.0, 1.0, 1.0])
        report = build_baseline_report(local, api, agreement_threshold=0.8)
        assert "m1" in report.not_recommended_metrics
        assert report.recommended_local_metrics == []

    def test_multiple_metrics(self):
        local = (
            _make_scores("loc", "good_metric", [0.5, 0.5, 0.5])
            + _make_scores("loc", "bad_metric", [0.0, 0.0, 0.0])
        )
        api = (
            _make_scores("api", "good_metric", [0.5, 0.5, 0.5])
            + _make_scores("api", "bad_metric", [1.0, 1.0, 1.0])
        )
        report = build_baseline_report(local, api, agreement_threshold=0.8)
        assert "good_metric" in report.recommended_local_metrics
        assert "bad_metric" in report.not_recommended_metrics

    def test_empty_inputs(self):
        report = build_baseline_report([], [])
        assert report.overall_agreement == 0.0
        assert report.agreements == []

    def test_generated_at_present(self):
        report = build_baseline_report([], [])
        assert report.generated_at != ""

    def test_overall_agreement_is_average(self):
        # Two metrics: one 100% agreement, one 0% agreement
        local = (
            _make_scores("loc", "m1", [0.5, 0.5])
            + _make_scores("loc", "m2", [0.0, 0.0])
        )
        api = (
            _make_scores("api", "m1", [0.5, 0.5])
            + _make_scores("api", "m2", [1.0, 1.0])
        )
        report = build_baseline_report(local, api)
        assert abs(report.overall_agreement - 0.5) < 1e-9

    def test_agreement_threshold_adjustable(self):
        local = _make_scores("loc", "m1", [0.5, 0.5])
        api = _make_scores("api", "m1", [0.7, 0.7])
        # Default threshold=0.5 -> within -> agreement=1.0
        report_lenient = build_baseline_report(local, api, agreement_threshold=0.5)
        assert "m1" in report_lenient.recommended_local_metrics

        # Very strict threshold
        report_strict = build_baseline_report(local, api, agreement_threshold=1.01)
        assert "m1" in report_strict.not_recommended_metrics


# ---------------------------------------------------------------------------
# format_baseline_report
# ---------------------------------------------------------------------------


class TestFormatBaselineReport:
    def test_empty_report(self):
        report = BaselineReport(
            agreements=[], overall_agreement=0.0,
            recommended_local_metrics=[], not_recommended_metrics=[],
            generated_at="2026-01-01",
        )
        text = format_baseline_report(report)
        assert "Model Baseline Report" in text
        assert "0.0%" in text

    def test_report_with_data(self):
        a = AgreementResult("helpfulness", "local-7b", "gpt-4", 0.9, 0.95, 0.02, 50)
        report = BaselineReport(
            agreements=[a], overall_agreement=0.9,
            recommended_local_metrics=["helpfulness"],
            not_recommended_metrics=[],
            generated_at="2026-01-01",
        )
        text = format_baseline_report(report)
        assert "helpfulness" in text
        assert "90.0%" in text
        assert "Recommended" in text

    def test_not_recommended_shown(self):
        a = AgreementResult("bad_metric", "local", "api", 0.3, 0.2, -0.5, 10)
        report = BaselineReport(
            agreements=[a], overall_agreement=0.3,
            recommended_local_metrics=[],
            not_recommended_metrics=["bad_metric"],
            generated_at="2026-01-01",
        )
        text = format_baseline_report(report)
        assert "Not recommended" in text
        assert "bad_metric" in text

    def test_generated_at_shown(self):
        report = BaselineReport(
            agreements=[], overall_agreement=0.0,
            recommended_local_metrics=[], not_recommended_metrics=[],
            generated_at="2026-03-28T12:00:00",
        )
        text = format_baseline_report(report)
        assert "2026-03-28" in text


# ---------------------------------------------------------------------------
# rank_metrics_by_agreement
# ---------------------------------------------------------------------------


class TestRankMetricsByAgreement:
    def test_sorted_descending(self):
        agreements = [
            AgreementResult("low", "l", "a", 0.3, 0.0, 0.0, 10),
            AgreementResult("high", "l", "a", 0.9, 0.0, 0.0, 10),
            AgreementResult("mid", "l", "a", 0.6, 0.0, 0.0, 10),
        ]
        report = BaselineReport(
            agreements=agreements, overall_agreement=0.6,
            recommended_local_metrics=[], not_recommended_metrics=[],
            generated_at="",
        )
        ranked = rank_metrics_by_agreement(report)
        assert ranked[0] == ("high", 0.9)
        assert ranked[1] == ("mid", 0.6)
        assert ranked[2] == ("low", 0.3)

    def test_empty(self):
        report = BaselineReport(
            agreements=[], overall_agreement=0.0,
            recommended_local_metrics=[], not_recommended_metrics=[],
            generated_at="",
        )
        assert rank_metrics_by_agreement(report) == []

    def test_single_metric(self):
        agreements = [AgreementResult("only", "l", "a", 0.75, 0.0, 0.0, 5)]
        report = BaselineReport(
            agreements=agreements, overall_agreement=0.75,
            recommended_local_metrics=[], not_recommended_metrics=[],
            generated_at="",
        )
        ranked = rank_metrics_by_agreement(report)
        assert len(ranked) == 1
        assert ranked[0] == ("only", 0.75)

    def test_equal_agreement_stable(self):
        agreements = [
            AgreementResult("a", "l", "api", 0.5, 0.0, 0.0, 10),
            AgreementResult("b", "l", "api", 0.5, 0.0, 0.0, 10),
        ]
        report = BaselineReport(
            agreements=agreements, overall_agreement=0.5,
            recommended_local_metrics=[], not_recommended_metrics=[],
            generated_at="",
        )
        ranked = rank_metrics_by_agreement(report)
        assert len(ranked) == 2
        names = [r[0] for r in ranked]
        assert "a" in names and "b" in names
