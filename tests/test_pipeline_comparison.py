from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.pipeline_comparison import (
    MetricDelta,
    PipelineComparisonReport,
    RunSummary,
    compare_runs,
    compute_metric_deltas,
    format_comparison_table,
    is_regression,
    rank_by_improvement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = "run-a",
    pipeline_name: str = "pipe",
    pass_rate: float = 0.8,
    total_items: int = 100,
    total_cost_usd: float = 1.0,
    duration_ms: float = 5000.0,
    metrics_scores: dict | None = None,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        pipeline_name=pipeline_name,
        pass_rate=pass_rate,
        total_items=total_items,
        total_cost_usd=total_cost_usd,
        duration_ms=duration_ms,
        metrics_scores=metrics_scores or {},
    )


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_as_dict(self):
        r = _make_run(metrics_scores={"helpfulness": 0.9})
        d = r.as_dict()
        assert d["run_id"] == "run-a"
        assert d["pass_rate"] == 0.8
        assert d["metrics_scores"] == {"helpfulness": 0.9}

    def test_from_dict_roundtrip(self):
        r = _make_run(metrics_scores={"m1": 0.5, "m2": 0.7})
        restored = RunSummary.from_dict(r.as_dict())
        assert restored == r

    def test_from_dict_defaults(self):
        r = RunSummary.from_dict({"run_id": "x", "pipeline_name": "p"})
        assert r.pass_rate == 0.0
        assert r.total_items == 0
        assert r.total_cost_usd == 0.0
        assert r.duration_ms == 0.0
        assert r.metrics_scores == {}

    def test_as_dict_copies_scores(self):
        scores = {"m1": 0.5}
        r = _make_run(metrics_scores=scores)
        d = r.as_dict()
        d["metrics_scores"]["m2"] = 0.9
        assert "m2" not in r.metrics_scores


# ---------------------------------------------------------------------------
# MetricDelta
# ---------------------------------------------------------------------------


class TestMetricDelta:
    def test_as_dict(self):
        d = MetricDelta("helpfulness", 0.7, 0.9, 0.2, True)
        out = d.as_dict()
        assert out["metric_id"] == "helpfulness"
        assert out["delta"] == 0.2
        assert out["improved"] is True

    def test_not_improved(self):
        d = MetricDelta("safety", 0.9, 0.7, -0.2, False)
        assert d.improved is False
        assert d.delta == -0.2


# ---------------------------------------------------------------------------
# compute_metric_deltas
# ---------------------------------------------------------------------------


class TestComputeMetricDeltas:
    def test_matching_metrics(self):
        deltas = compute_metric_deltas(
            {"m1": 0.5, "m2": 0.8}, {"m1": 0.7, "m2": 0.6}
        )
        by_id = {d.metric_id: d for d in deltas}
        assert by_id["m1"].delta == pytest.approx(0.2)
        assert by_id["m1"].improved is True
        assert by_id["m2"].delta == pytest.approx(-0.2)
        assert by_id["m2"].improved is False

    def test_missing_in_a(self):
        deltas = compute_metric_deltas({}, {"m1": 0.5})
        assert len(deltas) == 1
        assert deltas[0].score_a == 0.0
        assert deltas[0].score_b == 0.5
        assert deltas[0].delta == pytest.approx(0.5)

    def test_missing_in_b(self):
        deltas = compute_metric_deltas({"m1": 0.5}, {})
        assert len(deltas) == 1
        assert deltas[0].score_a == 0.5
        assert deltas[0].score_b == 0.0
        assert deltas[0].delta == pytest.approx(-0.5)

    def test_empty_both(self):
        assert compute_metric_deltas({}, {}) == []

    def test_sorted_by_key(self):
        deltas = compute_metric_deltas(
            {"z": 1.0, "a": 0.5}, {"z": 0.5, "a": 1.0}
        )
        assert deltas[0].metric_id == "a"
        assert deltas[1].metric_id == "z"


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_basic(self):
        a = _make_run("a", "pipe", 0.7, 100, 1.0, 5000, {"m1": 0.6})
        b = _make_run("b", "pipe", 0.9, 100, 1.5, 4000, {"m1": 0.8})
        report = compare_runs(a, b)
        assert report.pass_rate_delta == pytest.approx(0.2)
        assert report.cost_delta == pytest.approx(0.5)
        assert report.duration_delta == pytest.approx(-1000)
        assert report.overall_improved is True

    def test_regression(self):
        a = _make_run("a", pass_rate=0.9)
        b = _make_run("b", pass_rate=0.6)
        report = compare_runs(a, b)
        assert report.overall_improved is False
        assert report.pass_rate_delta == pytest.approx(-0.3)

    def test_identical_runs(self):
        a = _make_run("a", metrics_scores={"m1": 0.5})
        b = _make_run("b", metrics_scores={"m1": 0.5})
        report = compare_runs(a, b)
        assert report.pass_rate_delta == pytest.approx(0.0)
        assert report.cost_delta == pytest.approx(0.0)
        assert report.duration_delta == pytest.approx(0.0)
        assert report.overall_improved is False

    def test_empty_metrics(self):
        a = _make_run("a")
        b = _make_run("b")
        report = compare_runs(a, b)
        assert report.metric_deltas == []


# ---------------------------------------------------------------------------
# is_regression
# ---------------------------------------------------------------------------


class TestIsRegression:
    def test_regression_true(self):
        a = _make_run("a", pass_rate=0.9)
        b = _make_run("b", pass_rate=0.8)
        report = compare_runs(a, b)
        assert is_regression(report, threshold=0.05) is True

    def test_regression_false_improved(self):
        a = _make_run("a", pass_rate=0.7)
        b = _make_run("b", pass_rate=0.9)
        report = compare_runs(a, b)
        assert is_regression(report) is False

    def test_regression_false_within_threshold(self):
        a = _make_run("a", pass_rate=0.80)
        b = _make_run("b", pass_rate=0.76)
        report = compare_runs(a, b)
        assert is_regression(report, threshold=0.05) is False

    def test_regression_just_over_threshold(self):
        a = _make_run("a", pass_rate=0.80)
        b = _make_run("b", pass_rate=0.70)
        report = compare_runs(a, b)
        # delta is -0.10, well past threshold of 0.05
        assert is_regression(report, threshold=0.05) is True


# ---------------------------------------------------------------------------
# PipelineComparisonReport
# ---------------------------------------------------------------------------


class TestPipelineComparisonReport:
    def test_as_dict_from_dict_roundtrip(self):
        a = _make_run("a", "pipe-a", 0.7, 50, 0.5, 3000, {"m1": 0.6})
        b = _make_run("b", "pipe-b", 0.9, 50, 0.8, 2000, {"m1": 0.8})
        report = compare_runs(a, b)
        restored = PipelineComparisonReport.from_dict(report.as_dict())
        assert restored.run_a.run_id == "a"
        assert restored.run_b.run_id == "b"
        assert restored.pass_rate_delta == pytest.approx(0.2)
        assert len(restored.metric_deltas) == 1
        assert restored.overall_improved is True

    def test_format_text_contains_key_info(self):
        a = _make_run("run-1", "pipeline-x", 0.7, metrics_scores={"m1": 0.5})
        b = _make_run("run-2", "pipeline-y", 0.9, metrics_scores={"m1": 0.8})
        report = compare_runs(a, b)
        text = report.format_text()
        assert "pipeline-x" in text
        assert "pipeline-y" in text
        assert "run-1" in text
        assert "run-2" in text
        assert "IMPROVED" in text

    def test_format_text_not_improved(self):
        a = _make_run("a", pass_rate=0.9)
        b = _make_run("b", pass_rate=0.7)
        report = compare_runs(a, b)
        text = report.format_text()
        assert "NOT IMPROVED" in text


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------


class TestFormatComparisonTable:
    def test_output_contains_metrics(self):
        a = _make_run("a", metrics_scores={"helpfulness": 0.6, "safety": 0.9})
        b = _make_run("b", metrics_scores={"helpfulness": 0.8, "safety": 0.7})
        report = compare_runs(a, b)
        table = format_comparison_table(report)
        assert "helpfulness" in table
        assert "safety" in table
        assert "better" in table
        assert "worse" in table

    def test_output_contains_header(self):
        a = _make_run("a", metrics_scores={"m1": 0.5})
        b = _make_run("b", metrics_scores={"m1": 0.5})
        report = compare_runs(a, b)
        table = format_comparison_table(report)
        assert "Metric" in table
        assert "Run A" in table
        assert "Run B" in table
        assert "Delta" in table

    def test_same_score_shows_same(self):
        a = _make_run("a", metrics_scores={"m1": 0.5})
        b = _make_run("b", metrics_scores={"m1": 0.5})
        report = compare_runs(a, b)
        table = format_comparison_table(report)
        assert "same" in table

    def test_pass_rate_row(self):
        a = _make_run("a", pass_rate=0.7)
        b = _make_run("b", pass_rate=0.9)
        report = compare_runs(a, b)
        table = format_comparison_table(report)
        assert "Pass rate" in table


# ---------------------------------------------------------------------------
# rank_by_improvement
# ---------------------------------------------------------------------------


class TestRankByImprovement:
    def test_sorted_descending(self):
        deltas = [
            MetricDelta("a", 0.5, 0.6, 0.1, True),
            MetricDelta("b", 0.5, 0.9, 0.4, True),
            MetricDelta("c", 0.5, 0.3, -0.2, False),
        ]
        ranked = rank_by_improvement(deltas)
        assert ranked[0].metric_id == "b"
        assert ranked[1].metric_id == "a"
        assert ranked[2].metric_id == "c"

    def test_empty(self):
        assert rank_by_improvement([]) == []

    def test_single_element(self):
        deltas = [MetricDelta("x", 0.5, 0.7, 0.2, True)]
        ranked = rank_by_improvement(deltas)
        assert len(ranked) == 1
        assert ranked[0].metric_id == "x"
