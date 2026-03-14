"""Tests for the analysis engine: core, reports, and trends."""

from __future__ import annotations

import pytest

from evalyn_sdk.analysis.core import (
    ItemStats,
    MetricStats,
    RunAnalysis,
    analyze_run,
)
from evalyn_sdk.analysis.reports import (
    ascii_bar,
    ascii_score_distribution,
    format_pass_rate_bar,
    generate_comparison_report,
    generate_text_report,
)
from evalyn_sdk.analysis.trends import (
    TrendAnalysis,
    analyze_trends,
    generate_trend_text_report,
)
from evalyn_sdk.models import EvalRun, MetricResult, MetricSpec


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

ITEM_IDS = [f"item-{i:03d}" for i in range(10)]
METRIC_ACCURACY = "accuracy"
METRIC_RELEVANCE = "relevance"
METRIC_STYLE = "style"  # no pass/fail - score-only metric


def _make_metric_result(
    metric_id: str,
    item_id: str,
    score: float,
    passed: bool | None = None,
    call_id: str = "call-000",
) -> dict:
    """Build a raw metric result dict."""
    return {
        "metric_id": metric_id,
        "item_id": item_id,
        "call_id": call_id,
        "score": score,
        "passed": passed,
        "details": {"reason": f"eval for {metric_id} on {item_id}"},
    }


def _make_run_dict(
    run_id: str = "run-aaaa-bbbb-cccc",
    dataset_name: str = "test-dataset",
    created_at: str = "2026-03-01T10:00:00+00:00",
    items: int = 6,
    include_style: bool = False,
    accuracy_pass_rate: float = 0.8,
    relevance_pass_rate: float = 0.6,
    style_scores: list[float] | None = None,
    usage_summary: dict | None = None,
) -> dict:
    """Build a realistic run dict with configurable pass rates.

    Generates `items` items with accuracy and relevance metrics.
    Pass/fail is deterministic based on the requested pass rate
    (first N items pass, rest fail).
    """
    metrics = [
        {"id": METRIC_ACCURACY, "name": "Accuracy", "type": "objective"},
        {"id": METRIC_RELEVANCE, "name": "Relevance", "type": "subjective"},
    ]
    if include_style:
        metrics.append({"id": METRIC_STYLE, "name": "Style", "type": "subjective"})

    results = []
    acc_pass_count = int(items * accuracy_pass_rate)
    rel_pass_count = int(items * relevance_pass_rate)

    for idx in range(items):
        item_id = ITEM_IDS[idx] if idx < len(ITEM_IDS) else f"item-{idx:03d}"

        acc_passed = idx < acc_pass_count
        acc_score = 0.9 + 0.1 * (idx % 2) if acc_passed else 0.2 + 0.1 * (idx % 3)
        results.append(
            _make_metric_result(
                METRIC_ACCURACY, item_id, acc_score, passed=acc_passed
            )
        )

        rel_passed = idx < rel_pass_count
        rel_score = 0.7 + 0.05 * idx if rel_passed else 0.3 - 0.02 * idx
        results.append(
            _make_metric_result(
                METRIC_RELEVANCE, item_id, rel_score, passed=rel_passed
            )
        )

        if include_style:
            s = (style_scores or [0.5])[idx % len(style_scores or [0.5])]
            results.append(
                _make_metric_result(METRIC_STYLE, item_id, s, passed=None)
            )

    run = {
        "id": run_id,
        "dataset_name": dataset_name,
        "created_at": created_at,
        "metrics": metrics,
        "metric_results": results,
    }
    if usage_summary is not None:
        run["usage_summary"] = usage_summary
    return run


def _make_eval_run(
    run_id: str = "run-aaaa",
    dataset_name: str = "test-dataset",
    created_at_iso: str = "2026-03-01T10:00:00+00:00",
    items: int = 6,
    accuracy_pass_rate: float = 0.8,
    relevance_pass_rate: float = 0.6,
) -> EvalRun:
    """Build an EvalRun model instance (for trend tests)."""
    d = _make_run_dict(
        run_id=run_id,
        dataset_name=dataset_name,
        created_at=created_at_iso,
        items=items,
        accuracy_pass_rate=accuracy_pass_rate,
        relevance_pass_rate=relevance_pass_rate,
    )
    return EvalRun.from_dict(d)


# ---------------------------------------------------------------------------
# MetricStats tests
# ---------------------------------------------------------------------------


class TestMetricStats:
    def test_pass_rate_with_pass_fail(self):
        ms = MetricStats(
            metric_id="m", metric_type="objective",
            count=10, passed=7, failed=3, has_pass_fail=True,
        )
        assert ms.pass_rate == pytest.approx(0.7)

    def test_pass_rate_without_pass_fail_returns_none(self):
        ms = MetricStats(metric_id="m", metric_type="subjective", count=5)
        assert ms.pass_rate is None

    def test_pass_rate_count_zero(self):
        ms = MetricStats(
            metric_id="m", metric_type="objective",
            count=0, has_pass_fail=True,
        )
        assert ms.pass_rate == 0.0

    def test_avg_score(self):
        ms = MetricStats(
            metric_id="m", metric_type="objective",
            scores=[0.2, 0.4, 0.6, 0.8],
        )
        assert ms.avg_score == pytest.approx(0.5)

    def test_avg_score_empty(self):
        ms = MetricStats(metric_id="m", metric_type="objective")
        assert ms.avg_score == 0.0

    def test_min_max_score(self):
        ms = MetricStats(
            metric_id="m", metric_type="objective",
            scores=[0.1, 0.5, 0.9],
        )
        assert ms.min_score == pytest.approx(0.1)
        assert ms.max_score == pytest.approx(0.9)

    def test_std_dev_known_values(self):
        # scores [2, 4, 4, 4, 5, 5, 7, 9] - population std dev = 2.0
        ms = MetricStats(
            metric_id="m", metric_type="objective",
            scores=[2, 4, 4, 4, 5, 5, 7, 9],
        )
        assert ms.std_dev == pytest.approx(2.0)

    def test_std_dev_single_value(self):
        ms = MetricStats(
            metric_id="m", metric_type="objective",
            scores=[0.5],
        )
        assert ms.std_dev == 0.0

    def test_std_dev_empty(self):
        ms = MetricStats(metric_id="m", metric_type="objective")
        assert ms.std_dev == 0.0

    def test_std_dev_identical_values(self):
        ms = MetricStats(
            metric_id="m", metric_type="objective",
            scores=[0.5, 0.5, 0.5],
        )
        assert ms.std_dev == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ItemStats tests
# ---------------------------------------------------------------------------


class TestItemStats:
    def test_all_passed_true(self):
        ist = ItemStats(item_id="i1", metrics_passed=3, metrics_failed=0)
        assert ist.all_passed is True

    def test_all_passed_false_when_failures(self):
        ist = ItemStats(item_id="i1", metrics_passed=2, metrics_failed=1)
        assert ist.all_passed is False

    def test_all_passed_false_when_zero_passed(self):
        ist = ItemStats(item_id="i1", metrics_passed=0, metrics_failed=0)
        assert ist.all_passed is False


# ---------------------------------------------------------------------------
# RunAnalysis and analyze_run tests
# ---------------------------------------------------------------------------


class TestAnalyzeRun:
    def test_basic_analysis(self):
        d = _make_run_dict(items=6, accuracy_pass_rate=1.0, relevance_pass_rate=1.0)
        a = analyze_run(d)
        assert a.run_id == "run-aaaa-bbbb-cccc"
        assert a.dataset_name == "test-dataset"
        assert a.total_items == 6
        assert a.total_metrics == 2
        assert len(a.metric_stats) == 2
        assert METRIC_ACCURACY in a.metric_stats
        assert METRIC_RELEVANCE in a.metric_stats

    def test_overall_pass_rate_all_pass(self):
        d = _make_run_dict(items=5, accuracy_pass_rate=1.0, relevance_pass_rate=1.0)
        a = analyze_run(d)
        assert a.overall_pass_rate == pytest.approx(1.0)

    def test_overall_pass_rate_mixed(self):
        # 6 items: accuracy passes first 4, relevance passes first 3
        # Items 0,1,2 pass both. Items 3 passes acc only. 4,5 fail both.
        d = _make_run_dict(
            items=6, accuracy_pass_rate=4 / 6, relevance_pass_rate=3 / 6
        )
        a = analyze_run(d)
        # Items that pass ALL metrics: items 0,1,2 (3 out of 6)
        assert a.overall_pass_rate == pytest.approx(3 / 6)

    def test_overall_pass_rate_empty(self):
        d = {"id": "empty", "dataset_name": "d", "created_at": "", "metrics": [], "metric_results": []}
        a = analyze_run(d)
        assert a.overall_pass_rate == 0.0
        assert a.total_items == 0

    def test_failed_items_list(self):
        d = _make_run_dict(items=5, accuracy_pass_rate=0.6, relevance_pass_rate=0.4)
        a = analyze_run(d)
        # Items that fail at least one metric should appear in failed_items
        assert len(a.failed_items) > 0
        for fid in a.failed_items:
            assert a.item_stats[fid].all_passed is False

    def test_metric_pass_rate(self):
        d = _make_run_dict(items=10, accuracy_pass_rate=0.7, relevance_pass_rate=0.5)
        a = analyze_run(d)
        assert a.metric_stats[METRIC_ACCURACY].pass_rate == pytest.approx(0.7)
        assert a.metric_stats[METRIC_RELEVANCE].pass_rate == pytest.approx(0.5)

    def test_score_only_metric_has_no_pass_fail(self):
        d = _make_run_dict(items=4, include_style=True, style_scores=[0.3, 0.7])
        a = analyze_run(d)
        assert a.metric_stats[METRIC_STYLE].has_pass_fail is False
        assert a.metric_stats[METRIC_STYLE].pass_rate is None
        assert a.metric_stats[METRIC_STYLE].count == 4

    def test_cost_information(self):
        usage = {
            "total_cost_usd": 1.23,
            "has_unknown_pricing": True,
            "cost_by_metric": {"accuracy": 0.50, "relevance": 0.73},
        }
        d = _make_run_dict(items=3, usage_summary=usage)
        a = analyze_run(d)
        assert a.total_cost_usd == pytest.approx(1.23)
        assert a.has_unknown_pricing is True
        assert a.cost_by_metric["accuracy"] == pytest.approx(0.50)

    def test_none_score_handled(self):
        d = {
            "id": "r1", "dataset_name": "ds", "created_at": "",
            "metrics": [{"id": "m1", "type": "objective"}],
            "metric_results": [
                {"metric_id": "m1", "item_id": "i1", "call_id": "c1",
                 "score": None, "passed": True, "details": {}},
            ],
        }
        a = analyze_run(d)
        assert a.metric_stats["m1"].scores == [0.0]
        assert a.metric_stats["m1"].avg_score == 0.0


# ---------------------------------------------------------------------------
# Reports tests
# ---------------------------------------------------------------------------


class TestAsciiBar:
    def test_full_bar(self):
        bar = ascii_bar(1.0, max_width=10)
        assert bar.count("\u2588") == 10  # fill char
        assert bar.count("\u2591") == 0   # empty char

    def test_empty_bar(self):
        bar = ascii_bar(0.0, max_width=10)
        assert bar.count("\u2588") == 0
        assert bar.count("\u2591") == 10

    def test_half_bar(self):
        bar = ascii_bar(0.5, max_width=10)
        assert bar.count("\u2588") == 5
        assert bar.count("\u2591") == 5

    def test_custom_chars(self):
        bar = ascii_bar(0.3, max_width=10, fill="#", empty="-")
        assert bar.count("#") == 3
        assert bar.count("-") == 7

    def test_total_length(self):
        bar = ascii_bar(0.73, max_width=20)
        assert len(bar) == 20


class TestAsciiScoreDistribution:
    def test_empty_scores(self):
        result = ascii_score_distribution([], "m1")
        assert "no data" in result

    def test_contains_metric_id(self):
        result = ascii_score_distribution([0.5, 0.6], "my_metric")
        assert "my_metric" in result

    def test_contains_avg(self):
        result = ascii_score_distribution([0.4, 0.6], "m")
        assert "avg=0.50" in result


class TestFormatPassRateBar:
    def test_none_pass_rate(self):
        line = format_pass_rate_bar("style_metric", None, 10)
        assert "N/A" in line
        assert "no pass/fail" in line

    def test_valid_pass_rate(self):
        line = format_pass_rate_bar("accuracy", 0.75, 8)
        assert "75.0%" in line
        assert "n=8" in line
        assert "accuracy" in line

    def test_zero_pass_rate(self):
        line = format_pass_rate_bar("metric_x", 0.0, 5)
        assert "0.0%" in line


class TestGenerateTextReport:
    def test_normal_report_structure(self):
        d = _make_run_dict(items=6, accuracy_pass_rate=0.8, relevance_pass_rate=0.5)
        a = analyze_run(d)
        report = generate_text_report(a)
        assert "EVAL RUN ANALYSIS" in report
        assert "OVERALL SUMMARY" in report
        assert "METRIC PASS RATES" in report
        assert "SCORE STATISTICS" in report
        assert "SCORE DISTRIBUTIONS" in report
        assert a.dataset_name in report

    def test_verbose_shows_failed_items(self):
        d = _make_run_dict(items=6, accuracy_pass_rate=0.5, relevance_pass_rate=0.5)
        a = analyze_run(d)
        report_normal = generate_text_report(a, verbose=False)
        report_verbose = generate_text_report(a, verbose=True)
        assert "FAILED ITEMS" not in report_normal
        assert "FAILED ITEMS" in report_verbose

    def test_empty_run(self):
        d = {"id": "empty-run", "dataset_name": "empty", "created_at": "",
             "metrics": [], "metric_results": []}
        a = analyze_run(d)
        report = generate_text_report(a)
        assert "EVAL RUN ANALYSIS" in report
        assert "0" in report

    def test_report_contains_metric_names(self):
        d = _make_run_dict(items=4, include_style=True, style_scores=[0.5, 0.8])
        a = analyze_run(d)
        report = generate_text_report(a)
        assert METRIC_ACCURACY in report
        assert METRIC_RELEVANCE in report
        assert METRIC_STYLE in report


class TestGenerateComparisonReport:
    def test_needs_at_least_two_runs(self):
        d = _make_run_dict(items=3)
        a = analyze_run(d)
        result = generate_comparison_report([a])
        assert "Need at least 2 runs" in result

    def test_two_run_comparison(self):
        d1 = _make_run_dict(
            run_id="run-001", items=6,
            accuracy_pass_rate=0.5, relevance_pass_rate=0.5,
            created_at="2026-03-01T10:00:00+00:00",
        )
        d2 = _make_run_dict(
            run_id="run-002", items=6,
            accuracy_pass_rate=0.8, relevance_pass_rate=0.8,
            created_at="2026-03-02T10:00:00+00:00",
        )
        a1, a2 = analyze_run(d1), analyze_run(d2)
        report = generate_comparison_report([a1, a2])
        assert "EVAL RUN COMPARISON" in report
        assert "PASS RATE BY METRIC" in report
        assert "Run1" in report
        assert "Run2" in report
        assert "Delta" in report

    def test_three_run_comparison(self):
        runs = []
        for i in range(3):
            d = _make_run_dict(
                run_id=f"run-{i:03d}", items=5,
                accuracy_pass_rate=0.4 + 0.2 * i,
                relevance_pass_rate=0.3 + 0.2 * i,
                created_at=f"2026-03-0{i+1}T10:00:00+00:00",
            )
            runs.append(analyze_run(d))
        report = generate_comparison_report(runs)
        assert "Run3" in report


# ---------------------------------------------------------------------------
# Trends tests
# ---------------------------------------------------------------------------


class TestAnalyzeTrendsEmpty:
    def test_zero_runs(self):
        trend = analyze_trends([])
        assert trend.project_name == "unknown"
        assert trend.runs == []
        assert trend.metric_trends == {}
        assert trend.overall_trends == []
        assert trend.timestamps == []
        assert trend.run_ids == []

    def test_one_run(self):
        run = _make_eval_run(
            run_id="single", items=5,
            accuracy_pass_rate=0.8, relevance_pass_rate=0.6,
        )
        trend = analyze_trends([run])
        assert len(trend.runs) == 1
        assert len(trend.overall_trends) == 1
        assert trend.overall_delta == 0.0
        assert trend.improving_metrics == []
        assert trend.regressing_metrics == []
        assert trend.stable_metrics == []


class TestAnalyzeTrendsMultipleRuns:
    def _make_three_runs(self) -> list[EvalRun]:
        return [
            _make_eval_run(
                run_id="r1", items=5,
                accuracy_pass_rate=0.6, relevance_pass_rate=0.4,
                created_at_iso="2026-03-01T10:00:00+00:00",
            ),
            _make_eval_run(
                run_id="r2", items=5,
                accuracy_pass_rate=0.8, relevance_pass_rate=0.4,
                created_at_iso="2026-03-02T10:00:00+00:00",
            ),
            _make_eval_run(
                run_id="r3", items=5,
                accuracy_pass_rate=1.0, relevance_pass_rate=0.4,
                created_at_iso="2026-03-03T10:00:00+00:00",
            ),
        ]

    def test_three_runs_sorted(self):
        runs = self._make_three_runs()
        # Pass in reverse order - should still sort by date
        trend = analyze_trends(list(reversed(runs)))
        assert len(trend.runs) == 3
        assert trend.run_ids[0] == "r1"
        assert trend.run_ids[-1] == "r3"

    def test_overall_delta_positive(self):
        # Use runs where relevance also improves, so overall (all-metrics) rate increases
        runs = [
            _make_eval_run(
                run_id="r1", items=5,
                accuracy_pass_rate=0.4, relevance_pass_rate=0.4,
                created_at_iso="2026-03-01T10:00:00+00:00",
            ),
            _make_eval_run(
                run_id="r2", items=5,
                accuracy_pass_rate=0.8, relevance_pass_rate=0.8,
                created_at_iso="2026-03-02T10:00:00+00:00",
            ),
            _make_eval_run(
                run_id="r3", items=5,
                accuracy_pass_rate=1.0, relevance_pass_rate=1.0,
                created_at_iso="2026-03-03T10:00:00+00:00",
            ),
        ]
        trend = analyze_trends(runs)
        # Overall pass rate should improve from r1 to r3
        assert trend.overall_delta > 0

    def test_metric_deltas(self):
        runs = self._make_three_runs()
        trend = analyze_trends(runs)
        deltas = trend.metric_deltas
        # accuracy improved from 0.6 to 1.0
        assert deltas[METRIC_ACCURACY] is not None
        assert deltas[METRIC_ACCURACY] > 0
        # relevance stayed at 0.4
        assert deltas[METRIC_RELEVANCE] is not None

    def test_improving_metrics(self):
        runs = self._make_three_runs()
        trend = analyze_trends(runs)
        assert METRIC_ACCURACY in trend.improving_metrics

    def test_stable_metrics(self):
        runs = self._make_three_runs()
        trend = analyze_trends(runs)
        # relevance stayed at 0.4 in all three runs
        assert METRIC_RELEVANCE in trend.stable_metrics

    def test_regressing_metrics(self):
        runs = [
            _make_eval_run(
                run_id="r1", items=5,
                accuracy_pass_rate=1.0, relevance_pass_rate=0.8,
                created_at_iso="2026-03-01T10:00:00+00:00",
            ),
            _make_eval_run(
                run_id="r2", items=5,
                accuracy_pass_rate=0.4, relevance_pass_rate=0.2,
                created_at_iso="2026-03-02T10:00:00+00:00",
            ),
        ]
        trend = analyze_trends(runs)
        assert METRIC_ACCURACY in trend.regressing_metrics

    def test_item_count_trends(self):
        runs = [
            _make_eval_run(run_id="a", items=3, created_at_iso="2026-03-01T10:00:00+00:00"),
            _make_eval_run(run_id="b", items=7, created_at_iso="2026-03-02T10:00:00+00:00"),
        ]
        trend = analyze_trends(runs)
        assert trend.item_count_trends == [3, 7]

    def test_timestamps_populated(self):
        runs = [
            _make_eval_run(run_id="a", created_at_iso="2026-03-01T10:00:00+00:00"),
            _make_eval_run(run_id="b", created_at_iso="2026-03-05T10:00:00+00:00"),
        ]
        trend = analyze_trends(runs)
        assert len(trend.timestamps) == 2
        assert "2026-03-01" in trend.timestamps[0]
        assert "2026-03-05" in trend.timestamps[1]


class TestAnalyzeTrendsSixRuns:
    """Tests for 6+ runs (triggers summary mode in report)."""

    def _make_six_runs(self) -> list[EvalRun]:
        runs = []
        for i in range(6):
            runs.append(
                _make_eval_run(
                    run_id=f"run-{i:02d}",
                    items=5,
                    accuracy_pass_rate=0.4 + 0.1 * i,
                    relevance_pass_rate=0.5,
                    created_at_iso=f"2026-03-{i+1:02d}T10:00:00+00:00",
                )
            )
        return runs

    def test_six_runs_analysis(self):
        runs = self._make_six_runs()
        trend = analyze_trends(runs)
        assert len(trend.runs) == 6
        assert len(trend.overall_trends) == 6

    def test_six_runs_report_uses_summary(self):
        runs = self._make_six_runs()
        trend = analyze_trends(runs)
        report = generate_trend_text_report(trend)
        # Summary mode shows First/Latest columns instead of per-run columns
        assert "First" in report
        assert "Latest" in report


# ---------------------------------------------------------------------------
# TrendAnalysis property edge cases
# ---------------------------------------------------------------------------


class TestTrendAnalysisProperties:
    def test_metric_deltas_single_valid_rate(self):
        t = TrendAnalysis(
            project_name="p", runs=[], run_ids=["r1"],
            metric_trends={"m1": [0.5]},
            overall_trends=[0.5],
            item_count_trends=[5],
            timestamps=["2026-03-01"],
        )
        assert t.metric_deltas["m1"] is None

    def test_metric_deltas_with_nones(self):
        t = TrendAnalysis(
            project_name="p", runs=[], run_ids=["r1", "r2", "r3"],
            metric_trends={"m1": [None, 0.5, 0.8]},
            overall_trends=[0.0, 0.5, 0.8],
            item_count_trends=[1, 1, 1],
            timestamps=["t1", "t2", "t3"],
        )
        # Only two valid rates (0.5 and 0.8), delta = 0.3
        assert t.metric_deltas["m1"] == pytest.approx(0.3)

    def test_overall_delta_single_run(self):
        t = TrendAnalysis(
            project_name="p", runs=[], run_ids=["r1"],
            metric_trends={},
            overall_trends=[0.75],
            item_count_trends=[5],
            timestamps=["t1"],
        )
        assert t.overall_delta == 0.0

    def test_improving_regressing_stable_classification(self):
        t = TrendAnalysis(
            project_name="p", runs=[], run_ids=["r1", "r2"],
            metric_trends={
                "improving": [0.3, 0.8],
                "regressing": [0.9, 0.2],
                "stable": [0.5, 0.5005],  # within 0.001 threshold
            },
            overall_trends=[0.5, 0.5],
            item_count_trends=[5, 5],
            timestamps=["t1", "t2"],
        )
        assert "improving" in t.improving_metrics
        assert "regressing" in t.regressing_metrics
        assert "stable" in t.stable_metrics


# ---------------------------------------------------------------------------
# Trend report output tests
# ---------------------------------------------------------------------------


class TestGenerateTrendTextReport:
    def test_empty_report(self):
        trend = analyze_trends([])
        report = generate_trend_text_report(trend)
        assert "No runs found" in report

    def test_single_run_report(self):
        run = _make_eval_run(run_id="solo", items=4)
        trend = analyze_trends([run])
        report = generate_trend_text_report(trend)
        assert "EVALUATION TRENDS" in report
        assert "RUN OVERVIEW" in report
        assert "METRIC TRENDS" in report
        assert "SUMMARY" in report

    def test_three_run_report_sections(self):
        runs = [
            _make_eval_run(
                run_id=f"r{i}", items=5,
                accuracy_pass_rate=0.5 + 0.1 * i,
                relevance_pass_rate=0.4 + 0.1 * i,
                created_at_iso=f"2026-03-{i+1:02d}T10:00:00+00:00",
            )
            for i in range(3)
        ]
        trend = analyze_trends(runs)
        report = generate_trend_text_report(trend)
        assert "EVALUATION TRENDS" in report
        assert "RUN OVERVIEW" in report
        assert "METRIC TRENDS" in report
        assert "SUMMARY" in report
        assert "test-dataset" in report

    def test_report_contains_time_range(self):
        runs = [
            _make_eval_run(
                run_id="a", created_at_iso="2026-03-01T10:00:00+00:00",
            ),
            _make_eval_run(
                run_id="b", created_at_iso="2026-03-15T10:00:00+00:00",
            ),
        ]
        trend = analyze_trends(runs)
        report = generate_trend_text_report(trend)
        assert "2026-03-01" in report
        assert "2026-03-15" in report

    def test_report_shows_overall_change(self):
        runs = [
            _make_eval_run(
                run_id="a", items=5,
                accuracy_pass_rate=0.4, relevance_pass_rate=0.2,
                created_at_iso="2026-03-01T10:00:00+00:00",
            ),
            _make_eval_run(
                run_id="b", items=5,
                accuracy_pass_rate=1.0, relevance_pass_rate=1.0,
                created_at_iso="2026-03-02T10:00:00+00:00",
            ),
        ]
        trend = analyze_trends(runs)
        report = generate_trend_text_report(trend)
        assert "Overall change" in report
