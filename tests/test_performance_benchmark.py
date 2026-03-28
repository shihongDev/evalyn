"""Tests for the performance benchmark module."""

from __future__ import annotations

import sys
import time
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.performance_benchmark import (
    BenchmarkBaseline,
    BenchmarkReport,
    BenchmarkResult,
    build_benchmark_report,
    compare_to_baseline,
    render_benchmark_table,
    run_benchmark,
    save_baselines,
)


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_basic_timing(self):
        result = run_benchmark("noop", lambda: None, iterations=100)
        assert result.name == "noop"
        assert result.duration_ms >= 0
        assert result.items_per_second > 0
        assert result.timestamp != ""

    def test_slow_function(self):
        def slow():
            time.sleep(0.005)

        result = run_benchmark("slow", slow, iterations=3)
        # Each iteration ~5ms, so avg should be >= 4ms
        assert result.duration_ms >= 4.0

    def test_zero_iterations(self):
        result = run_benchmark("zero", lambda: None, iterations=0)
        assert result.duration_ms == 0.0
        assert result.items_per_second == 0.0

    def test_single_iteration(self):
        result = run_benchmark("single", lambda: None, iterations=1)
        assert result.name == "single"
        assert result.duration_ms >= 0

    def test_fast_function(self):
        result = run_benchmark("fast", lambda: 1 + 1, iterations=1000)
        assert result.items_per_second > 0
        # Fast function should complete quickly
        assert result.duration_ms < 100


# ---------------------------------------------------------------------------
# compare_to_baseline
# ---------------------------------------------------------------------------


class TestCompareToBaseline:
    def test_regression(self):
        result = BenchmarkResult(name="a", duration_ms=150.0)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=20.0)
        assert compare_to_baseline(result, baseline) == "regression"

    def test_improvement(self):
        result = BenchmarkResult(name="a", duration_ms=50.0)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=20.0)
        assert compare_to_baseline(result, baseline) == "improvement"

    def test_stable(self):
        result = BenchmarkResult(name="a", duration_ms=105.0)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=20.0)
        assert compare_to_baseline(result, baseline) == "stable"

    def test_at_upper_boundary(self):
        # Exactly at upper boundary (120) - should be stable
        result = BenchmarkResult(name="a", duration_ms=120.0)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=20.0)
        assert compare_to_baseline(result, baseline) == "stable"

    def test_at_lower_boundary(self):
        # Exactly at lower boundary (80) - should be stable
        result = BenchmarkResult(name="a", duration_ms=80.0)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=20.0)
        assert compare_to_baseline(result, baseline) == "stable"

    def test_just_above_upper(self):
        result = BenchmarkResult(name="a", duration_ms=120.01)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=20.0)
        assert compare_to_baseline(result, baseline) == "regression"

    def test_just_below_lower(self):
        result = BenchmarkResult(name="a", duration_ms=79.99)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=20.0)
        assert compare_to_baseline(result, baseline) == "improvement"

    def test_zero_tolerance(self):
        result = BenchmarkResult(name="a", duration_ms=100.1)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=0.0)
        assert compare_to_baseline(result, baseline) == "regression"

    def test_large_tolerance(self):
        result = BenchmarkResult(name="a", duration_ms=190.0)
        baseline = BenchmarkBaseline(name="a", expected_duration_ms=100.0, tolerance_pct=100.0)
        assert compare_to_baseline(result, baseline) == "stable"


# ---------------------------------------------------------------------------
# build_benchmark_report
# ---------------------------------------------------------------------------


class TestBuildBenchmarkReport:
    def test_basic(self):
        results = [BenchmarkResult(name="a", duration_ms=100.0)]
        report = build_benchmark_report(results)
        assert report.total_benchmarks == 1
        assert report.regressions == []
        assert report.improvements == []

    def test_with_baselines(self):
        results = [
            BenchmarkResult(name="fast", duration_ms=50.0),
            BenchmarkResult(name="slow", duration_ms=200.0),
            BenchmarkResult(name="same", duration_ms=100.0),
        ]
        baselines = [
            BenchmarkBaseline(name="fast", expected_duration_ms=100.0, tolerance_pct=20.0),
            BenchmarkBaseline(name="slow", expected_duration_ms=100.0, tolerance_pct=20.0),
            BenchmarkBaseline(name="same", expected_duration_ms=100.0, tolerance_pct=20.0),
        ]
        report = build_benchmark_report(results, baselines)
        assert "slow" in report.regressions
        assert "fast" in report.improvements
        assert "same" not in report.regressions
        assert "same" not in report.improvements
        assert report.total_benchmarks == 3

    def test_no_baselines(self):
        results = [BenchmarkResult(name="a", duration_ms=100.0)]
        report = build_benchmark_report(results)
        assert report.regressions == []
        assert report.improvements == []

    def test_empty_results(self):
        report = build_benchmark_report([])
        assert report.total_benchmarks == 0

    def test_unmatched_baseline(self):
        results = [BenchmarkResult(name="a", duration_ms=100.0)]
        baselines = [BenchmarkBaseline(name="b", expected_duration_ms=50.0)]
        report = build_benchmark_report(results, baselines)
        assert report.regressions == []
        assert report.improvements == []


# ---------------------------------------------------------------------------
# save_baselines
# ---------------------------------------------------------------------------


class TestSaveBaselines:
    def test_basic(self):
        results = [
            BenchmarkResult(name="a", duration_ms=100.0),
            BenchmarkResult(name="b", duration_ms=200.0),
        ]
        baselines = save_baselines(results)
        assert len(baselines) == 2
        assert baselines[0].name == "a"
        assert baselines[0].expected_duration_ms == 100.0
        assert baselines[0].tolerance_pct == 20.0

    def test_custom_tolerance(self):
        results = [BenchmarkResult(name="a", duration_ms=100.0)]
        baselines = save_baselines(results, tolerance_pct=10.0)
        assert baselines[0].tolerance_pct == 10.0

    def test_empty_results(self):
        baselines = save_baselines([])
        assert baselines == []

    def test_roundtrip_with_compare(self):
        results = [BenchmarkResult(name="a", duration_ms=100.0)]
        baselines = save_baselines(results, tolerance_pct=20.0)
        # Same result should be stable against its own baseline
        assert compare_to_baseline(results[0], baselines[0]) == "stable"


# ---------------------------------------------------------------------------
# render_benchmark_table
# ---------------------------------------------------------------------------


class TestRenderBenchmarkTable:
    def test_basic_output(self):
        report = BenchmarkReport(
            results=[BenchmarkResult(name="test", duration_ms=50.0, items_per_second=20.0)],
            total_benchmarks=1,
        )
        table = render_benchmark_table(report)
        assert "Name" in table
        assert "Duration(ms)" in table
        assert "test" in table
        assert "50.00" in table
        assert "stable" in table

    def test_empty_report(self):
        report = BenchmarkReport()
        table = render_benchmark_table(report)
        assert "No benchmark results" in table

    def test_regression_in_table(self):
        report = BenchmarkReport(
            results=[BenchmarkResult(name="slow", duration_ms=200.0, items_per_second=5.0)],
            regressions=["slow"],
            total_benchmarks=1,
        )
        table = render_benchmark_table(report)
        assert "REGRESSION" in table

    def test_improvement_in_table(self):
        report = BenchmarkReport(
            results=[BenchmarkResult(name="fast", duration_ms=10.0, items_per_second=100.0)],
            improvements=["fast"],
            total_benchmarks=1,
        )
        table = render_benchmark_table(report)
        assert "IMPROVEMENT" in table

    def test_multiple_results(self):
        report = BenchmarkReport(
            results=[
                BenchmarkResult(name="a", duration_ms=10.0, items_per_second=100.0),
                BenchmarkResult(name="b", duration_ms=20.0, items_per_second=50.0),
            ],
            total_benchmarks=2,
        )
        table = render_benchmark_table(report)
        lines = table.strip().split("\n")
        # header + separator + 2 result rows
        assert len(lines) >= 4


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_benchmark_result_roundtrip(self):
        r = BenchmarkResult(
            name="test", duration_ms=100.0, memory_mb=50.0,
            items_per_second=10.0, timestamp="2026-01-01",
        )
        restored = BenchmarkResult.from_dict(r.as_dict())
        assert restored.name == r.name
        assert restored.duration_ms == r.duration_ms
        assert restored.memory_mb == r.memory_mb
        assert restored.items_per_second == r.items_per_second
        assert restored.timestamp == r.timestamp

    def test_benchmark_baseline_roundtrip(self):
        b = BenchmarkBaseline(name="test", expected_duration_ms=100.0, tolerance_pct=15.0)
        restored = BenchmarkBaseline.from_dict(b.as_dict())
        assert restored.name == b.name
        assert restored.expected_duration_ms == b.expected_duration_ms
        assert restored.tolerance_pct == b.tolerance_pct

    def test_benchmark_baseline_from_dict_defaults(self):
        b = BenchmarkBaseline.from_dict({"name": "x"})
        assert b.expected_duration_ms == 0.0
        assert b.tolerance_pct == 20.0

    def test_benchmark_report_roundtrip(self):
        report = BenchmarkReport(
            results=[BenchmarkResult(name="a", duration_ms=100.0)],
            regressions=["slow"],
            improvements=["fast"],
            total_benchmarks=1,
        )
        restored = BenchmarkReport.from_dict(report.as_dict())
        assert restored.total_benchmarks == 1
        assert restored.regressions == ["slow"]
        assert restored.improvements == ["fast"]
        assert len(restored.results) == 1

    def test_benchmark_report_format_text(self):
        report = BenchmarkReport(
            results=[BenchmarkResult(name="a", duration_ms=100.0, items_per_second=10.0)],
            regressions=["a"],
            total_benchmarks=1,
        )
        text = report.format_text()
        assert "Benchmark Report" in text
        assert "total_benchmarks: 1" in text
        assert "REGRESSIONS" in text
        assert "a" in text

    def test_benchmark_report_format_text_empty(self):
        report = BenchmarkReport()
        text = report.format_text()
        assert "total_benchmarks: 0" in text

    def test_benchmark_report_format_text_improvements(self):
        report = BenchmarkReport(
            results=[BenchmarkResult(name="b", duration_ms=10.0, items_per_second=100.0)],
            improvements=["b"],
            total_benchmarks=1,
        )
        text = report.format_text()
        assert "IMPROVEMENTS" in text
