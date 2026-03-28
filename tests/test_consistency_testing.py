"""Tests for agent consistency testing.

All tests use synthetic data - no storage or ML deps required.
"""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.consistency_testing import (
    ConsistencyReport,
    ConsistencyResult,
    RunOutput,
    analyze_consistency,
    build_consistency_report,
    measure_output_consistency,
    measure_tool_call_consistency,
)


# ---------------------------------------------------------------------------
# measure_output_consistency
# ---------------------------------------------------------------------------


class TestMeasureOutputConsistency:
    def test_all_same(self):
        assert measure_output_consistency(["a", "a", "a"]) == 1.0

    def test_all_different(self):
        result = measure_output_consistency(["a", "b", "c"])
        assert abs(result - 1.0 / 3.0) < 1e-9

    def test_majority(self):
        result = measure_output_consistency(["a", "a", "b"])
        assert abs(result - 2.0 / 3.0) < 1e-9

    def test_two_outputs(self):
        result = measure_output_consistency(["x", "y"])
        assert result == 0.5

    def test_single_output(self):
        assert measure_output_consistency(["a"]) == 1.0

    def test_empty(self):
        assert measure_output_consistency([]) == 0.0

    def test_four_of_five(self):
        result = measure_output_consistency(["a", "a", "a", "a", "b"])
        assert abs(result - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# measure_tool_call_consistency
# ---------------------------------------------------------------------------


class TestMeasureToolCallConsistency:
    def test_perfect(self):
        sets = [["search", "read"], ["search", "read"], ["search", "read"]]
        assert measure_tool_call_consistency(sets) == 1.0

    def test_varied(self):
        sets = [["search"], ["search"], ["read"]]
        result = measure_tool_call_consistency(sets)
        assert abs(result - 2.0 / 3.0) < 1e-9

    def test_all_different(self):
        sets = [["a"], ["b"], ["c"]]
        result = measure_tool_call_consistency(sets)
        assert abs(result - 1.0 / 3.0) < 1e-9

    def test_empty_input(self):
        assert measure_tool_call_consistency([]) == 0.0

    def test_single_set(self):
        assert measure_tool_call_consistency([["search"]]) == 1.0

    def test_empty_tool_calls(self):
        sets = [[], [], []]
        assert measure_tool_call_consistency(sets) == 1.0

    def test_mixed_empty_and_nonempty(self):
        sets = [[], ["search"], []]
        result = measure_tool_call_consistency(sets)
        assert abs(result - 2.0 / 3.0) < 1e-9


# ---------------------------------------------------------------------------
# analyze_consistency
# ---------------------------------------------------------------------------


class TestAnalyzeConsistency:
    def test_full(self):
        runs = [
            RunOutput(run_index=0, output="hello", tool_calls=["search"], passed=True),
            RunOutput(run_index=1, output="hello", tool_calls=["search"], passed=True),
            RunOutput(run_index=2, output="world", tool_calls=["read"], passed=False),
        ]
        result = analyze_consistency("item1", runs)
        assert result.item_id == "item1"
        assert result.num_runs == 3
        assert abs(result.output_consistency - 2.0 / 3.0) < 1e-9
        assert abs(result.tool_call_consistency - 2.0 / 3.0) < 1e-9
        assert abs(result.pass_consistency - 2.0 / 3.0) < 1e-9
        assert len(result.outputs) == 3

    def test_single_run(self):
        runs = [RunOutput(run_index=0, output="only", passed=True)]
        result = analyze_consistency("single", runs)
        assert result.num_runs == 1
        assert result.output_consistency == 1.0
        assert result.pass_consistency == 1.0

    def test_empty_runs(self):
        result = analyze_consistency("empty", [])
        assert result.num_runs == 0
        assert result.output_consistency == 0.0
        assert result.tool_call_consistency == 0.0
        assert result.pass_consistency == 0.0

    def test_all_failed(self):
        runs = [
            RunOutput(run_index=0, output="x", passed=False),
            RunOutput(run_index=1, output="x", passed=False),
        ]
        result = analyze_consistency("fail", runs)
        assert result.pass_consistency == 0.0
        assert result.output_consistency == 1.0


# ---------------------------------------------------------------------------
# build_consistency_report
# ---------------------------------------------------------------------------


class TestBuildConsistencyReport:
    def test_averages(self):
        r1 = ConsistencyResult(
            item_id="a", num_runs=3,
            output_consistency=1.0, tool_call_consistency=1.0, pass_consistency=1.0,
        )
        r2 = ConsistencyResult(
            item_id="b", num_runs=3,
            output_consistency=0.5, tool_call_consistency=0.5, pass_consistency=0.5,
        )
        report = build_consistency_report([r1, r2])
        assert report.total_items == 2
        assert abs(report.avg_output_consistency - 0.75) < 1e-9
        assert abs(report.avg_pass_consistency - 0.75) < 1e-9

    def test_empty(self):
        report = build_consistency_report([])
        assert report.total_items == 0
        assert report.avg_output_consistency == 0.0
        assert report.avg_pass_consistency == 0.0

    def test_single_result(self):
        r = ConsistencyResult(
            item_id="x", num_runs=5,
            output_consistency=0.8, tool_call_consistency=0.6, pass_consistency=0.9,
        )
        report = build_consistency_report([r])
        assert report.total_items == 1
        assert abs(report.avg_output_consistency - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestConsistencyReportFormatText:
    def test_contains_header(self):
        report = ConsistencyReport(
            results=[], avg_output_consistency=0.5,
            avg_pass_consistency=0.8, total_items=0,
        )
        text = report.format_text()
        assert "Consistency Report" in text
        assert "50.0%" in text
        assert "80.0%" in text

    def test_includes_items(self):
        r = ConsistencyResult(
            item_id="item1", num_runs=3,
            output_consistency=1.0, tool_call_consistency=0.5, pass_consistency=0.75,
        )
        report = build_consistency_report([r])
        text = report.format_text()
        assert "item1" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_run_output_as_dict(self):
        ro = RunOutput(run_index=1, output="hi", tool_calls=["a"], passed=False, duration_ms=42.0)
        d = ro.as_dict()
        assert d["run_index"] == 1
        assert d["output"] == "hi"
        assert d["tool_calls"] == ["a"]
        assert d["passed"] is False
        assert d["duration_ms"] == 42.0

    def test_consistency_result_roundtrip(self):
        runs = [RunOutput(run_index=0, output="x", tool_calls=["s"])]
        cr = ConsistencyResult(
            item_id="id1", num_runs=1,
            output_consistency=1.0, tool_call_consistency=1.0, pass_consistency=1.0,
            outputs=runs,
        )
        d = cr.as_dict()
        restored = ConsistencyResult.from_dict(d)
        assert restored.item_id == cr.item_id
        assert restored.num_runs == cr.num_runs
        assert restored.output_consistency == cr.output_consistency
        assert len(restored.outputs) == 1
        assert restored.outputs[0].output == "x"

    def test_consistency_report_roundtrip(self):
        r = ConsistencyResult(
            item_id="a", num_runs=2,
            output_consistency=0.5, tool_call_consistency=0.5, pass_consistency=1.0,
        )
        report = build_consistency_report([r])
        d = report.as_dict()
        restored = ConsistencyReport.from_dict(d)
        assert restored.total_items == 1
        assert abs(restored.avg_output_consistency - 0.5) < 1e-9
        assert len(restored.results) == 1
