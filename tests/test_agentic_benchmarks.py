"""Tests for agentic benchmark integration."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.evaluation.agentic_benchmarks import (
    BASIC_AGENT_TASKS,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
    compare_benchmark_results,
    create_basic_suite,
    evaluate_task,
    run_benchmark_suite,
)


# ---------------------------------------------------------------------------
# create_basic_suite
# ---------------------------------------------------------------------------

class TestCreateBasicSuite:
    def test_has_tasks(self):
        suite = create_basic_suite()
        assert len(suite.tasks) == len(BASIC_AGENT_TASKS)
        assert suite.name == "basic_agent"

    def test_task_ids_unique(self):
        suite = create_basic_suite()
        ids = [t.task_id for t in suite.tasks]
        assert len(ids) == len(set(ids))

    def test_difficulties_valid(self):
        suite = create_basic_suite()
        for t in suite.tasks:
            assert t.difficulty in ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# evaluate_task
# ---------------------------------------------------------------------------

class TestEvaluateTask:
    def test_tools_correct(self):
        """All expected tools present should set tool_calls_correct=True."""
        task = BenchmarkTask("t1", "T", "", "", ["search"], ["answer"])
        result = evaluate_task(task, ["search", "browse"], "Here is the answer")
        assert result.tool_calls_correct is True

    def test_tools_incorrect(self):
        """Missing expected tool should set tool_calls_correct=False."""
        task = BenchmarkTask("t1", "T", "", "", ["search", "calculate"], ["answer"])
        result = evaluate_task(task, ["search"], "Here is the answer")
        assert result.tool_calls_correct is False

    def test_output_correct(self):
        """Keyword present in output should set output_correct=True."""
        task = BenchmarkTask("t1", "T", "", "", [], ["hello", "world"])
        result = evaluate_task(task, [], "Hello there!")
        assert result.output_correct is True

    def test_output_incorrect(self):
        """No keyword in output should set output_correct=False."""
        task = BenchmarkTask("t1", "T", "", "", [], ["hello"])
        result = evaluate_task(task, [], "goodbye")
        assert result.output_correct is False

    def test_both_correct(self):
        """Both checks passing should give score=1.0 and passed=True."""
        task = BenchmarkTask("t1", "T", "", "", ["search"], ["answer"])
        result = evaluate_task(task, ["search"], "the answer is 42")
        assert result.tool_calls_correct is True
        assert result.output_correct is True
        assert result.score == 1.0
        assert result.passed is True

    def test_both_wrong(self):
        """Both checks failing should give score=0.0 and passed=False."""
        task = BenchmarkTask("t1", "T", "", "", ["search"], ["answer"])
        result = evaluate_task(task, [], "nothing useful")
        assert result.score == 0.0
        assert result.passed is False

    def test_half_score(self):
        """One check passing should give score=0.5 and passed=True."""
        task = BenchmarkTask("t1", "T", "", "", ["search"], ["answer"])
        result = evaluate_task(task, ["search"], "nothing useful")
        assert result.score == 0.5
        assert result.passed is True

    def test_case_insensitive_keywords(self):
        """Keyword matching should be case-insensitive."""
        task = BenchmarkTask("t1", "T", "", "", [], ["ANSWER"])
        result = evaluate_task(task, [], "the answer is here")
        assert result.output_correct is True

    def test_empty_expected_tools(self):
        """No expected tools means tool check always passes."""
        task = BenchmarkTask("t1", "T", "", "", [], ["ok"])
        result = evaluate_task(task, [], "ok")
        assert result.tool_calls_correct is True

    def test_empty_expected_keywords(self):
        """No expected keywords means output check always fails."""
        task = BenchmarkTask("t1", "T", "", "", ["search"], [])
        result = evaluate_task(task, ["search"], "anything")
        assert result.output_correct is False


# ---------------------------------------------------------------------------
# run_benchmark_suite
# ---------------------------------------------------------------------------

class TestRunBenchmarkSuite:
    def test_all_pass(self):
        """All tasks matching should give 100% pass rate."""
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer is here"),
            "multi_step_tool": (["search", "calculate"], "the result is 42"),
            "error_recovery": (["search"], "sorry, I am unable to help"),
        }
        report = run_benchmark_suite(suite, outputs)
        assert report.pass_rate == 1.0
        assert report.avg_score == 1.0
        assert len(report.results) == 3

    def test_mixed_results(self):
        """Some failures should produce mixed pass rate."""
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": ([], "nothing here"),  # both wrong
            "error_recovery": (["search"], "sorry, unable"),
        }
        report = run_benchmark_suite(suite, outputs)
        # search_and_answer: pass, multi_step_tool: fail, error_recovery: pass
        assert 0.0 < report.pass_rate < 1.0

    def test_missing_task_in_outputs(self):
        """Missing task output should score as failure."""
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer"),
            # multi_step_tool missing
            # error_recovery missing
        }
        report = run_benchmark_suite(suite, outputs)
        assert len(report.results) == 3
        missing_results = [r for r in report.results if r.task_id != "search_and_answer"]
        for r in missing_results:
            assert r.passed is False
            assert r.score == 0.0

    def test_by_difficulty_breakdown(self):
        """Report should contain score breakdown by difficulty."""
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": (["search", "calculate"], "the result"),
            "error_recovery": (["search"], "sorry, unable"),
        }
        report = run_benchmark_suite(suite, outputs)
        assert "easy" in report.by_difficulty
        assert "medium" in report.by_difficulty
        assert "hard" in report.by_difficulty

    def test_by_category_breakdown(self):
        """Report should contain score breakdown by category."""
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": (["search", "calculate"], "the result"),
            "error_recovery": (["search"], "sorry, unable"),
        }
        report = run_benchmark_suite(suite, outputs)
        assert "retrieval" in report.by_category
        assert "tool_use" in report.by_category
        assert "robustness" in report.by_category

    def test_empty_suite(self):
        """Empty suite should produce zero-value report."""
        suite = BenchmarkSuite(name="empty")
        report = run_benchmark_suite(suite, {})
        assert report.pass_rate == 0.0
        assert report.avg_score == 0.0
        assert report.results == []


# ---------------------------------------------------------------------------
# compare_benchmark_results
# ---------------------------------------------------------------------------

class TestCompareBenchmarkResults:
    def test_compare_improvement(self):
        """Improvement should show positive deltas."""
        suite = create_basic_suite()
        outputs_a = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": ([], "no result"),
            "error_recovery": ([], "nothing"),
        }
        outputs_b = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": (["search", "calculate"], "the result"),
            "error_recovery": (["search"], "sorry, unable"),
        }
        report_a = run_benchmark_suite(suite, outputs_a)
        report_b = run_benchmark_suite(suite, outputs_b)
        diff = compare_benchmark_results(report_a, report_b)
        assert diff["pass_rate_delta"] > 0
        assert diff["avg_score_delta"] > 0

    def test_compare_same(self):
        """Comparing identical reports should show zero deltas."""
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": (["search", "calculate"], "the result"),
            "error_recovery": (["search"], "sorry, unable"),
        }
        report = run_benchmark_suite(suite, outputs)
        diff = compare_benchmark_results(report, report)
        assert diff["pass_rate_delta"] == 0.0
        assert diff["avg_score_delta"] == 0.0

    def test_per_task_deltas(self):
        """Per-task deltas should be present for each task."""
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": (["search", "calculate"], "the result"),
            "error_recovery": (["search"], "sorry, unable"),
        }
        report = run_benchmark_suite(suite, outputs)
        diff = compare_benchmark_results(report, report)
        assert "per_task" in diff
        assert "search_and_answer" in diff["per_task"]


# ---------------------------------------------------------------------------
# BenchmarkReport.format_text
# ---------------------------------------------------------------------------

class TestFormatText:
    def test_includes_header(self):
        report = BenchmarkReport(suite_name="test_suite", pass_rate=0.8, avg_score=0.75)
        text = report.format_text()
        assert "BENCHMARK REPORT: test_suite" in text
        assert "80.0%" in text

    def test_includes_difficulty(self):
        report = BenchmarkReport(
            suite_name="s", by_difficulty={"easy": 1.0, "hard": 0.5}
        )
        text = report.format_text()
        assert "BY DIFFICULTY" in text
        assert "easy" in text
        assert "hard" in text

    def test_includes_category(self):
        report = BenchmarkReport(
            suite_name="s", by_category={"retrieval": 0.9}
        )
        text = report.format_text()
        assert "BY CATEGORY" in text
        assert "retrieval" in text

    def test_includes_results(self):
        result = BenchmarkResult("t1", True, 1.0, True, True)
        report = BenchmarkReport(suite_name="s", results=[result])
        text = report.format_text()
        assert "RESULTS" in text
        assert "t1" in text
        assert "PASS" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------

class TestRoundtrips:
    def test_benchmark_task_roundtrip(self):
        original = BenchmarkTask("t1", "Task 1", "desc", "cat", ["a"], ["b"], "hard")
        d = original.as_dict()
        restored = BenchmarkTask.from_dict(d)
        assert restored.task_id == original.task_id
        assert restored.name == original.name
        assert restored.difficulty == "hard"
        assert restored.expected_tools == ["a"]
        assert restored.expected_output_keywords == ["b"]

    def test_benchmark_suite_roundtrip(self):
        suite = create_basic_suite()
        d = suite.as_dict()
        restored = BenchmarkSuite.from_dict(d)
        assert restored.name == suite.name
        assert len(restored.tasks) == len(suite.tasks)

    def test_benchmark_result_as_dict(self):
        result = BenchmarkResult("t1", True, 0.75, True, False, 123.4)
        d = result.as_dict()
        assert d["task_id"] == "t1"
        assert d["passed"] is True
        assert d["score"] == 0.75
        assert d["duration_ms"] == 123.4

    def test_benchmark_report_roundtrip(self):
        suite = create_basic_suite()
        outputs = {
            "search_and_answer": (["search"], "the answer"),
            "multi_step_tool": (["search", "calculate"], "the result"),
            "error_recovery": (["search"], "sorry, unable"),
        }
        original = run_benchmark_suite(suite, outputs)
        d = original.as_dict()
        restored = BenchmarkReport.from_dict(d)
        assert restored.suite_name == original.suite_name
        assert restored.pass_rate == original.pass_rate
        assert abs(restored.avg_score - original.avg_score) < 1e-9
        assert len(restored.results) == len(original.results)
        assert restored.by_difficulty == original.by_difficulty
        assert restored.by_category == original.by_category
