"""Tests for pytest plugin and auto-compare."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.pytest_plugin import evalyn_assert, evalyn_check, EvalynAssertionError
from evalyn_sdk.models import EvalRun, MetricResult
from evalyn_sdk.evaluation.auto_compare import auto_compare, AutoCompareResult


# =============================================================================
# Pytest plugin tests
# =============================================================================

class TestEvalynAssert:
    def test_passes_on_nonempty(self):
        results = evalyn_assert(
            input="What is Python?",
            output="Python is a programming language",
            metrics=["output_nonempty"],
        )
        assert len(results) >= 1
        assert all(r.passed for r in results)

    def test_fails_on_empty_output(self):
        with pytest.raises(EvalynAssertionError, match="FAIL"):
            evalyn_assert(
                input="test",
                output="",
                metrics=["output_nonempty"],
            )

    def test_multiple_metrics(self):
        results = evalyn_assert(
            input={"query": "hello"},
            output="world",
            metrics=["output_nonempty"],
        )
        assert len(results) >= 1

    def test_returns_results(self):
        results = evalyn_assert(
            input="test",
            output="some output",
            metrics=["output_nonempty"],
        )
        assert isinstance(results, list)
        assert all(hasattr(r, "metric_id") for r in results)

    def test_error_includes_results(self):
        try:
            evalyn_assert(input="test", output="", metrics=["output_nonempty"])
        except EvalynAssertionError as e:
            assert hasattr(e, "results")
            assert len(e.results) >= 1

    def test_dict_input(self):
        results = evalyn_assert(
            input={"query": "hello", "context": "world"},
            output="response",
            metrics=["output_nonempty"],
        )
        assert len(results) >= 1


class TestEvalynCheck:
    def test_passing_check(self):
        result = evalyn_check(
            input="test",
            output="some output",
            metrics=["output_nonempty"],
        )
        assert result["passed"] is True
        assert len(result["failures"]) == 0

    def test_failing_check(self):
        result = evalyn_check(
            input="test",
            output="",
            metrics=["output_nonempty"],
        )
        assert result["passed"] is False
        assert len(result["failures"]) > 0

    def test_check_returns_results(self):
        result = evalyn_check(input="test", output="ok", metrics=["output_nonempty"])
        assert "results" in result


# =============================================================================
# Auto-compare tests
# =============================================================================

def _run(run_id, pinned=False, results=None):
    now = datetime.now(timezone.utc)
    return EvalRun(
        id=run_id, dataset_name="test", created_at=now,
        metric_results=results or [], pinned=pinned,
    )


def _result(metric_id, passed=True):
    return MetricResult(
        metric_id=metric_id, item_id="i1", call_id="c1",
        score=1.0 if passed else 0.0, passed=passed, details={},
    )


class TestAutoCompare:
    def test_no_baseline(self):
        current = _run("r1")
        result = auto_compare(current, [current])
        assert result.status == "NO_BASELINE"
        assert not result.has_baseline

    def test_regression_detected(self):
        baseline_results = [_result("help", True)] * 10
        current_results = [_result("help", True)] * 3 + [_result("help", False)] * 7
        baseline = _run("base", pinned=True, results=baseline_results)
        current = _run("curr", results=current_results)

        result = auto_compare(current, [baseline, current])
        assert result.has_baseline
        assert result.has_regressions
        assert result.status == "REGRESSION"
        assert len(result.regressions) == 1
        assert result.regressions[0]["metric_id"] == "help"

    def test_improvement_detected(self):
        baseline_results = [_result("help", True)] * 5 + [_result("help", False)] * 5
        current_results = [_result("help", True)] * 9 + [_result("help", False)] * 1
        baseline = _run("base", pinned=True, results=baseline_results)
        current = _run("curr", results=current_results)

        result = auto_compare(current, [baseline, current])
        assert result.status == "IMPROVED"
        assert len(result.improvements) == 1

    def test_stable(self):
        results = [_result("help", True)] * 10
        baseline = _run("base", pinned=True, results=results)
        current = _run("curr", results=results)

        result = auto_compare(current, [baseline, current])
        assert result.status == "STABLE"
        assert result.unchanged == 1

    def test_multiple_metrics(self):
        base_results = [_result("a", True), _result("b", True)]
        curr_results = [_result("a", True), _result("b", False)]
        baseline = _run("base", pinned=True, results=base_results)
        current = _run("curr", results=curr_results)

        result = auto_compare(current, [baseline, current])
        assert result.has_regressions
        assert any(r["metric_id"] == "b" for r in result.regressions)

    def test_custom_threshold(self):
        base_results = [_result("help", True)] * 10
        curr_results = [_result("help", True)] * 9 + [_result("help", False)]
        baseline = _run("base", pinned=True, results=base_results)
        current = _run("curr", results=curr_results)

        # 10% regression threshold - 10% drop should trigger
        result = auto_compare(current, [baseline, current], regression_threshold=0.05)
        assert result.has_regressions

        # 15% threshold - 10% drop should NOT trigger
        result = auto_compare(current, [baseline, current], regression_threshold=0.15)
        assert not result.has_regressions

    def test_as_dict(self):
        result = auto_compare(_run("r1"), [_run("r1")])
        d = result.as_dict()
        assert "status" in d
        assert "has_baseline" in d

    def test_format_text_no_baseline(self):
        result = auto_compare(_run("r1"), [_run("r1")])
        assert "no pinned baseline" in result.format_text()

    def test_format_text_with_regression(self):
        base_results = [_result("help", True)] * 10
        curr_results = [_result("help", False)] * 10
        baseline = _run("base", pinned=True, results=base_results)
        current = _run("curr", results=curr_results)

        result = auto_compare(current, [baseline, current])
        text = result.format_text()
        assert "REGRESSION" in text
        assert "help" in text
