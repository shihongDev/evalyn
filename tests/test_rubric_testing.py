"""Tests for rubric_testing module."""

from __future__ import annotations

from evalyn_sdk.rubric_testing import (
    RubricTestCase,
    RubricTestResult,
    RubricTestSuite,
    _mean,
    _std_dev,
    compute_consistency,
    compare_rubric_versions,
    detect_edge_cases,
    format_test_report,
    run_rubric_test_suite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _constant_scorer(value: float):
    """Return a score_fn that always returns the same value."""
    def fn(rubric_id: str, input_text: str) -> float:
        return value
    return fn


def _alternating_scorer():
    """Return a score_fn that alternates between 0.0 and 1.0."""
    state = {"call": 0}
    def fn(rubric_id: str, input_text: str) -> float:
        state["call"] += 1
        return 0.0 if state["call"] % 2 == 0 else 1.0
    return fn


def _input_length_scorer(rubric_id: str, input_text: str) -> float:
    """Score based on input length (0-1 clamped)."""
    return min(len(input_text) / 100.0, 1.0)


# ---------------------------------------------------------------------------
# _mean / _std_dev
# ---------------------------------------------------------------------------


class TestMean:
    def test_empty(self):
        assert _mean([]) == 0.0

    def test_single(self):
        assert _mean([5.0]) == 5.0

    def test_multiple(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_negative(self):
        assert _mean([-1.0, 1.0]) == 0.0


class TestStdDev:
    def test_empty(self):
        assert _std_dev([]) == 0.0

    def test_single(self):
        assert _std_dev([42.0]) == 0.0

    def test_identical(self):
        assert _std_dev([3.0, 3.0, 3.0]) == 0.0

    def test_known_values(self):
        # population std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert abs(_std_dev(vals) - 2.0) < 1e-9

    def test_two_values(self):
        # std of [0, 10] = 5.0
        assert abs(_std_dev([0.0, 10.0]) - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# RubricTestCase
# ---------------------------------------------------------------------------


class TestRubricTestCase:
    def test_create(self):
        tc = RubricTestCase(
            item_id="a", input_text="hello", expected_score=0.8,
            expected_pass=True, tags=["t1"],
        )
        assert tc.item_id == "a"
        assert tc.tags == ["t1"]

    def test_as_dict(self):
        tc = RubricTestCase(
            item_id="a", input_text="hello", expected_score=0.8,
            expected_pass=True,
        )
        d = tc.as_dict()
        assert d["item_id"] == "a"
        assert d["tags"] == []

    def test_from_dict(self):
        d = {
            "item_id": "b", "input_text": "world",
            "expected_score": 0.5, "expected_pass": False,
            "tags": ["edge"],
        }
        tc = RubricTestCase.from_dict(d)
        assert tc.item_id == "b"
        assert tc.expected_pass is False
        assert tc.tags == ["edge"]

    def test_roundtrip(self):
        tc = RubricTestCase(
            item_id="rt", input_text="x", expected_score=0.9,
            expected_pass=True, tags=["a", "b"],
        )
        assert RubricTestCase.from_dict(tc.as_dict()).as_dict() == tc.as_dict()

    def test_default_tags(self):
        tc = RubricTestCase(
            item_id="a", input_text="x", expected_score=0.0, expected_pass=False,
        )
        assert tc.tags == []


# ---------------------------------------------------------------------------
# RubricTestResult
# ---------------------------------------------------------------------------


class TestRubricTestResult:
    def test_as_dict(self):
        r = RubricTestResult(
            item_id="r1", scores=[0.8, 0.9], mean_score=0.85,
            std_dev=0.05, passed_count=2, total_runs=2,
            consistency=1.0, is_edge_case=False,
        )
        d = r.as_dict()
        assert d["item_id"] == "r1"
        assert d["scores"] == [0.8, 0.9]

    def test_from_dict(self):
        d = {
            "item_id": "r2", "scores": [0.5], "mean_score": 0.5,
            "std_dev": 0.0, "passed_count": 1, "total_runs": 1,
            "consistency": 1.0, "is_edge_case": False,
        }
        r = RubricTestResult.from_dict(d)
        assert r.item_id == "r2"

    def test_roundtrip(self):
        r = RubricTestResult(
            item_id="x", scores=[0.1, 0.2, 0.3], mean_score=0.2,
            std_dev=0.08, passed_count=0, total_runs=3,
            consistency=1.0, is_edge_case=False,
        )
        assert RubricTestResult.from_dict(r.as_dict()).as_dict() == r.as_dict()


# ---------------------------------------------------------------------------
# RubricTestSuite
# ---------------------------------------------------------------------------


class TestRubricTestSuite:
    def test_as_dict(self):
        suite = RubricTestSuite(
            rubric_id="rb1", results=[], overall_consistency=1.0,
            edge_case_count=0, total_items=0,
        )
        d = suite.as_dict()
        assert d["rubric_id"] == "rb1"
        assert d["results"] == []

    def test_from_dict(self):
        d = {
            "rubric_id": "rb2", "results": [],
            "overall_consistency": 0.9, "edge_case_count": 1,
            "total_items": 5,
        }
        suite = RubricTestSuite.from_dict(d)
        assert suite.rubric_id == "rb2"
        assert suite.edge_case_count == 1

    def test_roundtrip(self):
        r = RubricTestResult(
            item_id="i", scores=[1.0], mean_score=1.0,
            std_dev=0.0, passed_count=1, total_runs=1,
            consistency=1.0, is_edge_case=False,
        )
        suite = RubricTestSuite(
            rubric_id="rb", results=[r], overall_consistency=1.0,
            edge_case_count=0, total_items=1,
        )
        assert RubricTestSuite.from_dict(suite.as_dict()).as_dict() == suite.as_dict()


# ---------------------------------------------------------------------------
# compute_consistency
# ---------------------------------------------------------------------------


class TestComputeConsistency:
    def test_empty(self):
        assert compute_consistency([]) == 1.0

    def test_single(self):
        assert compute_consistency([0.5]) == 1.0

    def test_identical_scores(self):
        assert compute_consistency([0.7, 0.7, 0.7]) == 1.0

    def test_all_within_threshold(self):
        assert compute_consistency([0.5, 0.8, 0.6], threshold=0.5) == 1.0

    def test_none_within_threshold(self):
        assert compute_consistency([0.0, 1.0], threshold=0.1) == 0.0

    def test_partial_consistency(self):
        # Pairs: (0.0,0.5) diff=0.5 within, (0.0,1.0) diff=1.0 not, (0.5,1.0) diff=0.5 within
        result = compute_consistency([0.0, 0.5, 1.0], threshold=0.5)
        assert abs(result - 2.0 / 3.0) < 1e-9

    def test_custom_threshold(self):
        assert compute_consistency([0.0, 0.2], threshold=0.3) == 1.0
        assert compute_consistency([0.0, 0.5], threshold=0.3) == 0.0


# ---------------------------------------------------------------------------
# detect_edge_cases
# ---------------------------------------------------------------------------


class TestDetectEdgeCases:
    def test_no_edge_cases(self):
        results = [
            RubricTestResult(
                item_id="a", scores=[0.5], mean_score=0.5,
                std_dev=0.1, passed_count=1, total_runs=1,
                consistency=1.0, is_edge_case=False,
            )
        ]
        assert detect_edge_cases(results) == []

    def test_detects_high_variance(self):
        r = RubricTestResult(
            item_id="b", scores=[0.0, 1.0], mean_score=0.5,
            std_dev=0.5, passed_count=1, total_runs=2,
            consistency=0.0, is_edge_case=True,
        )
        assert detect_edge_cases([r]) == [r]

    def test_custom_threshold(self):
        r = RubricTestResult(
            item_id="c", scores=[0.4, 0.6], mean_score=0.5,
            std_dev=0.1, passed_count=1, total_runs=2,
            consistency=1.0, is_edge_case=False,
        )
        assert detect_edge_cases([r], std_threshold=0.05) == [r]
        assert detect_edge_cases([r], std_threshold=0.2) == []

    def test_empty_list(self):
        assert detect_edge_cases([]) == []


# ---------------------------------------------------------------------------
# run_rubric_test_suite
# ---------------------------------------------------------------------------


class TestRunRubricTestSuite:
    def test_constant_scorer_full_consistency(self):
        cases = [
            RubricTestCase("a", "hello", 0.8, True),
            RubricTestCase("b", "world", 0.8, True),
        ]
        suite = run_rubric_test_suite("rb", cases, _constant_scorer(0.8), n_runs=3)
        assert suite.rubric_id == "rb"
        assert suite.total_items == 2
        assert suite.overall_consistency == 1.0
        assert suite.edge_case_count == 0
        for r in suite.results:
            assert abs(r.mean_score - 0.8) < 1e-9
            assert r.std_dev < 1e-9
            assert r.passed_count == 3
            assert r.total_runs == 3

    def test_alternating_scorer_low_consistency(self):
        cases = [RubricTestCase("a", "text", 0.5, True)]
        scorer = _alternating_scorer()
        suite = run_rubric_test_suite(
            "rb", cases, scorer, n_runs=4,
            consistency_threshold=0.3, edge_std_threshold=0.1,
        )
        r = suite.results[0]
        assert r.total_runs == 4
        # Alternating 1,0,1,0 - high std dev
        assert r.std_dev > 0.1
        assert r.is_edge_case is True
        assert suite.edge_case_count == 1

    def test_empty_cases(self):
        suite = run_rubric_test_suite("rb", [], _constant_scorer(0.5))
        assert suite.total_items == 0
        assert suite.results == []
        assert suite.overall_consistency == 0.0
        assert suite.edge_case_count == 0

    def test_single_run(self):
        cases = [RubricTestCase("a", "x", 0.5, True)]
        suite = run_rubric_test_suite("rb", cases, _constant_scorer(0.7), n_runs=1)
        r = suite.results[0]
        assert r.total_runs == 1
        assert r.scores == [0.7]
        assert r.std_dev == 0.0

    def test_pass_threshold(self):
        cases = [RubricTestCase("a", "x", 0.5, True)]
        suite = run_rubric_test_suite(
            "rb", cases, _constant_scorer(0.3), n_runs=3, pass_threshold=0.5,
        )
        assert suite.results[0].passed_count == 0

    def test_score_fn_receives_args(self):
        received = []
        def capturing_fn(rubric_id, input_text):
            received.append((rubric_id, input_text))
            return 0.5
        cases = [RubricTestCase("a", "my_input", 0.5, True)]
        run_rubric_test_suite("my_rubric", cases, capturing_fn, n_runs=2)
        assert len(received) == 2
        assert all(r == ("my_rubric", "my_input") for r in received)

    def test_input_length_scorer(self):
        cases = [
            RubricTestCase("short", "hi", 0.02, True),
            RubricTestCase("long", "x" * 200, 1.0, True),
        ]
        suite = run_rubric_test_suite("rb", cases, _input_length_scorer, n_runs=3)
        short_r = suite.results[0]
        long_r = suite.results[1]
        assert short_r.mean_score < 0.1
        assert long_r.mean_score == 1.0


# ---------------------------------------------------------------------------
# compare_rubric_versions
# ---------------------------------------------------------------------------


class TestCompareRubricVersions:
    def _make_suite(self, rubric_id, items):
        """Helper: items is list of (item_id, mean_score, consistency)."""
        results = []
        for item_id, mean, cons in items:
            results.append(RubricTestResult(
                item_id=item_id, scores=[mean], mean_score=mean,
                std_dev=0.0, passed_count=1, total_runs=1,
                consistency=cons, is_edge_case=False,
            ))
        overall = _mean([r.consistency for r in results]) if results else 0.0
        return RubricTestSuite(
            rubric_id=rubric_id, results=results,
            overall_consistency=overall, edge_case_count=0,
            total_items=len(results),
        )

    def test_identical_suites(self):
        suite = self._make_suite("v1", [("a", 0.8, 1.0)])
        result = compare_rubric_versions(suite, suite)
        assert result["consistency_change"] == 0.0
        assert "unchanged" in result["summary"]

    def test_improvement(self):
        a = self._make_suite("v1", [("a", 0.5, 0.6)])
        b = self._make_suite("v2", [("a", 0.8, 0.9)])
        result = compare_rubric_versions(a, b)
        assert result["consistency_change"] > 0
        assert "improved" in result["summary"]
        assert abs(result["items"][0]["score_drift"] - 0.3) < 1e-9

    def test_degradation(self):
        a = self._make_suite("v1", [("a", 0.8, 0.9)])
        b = self._make_suite("v2", [("a", 0.5, 0.6)])
        result = compare_rubric_versions(a, b)
        assert result["consistency_change"] < 0
        assert "degraded" in result["summary"]

    def test_new_and_removed_items(self):
        a = self._make_suite("v1", [("a", 0.8, 1.0)])
        b = self._make_suite("v2", [("b", 0.7, 0.9)])
        result = compare_rubric_versions(a, b)
        items_by_id = {it["item_id"]: it for it in result["items"]}
        assert items_by_id["a"]["status"] == "removed_in_b"
        assert items_by_id["b"]["status"] == "new_in_b"

    def test_rubric_ids_in_result(self):
        a = self._make_suite("old", [])
        b = self._make_suite("new", [])
        result = compare_rubric_versions(a, b)
        assert result["rubric_a"] == "old"
        assert result["rubric_b"] == "new"


# ---------------------------------------------------------------------------
# format_test_report
# ---------------------------------------------------------------------------


class TestFormatTestReport:
    def test_contains_rubric_id(self):
        suite = RubricTestSuite(
            rubric_id="my_rubric", results=[], overall_consistency=1.0,
            edge_case_count=0, total_items=0,
        )
        report = format_test_report(suite)
        assert "my_rubric" in report

    def test_contains_consistency(self):
        suite = RubricTestSuite(
            rubric_id="rb", results=[], overall_consistency=0.875,
            edge_case_count=0, total_items=0,
        )
        report = format_test_report(suite)
        assert "0.875" in report

    def test_edge_case_highlighted(self):
        r = RubricTestResult(
            item_id="edge_item", scores=[0.0, 1.0], mean_score=0.5,
            std_dev=0.5, passed_count=1, total_runs=2,
            consistency=0.0, is_edge_case=True,
        )
        suite = RubricTestSuite(
            rubric_id="rb", results=[r], overall_consistency=0.0,
            edge_case_count=1, total_items=1,
        )
        report = format_test_report(suite)
        assert "[EDGE CASE]" in report
        assert "edge_item" in report

    def test_non_edge_case_no_marker(self):
        r = RubricTestResult(
            item_id="ok_item", scores=[0.8], mean_score=0.8,
            std_dev=0.0, passed_count=1, total_runs=1,
            consistency=1.0, is_edge_case=False,
        )
        suite = RubricTestSuite(
            rubric_id="rb", results=[r], overall_consistency=1.0,
            edge_case_count=0, total_items=1,
        )
        report = format_test_report(suite)
        assert "[EDGE CASE]" not in report
        assert "ok_item" in report

    def test_report_has_separator_lines(self):
        suite = RubricTestSuite(
            rubric_id="rb", results=[], overall_consistency=1.0,
            edge_case_count=0, total_items=0,
        )
        report = format_test_report(suite)
        assert "=" * 50 in report
        assert "-" * 50 in report

    def test_report_shows_pass_ratio(self):
        r = RubricTestResult(
            item_id="x", scores=[0.8, 0.9], mean_score=0.85,
            std_dev=0.05, passed_count=2, total_runs=5,
            consistency=1.0, is_edge_case=False,
        )
        suite = RubricTestSuite(
            rubric_id="rb", results=[r], overall_consistency=1.0,
            edge_case_count=0, total_items=1,
        )
        report = format_test_report(suite)
        assert "passed=2/5" in report
