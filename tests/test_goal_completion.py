"""Tests for agent goal completion metrics."""

import pytest
from evalyn_sdk.metrics.goal_completion import (
    GoalResult,
    GoalReport,
    check_tool_call_accuracy,
    check_tool_call_f1,
    check_topic_adherence,
    check_goal_completion,
    evaluate_goals,
)


# ---------------------------------------------------------------------------
# GoalResult
# ---------------------------------------------------------------------------

class TestGoalResult:
    def test_as_dict(self):
        r = GoalResult(goal_id="g1", achieved=True, score=0.9, method="lcs",
                       details={"k": "v"})
        d = r.as_dict()
        assert d["goal_id"] == "g1"
        assert d["achieved"] is True
        assert d["score"] == 0.9
        assert d["method"] == "lcs"
        assert d["details"] == {"k": "v"}

    def test_default_details(self):
        r = GoalResult(goal_id="g1", achieved=False, score=0.0, method="test")
        assert r.details == {}


# ---------------------------------------------------------------------------
# check_tool_call_accuracy (LCS-based)
# ---------------------------------------------------------------------------

class TestToolCallAccuracy:
    def test_perfect_match(self):
        r = check_tool_call_accuracy(["a", "b", "c"], ["a", "b", "c"])
        assert r.score == 1.0
        assert r.achieved is True
        assert r.method == "lcs"

    def test_partial_match(self):
        r = check_tool_call_accuracy(["a", "b", "c"], ["a", "c"])
        # LCS = [a, c] = 2, max(3, 2) = 3, score = 2/3
        assert abs(r.score - 2 / 3) < 1e-9
        assert r.achieved is False

    def test_no_overlap(self):
        r = check_tool_call_accuracy(["a", "b"], ["x", "y"])
        assert r.score == 0.0
        assert r.achieved is False

    def test_empty_both(self):
        r = check_tool_call_accuracy([], [])
        assert r.score == 1.0
        assert r.achieved is True

    def test_empty_expected(self):
        r = check_tool_call_accuracy([], ["a", "b"])
        # LCS = 0, max(0, 2) = 2, score = 0
        assert r.score == 0.0

    def test_empty_actual(self):
        r = check_tool_call_accuracy(["a", "b"], [])
        assert r.score == 0.0

    def test_reversed_order(self):
        r = check_tool_call_accuracy(["a", "b", "c"], ["c", "b", "a"])
        # LCS of [a,b,c] and [c,b,a] = 1 (any single element), score = 1/3
        assert abs(r.score - 1 / 3) < 1e-9

    def test_repeated_elements(self):
        r = check_tool_call_accuracy(["a", "a", "b"], ["a", "b"])
        # LCS = [a, b] = 2, max(3, 2) = 3, score = 2/3
        assert abs(r.score - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# check_tool_call_f1 (set-based)
# ---------------------------------------------------------------------------

class TestToolCallF1:
    def test_perfect_match(self):
        r = check_tool_call_f1(["a", "b", "c"], ["a", "b", "c"])
        assert r.score == 1.0
        assert r.achieved is True

    def test_partial_overlap(self):
        r = check_tool_call_f1(["a", "b", "c"], ["b", "c", "d"])
        # intersection = {b, c}, precision = 2/3, recall = 2/3, f1 = 2/3
        assert abs(r.score - 2 / 3) < 1e-9

    def test_no_overlap(self):
        r = check_tool_call_f1(["a", "b"], ["x", "y"])
        assert r.score == 0.0
        assert r.achieved is False

    def test_empty_both(self):
        r = check_tool_call_f1([], [])
        assert r.score == 1.0
        assert r.achieved is True

    def test_empty_expected(self):
        r = check_tool_call_f1([], ["a"])
        # recall = 0/0 -> 0.0 (no expected), precision = 0/1 = 0
        # Actually: intersection = empty, precision = 0/1 = 0, recall = 0 (empty expected)
        assert r.score == 0.0

    def test_empty_actual(self):
        r = check_tool_call_f1(["a", "b"], [])
        assert r.score == 0.0

    def test_superset_actual(self):
        r = check_tool_call_f1(["a"], ["a", "b", "c"])
        # intersection = {a}, precision = 1/3, recall = 1/1 = 1.0
        # f1 = 2 * (1/3 * 1) / (1/3 + 1) = 2/3 / 4/3 = 0.5
        assert abs(r.score - 0.5) < 1e-9

    def test_details_contain_precision_recall(self):
        r = check_tool_call_f1(["a", "b"], ["a", "b"])
        assert r.details["precision"] == 1.0
        assert r.details["recall"] == 1.0
        assert r.details["f1"] == 1.0


# ---------------------------------------------------------------------------
# check_topic_adherence
# ---------------------------------------------------------------------------

class TestTopicAdherence:
    def test_all_allowed_present(self):
        r = check_topic_adherence("Python is great for data science",
                                  ["python", "data science"])
        assert r.achieved is True
        assert r.score == 1.0

    def test_some_allowed_missing(self):
        r = check_topic_adherence("Python is great",
                                  ["python", "data science", "machine learning"])
        assert r.achieved is False
        # 1 of 3 found
        assert abs(r.score - 1 / 3) < 1e-9

    def test_forbidden_topic_found(self):
        r = check_topic_adherence("Python is easy, Java is harder",
                                  ["python"], ["java"])
        assert r.achieved is False
        # allowed_score = 1.0, forbidden_penalty = 1.0, score = 0.0
        assert r.score == 0.0

    def test_forbidden_not_found(self):
        r = check_topic_adherence("Python is great",
                                  ["python"], ["java", "rust"])
        assert r.achieved is True
        assert r.score == 1.0

    def test_empty_topics(self):
        r = check_topic_adherence("any text", [], [])
        assert r.achieved is True
        assert r.score == 1.0

    def test_case_insensitive(self):
        r = check_topic_adherence("PYTHON and Data Science",
                                  ["python", "data science"])
        assert r.achieved is True


# ---------------------------------------------------------------------------
# check_goal_completion
# ---------------------------------------------------------------------------

class TestGoalCompletion:
    def test_all_keywords_found(self):
        r = check_goal_completion("The answer is 42 and it is correct",
                                  ["answer", "42", "correct"])
        assert r.achieved is True
        assert r.score == 1.0

    def test_partial_keywords(self):
        r = check_goal_completion("The answer is unknown",
                                  ["answer", "42", "correct"])
        assert r.achieved is False
        assert abs(r.score - 1 / 3) < 1e-9

    def test_no_keywords_found(self):
        r = check_goal_completion("nothing relevant here", ["alpha", "beta"])
        assert r.score == 0.0
        assert r.achieved is False

    def test_empty_keywords(self):
        r = check_goal_completion("any text", [])
        assert r.score == 1.0
        assert r.achieved is True

    def test_case_insensitive(self):
        r = check_goal_completion("The ANSWER is here", ["answer"])
        assert r.achieved is True

    def test_details_found_and_missing(self):
        r = check_goal_completion("hello world", ["hello", "goodbye"])
        assert "hello" in r.details["found"]
        assert "goodbye" in r.details["missing"]


# ---------------------------------------------------------------------------
# GoalReport
# ---------------------------------------------------------------------------

class TestGoalReport:
    def _make_report(self):
        results = [
            GoalResult(goal_id="g1", achieved=True, score=1.0, method="lcs"),
            GoalResult(goal_id="g2", achieved=False, score=0.5, method="f1"),
            GoalResult(goal_id="g3", achieved=True, score=0.8, method="keyword"),
        ]
        return GoalReport(
            results=results, total_goals=3,
            achieved_count=2, achievement_rate=2 / 3,
        )

    def test_achievement_rate(self):
        report = self._make_report()
        assert abs(report.achievement_rate - 2 / 3) < 1e-9

    def test_as_dict(self):
        report = self._make_report()
        d = report.as_dict()
        assert d["total_goals"] == 3
        assert d["achieved_count"] == 2
        assert len(d["results"]) == 3

    def test_from_dict_roundtrip(self):
        report = self._make_report()
        d = report.as_dict()
        rebuilt = GoalReport.from_dict(d)
        assert rebuilt.total_goals == report.total_goals
        assert rebuilt.achieved_count == report.achieved_count
        assert abs(rebuilt.achievement_rate - report.achievement_rate) < 1e-9
        assert len(rebuilt.results) == len(report.results)
        assert rebuilt.results[0].goal_id == "g1"
        assert rebuilt.results[1].achieved is False

    def test_format_text(self):
        report = self._make_report()
        text = report.format_text()
        assert "2/3" in text
        assert "PASS" in text
        assert "FAIL" in text
        assert "g1" in text
        assert "g2" in text

    def test_format_text_empty(self):
        report = GoalReport(results=[], total_goals=0, achieved_count=0,
                            achievement_rate=0.0)
        text = report.format_text()
        assert "0/0" in text


# ---------------------------------------------------------------------------
# evaluate_goals
# ---------------------------------------------------------------------------

class TestEvaluateGoals:
    def test_multiple_goals(self):
        goals = [
            ("accuracy", lambda: check_tool_call_accuracy(["a", "b"], ["a", "b"])),
            ("f1", lambda: check_tool_call_f1(["a"], ["a", "b"])),
            ("keywords", lambda: check_goal_completion("hello world", ["hello", "world"])),
        ]
        report = evaluate_goals(goals)
        assert report.total_goals == 3
        assert report.achieved_count == 2  # accuracy + keywords pass; f1 does not (score < 1)
        assert len(report.results) == 3

    def test_goal_id_override(self):
        """evaluate_goals sets goal_id from the tuple, overriding the default."""
        goals = [
            ("my_custom_id", lambda: check_goal_completion("hello", ["hello"])),
        ]
        report = evaluate_goals(goals)
        assert report.results[0].goal_id == "my_custom_id"

    def test_empty_goals(self):
        report = evaluate_goals([])
        assert report.total_goals == 0
        assert report.achieved_count == 0
        assert report.achievement_rate == 0.0

    def test_all_failing(self):
        goals = [
            ("g1", lambda: check_goal_completion("", ["x"])),
            ("g2", lambda: check_goal_completion("", ["y"])),
        ]
        report = evaluate_goals(goals)
        assert report.achieved_count == 0
        assert report.achievement_rate == 0.0

    def test_all_passing(self):
        goals = [
            ("g1", lambda: check_goal_completion("x y", ["x"])),
            ("g2", lambda: check_tool_call_accuracy(["a"], ["a"])),
        ]
        report = evaluate_goals(goals)
        assert report.achieved_count == 2
        assert report.achievement_rate == 1.0
