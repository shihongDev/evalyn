"""Tests for planning evaluation."""

import pytest
from evalyn_sdk.evaluation.planning_eval import (
    PlanStep,
    PlanEvalResult,
    evaluate_completeness,
    evaluate_ordering,
    evaluate_efficiency,
    evaluate_replanning,
    evaluate_plan,
)


# -- PlanStep as_dict / from_dict --


class TestPlanStepSerialization:
    def test_as_dict(self):
        step = PlanStep(index=0, description="setup env", dependencies=[])
        d = step.as_dict()
        assert d == {"index": 0, "description": "setup env", "dependencies": []}

    def test_from_dict(self):
        d = {"index": 1, "description": "build", "dependencies": [0]}
        step = PlanStep.from_dict(d)
        assert step.index == 1
        assert step.description == "build"
        assert step.dependencies == [0]

    def test_roundtrip(self):
        step = PlanStep(index=2, description="deploy", dependencies=[0, 1])
        assert PlanStep.from_dict(step.as_dict()).as_dict() == step.as_dict()

    def test_from_dict_missing_dependencies(self):
        d = {"index": 0, "description": "init"}
        step = PlanStep.from_dict(d)
        assert step.dependencies == []


# -- PlanEvalResult --


class TestPlanEvalResultSerialization:
    def test_as_dict(self):
        r = PlanEvalResult(
            completeness=0.8,
            ordering_correctness=1.0,
            efficiency=0.9,
            replanning_quality=0.5,
            overall=0.9,
        )
        d = r.as_dict()
        assert d["completeness"] == 0.8
        assert d["overall"] == 0.9
        assert d["details"] == {}

    def test_from_dict(self):
        d = {
            "completeness": 0.5,
            "ordering_correctness": 0.6,
            "efficiency": 0.7,
            "replanning_quality": 0.4,
            "overall": 0.6,
            "details": {"note": "test"},
        }
        r = PlanEvalResult.from_dict(d)
        assert r.completeness == 0.5
        assert r.details == {"note": "test"}

    def test_format_text(self):
        r = PlanEvalResult(
            completeness=1.0,
            ordering_correctness=1.0,
            efficiency=1.0,
            replanning_quality=1.0,
            overall=1.0,
        )
        text = r.format_text()
        assert "Plan Evaluation Result" in text
        assert "completeness" in text
        assert "1.000" in text


# -- evaluate_completeness --


class TestEvaluateCompleteness:
    def test_all_topics_covered(self):
        steps = [
            PlanStep(0, "setup the database"),
            PlanStep(1, "write the API layer"),
            PlanStep(2, "add authentication"),
        ]
        topics = ["database", "API", "authentication"]
        assert evaluate_completeness(steps, topics) == 1.0

    def test_no_topics_covered(self):
        steps = [PlanStep(0, "do something unrelated")]
        topics = ["database", "API"]
        assert evaluate_completeness(steps, topics) == 0.0

    def test_partial_coverage(self):
        steps = [PlanStep(0, "setup database"), PlanStep(1, "write tests")]
        topics = ["database", "API", "tests"]
        score = evaluate_completeness(steps, topics)
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_empty_topics(self):
        steps = [PlanStep(0, "anything")]
        assert evaluate_completeness(steps, []) == 1.0

    def test_case_insensitive(self):
        steps = [PlanStep(0, "Setup DATABASE")]
        assert evaluate_completeness(steps, ["database"]) == 1.0


# -- evaluate_ordering --


class TestEvaluateOrdering:
    def test_all_correct(self):
        steps = [
            PlanStep(0, "first", []),
            PlanStep(1, "second", [0]),
            PlanStep(2, "third", [0, 1]),
        ]
        assert evaluate_ordering(steps) == 1.0

    def test_violation(self):
        steps = [
            PlanStep(0, "first", [1]),  # dep 1 >= index 0 - violation
            PlanStep(1, "second", []),
        ]
        assert evaluate_ordering(steps) == 0.5

    def test_no_dependencies(self):
        steps = [PlanStep(0, "a"), PlanStep(1, "b"), PlanStep(2, "c")]
        assert evaluate_ordering(steps) == 1.0

    def test_empty_plan(self):
        assert evaluate_ordering([]) == 1.0

    def test_self_dependency_violation(self):
        steps = [PlanStep(0, "step", [0])]  # depends on itself - violation
        assert evaluate_ordering(steps) == 0.0


# -- evaluate_efficiency --


class TestEvaluateEfficiency:
    def test_short_plan(self):
        steps = [PlanStep(i, f"step {i}") for i in range(5)]
        score = evaluate_efficiency(steps, max_steps=20)
        assert score == 1.0

    def test_exact_half(self):
        steps = [PlanStep(i, f"step {i}") for i in range(10)]
        score = evaluate_efficiency(steps, max_steps=20)
        assert score == 1.0

    def test_long_plan(self):
        steps = [PlanStep(i, f"step {i}") for i in range(40)]
        score = evaluate_efficiency(steps, max_steps=20)
        assert score == 0.0

    def test_mid_range(self):
        # max_steps=20, half=10, double=40
        # 25 steps: 1.0 - (25-10)/(40-10) = 1.0 - 15/30 = 0.5
        steps = [PlanStep(i, f"step {i}") for i in range(25)]
        score = evaluate_efficiency(steps, max_steps=20)
        assert abs(score - 0.5) < 1e-9

    def test_empty_plan(self):
        assert evaluate_efficiency([], max_steps=20) == 1.0


# -- evaluate_replanning --


class TestEvaluateReplanning:
    def test_preserves_all_steps(self):
        original = [PlanStep(0, "setup db"), PlanStep(1, "write api")]
        revised = [PlanStep(0, "setup db"), PlanStep(1, "write api"), PlanStep(2, "test")]
        assert evaluate_replanning(original, revised) == 1.0

    def test_complete_rewrite(self):
        original = [PlanStep(0, "setup database"), PlanStep(1, "write api")]
        revised = [PlanStep(0, "completely different plan"), PlanStep(1, "nothing related")]
        assert evaluate_replanning(original, revised) == 0.0

    def test_partial_preservation(self):
        original = [PlanStep(0, "setup db"), PlanStep(1, "write api")]
        revised = [PlanStep(0, "setup db"), PlanStep(1, "rewrite everything")]
        assert evaluate_replanning(original, revised) == 0.5

    def test_empty_original(self):
        assert evaluate_replanning([], [PlanStep(0, "new step")]) == 1.0


# -- evaluate_plan (full pipeline) --


class TestEvaluatePlan:
    def test_full_pipeline(self):
        steps = [
            PlanStep(0, "setup database", []),
            PlanStep(1, "build API on top of database", [0]),
            PlanStep(2, "add authentication to API", [1]),
        ]
        result = evaluate_plan(steps, required_topics=["database", "API", "authentication"])
        assert result.completeness == 1.0
        assert result.ordering_correctness == 1.0
        assert result.efficiency == 1.0  # 3 steps well under max_steps/2
        assert result.replanning_quality == 0.0  # not evaluated in evaluate_plan
        assert abs(result.overall - 1.0) < 1e-9

    def test_overall_is_average(self):
        steps = [
            PlanStep(0, "setup database", [1]),  # ordering violation
        ]
        result = evaluate_plan(steps, required_topics=["database", "API"])
        # completeness = 0.5 (database found, API not)
        # ordering = 0.0 (dep 1 >= index 0)
        # efficiency = 1.0 (1 step)
        expected_overall = (0.5 + 0.0 + 1.0) / 3.0
        assert abs(result.overall - expected_overall) < 1e-9

    def test_empty_plan(self):
        result = evaluate_plan([])
        assert result.completeness == 1.0  # no required topics
        assert result.ordering_correctness == 1.0
        assert result.efficiency == 1.0
        assert abs(result.overall - 1.0) < 1e-9

    def test_details_populated(self):
        steps = [PlanStep(0, "step")]
        result = evaluate_plan(steps, required_topics=["topic"], max_steps=10)
        assert result.details["step_count"] == 1
        assert result.details["max_steps"] == 10
        assert result.details["required_topics"] == ["topic"]


# -- Edge cases --


class TestEdgeCases:
    def test_single_step_plan(self):
        steps = [PlanStep(0, "do everything")]
        result = evaluate_plan(steps, required_topics=["everything"])
        assert result.completeness == 1.0
        assert result.ordering_correctness == 1.0

    def test_single_step_ordering_no_deps(self):
        steps = [PlanStep(0, "solo")]
        assert evaluate_ordering(steps) == 1.0

    def test_efficiency_at_boundary_double(self):
        # Exactly at 2*max_steps should be 0.0
        steps = [PlanStep(i, f"s{i}") for i in range(40)]
        assert evaluate_efficiency(steps, max_steps=20) == 0.0

    def test_completeness_substring_match(self):
        steps = [PlanStep(0, "set up the authentication layer")]
        assert evaluate_completeness(steps, ["auth"]) == 1.0
