"""Tests for multi-objective calibration."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.multi_objective import (
    MultiObjectiveResult,
    ObjectiveWeights,
    ParetoPoint,
    compute_combined_score,
    find_pareto_front,
    optimize_multi_objective,
    select_best_balanced,
)


# -------------------------------------------------------------------
# ObjectiveWeights
# -------------------------------------------------------------------


class TestObjectiveWeights:
    def test_defaults(self):
        w = ObjectiveWeights()
        assert w.accuracy_weight == 0.7
        assert w.cost_weight == 0.3

    def test_custom(self):
        w = ObjectiveWeights(accuracy_weight=0.5, cost_weight=0.5)
        assert w.accuracy_weight == 0.5

    def test_as_dict(self):
        w = ObjectiveWeights(accuracy_weight=0.6, cost_weight=0.4)
        d = w.as_dict()
        assert d["accuracy_weight"] == 0.6
        assert d["cost_weight"] == 0.4

    def test_from_dict(self):
        d = {"accuracy_weight": 0.8, "cost_weight": 0.2}
        w = ObjectiveWeights.from_dict(d)
        assert w.accuracy_weight == 0.8
        assert w.cost_weight == 0.2

    def test_from_dict_defaults(self):
        w = ObjectiveWeights.from_dict({})
        assert w.accuracy_weight == 0.7
        assert w.cost_weight == 0.3

    def test_roundtrip(self):
        w = ObjectiveWeights(accuracy_weight=0.9, cost_weight=0.1)
        restored = ObjectiveWeights.from_dict(w.as_dict())
        assert restored.accuracy_weight == w.accuracy_weight
        assert restored.cost_weight == w.cost_weight


# -------------------------------------------------------------------
# ParetoPoint
# -------------------------------------------------------------------


class TestParetoPoint:
    def test_as_dict(self):
        p = ParetoPoint(prompt="v1", accuracy=0.85, cost_tokens=500, combined_score=0.8)
        d = p.as_dict()
        assert d["prompt"] == "v1"
        assert d["accuracy"] == 0.85
        assert d["cost_tokens"] == 500
        assert d["combined_score"] == 0.8

    def test_fields(self):
        p = ParetoPoint(prompt="test", accuracy=0.5, cost_tokens=100, combined_score=0.6)
        assert p.prompt == "test"
        assert p.cost_tokens == 100


# -------------------------------------------------------------------
# compute_combined_score
# -------------------------------------------------------------------


class TestComputeCombinedScore:
    def test_balanced(self):
        w = ObjectiveWeights(accuracy_weight=0.5, cost_weight=0.5)
        score = compute_combined_score(0.8, 5000, w, max_tokens=10000)
        # 0.5 * 0.8 + 0.5 * (1 - 0.5) = 0.4 + 0.25 = 0.65
        assert abs(score - 0.65) < 1e-9

    def test_accuracy_only(self):
        w = ObjectiveWeights(accuracy_weight=1.0, cost_weight=0.0)
        score = compute_combined_score(0.9, 5000, w, max_tokens=10000)
        assert abs(score - 0.9) < 1e-9

    def test_cost_only(self):
        w = ObjectiveWeights(accuracy_weight=0.0, cost_weight=1.0)
        score = compute_combined_score(0.9, 2000, w, max_tokens=10000)
        # 0 + 1.0 * (1 - 0.2) = 0.8
        assert abs(score - 0.8) < 1e-9

    def test_clamp_low(self):
        w = ObjectiveWeights(accuracy_weight=0.0, cost_weight=1.0)
        # cost_tokens > max_tokens: 1 - 20000/10000 = -1.0
        score = compute_combined_score(0.0, 20000, w, max_tokens=10000)
        assert score == 0.0

    def test_clamp_high(self):
        w = ObjectiveWeights(accuracy_weight=1.0, cost_weight=1.0)
        score = compute_combined_score(1.0, 0, w, max_tokens=10000)
        # 1.0 + 1.0 = 2.0 -> clamped to 1.0
        assert score == 1.0

    def test_zero_cost(self):
        w = ObjectiveWeights()
        score = compute_combined_score(1.0, 0, w, max_tokens=10000)
        # 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert abs(score - 1.0) < 1e-9

    def test_max_cost(self):
        w = ObjectiveWeights()
        score = compute_combined_score(1.0, 10000, w, max_tokens=10000)
        # 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert abs(score - 0.7) < 1e-9


# -------------------------------------------------------------------
# find_pareto_front
# -------------------------------------------------------------------


class TestFindParetoFront:
    def test_basic(self):
        points = [
            ParetoPoint("a", accuracy=0.9, cost_tokens=5000, combined_score=0.0),
            ParetoPoint("b", accuracy=0.7, cost_tokens=1000, combined_score=0.0),
            ParetoPoint("c", accuracy=0.5, cost_tokens=500, combined_score=0.0),
        ]
        front = find_pareto_front(points)
        prompts = [p.prompt for p in front]
        assert "a" in prompts  # highest accuracy
        assert "c" in prompts  # lowest cost

    def test_single_point(self):
        p = ParetoPoint("only", accuracy=0.8, cost_tokens=1000, combined_score=0.5)
        front = find_pareto_front([p])
        assert len(front) == 1
        assert front[0].prompt == "only"

    def test_dominated_removed(self):
        points = [
            ParetoPoint("good", accuracy=0.9, cost_tokens=1000, combined_score=0.0),
            ParetoPoint("bad", accuracy=0.5, cost_tokens=5000, combined_score=0.0),
        ]
        front = find_pareto_front(points)
        assert len(front) == 1
        assert front[0].prompt == "good"

    def test_empty(self):
        assert find_pareto_front([]) == []

    def test_sorted_by_accuracy_descending(self):
        points = [
            ParetoPoint("low", accuracy=0.3, cost_tokens=100, combined_score=0.0),
            ParetoPoint("high", accuracy=0.9, cost_tokens=9000, combined_score=0.0),
            ParetoPoint("mid", accuracy=0.6, cost_tokens=500, combined_score=0.0),
        ]
        front = find_pareto_front(points)
        for i in range(len(front) - 1):
            assert front[i].accuracy >= front[i + 1].accuracy

    def test_identical_points(self):
        p1 = ParetoPoint("a", accuracy=0.8, cost_tokens=1000, combined_score=0.0)
        p2 = ParetoPoint("b", accuracy=0.8, cost_tokens=1000, combined_score=0.0)
        front = find_pareto_front([p1, p2])
        # Neither dominates the other
        assert len(front) == 2


# -------------------------------------------------------------------
# select_best_balanced
# -------------------------------------------------------------------


class TestSelectBestBalanced:
    def test_picks_highest_combined(self):
        w = ObjectiveWeights()
        points = [
            ParetoPoint("a", accuracy=0.9, cost_tokens=8000, combined_score=0.0),
            ParetoPoint("b", accuracy=0.7, cost_tokens=1000, combined_score=0.0),
        ]
        best = select_best_balanced(points, w, max_tokens=10000)
        assert best is not None
        # a: 0.7*0.9 + 0.3*(1-0.8) = 0.63 + 0.06 = 0.69
        # b: 0.7*0.7 + 0.3*(1-0.1) = 0.49 + 0.27 = 0.76
        assert best.prompt == "b"

    def test_empty(self):
        assert select_best_balanced([], ObjectiveWeights()) is None

    def test_single_point(self):
        p = ParetoPoint("only", accuracy=0.5, cost_tokens=500, combined_score=0.0)
        best = select_best_balanced([p], ObjectiveWeights())
        assert best is not None
        assert best.prompt == "only"


# -------------------------------------------------------------------
# optimize_multi_objective
# -------------------------------------------------------------------


class TestOptimizeMultiObjective:
    def test_full_pipeline(self):
        candidates = [
            ("v1", 0.9, 5000),
            ("v2", 0.7, 1000),
            ("v3", 0.5, 500),
            ("v4", 0.4, 8000),  # dominated by v1 and v2
        ]
        result = optimize_multi_objective(candidates)
        assert len(result.pareto_front) >= 2
        assert result.best_balanced is not None
        assert result.best_accuracy is not None
        assert result.cheapest is not None
        # v4 should be dominated (lower accuracy, higher cost than v2/v3)
        front_prompts = [p.prompt for p in result.pareto_front]
        assert "v4" not in front_prompts

    def test_empty_candidates(self):
        result = optimize_multi_objective([])
        assert result.pareto_front == []
        assert result.best_balanced is None
        assert result.best_accuracy is None
        assert result.cheapest is None

    def test_single_candidate(self):
        result = optimize_multi_objective([("only", 0.8, 1000)])
        assert len(result.pareto_front) == 1
        assert result.best_balanced.prompt == "only"
        assert result.best_accuracy.prompt == "only"
        assert result.cheapest.prompt == "only"

    def test_custom_weights(self):
        w = ObjectiveWeights(accuracy_weight=1.0, cost_weight=0.0)
        candidates = [
            ("expensive_good", 0.95, 9000),
            ("cheap_bad", 0.3, 100),
        ]
        result = optimize_multi_objective(candidates, weights=w)
        assert result.best_balanced is not None
        # With accuracy_weight=1.0, the best balanced should be the most accurate
        assert result.best_balanced.prompt == "expensive_good"

    def test_best_accuracy_is_highest(self):
        candidates = [
            ("a", 0.9, 5000),
            ("b", 0.7, 1000),
        ]
        result = optimize_multi_objective(candidates)
        assert result.best_accuracy.prompt == "a"

    def test_cheapest_is_lowest_cost(self):
        candidates = [
            ("a", 0.9, 5000),
            ("b", 0.7, 1000),
        ]
        result = optimize_multi_objective(candidates)
        assert result.cheapest.prompt == "b"


# -------------------------------------------------------------------
# MultiObjectiveResult
# -------------------------------------------------------------------


class TestMultiObjectiveResult:
    def test_format_text(self):
        pp = ParetoPoint("v1", accuracy=0.9, cost_tokens=500, combined_score=0.8)
        result = MultiObjectiveResult(
            pareto_front=[pp],
            best_balanced=pp,
            best_accuracy=pp,
            cheapest=pp,
        )
        text = result.format_text()
        assert "Multi-Objective Optimization Result" in text
        assert "v1" in text
        assert "Best balanced" in text
        assert "Best accuracy" in text
        assert "Cheapest" in text

    def test_format_text_empty(self):
        result = MultiObjectiveResult(pareto_front=[])
        text = result.format_text()
        assert "0 points" in text

    def test_as_dict(self):
        pp = ParetoPoint("v1", accuracy=0.9, cost_tokens=500, combined_score=0.8)
        result = MultiObjectiveResult(
            pareto_front=[pp],
            best_balanced=pp,
            best_accuracy=pp,
            cheapest=pp,
        )
        d = result.as_dict()
        assert len(d["pareto_front"]) == 1
        assert d["best_balanced"]["prompt"] == "v1"
        assert d["best_accuracy"]["prompt"] == "v1"
        assert d["cheapest"]["prompt"] == "v1"

    def test_as_dict_none_fields(self):
        result = MultiObjectiveResult(pareto_front=[])
        d = result.as_dict()
        assert d["best_balanced"] is None
        assert d["best_accuracy"] is None
        assert d["cheapest"] is None

    def test_from_dict_roundtrip(self):
        pp = ParetoPoint("v1", accuracy=0.9, cost_tokens=500, combined_score=0.8)
        original = MultiObjectiveResult(
            pareto_front=[pp],
            best_balanced=pp,
            best_accuracy=pp,
            cheapest=pp,
        )
        d = original.as_dict()
        restored = MultiObjectiveResult.from_dict(d)
        assert len(restored.pareto_front) == 1
        assert restored.pareto_front[0].prompt == "v1"
        assert restored.best_balanced.prompt == "v1"
        assert restored.best_accuracy.prompt == "v1"
        assert restored.cheapest.prompt == "v1"

    def test_from_dict_empty(self):
        d = {
            "pareto_front": [],
            "best_balanced": None,
            "best_accuracy": None,
            "cheapest": None,
        }
        restored = MultiObjectiveResult.from_dict(d)
        assert restored.pareto_front == []
        assert restored.best_balanced is None
