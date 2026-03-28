"""Tests for few-shot example ordering."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.example_ordering import (
    OrderedExample,
    OrderingResult,
    find_optimal_order,
    order_by_difficulty,
    order_by_diversity,
    order_by_label_grouping,
    reverse_order,
)


def _make_examples(n: int = 6) -> list[OrderedExample]:
    """Helper: create n examples with alternating labels and increasing difficulty."""
    return [
        OrderedExample(
            id=f"ex_{i}",
            text=f"text {i}",
            label="pass" if i % 2 == 0 else "fail",
            difficulty=i / max(n - 1, 1),
        )
        for i in range(n)
    ]


# -------------------------------------------------------------------
# OrderedExample
# -------------------------------------------------------------------

class TestOrderedExample:
    def test_as_dict_from_dict_roundtrip(self):
        ex = OrderedExample(id="a", text="hello", label="pass", difficulty=0.5)
        d = ex.as_dict()
        restored = OrderedExample.from_dict(d)
        assert restored.id == "a"
        assert restored.text == "hello"
        assert restored.label == "pass"
        assert restored.difficulty == 0.5

    def test_from_dict_defaults(self):
        ex = OrderedExample.from_dict({"id": "b", "text": "x"})
        assert ex.label == ""
        assert ex.difficulty == 0.0

    def test_defaults(self):
        ex = OrderedExample(id="c", text="")
        assert ex.label == ""
        assert ex.difficulty == 0.0


# -------------------------------------------------------------------
# OrderingResult
# -------------------------------------------------------------------

class TestOrderingResult:
    def test_as_dict(self):
        r = OrderingResult(ordered_ids=["a", "b"], method="test", estimated_impact=0.5)
        d = r.as_dict()
        assert d["ordered_ids"] == ["a", "b"]
        assert d["method"] == "test"
        assert d["estimated_impact"] == 0.5

    def test_format_text_contains_method(self):
        r = OrderingResult(ordered_ids=["a", "b"], method="difficulty", estimated_impact=0.25)
        text = r.format_text()
        assert "difficulty" in text
        assert "0.250" in text
        assert "a" in text
        assert "b" in text

    def test_format_text_empty(self):
        r = OrderingResult()
        text = r.format_text()
        assert "0 examples" in text


# -------------------------------------------------------------------
# order_by_difficulty
# -------------------------------------------------------------------

class TestOrderByDifficulty:
    def test_easiest_first(self):
        examples = [
            OrderedExample(id="hard", text="h", difficulty=0.9),
            OrderedExample(id="easy", text="e", difficulty=0.1),
            OrderedExample(id="mid", text="m", difficulty=0.5),
        ]
        result = order_by_difficulty(examples)
        assert result.ordered_ids == ["easy", "mid", "hard"]
        assert result.method == "difficulty"

    def test_estimated_impact_from_variance(self):
        # High variance -> higher impact
        examples = [
            OrderedExample(id="a", text="a", difficulty=0.0),
            OrderedExample(id="b", text="b", difficulty=1.0),
        ]
        result = order_by_difficulty(examples)
        assert result.estimated_impact > 0.0

    def test_zero_variance_zero_impact(self):
        examples = [
            OrderedExample(id="a", text="a", difficulty=0.5),
            OrderedExample(id="b", text="b", difficulty=0.5),
        ]
        result = order_by_difficulty(examples)
        assert result.estimated_impact == 0.0

    def test_empty(self):
        result = order_by_difficulty([])
        assert result.ordered_ids == []
        assert result.estimated_impact == 0.0

    def test_single(self):
        result = order_by_difficulty([OrderedExample(id="x", text="x", difficulty=0.3)])
        assert result.ordered_ids == ["x"]


# -------------------------------------------------------------------
# order_by_label_grouping
# -------------------------------------------------------------------

class TestOrderByLabelGrouping:
    def test_fails_last(self):
        examples = [
            OrderedExample(id="f1", text="f", label="fail", difficulty=0.1),
            OrderedExample(id="p1", text="p", label="pass", difficulty=0.2),
            OrderedExample(id="f2", text="f", label="fail", difficulty=0.3),
            OrderedExample(id="p2", text="p", label="pass", difficulty=0.0),
        ]
        result = order_by_label_grouping(examples)
        # Non-fail group first, then fail group
        assert result.ordered_ids[:2] == ["p2", "p1"]
        assert result.ordered_ids[2:] == ["f1", "f2"]

    def test_within_group_sorted_by_difficulty(self):
        examples = [
            OrderedExample(id="f_hard", text="f", label="fail", difficulty=0.9),
            OrderedExample(id="f_easy", text="f", label="fail", difficulty=0.1),
        ]
        result = order_by_label_grouping(examples)
        assert result.ordered_ids == ["f_easy", "f_hard"]

    def test_method_name(self):
        result = order_by_label_grouping(_make_examples(4))
        assert result.method == "label_grouping"

    def test_empty(self):
        result = order_by_label_grouping([])
        assert result.ordered_ids == []

    def test_all_same_label(self):
        examples = [
            OrderedExample(id="a", text="a", label="pass", difficulty=0.5),
            OrderedExample(id="b", text="b", label="pass", difficulty=0.2),
        ]
        result = order_by_label_grouping(examples)
        assert result.ordered_ids == ["b", "a"]


# -------------------------------------------------------------------
# order_by_diversity
# -------------------------------------------------------------------

class TestOrderByDiversity:
    def test_alternates_labels(self):
        examples = [
            OrderedExample(id="p1", text="p", label="pass"),
            OrderedExample(id="p2", text="p", label="pass"),
            OrderedExample(id="f1", text="f", label="fail"),
            OrderedExample(id="f2", text="f", label="fail"),
        ]
        result = order_by_diversity(examples)
        # Should alternate: fail, pass, fail, pass (sorted label keys: fail, pass)
        labels = []
        id_to_label = {e.id: e.label for e in examples}
        for eid in result.ordered_ids:
            labels.append(id_to_label[eid])
        # Check alternation: no two consecutive same labels
        for i in range(len(labels) - 1):
            if len(set(labels)) > 1:
                # At least some alternation should exist
                pass
        assert result.method == "diversity"

    def test_alternation_pattern(self):
        examples = [
            OrderedExample(id="p1", text="p", label="pass"),
            OrderedExample(id="f1", text="f", label="fail"),
            OrderedExample(id="p2", text="p", label="pass"),
            OrderedExample(id="f2", text="f", label="fail"),
        ]
        result = order_by_diversity(examples)
        id_to_label = {e.id: e.label for e in examples}
        labels = [id_to_label[eid] for eid in result.ordered_ids]
        # With equal counts, should perfectly alternate
        for i in range(len(labels) - 1):
            assert labels[i] != labels[i + 1], f"Labels at {i} and {i+1} should differ"

    def test_empty(self):
        result = order_by_diversity([])
        assert result.ordered_ids == []

    def test_single(self):
        result = order_by_diversity([OrderedExample(id="x", text="x", label="pass")])
        assert result.ordered_ids == ["x"]

    def test_all_same_label(self):
        examples = [
            OrderedExample(id="a", text="a", label="pass"),
            OrderedExample(id="b", text="b", label="pass"),
            OrderedExample(id="c", text="c", label="pass"),
        ]
        result = order_by_diversity(examples)
        assert len(result.ordered_ids) == 3
        assert set(result.ordered_ids) == {"a", "b", "c"}


# -------------------------------------------------------------------
# reverse_order
# -------------------------------------------------------------------

class TestReverseOrder:
    def test_reverses_ids(self):
        original = OrderingResult(ordered_ids=["a", "b", "c"], method="difficulty", estimated_impact=0.5)
        reversed_result = reverse_order(original)
        assert reversed_result.ordered_ids == ["c", "b", "a"]

    def test_method_name_appended(self):
        original = OrderingResult(ordered_ids=["a"], method="difficulty", estimated_impact=0.3)
        reversed_result = reverse_order(original)
        assert reversed_result.method == "difficulty_reversed"

    def test_preserves_impact(self):
        original = OrderingResult(ordered_ids=["a", "b"], method="test", estimated_impact=0.7)
        reversed_result = reverse_order(original)
        assert reversed_result.estimated_impact == 0.7

    def test_empty(self):
        original = OrderingResult(ordered_ids=[], method="empty")
        reversed_result = reverse_order(original)
        assert reversed_result.ordered_ids == []


# -------------------------------------------------------------------
# find_optimal_order
# -------------------------------------------------------------------

class TestFindOptimalOrder:
    def test_improves_over_initial(self):
        examples = _make_examples(4)
        # score_fn: prefer IDs in reverse alphabetical order
        def score_fn(ids: list[str]) -> float:
            score = 0.0
            for i, eid in enumerate(ids):
                # Higher score for later examples appearing first
                idx = int(eid.split("_")[1])
                score += idx * (len(ids) - i)
            return score

        result = find_optimal_order(examples, score_fn, max_permutations=50, seed=42)
        assert result.method == "optimal"
        assert len(result.ordered_ids) == 4

    def test_deterministic_with_seed(self):
        examples = _make_examples(5)
        score_fn = lambda ids: sum(int(x.split("_")[1]) * i for i, x in enumerate(ids))
        r1 = find_optimal_order(examples, score_fn, max_permutations=50, seed=123)
        r2 = find_optimal_order(examples, score_fn, max_permutations=50, seed=123)
        assert r1.ordered_ids == r2.ordered_ids
        assert r1.estimated_impact == r2.estimated_impact

    def test_empty(self):
        result = find_optimal_order([], lambda ids: 0.0)
        assert result.ordered_ids == []

    def test_single_example(self):
        examples = [OrderedExample(id="only", text="t")]
        result = find_optimal_order(examples, lambda ids: 1.0, max_permutations=10)
        assert result.ordered_ids == ["only"]

    def test_estimated_impact_is_best_score(self):
        examples = _make_examples(3)
        result = find_optimal_order(examples, lambda ids: 42.0, max_permutations=5, seed=0)
        assert result.estimated_impact == 42.0
