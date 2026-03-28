"""Tests for few-shot example selection."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.few_shot_selection import (
    Example,
    SelectionResult,
    evaluate_leave_one_out,
    find_optimal_k,
    select_by_coverage,
    select_diverse,
    select_informative,
)


def _make_examples(n: int = 10) -> list[Example]:
    """Helper: create n examples with varied labels and scores."""
    examples = []
    for i in range(n):
        examples.append(
            Example(
                id=f"ex_{i}",
                input_text=f"input {i} word_{i}",
                output_text=f"output {i} unique_{i}",
                label="pass" if i % 2 == 0 else "fail",
                score=i / max(n - 1, 1),
            )
        )
    return examples


# -------------------------------------------------------------------
# Example
# -------------------------------------------------------------------

class TestExample:
    def test_as_dict_from_dict_roundtrip(self):
        ex = Example(id="a", input_text="in", output_text="out", label="pass", score=0.8)
        d = ex.as_dict()
        restored = Example.from_dict(d)
        assert restored.id == "a"
        assert restored.input_text == "in"
        assert restored.output_text == "out"
        assert restored.label == "pass"
        assert restored.score == 0.8

    def test_from_dict_defaults(self):
        ex = Example.from_dict({"id": "b", "input_text": "x", "output_text": "y"})
        assert ex.label == ""
        assert ex.score == 0.0

    def test_defaults(self):
        ex = Example(id="c", input_text="", output_text="")
        assert ex.label == ""
        assert ex.score == 0.0


# -------------------------------------------------------------------
# SelectionResult
# -------------------------------------------------------------------

class TestSelectionResult:
    def test_as_dict(self):
        ex = Example(id="a", input_text="in", output_text="out")
        sr = SelectionResult(selected=[ex], method="diverse", coverage_score=5.0)
        d = sr.as_dict()
        assert d["method"] == "diverse"
        assert len(d["selected"]) == 1
        assert d["coverage_score"] == 5.0

    def test_format_text(self):
        ex = Example(id="a", input_text="in", output_text="out", label="pass", score=0.7)
        sr = SelectionResult(selected=[ex], method="informative", coverage_score=3.0)
        text = sr.format_text()
        assert "informative" in text
        assert "1 examples" in text
        assert "a" in text

    def test_empty_result(self):
        sr = SelectionResult(selected=[], method="coverage", coverage_score=0.0)
        assert sr.as_dict()["selected"] == []


# -------------------------------------------------------------------
# select_diverse
# -------------------------------------------------------------------

class TestSelectDiverse:
    def test_covers_labels(self):
        examples = _make_examples(10)
        result = select_diverse(examples, k=4, seed=42)
        labels = {e.label for e in result.selected}
        assert "pass" in labels
        assert "fail" in labels
        assert len(result.selected) == 4

    def test_k_greater_than_examples(self):
        examples = _make_examples(3)
        result = select_diverse(examples, k=10, seed=42)
        assert len(result.selected) == 3

    def test_k_zero(self):
        examples = _make_examples(5)
        result = select_diverse(examples, k=0)
        assert result.selected == []

    def test_empty_examples(self):
        result = select_diverse([], k=5)
        assert result.selected == []
        assert result.method == "diverse"

    def test_deterministic_with_seed(self):
        examples = _make_examples(10)
        r1 = select_diverse(examples, k=3, seed=99)
        r2 = select_diverse(examples, k=3, seed=99)
        assert [e.id for e in r1.selected] == [e.id for e in r2.selected]

    def test_single_label_group(self):
        examples = [
            Example(id=f"e{i}", input_text=f"in{i}", output_text=f"out{i}", label="pass", score=i * 0.1)
            for i in range(5)
        ]
        result = select_diverse(examples, k=3, seed=1)
        assert len(result.selected) == 3


# -------------------------------------------------------------------
# select_informative
# -------------------------------------------------------------------

class TestSelectInformative:
    def test_picks_boundary_scores(self):
        examples = [
            Example(id="far_low", input_text="a", output_text="b", score=0.0),
            Example(id="far_high", input_text="a", output_text="b", score=1.0),
            Example(id="boundary", input_text="a", output_text="b", score=0.5),
            Example(id="near_boundary", input_text="a", output_text="b", score=0.45),
        ]
        result = select_informative(examples, k=2)
        ids = [e.id for e in result.selected]
        assert "boundary" in ids
        assert "near_boundary" in ids

    def test_k_greater_than_examples(self):
        examples = _make_examples(3)
        result = select_informative(examples, k=10)
        assert len(result.selected) == 3

    def test_empty_examples(self):
        result = select_informative([], k=5)
        assert result.selected == []

    def test_k_zero(self):
        result = select_informative(_make_examples(5), k=0)
        assert result.selected == []

    def test_method_name(self):
        result = select_informative(_make_examples(5), k=2)
        assert result.method == "informative"


# -------------------------------------------------------------------
# select_by_coverage
# -------------------------------------------------------------------

class TestSelectByCoverage:
    def test_maximizes_words(self):
        examples = [
            Example(id="a", input_text="hello world", output_text="foo"),
            Example(id="b", input_text="hello world", output_text="foo"),
            Example(id="c", input_text="unique novel rare", output_text="bar baz"),
        ]
        result = select_by_coverage(examples, k=2)
        ids = {e.id for e in result.selected}
        # Should pick one of a/b (hello world foo) and c (unique novel rare bar baz)
        assert "c" in ids
        assert len(result.selected) == 2

    def test_k_greater_than_examples(self):
        examples = _make_examples(2)
        result = select_by_coverage(examples, k=10)
        assert len(result.selected) == 2

    def test_empty_examples(self):
        result = select_by_coverage([], k=5)
        assert result.selected == []

    def test_k_zero(self):
        result = select_by_coverage(_make_examples(5), k=0)
        assert result.selected == []

    def test_method_name(self):
        result = select_by_coverage(_make_examples(5), k=2)
        assert result.method == "coverage"

    def test_coverage_score_is_word_count(self):
        examples = [
            Example(id="a", input_text="one two", output_text="three"),
            Example(id="b", input_text="four five", output_text="six"),
        ]
        result = select_by_coverage(examples, k=2)
        # All words: one, two, three, four, five, six = 6
        assert result.coverage_score == 6.0


# -------------------------------------------------------------------
# evaluate_leave_one_out
# -------------------------------------------------------------------

class TestEvaluateLeaveOneOut:
    def test_contribution_values(self):
        examples = [
            Example(id="a", input_text="x", output_text="y", score=1.0),
            Example(id="b", input_text="x", output_text="y", score=0.0),
        ]

        def score_fn(exs: list[Example]) -> float:
            return sum(e.score for e in exs)

        result = evaluate_leave_one_out(examples, score_fn)
        # full_score = 1.0, without a = 0.0, without b = 1.0
        assert abs(result["a"] - 1.0) < 1e-9
        assert abs(result["b"] - 0.0) < 1e-9

    def test_empty_examples(self):
        result = evaluate_leave_one_out([], lambda x: 0.0)
        assert result == {}

    def test_single_example(self):
        examples = [Example(id="only", input_text="x", output_text="y", score=5.0)]
        result = evaluate_leave_one_out(examples, lambda exs: sum(e.score for e in exs))
        # full_score=5.0, without "only"=0.0, contribution=5.0
        assert abs(result["only"] - 5.0) < 1e-9


# -------------------------------------------------------------------
# find_optimal_k
# -------------------------------------------------------------------

class TestFindOptimalK:
    def test_returns_best_k(self):
        examples = _make_examples(10)

        # score_fn: peaks at k=3
        def score_fn(exs: list[Example]) -> float:
            k = len(exs)
            return -(k - 3) ** 2  # max at k=3

        best = find_optimal_k(examples, score_fn, max_k=8)
        assert best == 3

    def test_monotonic_increasing(self):
        examples = _make_examples(10)

        def score_fn(exs: list[Example]) -> float:
            return float(len(exs))

        best = find_optimal_k(examples, score_fn, max_k=10)
        assert best == 10

    def test_empty_examples(self):
        best = find_optimal_k([], lambda x: 0.0, max_k=5)
        assert best == 0

    def test_max_k_zero(self):
        best = find_optimal_k(_make_examples(5), lambda x: 0.0, max_k=0)
        assert best == 0

    def test_max_k_exceeds_examples(self):
        examples = _make_examples(3)

        def score_fn(exs: list[Example]) -> float:
            return float(len(exs))

        best = find_optimal_k(examples, score_fn, max_k=100)
        assert best == 3
