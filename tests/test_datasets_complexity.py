"""Tests for evalyn_sdk.datasets_complexity."""

from __future__ import annotations

import pytest

from evalyn_sdk.datasets_complexity import (
    ComplexityFeatures,
    ComplexityReport,
    ComplexityScore,
    classify_difficulty,
    compute_complexity_score,
    extract_features,
    score_dataset,
    score_item,
    sort_by_complexity,
)


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_basic_text(self):
        features = extract_features("Hello world this is a test")
        assert features.word_count == 6
        assert features.sentence_count >= 1
        assert features.avg_word_length > 0
        assert 0 < features.unique_word_ratio <= 1.0
        assert features.has_code is False
        assert features.has_numbers is False
        assert features.question_count == 0

    def test_with_code(self):
        features = extract_features("Use def foo() to define a function")
        assert features.has_code is True

    def test_with_code_block(self):
        features = extract_features("Here is some code:\n```python\nprint('hi')\n```")
        assert features.has_code is True

    def test_with_numbers(self):
        features = extract_features("The value is 42 and the count is 100")
        assert features.has_numbers is True

    def test_with_questions(self):
        features = extract_features("What is this? How does it work? Why?")
        assert features.question_count == 3

    def test_multiple_sentences(self):
        features = extract_features("First sentence. Second sentence. Third one!")
        assert features.sentence_count == 3

    def test_empty_text(self):
        features = extract_features("")
        assert features.word_count == 0
        assert features.sentence_count == 0
        assert features.avg_word_length == 0.0
        assert features.unique_word_ratio == 0.0

    def test_whitespace_only(self):
        features = extract_features("   ")
        assert features.word_count == 0

    def test_single_word(self):
        features = extract_features("hello")
        assert features.word_count == 1
        assert features.unique_word_ratio == 1.0
        assert features.avg_word_length == 5.0

    def test_repeated_words(self):
        features = extract_features("the the the the")
        assert features.word_count == 4
        assert features.unique_word_ratio == pytest.approx(0.25)

    def test_import_detected_as_code(self):
        features = extract_features("import numpy as np")
        assert features.has_code is True

    def test_class_detected_as_code(self):
        features = extract_features("class MyClass inherits from Base")
        assert features.has_code is True


# ---------------------------------------------------------------------------
# compute_complexity_score
# ---------------------------------------------------------------------------


class TestComputeComplexityScore:
    def test_low_complexity(self):
        features = ComplexityFeatures(
            word_count=5,
            sentence_count=1,
            avg_word_length=3.0,
            unique_word_ratio=0.5,
            has_code=False,
            has_numbers=False,
            question_count=0,
        )
        score = compute_complexity_score(features)
        assert 0.0 <= score < 0.5

    def test_high_complexity(self):
        features = ComplexityFeatures(
            word_count=150,
            sentence_count=10,
            avg_word_length=7.0,
            unique_word_ratio=0.9,
            has_code=True,
            has_numbers=True,
            question_count=5,
        )
        score = compute_complexity_score(features)
        assert score > 0.6

    def test_zero_features(self):
        features = ComplexityFeatures()
        score = compute_complexity_score(features)
        assert score == pytest.approx(0.0)

    def test_score_bounded_0_1(self):
        features = ComplexityFeatures(
            word_count=10000,
            sentence_count=100,
            avg_word_length=20.0,
            unique_word_ratio=1.0,
            has_code=True,
            has_numbers=True,
            question_count=100,
        )
        score = compute_complexity_score(features)
        assert 0.0 <= score <= 1.0

    def test_code_increases_score(self):
        base = ComplexityFeatures(
            word_count=20,
            sentence_count=2,
            avg_word_length=4.0,
            unique_word_ratio=0.8,
        )
        with_code = ComplexityFeatures(
            word_count=20,
            sentence_count=2,
            avg_word_length=4.0,
            unique_word_ratio=0.8,
            has_code=True,
        )
        assert compute_complexity_score(with_code) > compute_complexity_score(base)


# ---------------------------------------------------------------------------
# classify_difficulty
# ---------------------------------------------------------------------------


class TestClassifyDifficulty:
    def test_easy(self):
        assert classify_difficulty(0.0) == "easy"
        assert classify_difficulty(0.1) == "easy"
        assert classify_difficulty(0.32) == "easy"

    def test_medium(self):
        assert classify_difficulty(0.33) == "medium"
        assert classify_difficulty(0.5) == "medium"
        assert classify_difficulty(0.65) == "medium"

    def test_hard(self):
        assert classify_difficulty(0.66) == "hard"
        assert classify_difficulty(0.8) == "hard"
        assert classify_difficulty(1.0) == "hard"

    def test_boundary_easy_medium(self):
        assert classify_difficulty(0.329) == "easy"
        assert classify_difficulty(0.33) == "medium"

    def test_boundary_medium_hard(self):
        assert classify_difficulty(0.659) == "medium"
        assert classify_difficulty(0.66) == "hard"


# ---------------------------------------------------------------------------
# score_item
# ---------------------------------------------------------------------------


class TestScoreItem:
    def test_basic_item(self):
        item = {"id": "item-1", "input": "What is the capital of France?"}
        result = score_item(item)
        assert result.item_id == "item-1"
        assert 0.0 <= result.score <= 1.0
        assert result.difficulty in ("easy", "medium", "hard")
        assert result.features is not None

    def test_custom_input_field(self):
        item = {"id": "item-1", "text": "Explain quantum computing in detail."}
        result = score_item(item, input_field="text")
        assert result.item_id == "item-1"
        assert result.features is not None
        assert result.features.word_count > 0

    def test_missing_input_field(self):
        item = {"id": "item-1"}
        result = score_item(item)
        assert result.score == pytest.approx(0.0)

    def test_missing_id(self):
        item = {"input": "some text here"}
        result = score_item(item)
        assert result.item_id == ""


# ---------------------------------------------------------------------------
# score_dataset
# ---------------------------------------------------------------------------


class TestScoreDataset:
    def test_distribution_sums_to_total(self):
        items = [
            {"id": "1", "input": "hi"},
            {"id": "2", "input": "What is the meaning of life? How would you explain it? Can you elaborate on each point in detail with references?"},
            {"id": "3", "input": "Write a function def solve(x): return x**2 with import math and class Foo"},
        ]
        report = score_dataset(items)
        total = sum(report.distribution.values())
        assert total == len(items)

    def test_avg_complexity(self):
        items = [
            {"id": "1", "input": "hello"},
            {"id": "2", "input": "world"},
        ]
        report = score_dataset(items)
        expected_avg = sum(s.score for s in report.scores) / len(report.scores)
        assert report.avg_complexity == pytest.approx(expected_avg)

    def test_empty_dataset(self):
        report = score_dataset([])
        assert report.scores == []
        assert report.avg_complexity == 0.0
        assert report.distribution == {"easy": 0, "medium": 0, "hard": 0}

    def test_custom_input_field(self):
        items = [{"id": "1", "prompt": "Explain this"}]
        report = score_dataset(items, input_field="prompt")
        assert len(report.scores) == 1
        assert report.scores[0].features.word_count == 2


# ---------------------------------------------------------------------------
# sort_by_complexity
# ---------------------------------------------------------------------------


class TestSortByComplexity:
    def test_ascending(self):
        scores = [
            ComplexityScore(item_id="a", score=0.8, difficulty="hard"),
            ComplexityScore(item_id="b", score=0.2, difficulty="easy"),
            ComplexityScore(item_id="c", score=0.5, difficulty="medium"),
        ]
        result = sort_by_complexity(scores, ascending=True)
        assert [s.item_id for s in result] == ["b", "c", "a"]

    def test_descending(self):
        scores = [
            ComplexityScore(item_id="a", score=0.8, difficulty="hard"),
            ComplexityScore(item_id="b", score=0.2, difficulty="easy"),
            ComplexityScore(item_id="c", score=0.5, difficulty="medium"),
        ]
        result = sort_by_complexity(scores, ascending=False)
        assert [s.item_id for s in result] == ["a", "c", "b"]

    def test_empty_list(self):
        result = sort_by_complexity([])
        assert result == []

    def test_single_item(self):
        scores = [ComplexityScore(item_id="x", score=0.5)]
        result = sort_by_complexity(scores)
        assert len(result) == 1
        assert result[0].item_id == "x"

    def test_does_not_mutate_original(self):
        scores = [
            ComplexityScore(item_id="a", score=0.9),
            ComplexityScore(item_id="b", score=0.1),
        ]
        original_order = [s.item_id for s in scores]
        sort_by_complexity(scores)
        assert [s.item_id for s in scores] == original_order


# ---------------------------------------------------------------------------
# ComplexityReport format_text
# ---------------------------------------------------------------------------


class TestComplexityReportFormatText:
    def test_contains_header(self):
        report = ComplexityReport(
            scores=[],
            avg_complexity=0.45,
            distribution={"easy": 2, "medium": 3, "hard": 1},
        )
        text = report.format_text()
        assert "Complexity Report" in text
        assert "0.450" in text
        assert "easy: 2" in text
        assert "medium: 3" in text
        assert "hard: 1" in text

    def test_empty_report(self):
        report = ComplexityReport()
        text = report.format_text()
        assert "Complexity Report" in text
        assert "Items scored: 0" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_complexity_features_as_dict(self):
        f = ComplexityFeatures(
            word_count=10,
            sentence_count=2,
            avg_word_length=4.5,
            unique_word_ratio=0.8,
            has_code=True,
            has_numbers=True,
            question_count=1,
        )
        d = f.as_dict()
        assert d["word_count"] == 10
        assert d["has_code"] is True
        assert d["question_count"] == 1

    def test_complexity_score_roundtrip(self):
        score = ComplexityScore(
            item_id="item-1",
            score=0.75,
            difficulty="hard",
            features=ComplexityFeatures(word_count=50, has_code=True),
        )
        d = score.as_dict()
        restored = ComplexityScore.from_dict(d)
        assert restored.item_id == "item-1"
        assert restored.score == 0.75
        assert restored.difficulty == "hard"
        assert restored.features is not None
        assert restored.features.word_count == 50
        assert restored.features.has_code is True

    def test_complexity_score_from_dict_no_features(self):
        d = {"item_id": "x", "score": 0.3, "difficulty": "easy", "features": None}
        score = ComplexityScore.from_dict(d)
        assert score.features is None

    def test_complexity_report_roundtrip(self):
        report = ComplexityReport(
            scores=[
                ComplexityScore(item_id="1", score=0.2, difficulty="easy"),
                ComplexityScore(item_id="2", score=0.7, difficulty="hard"),
            ],
            avg_complexity=0.45,
            distribution={"easy": 1, "medium": 0, "hard": 1},
        )
        d = report.as_dict()
        restored = ComplexityReport.from_dict(d)
        assert len(restored.scores) == 2
        assert restored.avg_complexity == 0.45
        assert restored.distribution["easy"] == 1
        assert restored.distribution["hard"] == 1

    def test_complexity_report_from_dict_empty(self):
        report = ComplexityReport.from_dict({})
        assert report.scores == []
        assert report.avg_complexity == 0.0
        assert report.distribution == {}
