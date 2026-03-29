"""Tests for difficulty_grading module."""

from __future__ import annotations

import pytest

from evalyn_sdk.difficulty_grading import (
    DifficultyDistribution,
    DifficultyFactors,
    DifficultyGrade,
    compute_difficulty_score,
    compute_distribution,
    extract_factors,
    format_difficulty_report,
    grade_batch,
    grade_difficulty,
    rebalance_suggestions,
)


# ---------------------------------------------------------------------------
# DifficultyFactors dataclass
# ---------------------------------------------------------------------------


class TestDifficultyFactors:
    def test_basic_fields(self):
        f = DifficultyFactors(
            word_count=10,
            avg_word_length=4.5,
            unique_word_ratio=0.8,
            sentence_count=2,
            question_marks=1,
            conditional_count=0,
            negation_count=1,
        )
        assert f.word_count == 10
        assert f.avg_word_length == 4.5
        assert f.unique_word_ratio == 0.8
        assert f.sentence_count == 2
        assert f.question_marks == 1
        assert f.conditional_count == 0
        assert f.negation_count == 1

    def test_as_dict(self):
        f = DifficultyFactors(5, 3.0, 1.0, 1, 0, 0, 0)
        d = f.as_dict()
        assert d["word_count"] == 5
        assert d["avg_word_length"] == 3.0
        assert d["unique_word_ratio"] == 1.0

    def test_from_dict(self):
        data = {
            "word_count": 20,
            "avg_word_length": 5.0,
            "unique_word_ratio": 0.7,
            "sentence_count": 3,
            "question_marks": 2,
            "conditional_count": 1,
            "negation_count": 0,
        }
        f = DifficultyFactors.from_dict(data)
        assert f.word_count == 20
        assert f.conditional_count == 1

    def test_roundtrip(self):
        original = DifficultyFactors(15, 4.2, 0.85, 3, 1, 2, 1)
        restored = DifficultyFactors.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# DifficultyGrade dataclass
# ---------------------------------------------------------------------------


class TestDifficultyGrade:
    def test_basic_fields(self):
        factors = DifficultyFactors(10, 4.0, 0.8, 2, 0, 0, 0)
        g = DifficultyGrade(
            item_id="a1", level="easy", score=0.2, factors=factors, tags=["short"]
        )
        assert g.item_id == "a1"
        assert g.level == "easy"
        assert g.tags == ["short"]

    def test_default_tags(self):
        factors = DifficultyFactors(10, 4.0, 0.8, 2, 0, 0, 0)
        g = DifficultyGrade(item_id="x", level="easy", score=0.1, factors=factors)
        assert g.tags == []

    def test_as_dict(self):
        factors = DifficultyFactors(10, 4.0, 0.8, 2, 0, 0, 0)
        g = DifficultyGrade("id1", "medium", 0.5, factors, ["tag1"])
        d = g.as_dict()
        assert d["item_id"] == "id1"
        assert d["level"] == "medium"
        assert d["factors"]["word_count"] == 10
        assert d["tags"] == ["tag1"]

    def test_from_dict(self):
        data = {
            "item_id": "z",
            "level": "hard",
            "score": 0.9,
            "factors": {
                "word_count": 100,
                "avg_word_length": 6.0,
                "unique_word_ratio": 0.6,
                "sentence_count": 8,
                "question_marks": 3,
                "conditional_count": 4,
                "negation_count": 2,
            },
            "tags": ["long"],
        }
        g = DifficultyGrade.from_dict(data)
        assert g.item_id == "z"
        assert g.factors.word_count == 100

    def test_roundtrip(self):
        factors = DifficultyFactors(30, 5.1, 0.75, 4, 2, 1, 1)
        original = DifficultyGrade("abc", "medium", 0.55, factors, ["multi-question"])
        restored = DifficultyGrade.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# DifficultyDistribution dataclass
# ---------------------------------------------------------------------------


class TestDifficultyDistribution:
    def test_basic_fields(self):
        d = DifficultyDistribution(5, 5, 5, 15, True)
        assert d.easy_count == 5
        assert d.total == 15
        assert d.balanced is True

    def test_as_dict(self):
        d = DifficultyDistribution(3, 4, 3, 10, False)
        out = d.as_dict()
        assert out["easy_count"] == 3
        assert out["balanced"] is False

    def test_from_dict(self):
        data = {
            "easy_count": 1,
            "medium_count": 2,
            "hard_count": 3,
            "total": 6,
            "balanced": False,
        }
        d = DifficultyDistribution.from_dict(data)
        assert d.hard_count == 3

    def test_roundtrip(self):
        original = DifficultyDistribution(4, 4, 4, 12, True)
        restored = DifficultyDistribution.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# extract_factors
# ---------------------------------------------------------------------------


class TestExtractFactors:
    def test_empty_text(self):
        f = extract_factors("")
        assert f.word_count == 0
        assert f.avg_word_length == 0.0
        assert f.unique_word_ratio == 0.0
        assert f.sentence_count == 0

    def test_simple_sentence(self):
        f = extract_factors("The cat sat on the mat.")
        assert f.word_count == 6
        assert f.sentence_count == 1
        assert f.question_marks == 0

    def test_question_marks(self):
        f = extract_factors("Why? What? How?")
        assert f.question_marks == 3

    def test_conditionals(self):
        f = extract_factors("If you go, unless it rains, when you arrive.")
        assert f.conditional_count == 3

    def test_negations(self):
        f = extract_factors("Not now. No way. I never said don't go.")
        assert f.negation_count == 4

    def test_unique_word_ratio(self):
        f = extract_factors("the the the the")
        assert f.unique_word_ratio == pytest.approx(0.25, abs=0.01)

    def test_all_unique(self):
        f = extract_factors("alpha beta gamma delta")
        assert f.unique_word_ratio == pytest.approx(1.0, abs=0.01)

    def test_multiple_sentences(self):
        f = extract_factors("First sentence. Second sentence! Third sentence?")
        assert f.sentence_count == 3

    def test_avg_word_length(self):
        f = extract_factors("hi go")
        assert f.avg_word_length == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# compute_difficulty_score
# ---------------------------------------------------------------------------


class TestComputeDifficultyScore:
    def test_zero_factors(self):
        f = DifficultyFactors(0, 0.0, 0.0, 0, 0, 0, 0)
        assert compute_difficulty_score(f) == 0.0

    def test_maxed_factors(self):
        f = DifficultyFactors(200, 8.0, 1.0, 15, 5, 5, 5)
        assert compute_difficulty_score(f) == pytest.approx(1.0, abs=0.01)

    def test_score_in_range(self):
        f = DifficultyFactors(50, 4.0, 0.7, 3, 1, 1, 1)
        score = compute_difficulty_score(f)
        assert 0.0 <= score <= 1.0

    def test_higher_factors_higher_score(self):
        low = DifficultyFactors(10, 3.0, 0.5, 1, 0, 0, 0)
        high = DifficultyFactors(150, 7.0, 0.9, 10, 4, 4, 4)
        assert compute_difficulty_score(low) < compute_difficulty_score(high)

    def test_beyond_ceiling_capped(self):
        f = DifficultyFactors(500, 15.0, 1.0, 50, 20, 20, 20)
        assert compute_difficulty_score(f) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# grade_difficulty
# ---------------------------------------------------------------------------


class TestGradeDifficulty:
    def test_easy_text(self):
        g = grade_difficulty("Hi.", item_id="e1")
        assert g.level == "easy"
        assert g.score < 0.33
        assert g.item_id == "e1"

    def test_medium_text(self):
        text = (
            "If you consider the implications of this policy, "
            "you will not find a simple answer. When thinking about "
            "the alternatives, unless there is a better option, "
            "the decision is never straightforward."
        )
        g = grade_difficulty(text, item_id="m1")
        assert g.level in ("medium", "hard")

    def test_hard_text_long(self):
        # Very long text with lots of conditionals and negations
        base = "If not, unless never, when don't. " * 20
        g = grade_difficulty(base, item_id="h1")
        assert g.score >= 0.33  # at least medium

    def test_default_item_id(self):
        g = grade_difficulty("Hello world.")
        assert g.item_id == ""

    def test_tags_short(self):
        g = grade_difficulty("Hi.")
        assert "short" in g.tags

    def test_tags_conditional_heavy(self):
        text = "If this, unless that, when those, if more, unless again."
        g = grade_difficulty(text)
        assert "conditional-heavy" in g.tags

    def test_factors_populated(self):
        g = grade_difficulty("The quick brown fox.")
        assert g.factors.word_count > 0


# ---------------------------------------------------------------------------
# grade_batch
# ---------------------------------------------------------------------------


class TestGradeBatch:
    def test_basic_batch(self):
        items = [
            {"input": "Hello.", "id": "a"},
            {"input": "This is a more complex sentence with several words.", "id": "b"},
        ]
        grades = grade_batch(items)
        assert len(grades) == 2
        assert grades[0].item_id == "a"
        assert grades[1].item_id == "b"

    def test_custom_text_field(self):
        items = [{"prompt": "Hello world.", "id": "x"}]
        grades = grade_batch(items, text_field="prompt")
        assert len(grades) == 1
        assert grades[0].factors.word_count == 2

    def test_missing_text_field(self):
        items = [{"other": "data"}]
        grades = grade_batch(items)
        assert len(grades) == 1
        assert grades[0].factors.word_count == 0

    def test_auto_id(self):
        items = [{"input": "Test."}]
        grades = grade_batch(items)
        assert grades[0].item_id == "0"

    def test_empty_batch(self):
        grades = grade_batch([])
        assert grades == []


# ---------------------------------------------------------------------------
# compute_distribution
# ---------------------------------------------------------------------------


class TestComputeDistribution:
    def _make_grade(self, level: str) -> DifficultyGrade:
        f = DifficultyFactors(10, 4.0, 0.8, 2, 0, 0, 0)
        return DifficultyGrade("x", level, 0.5, f)

    def test_balanced(self):
        grades = (
            [self._make_grade("easy")] * 10
            + [self._make_grade("medium")] * 10
            + [self._make_grade("hard")] * 10
        )
        dist = compute_distribution(grades)
        assert dist.easy_count == 10
        assert dist.medium_count == 10
        assert dist.hard_count == 10
        assert dist.total == 30
        assert dist.balanced is True

    def test_unbalanced(self):
        grades = [self._make_grade("easy")] * 10
        dist = compute_distribution(grades)
        assert dist.easy_count == 10
        assert dist.medium_count == 0
        assert dist.hard_count == 0
        assert dist.balanced is False

    def test_empty(self):
        dist = compute_distribution([])
        assert dist.total == 0
        assert dist.balanced is False


# ---------------------------------------------------------------------------
# rebalance_suggestions
# ---------------------------------------------------------------------------


class TestRebalanceSuggestions:
    def test_empty_distribution(self):
        d = DifficultyDistribution(0, 0, 0, 0, False)
        s = rebalance_suggestions(d)
        assert len(s) == 1
        assert "all difficulty levels" in s[0]

    def test_balanced_no_action(self):
        d = DifficultyDistribution(10, 10, 10, 30, True)
        s = rebalance_suggestions(d)
        assert any("balanced" in msg.lower() for msg in s)

    def test_missing_hard(self):
        d = DifficultyDistribution(10, 10, 0, 20, False)
        s = rebalance_suggestions(d)
        assert any("hard" in msg for msg in s)

    def test_missing_easy_and_medium(self):
        d = DifficultyDistribution(0, 0, 10, 10, False)
        s = rebalance_suggestions(d)
        assert any("easy" in msg for msg in s)
        assert any("medium" in msg for msg in s)


# ---------------------------------------------------------------------------
# format_difficulty_report
# ---------------------------------------------------------------------------


class TestFormatDifficultyReport:
    def test_report_contains_header(self):
        dist = DifficultyDistribution(1, 1, 1, 3, True)
        f = DifficultyFactors(10, 4.0, 0.8, 2, 0, 0, 0)
        grades = [DifficultyGrade("a", "easy", 0.1, f)]
        report = format_difficulty_report(grades, dist)
        assert "Difficulty Grading Report" in report

    def test_report_contains_distribution(self):
        dist = DifficultyDistribution(2, 3, 1, 6, False)
        report = format_difficulty_report([], dist)
        assert "Easy:   2" in report
        assert "Medium: 3" in report
        assert "Hard:   1" in report

    def test_report_contains_items(self):
        f = DifficultyFactors(10, 4.0, 0.8, 2, 0, 0, 0)
        g = DifficultyGrade("item1", "medium", 0.5, f, ["tag1"])
        dist = DifficultyDistribution(0, 1, 0, 1, False)
        report = format_difficulty_report([g], dist)
        assert "item1" in report
        assert "medium" in report
        assert "tag1" in report

    def test_report_contains_suggestions(self):
        dist = DifficultyDistribution(10, 0, 0, 10, False)
        report = format_difficulty_report([], dist)
        assert "Suggestions" in report
