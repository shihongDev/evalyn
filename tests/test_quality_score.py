"""Tests for quality_score module."""

from __future__ import annotations

import pytest

from evalyn_sdk.quality_score import (
    ItemQualityScore,
    QualityDimension,
    QualityReport,
    build_quality_report,
    compute_quality_score,
    filter_high_quality,
    format_quality_report,
    score_coherence,
    score_length_naturalness,
    score_repetitiveness,
    score_vocabulary_naturalness,
)


# ---------------------------------------------------------------------------
# QualityDimension dataclass
# ---------------------------------------------------------------------------


class TestQualityDimension:
    def test_basic_fields(self):
        d = QualityDimension(name="vocab", score=0.8)
        assert d.name == "vocab"
        assert d.score == 0.8
        assert d.weight == 1.0
        assert d.details == ""

    def test_custom_weight(self):
        d = QualityDimension("x", 0.5, weight=2.0, details="info")
        assert d.weight == 2.0
        assert d.details == "info"

    def test_as_dict(self):
        d = QualityDimension("len", 0.9, 1.5, "test")
        out = d.as_dict()
        assert out == {"name": "len", "score": 0.9, "weight": 1.5, "details": "test"}

    def test_from_dict(self):
        d = QualityDimension.from_dict({"name": "a", "score": 0.3})
        assert d.name == "a"
        assert d.weight == 1.0

    def test_roundtrip(self):
        original = QualityDimension("coherence", 0.75, 2.0, "good")
        restored = QualityDimension.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# ItemQualityScore dataclass
# ---------------------------------------------------------------------------


class TestItemQualityScore:
    def test_basic_fields(self):
        s = ItemQualityScore(item_id="i1", overall_score=0.7)
        assert s.item_id == "i1"
        assert s.passed is True
        assert s.dimensions == []

    def test_as_dict(self):
        dim = QualityDimension("x", 0.5)
        s = ItemQualityScore("a", 0.6, [dim], True)
        out = s.as_dict()
        assert out["item_id"] == "a"
        assert len(out["dimensions"]) == 1

    def test_from_dict(self):
        data = {
            "item_id": "b",
            "overall_score": 0.4,
            "dimensions": [{"name": "y", "score": 0.4}],
            "passed": False,
        }
        s = ItemQualityScore.from_dict(data)
        assert s.item_id == "b"
        assert s.passed is False
        assert len(s.dimensions) == 1

    def test_roundtrip(self):
        dim = QualityDimension("vocab", 0.8, 1.0, "d")
        original = ItemQualityScore("z", 0.8, [dim], True)
        restored = ItemQualityScore.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# QualityReport dataclass
# ---------------------------------------------------------------------------


class TestQualityReport:
    def test_defaults(self):
        r = QualityReport()
        assert r.scores == []
        assert r.mean_score == 0.0
        assert r.total_items == 0

    def test_as_dict(self):
        r = QualityReport(
            scores=[],
            mean_score=0.5,
            rejection_count=1,
            acceptance_rate=0.8,
            total_items=5,
        )
        out = r.as_dict()
        assert out["mean_score"] == 0.5
        assert out["total_items"] == 5

    def test_from_dict(self):
        data = {
            "scores": [],
            "mean_score": 0.6,
            "rejection_count": 2,
            "acceptance_rate": 0.7,
            "total_items": 10,
        }
        r = QualityReport.from_dict(data)
        assert r.rejection_count == 2

    def test_roundtrip(self):
        original = QualityReport([], 0.55, 3, 0.7, 10)
        restored = QualityReport.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# score_length_naturalness
# ---------------------------------------------------------------------------


class TestScoreLengthNaturalness:
    def test_empty_seeds(self):
        assert score_length_naturalness("hello", []) == 0.5

    def test_exact_match(self):
        # Text same length as single seed
        score = score_length_naturalness("hello", [5])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_far_from_mean(self):
        # Very short text vs long seeds
        score = score_length_naturalness("hi", [100, 120, 110])
        assert score < 0.5

    def test_close_to_mean(self):
        # Text length near seed mean
        text = "a" * 50
        score = score_length_naturalness(text, [45, 50, 55])
        assert score > 0.7

    def test_returns_0_to_1(self):
        score = score_length_naturalness("some text", [10, 20, 30])
        assert 0.0 <= score <= 1.0

    def test_single_seed_decay(self):
        score_close = score_length_naturalness("abc", [3])
        score_far = score_length_naturalness("a" * 100, [3])
        assert score_close > score_far


# ---------------------------------------------------------------------------
# score_vocabulary_naturalness
# ---------------------------------------------------------------------------


class TestScoreVocabularyNaturalness:
    def test_full_overlap(self):
        vocab = {"hello", "world"}
        assert score_vocabulary_naturalness("hello world", vocab) == pytest.approx(
            1.0, abs=0.01
        )

    def test_no_overlap(self):
        vocab = {"alpha", "beta"}
        score = score_vocabulary_naturalness("gamma delta", vocab)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        vocab = {"the", "cat"}
        score = score_vocabulary_naturalness("the cat sat", vocab)
        assert 0.5 < score < 1.0

    def test_empty_text(self):
        assert score_vocabulary_naturalness("", {"a"}) == 0.0

    def test_empty_vocab(self):
        assert score_vocabulary_naturalness("hello", set()) == 0.5

    def test_returns_0_to_1(self):
        score = score_vocabulary_naturalness("test words here", {"test", "other"})
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_repetitiveness
# ---------------------------------------------------------------------------


class TestScoreRepetitiveness:
    def test_no_repetition(self):
        score = score_repetitiveness("alpha beta gamma delta epsilon")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_high_repetition(self):
        score = score_repetitiveness("the the the the the the the the")
        assert score < 0.5

    def test_single_word(self):
        assert score_repetitiveness("hello") == 1.0

    def test_empty_text(self):
        assert score_repetitiveness("") == 1.0

    def test_two_words(self):
        score = score_repetitiveness("hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_returns_0_to_1(self):
        score = score_repetitiveness("some repeated repeated text text here")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_coherence
# ---------------------------------------------------------------------------


class TestScoreCoherence:
    def test_empty_text(self):
        assert score_coherence("") == 0.0

    def test_well_formed(self):
        score = score_coherence(
            "This is a test. The second sentence follows. A third is here."
        )
        assert score > 0.7

    def test_lowercase_start_penalty(self):
        good = score_coherence("First sentence. Second sentence.")
        bad = score_coherence("First sentence. second sentence.")
        assert good > bad

    def test_very_short_sentences(self):
        score = score_coherence("A. B. C.")
        assert score < 1.0

    def test_single_long_sentence(self):
        # 50+ words in single sentence should get penalty
        words = " ".join(f"word{i}" for i in range(60))
        # score_coherence looks at _tokenize which finds alpha words
        score = score_coherence(words + ".")
        assert score < 1.0

    def test_returns_0_to_1(self):
        score = score_coherence("Hello world. This is a test.")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    def test_basic(self):
        seeds = ["The cat sat on the mat.", "A dog ran in the park."]
        result = compute_quality_score("The cat ran in the yard.", seeds, item_id="t1")
        assert result.item_id == "t1"
        assert 0.0 <= result.overall_score <= 1.0
        assert len(result.dimensions) == 4

    def test_default_item_id(self):
        result = compute_quality_score("hello", ["hello"])
        assert result.item_id == ""

    def test_pass_at_threshold(self):
        seeds = ["Hello world."]
        result = compute_quality_score("Hello world.", seeds, threshold=0.0)
        assert result.passed is True

    def test_fail_below_threshold(self):
        # gibberish vs real seeds, high threshold
        result = compute_quality_score(
            "", ["Normal sentence. Another one."], threshold=0.9
        )
        assert result.passed is False

    def test_dimension_names(self):
        result = compute_quality_score("Test.", ["Test."])
        names = {d.name for d in result.dimensions}
        assert "length_naturalness" in names
        assert "vocabulary_naturalness" in names
        assert "repetitiveness" in names
        assert "coherence" in names

    def test_score_0_to_1(self):
        result = compute_quality_score("A sentence.", ["Another sentence."])
        assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# build_quality_report
# ---------------------------------------------------------------------------


class TestBuildQualityReport:
    def test_basic_report(self):
        seeds = ["The quick brown fox.", "A lazy dog sleeps."]
        gen = ["The slow brown fox.", "Random unrelated text here."]
        report = build_quality_report(gen, seeds)
        assert report.total_items == 2
        assert len(report.scores) == 2
        assert 0.0 <= report.mean_score <= 1.0
        assert 0.0 <= report.acceptance_rate <= 1.0

    def test_empty_gen(self):
        report = build_quality_report([], ["seed"])
        assert report.total_items == 0
        assert report.mean_score == 0.0

    def test_rejection_count(self):
        seeds = ["Good normal text here."]
        gen = ["a", "b", "c"]  # very short, likely low quality
        report = build_quality_report(gen, seeds, threshold=0.99)
        assert report.rejection_count == 3
        assert report.acceptance_rate == 0.0

    def test_item_ids_sequential(self):
        report = build_quality_report(["a", "b"], ["seed"])
        assert report.scores[0].item_id == "0"
        assert report.scores[1].item_id == "1"


# ---------------------------------------------------------------------------
# filter_high_quality
# ---------------------------------------------------------------------------


class TestFilterHighQuality:
    def test_all_pass(self):
        scores = [
            ItemQualityScore("a", 0.8, [], True),
            ItemQualityScore("b", 0.7, [], True),
        ]
        result = filter_high_quality(scores)
        assert len(result) == 2

    def test_some_fail(self):
        scores = [
            ItemQualityScore("a", 0.8, [], True),
            ItemQualityScore("b", 0.2, [], False),
            ItemQualityScore("c", 0.9, [], True),
        ]
        result = filter_high_quality(scores)
        assert len(result) == 2
        assert all(s.passed for s in result)

    def test_all_fail(self):
        scores = [
            ItemQualityScore("a", 0.1, [], False),
            ItemQualityScore("b", 0.2, [], False),
        ]
        result = filter_high_quality(scores)
        assert result == []

    def test_empty_input(self):
        assert filter_high_quality([]) == []


# ---------------------------------------------------------------------------
# format_quality_report
# ---------------------------------------------------------------------------


class TestFormatQualityReport:
    def test_header(self):
        r = QualityReport([], 0.5, 0, 1.0, 0)
        text = format_quality_report(r)
        assert "Quality Score Report" in text

    def test_contains_stats(self):
        r = QualityReport([], 0.75, 2, 0.8, 10)
        text = format_quality_report(r)
        assert "0.75" in text
        assert "10" in text

    def test_contains_items(self):
        dim = QualityDimension("vocab", 0.8, 1.0, "d")
        score = ItemQualityScore("item1", 0.8, [dim], True)
        r = QualityReport([score], 0.8, 0, 1.0, 1)
        text = format_quality_report(r)
        assert "item1" in text
        assert "PASS" in text
        assert "vocab" in text

    def test_fail_shown(self):
        score = ItemQualityScore("bad", 0.1, [], False)
        r = QualityReport([score], 0.1, 1, 0.0, 1)
        text = format_quality_report(r)
        assert "FAIL" in text
