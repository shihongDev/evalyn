"""Tests for reference-free evaluation metrics."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.reference_free import (
    ReferenceFreeReport,
    ReferenceFreeScore,
    check_coherence,
    check_completeness,
    check_fluency,
    check_informativeness,
    check_safety,
    evaluate_reference_free,
)


# -- ReferenceFreeScore as_dict --


class TestReferenceFreeScoreAsDict:
    def test_basic(self):
        s = ReferenceFreeScore("fluency", 0.8, True, {"k": "v"})
        d = s.as_dict()
        assert d["metric_name"] == "fluency"
        assert d["score"] == 0.8
        assert d["passed"] is True
        assert d["details"] == {"k": "v"}

    def test_default_details(self):
        s = ReferenceFreeScore("x", 0.5, False)
        assert s.as_dict()["details"] == {}


# -- ReferenceFreeReport as_dict / from_dict roundtrip --


class TestReferenceFreeReportRoundtrip:
    def test_roundtrip(self):
        scores = [
            ReferenceFreeScore("a", 0.9, True, {"x": 1}),
            ReferenceFreeScore("b", 0.3, False),
        ]
        report = ReferenceFreeReport(scores=scores, overall_score=0.6, pass_rate=0.5)
        d = report.as_dict()
        restored = ReferenceFreeReport.from_dict(d)
        assert restored.overall_score == 0.6
        assert restored.pass_rate == 0.5
        assert len(restored.scores) == 2
        assert restored.scores[0].metric_name == "a"
        assert restored.scores[1].passed is False

    def test_format_text_contains_header(self):
        scores = [ReferenceFreeScore("fluency", 0.8, True)]
        report = ReferenceFreeReport(scores=scores, overall_score=0.8, pass_rate=1.0)
        text = report.format_text()
        assert "Reference-Free Evaluation Report" in text
        assert "PASS" in text
        assert "0.80" in text


# -- check_fluency --


class TestCheckFluency:
    def test_well_formed(self):
        result = check_fluency("The quick brown fox jumps over the lazy dog.")
        assert result.score >= 0.66
        assert result.passed is True

    def test_poorly_formed(self):
        result = check_fluency("no cap no punct")
        assert result.score < 1.0
        # No capitalization, no ending punctuation
        assert result.details["capitalized"] is False
        assert result.details["ends_with_punctuation"] is False

    def test_only_capitalized(self):
        result = check_fluency("Hello world")
        assert result.details["capitalized"] is True
        assert result.details["ends_with_punctuation"] is False


# -- check_coherence --


class TestCheckCoherence:
    def test_consistent_topic(self):
        text = (
            "The cat sat on the mat. The cat then jumped off the mat. "
            "The mat was soft and the cat liked it."
        )
        result = check_coherence(text)
        assert result.score > 0.0
        assert result.passed is True

    def test_random_sentences(self):
        text = "Apples are red. Quantum physics is strange. Basketball courts are large."
        result = check_coherence(text)
        # Very low overlap expected between unrelated sentences
        assert result.score < 0.5

    def test_single_sentence(self):
        result = check_coherence("Just one sentence.")
        assert result.score == 1.0
        assert result.passed is True


# -- check_informativeness --


class TestCheckInformativeness:
    def test_unique_words(self):
        text = "Every word in this particular sentence happens to be unique and informative today"
        result = check_informativeness(text)
        assert result.score > 0.8
        assert result.passed is True

    def test_repetitive(self):
        text = "the the the the the the the the the the the the"
        result = check_informativeness(text)
        assert result.score < 0.3
        assert result.passed is False

    def test_short_text_penalty(self):
        text = "Short text"
        result = check_informativeness(text)
        assert result.details["short_penalty"] is True
        # Penalized by 0.5x
        assert result.score <= 0.5


# -- check_safety --


class TestCheckSafety:
    def test_clean_text(self):
        result = check_safety("This is a perfectly clean and professional response.")
        assert result.score == 1.0
        assert result.passed is True
        assert result.details["flagged_words"] == []

    def test_with_profanity(self):
        result = check_safety("What the hell is going on here?")
        assert result.score == 0.0
        assert result.passed is False
        assert "hell" in result.details["flagged_words"]

    def test_multiple_unsafe_words(self):
        result = check_safety("damn this shit")
        assert result.score == 0.0
        assert len(result.details["flagged_words"]) == 2


# -- check_completeness --


class TestCheckCompleteness:
    def test_long_enough(self):
        text = "A" * 100
        result = check_completeness(text, min_length=50)
        assert result.score == 1.0
        assert result.passed is True

    def test_too_short(self):
        text = "Hi"
        result = check_completeness(text, min_length=50)
        assert result.score < 0.5
        assert result.passed is False

    def test_exact_min(self):
        text = "A" * 50
        result = check_completeness(text, min_length=50)
        assert result.score == 1.0

    def test_custom_min_length(self):
        text = "A" * 10
        result = check_completeness(text, min_length=10)
        assert result.score == 1.0
        assert result.passed is True


# -- evaluate_reference_free --


class TestEvaluateReferenceFree:
    def test_full_pipeline(self):
        text = (
            "The evaluation framework provides comprehensive tools for "
            "assessing model quality. It includes multiple metrics and "
            "supports various evaluation strategies."
        )
        report = evaluate_reference_free(text, min_length=50)
        assert len(report.scores) == 5
        assert 0.0 <= report.overall_score <= 1.0
        assert 0.0 <= report.pass_rate <= 1.0
        metric_names = {s.metric_name for s in report.scores}
        assert metric_names == {
            "fluency",
            "coherence",
            "informativeness",
            "safety",
            "completeness",
        }

    def test_good_text_mostly_passes(self):
        text = (
            "Natural language processing has advanced significantly. "
            "Modern models can understand context and generate fluent text. "
            "These capabilities enable many practical applications."
        )
        report = evaluate_reference_free(text, min_length=50)
        assert report.pass_rate >= 0.6


# -- Edge cases --


class TestEdgeCases:
    def test_empty_text_fluency(self):
        result = check_fluency("")
        assert result.score == 0.0
        assert result.passed is False

    def test_empty_text_coherence(self):
        result = check_coherence("")
        assert result.score == 0.0
        assert result.passed is False

    def test_empty_text_informativeness(self):
        result = check_informativeness("")
        assert result.score == 0.0
        assert result.passed is False

    def test_empty_text_safety(self):
        # Empty text is considered safe
        result = check_safety("")
        assert result.score == 1.0
        assert result.passed is True

    def test_empty_text_completeness(self):
        result = check_completeness("")
        assert result.score == 0.0
        assert result.passed is False

    def test_whitespace_only(self):
        result = check_fluency("   ")
        assert result.score == 0.0
        assert result.passed is False

    def test_evaluate_empty(self):
        report = evaluate_reference_free("")
        assert report.overall_score < 0.5
        # Safety passes even for empty text
        assert any(
            s.metric_name == "safety" and s.passed for s in report.scores
        )
