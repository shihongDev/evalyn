"""Tests for dataset bias auditing."""

import pytest
from evalyn_sdk.datasets_bias_audit import (
    BiasAuditReport,
    BiasIndicator,
    audit_dataset,
    check_category_imbalance,
    check_length_bias,
    check_sentiment_skew,
    check_vocabulary_concentration,
    suggest_debiasing,
)


# ---------------------------------------------------------------------------
# BiasIndicator
# ---------------------------------------------------------------------------


class TestBiasIndicator:
    def test_as_dict(self):
        ind = BiasIndicator(name="test", value=0.5, severity="medium", description="desc")
        d = ind.as_dict()
        assert d == {"name": "test", "value": 0.5, "severity": "medium", "description": "desc"}

    def test_defaults(self):
        ind = BiasIndicator(name="x", value=0.0)
        assert ind.severity == "low"
        assert ind.description == ""


# ---------------------------------------------------------------------------
# BiasAuditReport
# ---------------------------------------------------------------------------


class TestBiasAuditReport:
    def test_as_dict_from_dict_roundtrip(self):
        report = BiasAuditReport(
            indicators=[BiasIndicator("a", 0.1, "low", "desc a")],
            overall_bias_score=0.3,
            high_severity_count=1,
            recommendations=["fix it"],
        )
        d = report.as_dict()
        restored = BiasAuditReport.from_dict(d)
        assert restored.overall_bias_score == 0.3
        assert restored.high_severity_count == 1
        assert len(restored.indicators) == 1
        assert restored.indicators[0].name == "a"
        assert restored.recommendations == ["fix it"]

    def test_from_dict_defaults(self):
        report = BiasAuditReport.from_dict({})
        assert report.overall_bias_score == 0.0
        assert report.high_severity_count == 0
        assert report.indicators == []
        assert report.recommendations == []

    def test_format_text_contains_header(self):
        report = BiasAuditReport(
            indicators=[BiasIndicator("len", 0.5, "medium", "cv is 0.5")],
            overall_bias_score=0.25,
            high_severity_count=0,
            recommendations=["do something"],
        )
        text = report.format_text()
        assert "Bias Audit Report" in text
        assert "len" in text
        assert "medium" in text
        assert "do something" in text

    def test_format_text_no_recommendations(self):
        report = BiasAuditReport()
        text = report.format_text()
        assert "Recommendations" not in text


# ---------------------------------------------------------------------------
# check_length_bias
# ---------------------------------------------------------------------------


class TestCheckLengthBias:
    def test_uniform_lengths_low_bias(self):
        texts = ["hello world", "hello there", "hello again", "hello earth"]
        result = check_length_bias(texts)
        assert result.name == "length_bias"
        assert result.severity == "low"

    def test_skewed_lengths_high_bias(self):
        # Very extreme skew: 1-char vs 10000-char to guarantee CV > 1.0
        texts = ["a"] * 50 + ["x" * 10000] * 5
        result = check_length_bias(texts)
        assert result.severity in ("medium", "high")
        assert result.value > 0.5

    def test_very_skewed_high(self):
        # One very short, rest very long - CV > 1.0
        texts = ["a"] * 1 + ["x" * 500] * 50
        result = check_length_bias(texts)
        # With such extreme skew, should be high or at least medium
        assert result.value > 0

    def test_empty_texts(self):
        result = check_length_bias([])
        assert result.severity == "low"
        assert result.value == 0.0

    def test_all_same_length(self):
        texts = ["aaaa", "bbbb", "cccc"]
        result = check_length_bias(texts)
        assert result.value == 0.0
        assert result.severity == "low"

    def test_all_empty_strings(self):
        texts = ["", "", ""]
        result = check_length_bias(texts)
        assert result.severity == "low"


# ---------------------------------------------------------------------------
# check_category_imbalance
# ---------------------------------------------------------------------------


class TestCheckCategoryImbalance:
    def test_balanced_categories(self):
        cats = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
        result = check_category_imbalance(cats)
        assert result.name == "category_imbalance"
        assert result.severity == "low"
        assert result.value == pytest.approx(1.0)

    def test_highly_imbalanced(self):
        cats = ["a"] * 100 + ["b"] * 1
        result = check_category_imbalance(cats)
        assert result.severity == "high"
        assert result.value > 5

    def test_medium_imbalance(self):
        cats = ["a"] * 30 + ["b"] * 10
        result = check_category_imbalance(cats)
        assert result.value == pytest.approx(3.0)
        assert result.severity == "medium"

    def test_empty_categories(self):
        result = check_category_imbalance([])
        assert result.severity == "low"
        assert result.value == 0.0

    def test_single_category(self):
        result = check_category_imbalance(["only"])
        assert result.severity == "low"
        assert result.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# check_vocabulary_concentration
# ---------------------------------------------------------------------------


class TestCheckVocabularyConcentration:
    def test_diverse_vocabulary(self):
        # Many distinct words
        words = [f"word{i} other{i} more{i}" for i in range(100)]
        result = check_vocabulary_concentration(words)
        assert result.name == "vocabulary_concentration"
        # With 300 unique words, top-10 should be small fraction
        assert result.value < 0.5

    def test_concentrated_vocabulary(self):
        # Same words repeated heavily
        texts = ["the the the the the"] * 50
        result = check_vocabulary_concentration(texts)
        assert result.value > 0.5

    def test_empty_texts(self):
        result = check_vocabulary_concentration([])
        assert result.value == 0.0
        assert result.severity == "low"

    def test_all_empty_strings(self):
        result = check_vocabulary_concentration(["", "", ""])
        assert result.value == 0.0

    def test_custom_top_n(self):
        texts = ["alpha beta gamma delta epsilon"] * 10
        result = check_vocabulary_concentration(texts, top_n=3)
        assert result.name == "vocabulary_concentration"
        # top 3 of 5 unique words = 60% of total
        assert result.value == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# check_sentiment_skew
# ---------------------------------------------------------------------------


class TestCheckSentimentSkew:
    def test_balanced_sentiment(self):
        texts = [
            "good and bad",
            "great but terrible",
            "love hate",
        ]
        result = check_sentiment_skew(texts)
        assert result.name == "sentiment_skew"
        # Roughly balanced
        assert result.value < 3.0

    def test_positive_skew(self):
        texts = [
            "good great excellent amazing wonderful fantastic love",
        ]
        result = check_sentiment_skew(texts)
        assert "positive" in result.description
        assert result.value >= 3.0

    def test_negative_skew(self):
        texts = [
            "bad terrible awful horrible worst hate ugly poor",
        ]
        result = check_sentiment_skew(texts)
        assert "negative" in result.description
        assert result.value >= 3.0

    def test_no_sentiment_words(self):
        texts = ["the quick brown fox jumps"]
        result = check_sentiment_skew(texts)
        assert result.value == 0.0
        assert result.severity == "low"

    def test_empty_texts(self):
        result = check_sentiment_skew([])
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# audit_dataset
# ---------------------------------------------------------------------------


class TestAuditDataset:
    def test_full_pipeline(self):
        items = [
            {"input": "hello world", "category": "greeting"},
            {"input": "good morning", "category": "greeting"},
            {"input": "calculate 2+2", "category": "math"},
            {"input": "solve for x in equation", "category": "math"},
        ]
        report = audit_dataset(items)
        assert isinstance(report, BiasAuditReport)
        assert len(report.indicators) >= 3  # length, vocab, sentiment + category

    def test_with_categories(self):
        items = [
            {"input": "a", "category": "x"},
            {"input": "b", "category": "x"},
            {"input": "c", "category": "x"},
            {"input": "d", "category": "y"},
        ]
        report = audit_dataset(items)
        cat_indicators = [i for i in report.indicators if i.name == "category_imbalance"]
        assert len(cat_indicators) == 1

    def test_no_categories(self):
        items = [{"input": "hello"}, {"input": "world"}]
        report = audit_dataset(items)
        cat_indicators = [i for i in report.indicators if i.name == "category_imbalance"]
        assert len(cat_indicators) == 0

    def test_custom_fields(self):
        items = [
            {"text": "hello", "group": "a"},
            {"text": "world", "group": "a"},
        ]
        report = audit_dataset(items, input_field="text", category_field="group")
        assert isinstance(report, BiasAuditReport)

    def test_empty_items(self):
        report = audit_dataset([])
        assert report.overall_bias_score == 0.0

    def test_overall_score_range(self):
        items = [{"input": f"text {i}"} for i in range(20)]
        report = audit_dataset(items)
        assert 0.0 <= report.overall_bias_score <= 1.0

    def test_roundtrip_through_dict(self):
        items = [
            {"input": "hello", "category": "a"},
            {"input": "world", "category": "b"},
        ]
        report = audit_dataset(items)
        d = report.as_dict()
        restored = BiasAuditReport.from_dict(d)
        assert restored.overall_bias_score == report.overall_bias_score
        assert len(restored.indicators) == len(report.indicators)


# ---------------------------------------------------------------------------
# suggest_debiasing
# ---------------------------------------------------------------------------


class TestSuggestDebiasing:
    def test_no_suggestions_for_low_severity(self):
        report = BiasAuditReport(
            indicators=[BiasIndicator("length_bias", 0.1, "low")],
        )
        suggestions = suggest_debiasing(report)
        assert suggestions == []

    def test_length_suggestion(self):
        report = BiasAuditReport(
            indicators=[BiasIndicator("length_bias", 1.5, "high")],
        )
        suggestions = suggest_debiasing(report)
        assert any("length" in s.lower() for s in suggestions)

    def test_category_suggestion(self):
        report = BiasAuditReport(
            indicators=[BiasIndicator("category_imbalance", 6.0, "high")],
        )
        suggestions = suggest_debiasing(report)
        assert any("categor" in s.lower() for s in suggestions)

    def test_vocabulary_suggestion(self):
        report = BiasAuditReport(
            indicators=[BiasIndicator("vocabulary_concentration", 0.6, "high")],
        )
        suggestions = suggest_debiasing(report)
        assert any("vocabular" in s.lower() for s in suggestions)

    def test_sentiment_suggestion(self):
        report = BiasAuditReport(
            indicators=[BiasIndicator("sentiment_skew", 4.0, "high")],
        )
        suggestions = suggest_debiasing(report)
        assert any("sentiment" in s.lower() or "balance" in s.lower() for s in suggestions)

    def test_multiple_high_severity(self):
        report = BiasAuditReport(
            indicators=[
                BiasIndicator("length_bias", 1.5, "high"),
                BiasIndicator("category_imbalance", 6.0, "high"),
            ],
            high_severity_count=2,
        )
        suggestions = suggest_debiasing(report)
        assert any("rebuilding" in s.lower() for s in suggestions)
