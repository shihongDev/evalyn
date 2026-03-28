"""Tests for evalyn_sdk.datasets_drift: drift detection between dataset versions."""

from __future__ import annotations

import pytest

from evalyn_sdk.datasets_drift import (
    DriftMetric,
    DriftReport,
    compute_length_distribution,
    compute_vocabulary_overlap,
    detect_category_drift,
    detect_drift,
    detect_length_drift,
    detect_vocabulary_drift,
)


# ---------------------------------------------------------------------------
# compute_length_distribution
# ---------------------------------------------------------------------------


class TestComputeLengthDistribution:
    def test_basic(self):
        texts = ["hello", "hi", "hey there"]
        dist = compute_length_distribution(texts)
        assert dist["mean"] == pytest.approx((5 + 2 + 9) / 3)
        assert dist["min"] == 2.0
        assert dist["max"] == 9.0
        assert dist["std"] >= 0

    def test_single_item(self):
        dist = compute_length_distribution(["abc"])
        assert dist["mean"] == 3.0
        assert dist["std"] == 0.0
        assert dist["min"] == 3.0
        assert dist["max"] == 3.0

    def test_empty_texts(self):
        dist = compute_length_distribution([])
        assert dist["mean"] == 0.0
        assert dist["std"] == 0.0
        assert dist["min"] == 0.0
        assert dist["max"] == 0.0

    def test_empty_strings(self):
        dist = compute_length_distribution(["", "", ""])
        assert dist["mean"] == 0.0
        assert dist["min"] == 0.0
        assert dist["max"] == 0.0


# ---------------------------------------------------------------------------
# compute_vocabulary_overlap
# ---------------------------------------------------------------------------


class TestComputeVocabularyOverlap:
    def test_identical(self):
        texts = ["hello world foo"]
        overlap = compute_vocabulary_overlap(texts, texts)
        assert overlap == 1.0

    def test_no_overlap(self):
        a = ["hello world"]
        b = ["foo bar"]
        overlap = compute_vocabulary_overlap(a, b)
        assert overlap == 0.0

    def test_partial_overlap(self):
        a = ["hello world"]
        b = ["hello foo"]
        # words_a = {hello, world}, words_b = {hello, foo}
        # intersection = {hello}, union = {hello, world, foo}
        overlap = compute_vocabulary_overlap(a, b)
        assert overlap == pytest.approx(1 / 3)

    def test_both_empty(self):
        overlap = compute_vocabulary_overlap([], [])
        assert overlap == 1.0

    def test_one_empty(self):
        overlap = compute_vocabulary_overlap(["hello"], [])
        assert overlap == 0.0


# ---------------------------------------------------------------------------
# detect_length_drift
# ---------------------------------------------------------------------------


class TestDetectLengthDrift:
    def test_no_drift(self):
        a = ["hello", "world", "test!"]
        b = ["abcde", "fghij", "klmno"]
        metric = detect_length_drift(a, b)
        assert metric.name == "length_drift"
        assert metric.drifted is False
        assert metric.value == pytest.approx(0.0)

    def test_significant_drift(self):
        a = ["hi"]  # mean length 2
        b = ["a very long sentence here"]  # mean length 25
        metric = detect_length_drift(a, b, threshold=0.2)
        assert metric.drifted is True
        assert metric.value > 0.2

    def test_custom_threshold(self):
        a = ["abcd"]  # mean 4
        b = ["abcdef"]  # mean 6
        # value = |4-6| / max(4,6,1) = 2/6 = 0.333
        metric = detect_length_drift(a, b, threshold=0.5)
        assert metric.drifted is False
        metric2 = detect_length_drift(a, b, threshold=0.2)
        assert metric2.drifted is True

    def test_empty_texts(self):
        metric = detect_length_drift([], [])
        assert metric.value == pytest.approx(0.0)
        assert metric.drifted is False


# ---------------------------------------------------------------------------
# detect_vocabulary_drift
# ---------------------------------------------------------------------------


class TestDetectVocabularyDrift:
    def test_no_drift_identical(self):
        texts = ["hello world foo bar"]
        metric = detect_vocabulary_drift(texts, texts)
        assert metric.name == "vocabulary_drift"
        assert metric.drifted is False
        assert metric.value == pytest.approx(0.0)

    def test_drift_no_overlap(self):
        a = ["alpha beta"]
        b = ["gamma delta"]
        metric = detect_vocabulary_drift(a, b, threshold=0.3)
        assert metric.drifted is True
        assert metric.value == pytest.approx(1.0)

    def test_partial(self):
        a = ["hello world"]
        b = ["hello foo"]
        metric = detect_vocabulary_drift(a, b, threshold=0.5)
        # overlap = 1/3, drift_value = 1 - 1/3 = 2/3 ~ 0.667
        assert metric.value == pytest.approx(2 / 3)
        assert metric.drifted is True


# ---------------------------------------------------------------------------
# detect_category_drift
# ---------------------------------------------------------------------------


class TestDetectCategoryDrift:
    def test_identical_distributions(self):
        cats = ["a", "b", "a", "b"]
        metric = detect_category_drift(cats, cats)
        assert metric.name == "category_drift"
        assert metric.value == pytest.approx(0.0)
        assert metric.drifted is False

    def test_different_distributions(self):
        a = ["x", "x", "x", "x"]  # 100% x
        b = ["y", "y", "y", "y"]  # 100% y
        metric = detect_category_drift(a, b, threshold=0.15)
        # pct diff: x -> |1.0 - 0.0| + y -> |0.0 - 1.0| = 2.0
        assert metric.value == pytest.approx(2.0)
        assert metric.drifted is True

    def test_both_empty(self):
        metric = detect_category_drift([], [])
        assert metric.value == pytest.approx(0.0)
        assert metric.drifted is False

    def test_partial_overlap(self):
        a = ["a", "a", "b", "b"]  # a=50%, b=50%
        b = ["a", "b", "b", "c"]  # a=25%, b=50%, c=25%
        metric = detect_category_drift(a, b, threshold=0.15)
        # a: |0.5 - 0.25| = 0.25, b: |0.5 - 0.5| = 0, c: |0 - 0.25| = 0.25
        assert metric.value == pytest.approx(0.5)
        assert metric.drifted is True


# ---------------------------------------------------------------------------
# detect_drift (full pipeline)
# ---------------------------------------------------------------------------


class TestDetectDrift:
    def test_full_pipeline_no_drift(self):
        items = [{"input": "hello world"}, {"input": "foo bar baz"}]
        report = detect_drift(items, items)
        assert isinstance(report, DriftReport)
        assert report.overall_drifted is False
        assert len(report.metrics) >= 2

    def test_full_pipeline_with_drift(self):
        a = [{"input": "hi"}]
        b = [{"input": "a very long completely different sentence with many words"}]
        report = detect_drift(a, b)
        assert report.overall_drifted is True
        assert report.drift_score > 0

    def test_with_categories(self):
        a = [{"input": "x", "category": "qa"}, {"input": "y", "category": "qa"}]
        b = [{"input": "x", "category": "sum"}, {"input": "y", "category": "sum"}]
        report = detect_drift(a, b)
        cat_metrics = [m for m in report.metrics if m.name == "category_drift"]
        assert len(cat_metrics) == 1

    def test_custom_input_field(self):
        a = [{"text": "hello world"}]
        b = [{"text": "hello world"}]
        report = detect_drift(a, b, input_field="text")
        assert report.overall_drifted is False

    def test_empty_items(self):
        report = detect_drift([], [])
        assert report.overall_drifted is False
        assert report.drift_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------


class TestDriftReport:
    def test_format_text(self):
        report = DriftReport(
            metrics=[
                DriftMetric(name="length_drift", value=0.05, threshold=0.2, drifted=False),
                DriftMetric(name="vocab_drift", value=0.4, threshold=0.3, drifted=True),
            ],
            overall_drifted=True,
            drift_score=0.225,
            summary="Drift detected in: vocab_drift",
        )
        text = report.format_text()
        assert "Drift Report" in text
        assert "length_drift" in text
        assert "vocab_drift" in text
        assert "DRIFT" in text
        assert "ok" in text

    def test_as_dict_from_dict_roundtrip(self):
        report = DriftReport(
            metrics=[
                DriftMetric(name="test", value=0.1, threshold=0.2, drifted=False),
            ],
            overall_drifted=False,
            drift_score=0.1,
            summary="No drift",
        )
        d = report.as_dict()
        restored = DriftReport.from_dict(d)
        assert restored.overall_drifted == report.overall_drifted
        assert restored.drift_score == report.drift_score
        assert restored.summary == report.summary
        assert len(restored.metrics) == 1
        assert restored.metrics[0].name == "test"
        assert restored.metrics[0].value == 0.1

    def test_from_dict_empty(self):
        report = DriftReport.from_dict({})
        assert report.metrics == []
        assert report.overall_drifted is False
        assert report.drift_score == 0.0


# ---------------------------------------------------------------------------
# DriftMetric
# ---------------------------------------------------------------------------


class TestDriftMetric:
    def test_as_dict(self):
        m = DriftMetric(name="foo", value=0.5, threshold=0.3, drifted=True)
        d = m.as_dict()
        assert d == {
            "name": "foo",
            "value": 0.5,
            "threshold": 0.3,
            "drifted": True,
        }

    def test_defaults(self):
        m = DriftMetric(name="bar", value=0.0)
        assert m.threshold == 0.1
        assert m.drifted is False
