"""Tests for language detection and metric priority queue."""

import pytest
from evalyn_sdk.models import DatasetItem, Metric, MetricSpec, MetricResult, EvalRun
from evalyn_sdk.analysis.language_detect import (
    detect_language, analyze_dataset_languages, DatasetLanguageReport,
)
from evalyn_sdk.evaluation.priority import (
    compute_priorities, sort_metrics_by_priority, extract_fail_rates_from_run,
)
from datetime import datetime, timezone


# =============================================================================
# Language detection tests
# =============================================================================

class TestDetectLanguage:
    def test_english(self):
        result = detect_language("The quick brown fox jumps over the lazy dog and they were happy with this")
        assert result.language == "en"
        assert result.script == "Latin"
        assert result.confidence > 0.3

    def test_spanish(self):
        result = detect_language("El gato es un animal muy bonito para la casa")
        assert result.language == "es"

    def test_french(self):
        result = detect_language("Le chat est sur la table dans le jardin")
        assert result.language == "fr"

    def test_german(self):
        result = detect_language("Der Hund ist nicht auf der Strasse und auch nicht im Park")
        assert result.language == "de"

    def test_japanese(self):
        result = detect_language("こんにちは世界")
        assert result.language == "ja"
        assert result.script == "CJK"

    def test_chinese(self):
        result = detect_language("你好世界这是中文")
        assert result.language == "zh"
        assert result.script == "CJK"

    def test_korean(self):
        result = detect_language("안녕하세요 세계")
        assert result.language == "ko"

    def test_russian(self):
        result = detect_language("Привет мир это русский текст")
        assert result.language == "ru"
        assert result.script == "Cyrillic"

    def test_arabic(self):
        result = detect_language("مرحبا بالعالم")
        assert result.language == "ar"
        assert result.script == "Arabic"

    def test_empty_string(self):
        result = detect_language("")
        assert result.language == "unknown"
        assert result.confidence == 0.0

    def test_numbers_only(self):
        result = detect_language("12345 67890")
        assert result.language == "unknown"

    def test_as_dict(self):
        result = detect_language("Hello world")
        d = result.as_dict()
        assert "language" in d
        assert "confidence" in d
        assert "script" in d


class TestAnalyzeDatasetLanguages:
    def test_monolingual(self):
        items = [
            DatasetItem(id="1", input="q", output="This is English text"),
            DatasetItem(id="2", input="q", output="The weather is nice today"),
        ]
        report = analyze_dataset_languages(items)
        assert report.primary_language == "en"
        assert not report.is_multilingual

    def test_multilingual(self):
        items = [
            DatasetItem(id="1", input="q", output="Hello world this is English"),
            DatasetItem(id="2", input="q", output="こんにちは世界"),
        ]
        report = analyze_dataset_languages(items)
        assert report.is_multilingual
        assert len(report.language_counts) >= 2

    def test_empty(self):
        report = analyze_dataset_languages([])
        assert report.total_items == 0
        assert report.primary_language is None

    def test_distribution(self):
        items = [
            DatasetItem(id="1", input="q", output="Hello world from here"),
            DatasetItem(id="2", input="q", output="The weather is great today"),
            DatasetItem(id="3", input="q", output="こんにちは世界"),
        ]
        report = analyze_dataset_languages(items)
        dist = report.distribution()
        assert sum(dist.values()) == pytest.approx(1.0, abs=0.01)

    def test_format_text(self):
        items = [DatasetItem(id="1", input="q", output="Hello world")]
        text = analyze_dataset_languages(items).format_text()
        assert "Language Distribution" in text

    def test_as_dict(self):
        items = [DatasetItem(id="1", input="q", output="Hello world")]
        d = analyze_dataset_languages(items).as_dict()
        assert "primary_language" in d
        assert "distribution" in d


# =============================================================================
# Priority queue tests
# =============================================================================

def _metric(mid, mtype="objective"):
    spec = MetricSpec(id=mid, name=mid, type=mtype)
    return Metric(spec, lambda c, i: MetricResult(
        metric_id=mid, item_id=i.id, call_id=c.id,
        score=1.0, passed=True, details={},
    ))


class TestComputePriorities:
    def test_high_fail_rate_first(self):
        metrics = [_metric("a"), _metric("b")]
        rates = {"a": 0.1, "b": 0.9}
        priorities = compute_priorities(metrics, historical_fail_rates=rates)
        assert priorities[0].metric_id == "b"  # higher fail rate first

    def test_objective_bonus(self):
        metrics = [_metric("obj", "objective"), _metric("subj", "subjective")]
        rates = {"obj": 0.5, "subj": 0.5}
        types = {"obj": "objective", "subj": "subjective"}
        priorities = compute_priorities(metrics, rates, types)
        # obj gets 1.5x bonus, so priority = 0.5*1.5 = 0.75 vs 0.5*1.0 = 0.5
        assert priorities[0].metric_id == "obj"

    def test_default_fail_rate(self):
        metrics = [_metric("a")]
        priorities = compute_priorities(metrics)
        assert priorities[0].historical_fail_rate == 0.5  # default

    def test_empty(self):
        assert compute_priorities([]) == []

    def test_as_dict(self):
        metrics = [_metric("a")]
        priorities = compute_priorities(metrics, {"a": 0.3})
        d = priorities[0].as_dict()
        assert d["metric_id"] == "a"
        assert d["historical_fail_rate"] == 0.3


class TestSortByPriority:
    def test_sort_order(self):
        metrics = [_metric("low_fail"), _metric("high_fail")]
        rates = {"low_fail": 0.1, "high_fail": 0.9}
        sorted_m = sort_metrics_by_priority(metrics, rates)
        assert sorted_m[0].spec.id == "high_fail"

    def test_no_history(self):
        metrics = [_metric("a"), _metric("b")]
        sorted_m = sort_metrics_by_priority(metrics)
        assert len(sorted_m) == 2  # still returns all metrics


class TestExtractFailRates:
    def test_basic(self):
        now = datetime.now(timezone.utc)
        run = EvalRun(
            id="r1", dataset_name="test", created_at=now,
            metric_results=[
                MetricResult(metric_id="a", item_id="i1", call_id="c1", score=1.0, passed=True, details={}),
                MetricResult(metric_id="a", item_id="i2", call_id="c1", score=0.0, passed=False, details={}),
                MetricResult(metric_id="b", item_id="i1", call_id="c1", score=1.0, passed=True, details={}),
                MetricResult(metric_id="b", item_id="i2", call_id="c1", score=1.0, passed=True, details={}),
            ],
        )
        rates = extract_fail_rates_from_run(run)
        assert rates["a"] == 0.5   # 1/2 failed
        assert rates["b"] == 0.0   # 0/2 failed

    def test_empty_run(self):
        now = datetime.now(timezone.utc)
        run = EvalRun(id="r1", dataset_name="test", created_at=now, metric_results=[])
        rates = extract_fail_rates_from_run(run)
        assert rates == {}
