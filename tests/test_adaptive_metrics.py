"""Tests for the reference-adaptive metrics module."""

from __future__ import annotations

import pytest

from evalyn_sdk.adaptive_metrics import (
    DEFAULT_FALLBACKS,
    AdaptiveConfig,
    AdaptiveResult,
    MetricMode,
    adapt_metrics_for_dataset,
    adapt_metrics_for_item,
    compute_adaptation_stats,
    format_adaptation_report,
    has_reference,
    select_metric_mode,
)


# ---------------------------------------------------------------------------
# MetricMode
# ---------------------------------------------------------------------------


class TestMetricMode:
    def test_basic_creation(self):
        m = MetricMode(mode="reference", metric_id="exact_match")
        assert m.mode == "reference"
        assert m.metric_id == "exact_match"
        assert m.rubric_variant == ""

    def test_with_variant(self):
        m = MetricMode(mode="no_reference", metric_id="coherence", rubric_variant="fallback")
        assert m.rubric_variant == "fallback"

    def test_as_dict(self):
        m = MetricMode(mode="reference", metric_id="bleu")
        d = m.as_dict()
        assert d == {"mode": "reference", "metric_id": "bleu", "rubric_variant": ""}

    def test_from_dict(self):
        d = {"mode": "no_reference", "metric_id": "coherence", "rubric_variant": "fallback"}
        m = MetricMode.from_dict(d)
        assert m.mode == "no_reference"
        assert m.metric_id == "coherence"
        assert m.rubric_variant == "fallback"

    def test_from_dict_defaults(self):
        d = {"mode": "reference", "metric_id": "bleu"}
        m = MetricMode.from_dict(d)
        assert m.rubric_variant == ""

    def test_roundtrip(self):
        original = MetricMode(mode="no_reference", metric_id="x", rubric_variant="v")
        restored = MetricMode.from_dict(original.as_dict())
        assert restored.mode == original.mode
        assert restored.metric_id == original.metric_id
        assert restored.rubric_variant == original.rubric_variant


# ---------------------------------------------------------------------------
# AdaptiveConfig
# ---------------------------------------------------------------------------


class TestAdaptiveConfig:
    def test_defaults(self):
        c = AdaptiveConfig()
        assert c.reference_field == "expected_output"
        assert c.fallback_metrics == {}

    def test_custom(self):
        c = AdaptiveConfig(reference_field="answer", fallback_metrics={"a": "b"})
        assert c.reference_field == "answer"
        assert c.fallback_metrics == {"a": "b"}

    def test_as_dict(self):
        c = AdaptiveConfig(fallback_metrics={"x": "y"})
        d = c.as_dict()
        assert d["reference_field"] == "expected_output"
        assert d["fallback_metrics"] == {"x": "y"}

    def test_from_dict(self):
        d = {"reference_field": "ref", "fallback_metrics": {"a": "b"}}
        c = AdaptiveConfig.from_dict(d)
        assert c.reference_field == "ref"
        assert c.fallback_metrics == {"a": "b"}

    def test_from_dict_defaults(self):
        c = AdaptiveConfig.from_dict({})
        assert c.reference_field == "expected_output"
        assert c.fallback_metrics == {}

    def test_roundtrip(self):
        original = AdaptiveConfig(reference_field="gold", fallback_metrics={"m": "n"})
        restored = AdaptiveConfig.from_dict(original.as_dict())
        assert restored.reference_field == original.reference_field
        assert restored.fallback_metrics == original.fallback_metrics


# ---------------------------------------------------------------------------
# AdaptiveResult
# ---------------------------------------------------------------------------


class TestAdaptiveResult:
    def test_creation(self):
        r = AdaptiveResult(item_id="1", mode_used="reference", metric_id="bleu", original_metric="bleu")
        assert r.item_id == "1"
        assert r.mode_used == "reference"

    def test_as_dict(self):
        r = AdaptiveResult(item_id="2", mode_used="no_reference", metric_id="coherence", original_metric="bleu")
        d = r.as_dict()
        assert d["item_id"] == "2"
        assert d["original_metric"] == "bleu"
        assert d["metric_id"] == "coherence"

    def test_from_dict(self):
        d = {"item_id": "3", "mode_used": "reference", "metric_id": "rouge", "original_metric": "rouge"}
        r = AdaptiveResult.from_dict(d)
        assert r.item_id == "3"
        assert r.metric_id == "rouge"

    def test_roundtrip(self):
        original = AdaptiveResult(item_id="x", mode_used="no_reference", metric_id="a", original_metric="b")
        restored = AdaptiveResult.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.mode_used == original.mode_used
        assert restored.metric_id == original.metric_id
        assert restored.original_metric == original.original_metric


# ---------------------------------------------------------------------------
# has_reference
# ---------------------------------------------------------------------------


class TestHasReference:
    def test_present_and_nonempty(self):
        assert has_reference({"expected_output": "hello"}) is True

    def test_missing_field(self):
        assert has_reference({"input": "hi"}) is False

    def test_none_value(self):
        assert has_reference({"expected_output": None}) is False

    def test_empty_string(self):
        assert has_reference({"expected_output": ""}) is False

    def test_whitespace_only(self):
        assert has_reference({"expected_output": "   "}) is False

    def test_custom_field(self):
        assert has_reference({"answer": "yes"}, reference_field="answer") is True

    def test_custom_field_missing(self):
        assert has_reference({"expected_output": "yes"}, reference_field="answer") is False

    def test_non_string_value(self):
        assert has_reference({"expected_output": 42}) is True

    def test_list_value(self):
        assert has_reference({"expected_output": [1, 2]}) is True

    def test_empty_list(self):
        # Empty list is truthy in has_reference (only str emptiness is checked)
        assert has_reference({"expected_output": []}) is True

    def test_zero_value(self):
        assert has_reference({"expected_output": 0}) is True


# ---------------------------------------------------------------------------
# select_metric_mode
# ---------------------------------------------------------------------------


class TestSelectMetricMode:
    def test_reference_mode(self):
        config = AdaptiveConfig()
        item = {"expected_output": "hello"}
        mode = select_metric_mode(item, "exact_match", config)
        assert mode.mode == "reference"
        assert mode.metric_id == "exact_match"
        assert mode.rubric_variant == ""

    def test_no_reference_no_fallback(self):
        config = AdaptiveConfig()
        item = {"input": "hi"}
        mode = select_metric_mode(item, "custom_metric", config)
        assert mode.mode == "no_reference"
        assert mode.metric_id == "custom_metric"
        assert mode.rubric_variant == ""

    def test_no_reference_with_fallback(self):
        config = AdaptiveConfig(fallback_metrics={"exact_match": "semantic_similarity"})
        item = {"input": "hi"}
        mode = select_metric_mode(item, "exact_match", config)
        assert mode.mode == "no_reference"
        assert mode.metric_id == "semantic_similarity"
        assert mode.rubric_variant == "fallback"

    def test_custom_reference_field(self):
        config = AdaptiveConfig(reference_field="gold")
        item = {"gold": "answer"}
        mode = select_metric_mode(item, "bleu", config)
        assert mode.mode == "reference"


# ---------------------------------------------------------------------------
# adapt_metrics_for_item
# ---------------------------------------------------------------------------


class TestAdaptMetricsForItem:
    def test_all_reference(self):
        config = AdaptiveConfig(fallback_metrics=DEFAULT_FALLBACKS)
        item = {"id": "item1", "expected_output": "hello"}
        results = adapt_metrics_for_item(item, ["exact_match", "bleu"], config)
        assert len(results) == 2
        assert all(r.mode_used == "reference" for r in results)
        assert results[0].item_id == "item1"

    def test_all_no_reference(self):
        config = AdaptiveConfig(fallback_metrics=DEFAULT_FALLBACKS)
        item = {"id": "item2", "input": "hi"}
        results = adapt_metrics_for_item(item, ["exact_match", "bleu"], config)
        assert len(results) == 2
        assert all(r.mode_used == "no_reference" for r in results)
        assert results[0].metric_id == "semantic_similarity"
        assert results[1].metric_id == "coherence"

    def test_preserves_original_metric(self):
        config = AdaptiveConfig(fallback_metrics={"bleu": "coherence"})
        item = {"id": "x"}
        results = adapt_metrics_for_item(item, ["bleu"], config)
        assert results[0].original_metric == "bleu"
        assert results[0].metric_id == "coherence"

    def test_item_id_from_item_id_field(self):
        config = AdaptiveConfig()
        item = {"item_id": "abc"}
        results = adapt_metrics_for_item(item, ["m"], config)
        assert results[0].item_id == "abc"

    def test_empty_metrics_list(self):
        config = AdaptiveConfig()
        item = {"id": "1", "expected_output": "x"}
        results = adapt_metrics_for_item(item, [], config)
        assert results == []


# ---------------------------------------------------------------------------
# adapt_metrics_for_dataset
# ---------------------------------------------------------------------------


class TestAdaptMetricsForDataset:
    def test_mixed_dataset(self):
        items = [
            {"id": "1", "expected_output": "yes"},
            {"id": "2", "input": "no ref"},
        ]
        config = AdaptiveConfig(fallback_metrics=DEFAULT_FALLBACKS)
        results = adapt_metrics_for_dataset(items, ["exact_match"], config)
        assert "1" in results
        assert "2" in results
        assert results["1"][0].mode_used == "reference"
        assert results["2"][0].mode_used == "no_reference"

    def test_default_config(self):
        items = [{"id": "a", "input": "test"}]
        results = adapt_metrics_for_dataset(items, ["exact_match"])
        assert results["a"][0].metric_id == "semantic_similarity"

    def test_empty_dataset(self):
        results = adapt_metrics_for_dataset([], ["exact_match"])
        assert results == {}

    def test_multiple_metrics(self):
        items = [{"id": "1"}]
        config = AdaptiveConfig(fallback_metrics=DEFAULT_FALLBACKS)
        results = adapt_metrics_for_dataset(items, ["exact_match", "bleu", "rouge"], config)
        assert len(results["1"]) == 3
        metrics_used = [r.metric_id for r in results["1"]]
        assert "semantic_similarity" in metrics_used
        assert "coherence" in metrics_used
        assert "completeness" in metrics_used


# ---------------------------------------------------------------------------
# compute_adaptation_stats
# ---------------------------------------------------------------------------


class TestComputeAdaptationStats:
    def test_all_reference(self):
        results = {
            "1": [AdaptiveResult("1", "reference", "bleu", "bleu")],
            "2": [AdaptiveResult("2", "reference", "bleu", "bleu")],
        }
        stats = compute_adaptation_stats(results)
        assert stats["reference_items"] == 2
        assert stats["no_reference_items"] == 0
        assert stats["metrics_adapted"] == 0
        assert stats["fallbacks_used"] == {}

    def test_all_no_reference(self):
        results = {
            "1": [AdaptiveResult("1", "no_reference", "coherence", "bleu")],
            "2": [AdaptiveResult("2", "no_reference", "coherence", "bleu")],
        }
        stats = compute_adaptation_stats(results)
        assert stats["no_reference_items"] == 2
        assert stats["metrics_adapted"] == 2
        assert stats["fallbacks_used"] == {"bleu": "coherence"}

    def test_mixed(self):
        results = {
            "1": [AdaptiveResult("1", "reference", "bleu", "bleu")],
            "2": [AdaptiveResult("2", "no_reference", "coherence", "bleu")],
        }
        stats = compute_adaptation_stats(results)
        assert stats["reference_items"] == 1
        assert stats["no_reference_items"] == 1

    def test_empty(self):
        stats = compute_adaptation_stats({})
        assert stats["reference_items"] == 0
        assert stats["no_reference_items"] == 0
        assert stats["metrics_adapted"] == 0

    def test_no_fallback_used(self):
        results = {
            "1": [AdaptiveResult("1", "no_reference", "custom", "custom")],
        }
        stats = compute_adaptation_stats(results)
        assert stats["metrics_adapted"] == 1
        assert stats["fallbacks_used"] == {}


# ---------------------------------------------------------------------------
# format_adaptation_report
# ---------------------------------------------------------------------------


class TestFormatAdaptationReport:
    def test_basic_report(self):
        stats = {
            "reference_items": 5,
            "no_reference_items": 3,
            "metrics_adapted": 3,
            "fallbacks_used": {},
        }
        report = format_adaptation_report(stats)
        assert "Reference items:" in report
        assert "5" in report
        assert "3" in report

    def test_with_fallbacks(self):
        stats = {
            "reference_items": 1,
            "no_reference_items": 1,
            "metrics_adapted": 1,
            "fallbacks_used": {"bleu": "coherence"},
        }
        report = format_adaptation_report(stats)
        assert "bleu -> coherence" in report
        assert "Fallbacks applied:" in report

    def test_empty_stats(self):
        stats = {
            "reference_items": 0,
            "no_reference_items": 0,
            "metrics_adapted": 0,
            "fallbacks_used": {},
        }
        report = format_adaptation_report(stats)
        assert "0" in report
        assert "Fallbacks applied:" not in report

    def test_report_header(self):
        report = format_adaptation_report({"fallbacks_used": {}})
        assert "Adaptive Metrics Report" in report


# ---------------------------------------------------------------------------
# DEFAULT_FALLBACKS
# ---------------------------------------------------------------------------


class TestDefaultFallbacks:
    def test_known_keys(self):
        assert "exact_match" in DEFAULT_FALLBACKS
        assert "bleu" in DEFAULT_FALLBACKS
        assert "rouge" in DEFAULT_FALLBACKS
        assert "string_distance" in DEFAULT_FALLBACKS

    def test_values(self):
        assert DEFAULT_FALLBACKS["exact_match"] == "semantic_similarity"
        assert DEFAULT_FALLBACKS["bleu"] == "coherence"
        assert DEFAULT_FALLBACKS["rouge"] == "completeness"
        assert DEFAULT_FALLBACKS["string_distance"] == "relevance"

    def test_is_dict(self):
        assert isinstance(DEFAULT_FALLBACKS, dict)
        assert len(DEFAULT_FALLBACKS) == 4
