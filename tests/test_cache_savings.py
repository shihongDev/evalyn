"""Tests for cache_savings analysis module."""

from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.analysis.cache_savings import (
    CacheRecommendation,
    CacheSavings,
    analyze_prompt_patterns,
    build_cache_report,
    compute_cache_savings,
)
from evalyn_sdk.models import Span


def _span(sid, **attrs):
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# -- CacheSavings dataclass --


class TestCacheSavingsDataclass:
    def test_defaults(self):
        cs = CacheSavings()
        assert cs.total_cache_creation_tokens == 0
        assert cs.total_cache_read_tokens == 0
        assert cs.total_input_tokens == 0
        assert cs.actual_cost_usd == 0.0
        assert cs.hypothetical_cost_usd == 0.0
        assert cs.savings_usd == 0.0
        assert cs.savings_pct == 0.0
        assert cs.cache_hit_rate == 0.0

    def test_as_dict(self):
        cs = CacheSavings(
            total_cache_creation_tokens=100,
            total_cache_read_tokens=200,
            total_input_tokens=500,
            actual_cost_usd=0.05,
            hypothetical_cost_usd=0.08,
            savings_usd=0.03,
            savings_pct=37.5,
            cache_hit_rate=0.6667,
        )
        d = cs.as_dict()
        assert d["total_cache_creation_tokens"] == 100
        assert d["total_cache_read_tokens"] == 200
        assert d["total_input_tokens"] == 500
        assert d["actual_cost_usd"] == 0.05
        assert d["savings_pct"] == 37.5
        assert d["cache_hit_rate"] == 0.6667

    def test_from_dict_roundtrip(self):
        cs = CacheSavings(
            total_cache_creation_tokens=50,
            total_cache_read_tokens=150,
            total_input_tokens=300,
            actual_cost_usd=0.01,
            hypothetical_cost_usd=0.02,
            savings_usd=0.01,
            savings_pct=50.0,
            cache_hit_rate=0.75,
        )
        d = cs.as_dict()
        restored = CacheSavings.from_dict(d)
        assert restored.total_cache_creation_tokens == 50
        assert restored.total_cache_read_tokens == 150
        assert restored.savings_pct == 50.0
        assert restored.cache_hit_rate == 0.75

    def test_from_dict_missing_keys(self):
        cs = CacheSavings.from_dict({})
        assert cs.total_cache_creation_tokens == 0
        assert cs.savings_usd == 0.0

    def test_format_text_contains_sections(self):
        cs = CacheSavings(
            total_cache_creation_tokens=100,
            total_cache_read_tokens=400,
            total_input_tokens=1000,
            actual_cost_usd=0.05,
            hypothetical_cost_usd=0.09,
            savings_usd=0.04,
            savings_pct=44.4,
            cache_hit_rate=0.8,
        )
        text = cs.format_text()
        assert "Prompt Cache Savings" in text
        assert "Cache creation tokens" in text
        assert "Cache read tokens" in text
        assert "Savings" in text
        assert "Cache hit rate" in text


# -- CacheRecommendation dataclass --


class TestCacheRecommendation:
    def test_as_dict(self):
        rec = CacheRecommendation(
            prompt_pattern="You are a helpful",
            repetition_count=5,
            cacheable=True,
            estimated_savings_pct=72.0,
        )
        d = rec.as_dict()
        assert d["prompt_pattern"] == "You are a helpful"
        assert d["repetition_count"] == 5
        assert d["cacheable"] is True
        assert d["estimated_savings_pct"] == 72.0


# -- compute_cache_savings --


class TestComputeCacheSavings:
    def test_no_cache_tokens_zero_savings(self):
        spans = [
            _span("1", input_tokens=100, cost=0.01),
            _span("2", input_tokens=200, cost=0.02),
        ]
        cs = compute_cache_savings(spans)
        assert cs.total_cache_creation_tokens == 0
        assert cs.total_cache_read_tokens == 0
        assert cs.savings_usd == 0.0
        assert cs.savings_pct == 0.0

    def test_cache_read_tokens_positive_savings(self):
        spans = [
            _span("1", input_tokens=1000, cost=0.10, cache_creation_tokens=500, cache_read_tokens=0),
            _span("2", input_tokens=1000, cost=0.05, cache_creation_tokens=0, cache_read_tokens=500),
        ]
        cs = compute_cache_savings(spans)
        assert cs.total_cache_creation_tokens == 500
        assert cs.total_cache_read_tokens == 500
        assert cs.savings_usd > 0
        assert cs.savings_pct > 0

    def test_cache_hit_rate_calculation(self):
        spans = [
            _span("1", cache_creation_tokens=200, cache_read_tokens=800),
        ]
        cs = compute_cache_savings(spans)
        assert cs.cache_hit_rate == 0.8

    def test_cache_hit_rate_no_cache_tokens(self):
        spans = [_span("1", input_tokens=100)]
        cs = compute_cache_savings(spans)
        assert cs.cache_hit_rate == 0.0

    def test_empty_spans(self):
        cs = compute_cache_savings([])
        assert cs.total_cache_creation_tokens == 0
        assert cs.total_cache_read_tokens == 0
        assert cs.total_input_tokens == 0
        assert cs.actual_cost_usd == 0.0
        assert cs.savings_usd == 0.0
        assert cs.cache_hit_rate == 0.0

    def test_no_cost_attributes(self):
        spans = [
            _span("1", cache_creation_tokens=100, cache_read_tokens=50),
        ]
        cs = compute_cache_savings(spans)
        # No cost info means avg_cost_per_token is 0, so savings_usd is 0
        assert cs.savings_usd == 0.0
        # But cache hit rate should still be computed
        assert cs.cache_hit_rate == 50 / 150

    def test_aggregates_multiple_spans(self):
        spans = [
            _span("1", input_tokens=100, cache_creation_tokens=10, cache_read_tokens=20, cost=0.01),
            _span("2", input_tokens=200, cache_creation_tokens=30, cache_read_tokens=40, cost=0.02),
            _span("3", input_tokens=300, cache_creation_tokens=50, cache_read_tokens=60, cost=0.03),
        ]
        cs = compute_cache_savings(spans)
        assert cs.total_input_tokens == 600
        assert cs.total_cache_creation_tokens == 90
        assert cs.total_cache_read_tokens == 120
        assert cs.actual_cost_usd == 0.06

    def test_hypothetical_cost_greater_than_actual(self):
        spans = [
            _span("1", input_tokens=1000, cost=0.10, cache_read_tokens=500),
        ]
        cs = compute_cache_savings(spans)
        assert cs.hypothetical_cost_usd >= cs.actual_cost_usd


# -- analyze_prompt_patterns --


class TestAnalyzePromptPatterns:
    def test_high_repetition_recommended(self):
        spans = [
            _span("1", input_text="You are a helpful assistant. Please answer the following question."),
            _span("2", input_text="You are a helpful assistant. Please answer the following question."),
            _span("3", input_text="You are a helpful assistant. Please answer the following question."),
        ]
        recs = analyze_prompt_patterns(spans)
        assert len(recs) == 1
        assert recs[0].cacheable is True
        assert recs[0].repetition_count == 3
        assert recs[0].estimated_savings_pct > 0

    def test_low_repetition_not_recommended(self):
        spans = [
            _span("1", input_text="Question one about topic A"),
            _span("2", input_text="Question two about topic B"),
        ]
        recs = analyze_prompt_patterns(spans)
        assert all(not r.cacheable for r in recs)

    def test_empty_spans(self):
        recs = analyze_prompt_patterns([])
        assert recs == []

    def test_no_input_text_skipped(self):
        spans = [_span("1"), _span("2")]
        recs = analyze_prompt_patterns(spans)
        assert recs == []

    def test_sorted_by_repetition_count(self):
        spans = [
            _span("1", input_text="Pattern A repeated"),
            _span("2", input_text="Pattern A repeated"),
            _span("3", input_text="Pattern A repeated"),
            _span("4", input_text="Pattern A repeated"),
            _span("5", input_text="Pattern B once"),
        ]
        recs = analyze_prompt_patterns(spans)
        assert recs[0].repetition_count >= recs[-1].repetition_count

    def test_pattern_truncated_to_100_chars(self):
        long_text = "A" * 200
        spans = [
            _span("1", input_text=long_text),
            _span("2", input_text=long_text),
            _span("3", input_text=long_text),
        ]
        recs = analyze_prompt_patterns(spans)
        assert len(recs) == 1
        assert len(recs[0].prompt_pattern) == 100


# -- build_cache_report --


class TestBuildCacheReport:
    def test_report_structure(self):
        spans = [
            _span("1", input_tokens=100, cost=0.01, cache_creation_tokens=50, cache_read_tokens=30),
        ]
        report = build_cache_report(spans)
        assert "savings" in report
        assert "recommendations" in report
        assert isinstance(report["savings"], dict)
        assert isinstance(report["recommendations"], list)

    def test_empty_report(self):
        report = build_cache_report([])
        assert report["savings"]["total_input_tokens"] == 0
        assert report["recommendations"] == []
