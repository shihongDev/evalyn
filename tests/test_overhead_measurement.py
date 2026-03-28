from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.overhead_measurement import (
    OverheadSample,
    OverheadReport,
    compute_overhead,
    record_sample,
    analyze_overhead,
    estimate_overhead_from_spans,
)


def _span(sid, span_type="llm_call", duration_ms=100, status="ok", **attrs):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        status=status,
        attributes=attrs,
    )


# --- compute_overhead ---


class TestComputeOverhead:
    def test_basic(self):
        ms, pct = compute_overhead(110.0, 100.0)
        assert ms == pytest.approx(10.0)
        assert pct == pytest.approx(10.0)

    def test_zero_baseline(self):
        ms, pct = compute_overhead(5.0, 0.0)
        assert ms == pytest.approx(5.0)
        assert pct == 0.0

    def test_no_overhead(self):
        ms, pct = compute_overhead(100.0, 100.0)
        assert ms == pytest.approx(0.0)
        assert pct == pytest.approx(0.0)

    def test_negative_overhead(self):
        ms, pct = compute_overhead(90.0, 100.0)
        assert ms == pytest.approx(-10.0)
        assert pct == pytest.approx(-10.0)

    def test_large_overhead(self):
        ms, pct = compute_overhead(200.0, 100.0)
        assert ms == pytest.approx(100.0)
        assert pct == pytest.approx(100.0)


# --- record_sample ---


class TestRecordSample:
    def test_creates_correctly(self):
        sample = record_sample("op1", 110.0, 100.0)
        assert sample.operation == "op1"
        assert sample.with_tracing_ms == 110.0
        assert sample.without_tracing_ms == 100.0
        assert sample.overhead_ms == pytest.approx(10.0)
        assert sample.overhead_pct == pytest.approx(10.0)

    def test_zero_baseline(self):
        sample = record_sample("op2", 5.0, 0.0)
        assert sample.overhead_pct == 0.0


# --- OverheadSample ---


class TestOverheadSample:
    def test_as_dict(self):
        sample = record_sample("op", 110.0, 100.0)
        d = sample.as_dict()
        assert d["operation"] == "op"
        assert d["with_tracing_ms"] == 110.0
        assert d["overhead_ms"] == pytest.approx(10.0)


# --- analyze_overhead ---


class TestAnalyzeOverhead:
    def test_averages(self):
        s1 = record_sample("a", 110.0, 100.0)  # 10ms, 10%
        s2 = record_sample("b", 120.0, 100.0)  # 20ms, 20%
        report = analyze_overhead([s1, s2])
        assert report.avg_overhead_ms == pytest.approx(15.0)
        assert report.avg_overhead_pct == pytest.approx(15.0)
        assert report.max_overhead_ms == pytest.approx(20.0)

    def test_recommendation_when_high(self):
        s = record_sample("slow", 200.0, 100.0)  # 100%
        report = analyze_overhead([s], max_acceptable_pct=5.0)
        assert report.recommendation != ""
        assert "disabling" in report.recommendation.lower()

    def test_no_recommendation_when_low(self):
        s = record_sample("fast", 101.0, 100.0)  # 1%
        report = analyze_overhead([s], max_acceptable_pct=5.0)
        assert report.recommendation == ""

    def test_empty_samples(self):
        report = analyze_overhead([])
        assert report.avg_overhead_ms == 0.0
        assert report.max_overhead_ms == 0.0
        assert report.recommendation != ""

    def test_single_sample(self):
        s = record_sample("only", 105.0, 100.0)
        report = analyze_overhead([s])
        assert report.avg_overhead_ms == pytest.approx(5.0)
        assert report.max_overhead_ms == pytest.approx(5.0)


# --- estimate_overhead_from_spans ---


class TestEstimateOverheadFromSpans:
    def test_basic(self):
        spans = [_span("a", duration_ms=150), _span("b", duration_ms=120)]
        report = estimate_overhead_from_spans(spans, baseline_duration_ms=100.0)
        assert len(report.samples) == 2
        assert report.avg_overhead_ms == pytest.approx(35.0)
        assert report.max_overhead_ms == pytest.approx(50.0)

    def test_no_overhead_below_baseline(self):
        spans = [_span("a", duration_ms=80)]
        report = estimate_overhead_from_spans(spans, baseline_duration_ms=100.0)
        assert report.samples[0].overhead_ms == 0.0

    def test_skips_spans_without_duration(self):
        s = _span("a")
        s.end_time = None  # no duration
        report = estimate_overhead_from_spans([s], baseline_duration_ms=100.0)
        assert len(report.samples) == 0

    def test_empty_spans(self):
        report = estimate_overhead_from_spans([], baseline_duration_ms=100.0)
        assert report.avg_overhead_ms == 0.0
        assert report.max_overhead_ms == 0.0

    def test_zero_baseline(self):
        spans = [_span("a", duration_ms=50)]
        report = estimate_overhead_from_spans(spans, baseline_duration_ms=0.0)
        # overhead_pct should be 0 when baseline is 0
        assert report.samples[0].overhead_pct == 0.0


# --- OverheadReport ---


class TestOverheadReport:
    def test_format_text(self):
        s = record_sample("op", 110.0, 100.0)
        report = analyze_overhead([s])
        text = report.format_text()
        assert "Overhead" in text
        assert "op" in text

    def test_format_text_with_recommendation(self):
        s = record_sample("slow", 200.0, 100.0)
        report = analyze_overhead([s], max_acceptable_pct=5.0)
        text = report.format_text()
        assert "Recommendation" in text

    def test_format_text_no_recommendation(self):
        s = record_sample("fast", 101.0, 100.0)
        report = analyze_overhead([s])
        text = report.format_text()
        assert "Recommendation" not in text

    def test_as_dict_roundtrip(self):
        s = record_sample("op", 110.0, 100.0)
        report = analyze_overhead([s])
        d = report.as_dict()
        restored = OverheadReport.from_dict(d)
        assert restored.avg_overhead_ms == pytest.approx(report.avg_overhead_ms)
        assert restored.avg_overhead_pct == pytest.approx(report.avg_overhead_pct)
        assert restored.max_overhead_ms == pytest.approx(report.max_overhead_ms)
        assert len(restored.samples) == len(report.samples)

    def test_from_dict_empty(self):
        report = OverheadReport.from_dict({})
        assert report.samples == []
        assert report.avg_overhead_ms == 0.0
