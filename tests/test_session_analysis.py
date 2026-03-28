"""Tests for session_analysis module."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from evalyn_sdk.models import Span
from evalyn_sdk.analysis.session_analysis import (
    SessionSummary,
    SessionComparisonResult,
    SessionReport,
    analyze_session,
    analyze_sessions,
    compare_sessions,
)


def _span(sid, session_id="s1", cost=0.0, tokens=0, status="ok", duration_ms=100, **attrs):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    attrs.update({"session_id": session_id, "cost": cost, "tokens": tokens})
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type="llm_call",
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        status=status,
        attributes=attrs,
    )


# --- analyze_session ---

class TestAnalyzeSession:
    def test_single_span(self):
        spans = [_span("1", cost=0.01, tokens=100, duration_ms=200)]
        s = analyze_session("s1", spans)
        assert s.session_id == "s1"
        assert s.span_count == 1
        assert s.total_cost_usd == 0.01
        assert s.total_tokens == 100
        assert s.total_duration_ms == 200.0
        assert s.error_count == 0
        assert s.pass_rate is None

    def test_multiple_spans_cost_aggregation(self):
        spans = [
            _span("1", cost=0.01, tokens=50, duration_ms=100),
            _span("2", cost=0.02, tokens=150, duration_ms=300),
        ]
        s = analyze_session("s1", spans)
        assert abs(s.total_cost_usd - 0.03) < 1e-9
        assert s.total_tokens == 200
        assert s.total_duration_ms == 400.0
        assert s.span_count == 2

    def test_error_count(self):
        spans = [
            _span("1", status="ok"),
            _span("2", status="error"),
            _span("3", status="error"),
        ]
        s = analyze_session("s1", spans)
        assert s.error_count == 2

    def test_pass_rate_all_pass(self):
        spans = [
            _span("1", passed=True),
            _span("2", passed=True),
        ]
        s = analyze_session("s1", spans)
        assert s.pass_rate == 1.0

    def test_pass_rate_mixed(self):
        spans = [
            _span("1", passed=True),
            _span("2", passed=False),
            _span("3", passed=True),
            _span("4", passed=False),
        ]
        s = analyze_session("s1", spans)
        assert s.pass_rate == 0.5

    def test_pass_rate_none_when_no_passed_attr(self):
        spans = [_span("1"), _span("2")]
        s = analyze_session("s1", spans)
        assert s.pass_rate is None

    def test_pass_rate_partial_passed_attrs(self):
        spans = [
            _span("1", passed=True),
            _span("2"),  # no passed attribute
            _span("3", passed=False),
        ]
        s = analyze_session("s1", spans)
        assert s.pass_rate == 0.5  # 1 out of 2 that have passed attr

    def test_empty_spans(self):
        s = analyze_session("empty", [])
        assert s.span_count == 0
        assert s.total_cost_usd == 0.0
        assert s.total_tokens == 0
        assert s.total_duration_ms == 0.0
        assert s.pass_rate is None
        assert s.error_count == 0

    def test_zero_cost_tokens(self):
        spans = [_span("1", cost=0, tokens=0)]
        s = analyze_session("s1", spans)
        assert s.total_cost_usd == 0.0
        assert s.total_tokens == 0


# --- analyze_sessions ---

class TestAnalyzeSessions:
    def test_group_by_session_id(self):
        spans = [
            _span("1", session_id="a", cost=0.01),
            _span("2", session_id="b", cost=0.02),
            _span("3", session_id="a", cost=0.03),
        ]
        report = analyze_sessions(spans)
        assert report.total_sessions == 2
        by_id = {s.session_id: s for s in report.sessions}
        assert by_id["a"].span_count == 2
        assert abs(by_id["a"].total_cost_usd - 0.04) < 1e-9
        assert by_id["b"].span_count == 1

    def test_unknown_session_id(self):
        spans = [_span("1", session_id="x")]
        # Remove session_id from attributes to test "unknown" fallback
        spans[0].attributes.pop("session_id")
        report = analyze_sessions(spans)
        assert report.total_sessions == 1
        assert report.sessions[0].session_id == "unknown"

    def test_empty_spans(self):
        report = analyze_sessions([])
        assert report.total_sessions == 0
        assert report.sessions == []

    def test_single_session(self):
        spans = [_span("1", session_id="only"), _span("2", session_id="only")]
        report = analyze_sessions(spans)
        assert report.total_sessions == 1
        assert report.sessions[0].session_id == "only"
        assert report.sessions[0].span_count == 2


# --- compare_sessions ---

class TestCompareSessions:
    def test_cost_delta(self):
        a = SessionSummary("a", 1, 0.10, 100, 50, 0.8, 0)
        b = SessionSummary("b", 1, 0.15, 200, 80, 0.9, 1)
        result = compare_sessions(a, b)
        assert abs(result.cost_delta - 0.05) < 1e-9
        assert result.duration_delta == 100.0
        assert abs(result.pass_rate_delta - 0.1) < 1e-9

    def test_negative_delta(self):
        a = SessionSummary("a", 2, 0.20, 500, 100, 1.0, 0)
        b = SessionSummary("b", 1, 0.05, 100, 20, 0.5, 0)
        result = compare_sessions(a, b)
        assert result.cost_delta < 0
        assert result.duration_delta < 0
        assert result.pass_rate_delta < 0

    def test_pass_rate_delta_none(self):
        a = SessionSummary("a", 1, 0.10, 100, 50, None, 0)
        b = SessionSummary("b", 1, 0.15, 200, 80, 0.9, 0)
        result = compare_sessions(a, b)
        assert result.pass_rate_delta is None

    def test_both_none_pass_rate(self):
        a = SessionSummary("a", 1, 0.10, 100, 50, None, 0)
        b = SessionSummary("b", 1, 0.15, 200, 80, None, 0)
        result = compare_sessions(a, b)
        assert result.pass_rate_delta is None


# --- format_text ---

class TestFormatText:
    def test_comparison_format(self):
        result = SessionComparisonResult("a", "b", 0.05, 100.0, 0.1)
        text = result.format_text()
        assert "a vs b" in text
        assert "Cost delta" in text
        assert "Duration delta" in text
        assert "Pass rate delta" in text

    def test_comparison_format_no_pass_rate(self):
        result = SessionComparisonResult("a", "b", 0.05, 100.0, None)
        text = result.format_text()
        assert "Pass rate delta" not in text

    def test_report_format(self):
        summaries = [
            SessionSummary("s1", 3, 0.05, 300, 100, 0.8, 1),
            SessionSummary("s2", 2, 0.02, 150, 50, None, 0),
        ]
        report = SessionReport(summaries, 2)
        text = report.format_text()
        assert "2 session(s)" in text
        assert "s1" in text
        assert "s2" in text
        assert "N/A" in text  # s2 has no pass_rate


# --- as_dict / from_dict roundtrip ---

class TestRoundtrip:
    def test_session_summary_roundtrip(self):
        original = SessionSummary("s1", 5, 0.123, 456.7, 890, 0.75, 2)
        d = original.as_dict()
        restored = SessionSummary.from_dict(d)
        assert restored.session_id == original.session_id
        assert restored.span_count == original.span_count
        assert restored.total_cost_usd == original.total_cost_usd
        assert restored.total_duration_ms == original.total_duration_ms
        assert restored.total_tokens == original.total_tokens
        assert restored.pass_rate == original.pass_rate
        assert restored.error_count == original.error_count

    def test_session_summary_roundtrip_none_pass_rate(self):
        original = SessionSummary("s1", 1, 0.0, 0.0, 0, None, 0)
        d = original.as_dict()
        restored = SessionSummary.from_dict(d)
        assert restored.pass_rate is None

    def test_session_report_roundtrip(self):
        summaries = [
            SessionSummary("s1", 3, 0.05, 300, 100, 0.8, 1),
            SessionSummary("s2", 2, 0.02, 150, 50, None, 0),
        ]
        original = SessionReport(summaries, 2)
        d = original.as_dict()
        restored = SessionReport.from_dict(d)
        assert restored.total_sessions == 2
        assert len(restored.sessions) == 2
        assert restored.sessions[0].session_id == "s1"
        assert restored.sessions[1].pass_rate is None

    def test_comparison_as_dict(self):
        result = SessionComparisonResult("a", "b", 0.05, 100.0, 0.1)
        d = result.as_dict()
        assert d["session_a_id"] == "a"
        assert d["session_b_id"] == "b"
        assert d["cost_delta"] == 0.05
        assert d["duration_delta"] == 100.0
        assert d["pass_rate_delta"] == 0.1
