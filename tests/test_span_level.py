from __future__ import annotations

from datetime import datetime, timezone, timedelta

from evalyn_sdk.models import Span
from evalyn_sdk.evaluation.span_level import (
    SpanScore,
    SpanEvalConfig,
    SpanEvalReport,
    evaluate_span,
    evaluate_spans,
    find_failing_spans,
    compute_type_breakdown,
)


def _span(sid, span_type="llm_call", status="ok", duration_ms=100, **attrs):
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


# --- evaluate_span: llm_call ---

def test_evaluate_span_llm_call_passes():
    span = _span("s1", span_type="llm_call", output="This is a sufficiently long output text")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.passed is True
    assert score.score == 1.0
    assert score.checks["output_exists"] is True
    assert score.checks["output_length"] is True
    assert score.checks["no_error"] is True


def test_evaluate_span_llm_call_short_output():
    span = _span("s2", span_type="llm_call", output="Hi")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.checks["output_exists"] is True
    assert score.checks["output_length"] is False
    assert score.score < 1.0


def test_evaluate_span_llm_call_no_output():
    span = _span("s3", span_type="llm_call")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.checks["output_exists"] is False
    assert score.checks["output_length"] is False


def test_evaluate_span_llm_call_error_status():
    span = _span("s4", span_type="llm_call", status="error",
                  output="This is enough output text for the check")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.checks["no_error"] is False


def test_evaluate_span_llm_call_high_latency():
    span = _span("s5", span_type="llm_call", duration_ms=10000,
                  output="Sufficiently long output text here")
    config = SpanEvalConfig(max_latency_ms=5000.0)
    score = evaluate_span(span, config)
    assert score.checks["latency_ok"] is False


def test_evaluate_span_llm_call_latency_ok():
    span = _span("s6", span_type="llm_call", duration_ms=100,
                  output="Sufficiently long output text here")
    config = SpanEvalConfig(max_latency_ms=5000.0)
    score = evaluate_span(span, config)
    assert score.checks["latency_ok"] is True


def test_evaluate_span_llm_call_disable_quality_check():
    span = _span("s7", span_type="llm_call", output="Hi")
    config = SpanEvalConfig(check_llm_quality=False)
    score = evaluate_span(span, config)
    assert "output_exists" not in score.checks
    assert "output_length" not in score.checks


def test_evaluate_span_llm_call_disable_latency_check():
    span = _span("s8", span_type="llm_call", duration_ms=99999,
                  output="Sufficiently long output text here")
    config = SpanEvalConfig(check_latency=False)
    score = evaluate_span(span, config)
    assert "latency_ok" not in score.checks


# --- evaluate_span: tool_call ---

def test_evaluate_span_tool_call_passes():
    span = _span("t1", span_type="tool_call", output="tool result data")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.passed is True
    assert score.checks["no_error"] is True
    assert score.checks["has_output"] is True


def test_evaluate_span_tool_call_error():
    span = _span("t2", span_type="tool_call", status="error", output="some output")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.checks["no_error"] is False


def test_evaluate_span_tool_call_no_output():
    span = _span("t3", span_type="tool_call")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.checks["has_output"] is False


def test_evaluate_span_tool_call_disabled():
    span = _span("t4", span_type="tool_call", status="error")
    config = SpanEvalConfig(check_tool_success=False)
    score = evaluate_span(span, config)
    # With tool checks disabled, no checks are added for tool_call
    assert score.score == 1.0
    assert score.passed is True


# --- evaluate_span: retrieval ---

def test_evaluate_span_retrieval_passes():
    span = _span("r1", span_type="retrieval", output="retrieved document")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.passed is True
    assert score.checks["has_output"] is True


def test_evaluate_span_retrieval_no_output():
    span = _span("r2", span_type="retrieval")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.passed is False
    assert score.checks["has_output"] is False


# --- evaluate_span: default type ---

def test_evaluate_span_default_type_ok():
    span = _span("d1", span_type="custom")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.passed is True
    assert score.checks["no_error"] is True


def test_evaluate_span_default_type_error():
    span = _span("d2", span_type="custom", status="error")
    config = SpanEvalConfig()
    score = evaluate_span(span, config)
    assert score.passed is False
    assert score.checks["no_error"] is False


# --- evaluate_spans batch ---

def test_evaluate_spans_batch():
    spans = [
        _span("b1", span_type="llm_call", output="A long enough output for the check"),
        _span("b2", span_type="tool_call", output="result"),
        _span("b3", span_type="retrieval", output="doc"),
    ]
    report = evaluate_spans(spans)
    assert report.total_spans == 3
    assert report.passed_count + report.failed_count == 3


def test_evaluate_spans_default_config():
    spans = [_span("x1", span_type="custom")]
    report = evaluate_spans(spans)
    assert report.total_spans == 1
    assert report.pass_rate == 1.0


def test_evaluate_spans_with_config():
    spans = [
        _span("c1", span_type="llm_call", output="short"),
    ]
    config = SpanEvalConfig(min_output_length=100)
    report = evaluate_spans(spans, config)
    assert report.total_spans == 1


# --- find_failing_spans ---

def test_find_failing_spans():
    spans = [
        _span("f1", span_type="llm_call", output="Long enough output for check"),
        _span("f2", span_type="retrieval"),  # No output - will fail
    ]
    report = evaluate_spans(spans)
    failing = find_failing_spans(report)
    assert len(failing) >= 1
    fail_ids = [s.span_id for s in failing]
    assert "f2" in fail_ids


def test_find_failing_spans_none_failing():
    spans = [
        _span("p1", span_type="llm_call",
              output="A sufficiently long output for the check"),
    ]
    report = evaluate_spans(spans)
    failing = find_failing_spans(report)
    assert failing == []


# --- compute_type_breakdown ---

def test_compute_type_breakdown():
    scores = [
        SpanScore(span_id="a", span_name="a", span_type="llm_call", score=1.0, passed=True),
        SpanScore(span_id="b", span_name="b", span_type="llm_call", score=0.5, passed=True),
        SpanScore(span_id="c", span_name="c", span_type="tool_call", score=0.0, passed=False),
    ]
    breakdown = compute_type_breakdown(scores)
    assert "llm_call" in breakdown
    assert "tool_call" in breakdown
    assert breakdown["llm_call"]["count"] == 2
    assert breakdown["llm_call"]["passed"] == 2
    assert breakdown["tool_call"]["count"] == 1
    assert breakdown["tool_call"]["passed"] == 0


def test_compute_type_breakdown_empty():
    breakdown = compute_type_breakdown([])
    assert breakdown == {}


# --- SpanEvalReport format_text ---

def test_span_eval_report_format_text():
    spans = [
        _span("ft1", span_type="llm_call", output="Output that is long enough"),
        _span("ft2", span_type="tool_call", status="error"),
    ]
    report = evaluate_spans(spans)
    text = report.format_text()
    assert "Span Evaluation" in text
    assert "Passed" in text
    assert "llm_call" in text or "tool_call" in text


# --- Config as_dict / from_dict ---

def test_config_as_dict():
    config = SpanEvalConfig(max_latency_ms=3000.0, min_output_length=20)
    d = config.as_dict()
    assert d["max_latency_ms"] == 3000.0
    assert d["min_output_length"] == 20
    assert d["check_llm_quality"] is True


def test_config_from_dict():
    d = {
        "check_llm_quality": False,
        "check_tool_success": True,
        "check_latency": False,
        "max_latency_ms": 1000.0,
        "min_output_length": 5,
    }
    config = SpanEvalConfig.from_dict(d)
    assert config.check_llm_quality is False
    assert config.max_latency_ms == 1000.0
    assert config.min_output_length == 5


def test_config_roundtrip():
    original = SpanEvalConfig(
        check_llm_quality=False, check_latency=True,
        max_latency_ms=2000.0, min_output_length=15,
    )
    restored = SpanEvalConfig.from_dict(original.as_dict())
    assert restored.check_llm_quality == original.check_llm_quality
    assert restored.max_latency_ms == original.max_latency_ms


# --- Default config ---

def test_default_config_values():
    config = SpanEvalConfig()
    assert config.check_llm_quality is True
    assert config.check_tool_success is True
    assert config.check_latency is True
    assert config.max_latency_ms == 5000.0
    assert config.min_output_length == 10


# --- Edge: empty spans ---

def test_evaluate_spans_empty():
    report = evaluate_spans([])
    assert report.total_spans == 0
    assert report.passed_count == 0
    assert report.failed_count == 0
    assert report.pass_rate == 0.0
    assert report.by_type == {}


# --- by_type breakdown correct ---

def test_by_type_breakdown_in_report():
    spans = [
        _span("bt1", span_type="llm_call",
              output="A long enough output for check purposes"),
        _span("bt2", span_type="llm_call",
              output="Another long enough output for check"),
        _span("bt3", span_type="tool_call", output="result"),
        _span("bt4", span_type="retrieval", output="doc"),
    ]
    report = evaluate_spans(spans)
    assert "llm_call" in report.by_type
    assert "tool_call" in report.by_type
    assert "retrieval" in report.by_type
    assert report.by_type["llm_call"]["count"] == 2
    assert report.by_type["tool_call"]["count"] == 1
    assert report.by_type["retrieval"]["count"] == 1


# --- SpanEvalReport as_dict / from_dict ---

def test_span_eval_report_roundtrip():
    spans = [
        _span("rr1", span_type="llm_call",
              output="A long enough output for check purposes"),
        _span("rr2", span_type="tool_call", output="result"),
    ]
    report = evaluate_spans(spans)
    d = report.as_dict()
    restored = SpanEvalReport.from_dict(d)
    assert restored.total_spans == report.total_spans
    assert restored.passed_count == report.passed_count
    assert restored.pass_rate == report.pass_rate
    assert len(restored.span_scores) == len(report.span_scores)
