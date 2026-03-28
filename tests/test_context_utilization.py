"""Tests for context_utilization analysis module."""

from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.analysis.context_utilization import (
    DEFAULT_CONTEXT_LIMIT,
    MODEL_CONTEXT_LIMITS,
    ContextAlert,
    ContextReport,
    analyze_context_utilization,
    check_context_utilization,
    get_context_limit,
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


# -- get_context_limit --


class TestGetContextLimit:
    def test_known_model_gpt4o(self):
        assert get_context_limit("gpt-4o") == 128000

    def test_known_model_gemini_flash(self):
        assert get_context_limit("gemini-2.5-flash") == 1048576

    def test_known_model_claude(self):
        assert get_context_limit("claude-sonnet-4-20250514") == 200000

    def test_unknown_model_returns_default(self):
        assert get_context_limit("unknown-model-xyz") == DEFAULT_CONTEXT_LIMIT

    def test_empty_string_returns_default(self):
        assert get_context_limit("") == DEFAULT_CONTEXT_LIMIT


# -- check_context_utilization --


class TestCheckContextUtilization:
    def test_below_threshold_no_alert(self):
        span = _span("1", model="gpt-4o", total_tokens=50000)
        alert = check_context_utilization(span, threshold_pct=80.0)
        assert alert is None

    def test_above_threshold_warning(self):
        # 110000 / 128000 = 85.9%
        span = _span("1", model="gpt-4o", total_tokens=110000)
        alert = check_context_utilization(span, threshold_pct=80.0)
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.span_id == "1"
        assert alert.model == "gpt-4o"

    def test_above_95_critical(self):
        # 125000 / 128000 = 97.7%
        span = _span("1", model="gpt-4o", total_tokens=125000)
        alert = check_context_utilization(span, threshold_pct=80.0)
        assert alert is not None
        assert alert.severity == "critical"

    def test_exactly_at_threshold_no_alert(self):
        # 80% of 128000 = 102400
        span = _span("1", model="gpt-4o", total_tokens=102400)
        alert = check_context_utilization(span, threshold_pct=80.0)
        assert alert is None

    def test_no_tokens_no_alert(self):
        span = _span("1", model="gpt-4o", total_tokens=0)
        alert = check_context_utilization(span)
        assert alert is None

    def test_no_model_attribute_uses_default(self):
        # total_tokens high relative to DEFAULT_CONTEXT_LIMIT
        high_tokens = int(DEFAULT_CONTEXT_LIMIT * 0.9)
        span = _span("1", total_tokens=high_tokens)
        alert = check_context_utilization(span, threshold_pct=80.0)
        assert alert is not None
        assert alert.context_limit == DEFAULT_CONTEXT_LIMIT

    def test_no_total_tokens_attribute(self):
        span = _span("1", model="gpt-4o")
        alert = check_context_utilization(span)
        assert alert is None

    def test_custom_threshold(self):
        # 70000 / 128000 = 54.7% - below 80 but above 50
        span = _span("1", model="gpt-4o", total_tokens=70000)
        assert check_context_utilization(span, threshold_pct=80.0) is None
        assert check_context_utilization(span, threshold_pct=50.0) is not None


# -- ContextAlert --


class TestContextAlert:
    def test_as_dict(self):
        alert = ContextAlert(
            span_id="s1",
            model="gpt-4o",
            tokens_used=110000,
            context_limit=128000,
            utilization_pct=85.94,
            severity="warning",
        )
        d = alert.as_dict()
        assert d["span_id"] == "s1"
        assert d["model"] == "gpt-4o"
        assert d["tokens_used"] == 110000
        assert d["context_limit"] == 128000
        assert d["utilization_pct"] == 85.94
        assert d["severity"] == "warning"


# -- ContextReport --


class TestContextReport:
    def test_defaults(self):
        r = ContextReport()
        assert r.alerts == []
        assert r.max_utilization_pct == 0.0
        assert r.mean_utilization_pct == 0.0
        assert r.models_at_risk == []
        assert r.recommendation == ""

    def test_as_dict(self):
        alert = ContextAlert(
            span_id="s1", model="gpt-4o",
            tokens_used=110000, context_limit=128000,
            utilization_pct=85.9, severity="warning",
        )
        r = ContextReport(
            alerts=[alert],
            max_utilization_pct=85.9,
            mean_utilization_pct=60.0,
            models_at_risk=["gpt-4o"],
            recommendation="Upgrade model.",
        )
        d = r.as_dict()
        assert len(d["alerts"]) == 1
        assert d["max_utilization_pct"] == 85.9
        assert d["models_at_risk"] == ["gpt-4o"]

    def test_from_dict_roundtrip(self):
        alert = ContextAlert(
            span_id="s1", model="gpt-4o",
            tokens_used=110000, context_limit=128000,
            utilization_pct=85.94, severity="warning",
        )
        r = ContextReport(
            alerts=[alert],
            max_utilization_pct=85.94,
            mean_utilization_pct=42.0,
            models_at_risk=["gpt-4o"],
            recommendation="Consider larger context.",
        )
        d = r.as_dict()
        restored = ContextReport.from_dict(d)
        assert len(restored.alerts) == 1
        assert restored.alerts[0].span_id == "s1"
        assert restored.max_utilization_pct == 85.94
        assert restored.recommendation == "Consider larger context."

    def test_from_dict_empty(self):
        r = ContextReport.from_dict({})
        assert r.alerts == []
        assert r.max_utilization_pct == 0.0

    def test_format_text_basic(self):
        r = ContextReport(
            max_utilization_pct=85.0,
            mean_utilization_pct=60.0,
        )
        text = r.format_text()
        assert "Context Window Utilization" in text
        assert "Max utilization" in text
        assert "Mean utilization" in text

    def test_format_text_with_alerts(self):
        alert = ContextAlert(
            span_id="s1", model="gpt-4o",
            tokens_used=110000, context_limit=128000,
            utilization_pct=85.9, severity="warning",
        )
        r = ContextReport(
            alerts=[alert],
            max_utilization_pct=85.9,
            mean_utilization_pct=85.9,
            models_at_risk=["gpt-4o"],
            recommendation="Upgrade model.",
        )
        text = r.format_text()
        assert "WARNING" in text
        assert "gpt-4o" in text
        assert "Recommendation" in text

    def test_format_text_with_models_at_risk(self):
        r = ContextReport(
            models_at_risk=["gpt-4o", "claude-sonnet-4-20250514"],
            max_utilization_pct=90.0,
            mean_utilization_pct=80.0,
        )
        text = r.format_text()
        assert "Models at risk" in text
        assert "gpt-4o" in text


# -- analyze_context_utilization --


class TestAnalyzeContextUtilization:
    def test_no_spans(self):
        r = analyze_context_utilization([])
        assert r.alerts == []
        assert r.max_utilization_pct == 0.0
        assert r.mean_utilization_pct == 0.0

    def test_all_below_threshold(self):
        spans = [
            _span("1", model="gpt-4o", total_tokens=10000),
            _span("2", model="gpt-4o", total_tokens=20000),
        ]
        r = analyze_context_utilization(spans, threshold_pct=80.0)
        assert r.alerts == []
        assert r.max_utilization_pct > 0
        assert r.models_at_risk == []

    def test_some_above_threshold(self):
        spans = [
            _span("1", model="gpt-4o", total_tokens=10000),
            _span("2", model="gpt-4o", total_tokens=110000),
        ]
        r = analyze_context_utilization(spans, threshold_pct=80.0)
        assert len(r.alerts) == 1
        assert r.alerts[0].span_id == "2"

    def test_models_at_risk(self):
        spans = [
            _span("1", model="gpt-4o", total_tokens=110000),
            _span("2", model="gpt-4o-mini", total_tokens=115000),
        ]
        r = analyze_context_utilization(spans, threshold_pct=80.0)
        assert "gpt-4o" in r.models_at_risk
        assert "gpt-4o-mini" in r.models_at_risk

    def test_recommendation_when_mean_above_70(self):
        # All spans at ~90% of 128000
        spans = [
            _span("1", model="gpt-4o", total_tokens=115000),
            _span("2", model="gpt-4o", total_tokens=115000),
        ]
        r = analyze_context_utilization(spans, threshold_pct=80.0)
        assert "larger context window" in r.recommendation

    def test_no_recommendation_when_mean_below_70(self):
        spans = [
            _span("1", model="gpt-4o", total_tokens=50000),
            _span("2", model="gpt-4o", total_tokens=50000),
        ]
        r = analyze_context_utilization(spans, threshold_pct=80.0)
        assert r.recommendation == ""

    def test_spans_with_no_tokens_skipped(self):
        spans = [
            _span("1", model="gpt-4o", total_tokens=0),
            _span("2", model="gpt-4o"),
        ]
        r = analyze_context_utilization(spans)
        assert r.alerts == []
        assert r.max_utilization_pct == 0.0
        assert r.mean_utilization_pct == 0.0

    def test_mixed_models(self):
        spans = [
            _span("1", model="gemini-2.5-flash", total_tokens=950000),  # ~90.6% of 1048576
            _span("2", model="gpt-4o", total_tokens=110000),  # ~85.9% of 128000
        ]
        r = analyze_context_utilization(spans, threshold_pct=80.0)
        assert len(r.alerts) == 2

    def test_max_utilization(self):
        spans = [
            _span("1", model="gpt-4o", total_tokens=50000),
            _span("2", model="gpt-4o", total_tokens=120000),
        ]
        r = analyze_context_utilization(spans, threshold_pct=80.0)
        # 120000 / 128000 = 93.75%
        assert abs(r.max_utilization_pct - 93.75) < 0.1
