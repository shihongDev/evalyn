from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.health_check import (
    HealthCheckItem,
    HealthCheckReport,
    check_spans_captured,
    check_span_types_present,
    check_no_orphans,
    check_timestamps_valid,
    check_attributes_present,
    run_health_check,
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


# --- HealthCheckItem ---


class TestHealthCheckItem:
    def test_as_dict(self):
        item = HealthCheckItem(name="test", passed=True, details="ok")
        d = item.as_dict()
        assert d == {"name": "test", "passed": True, "details": "ok"}

    def test_as_dict_no_details(self):
        item = HealthCheckItem(name="test", passed=False)
        d = item.as_dict()
        assert d["details"] == ""


# --- check_spans_captured ---


class TestCheckSpansCaptured:
    def test_passes_with_spans(self):
        spans = [_span("a"), _span("b")]
        result = check_spans_captured(spans)
        assert result.passed is True
        assert result.name == "spans_captured"

    def test_fails_when_empty(self):
        result = check_spans_captured([])
        assert result.passed is False

    def test_fails_below_min(self):
        result = check_spans_captured([_span("a")], min_expected=3)
        assert result.passed is False

    def test_passes_exact_min(self):
        result = check_spans_captured([_span("a"), _span("b")], min_expected=2)
        assert result.passed is True

    def test_details_contain_count(self):
        result = check_spans_captured([_span("a")])
        assert "1" in result.details


# --- check_span_types_present ---


class TestCheckSpanTypesPresent:
    def test_passes_with_matching_type(self):
        spans = [_span("a", span_type="llm_call")]
        result = check_span_types_present(spans)
        assert result.passed is True

    def test_fails_missing_type(self):
        spans = [_span("a", span_type="custom")]
        result = check_span_types_present(spans, required_types=["llm_call"])
        assert result.passed is False
        assert "llm_call" in result.details

    def test_passes_multiple_types(self):
        spans = [_span("a", span_type="llm_call"), _span("b", span_type="tool_call")]
        result = check_span_types_present(
            spans, required_types=["llm_call", "tool_call"]
        )
        assert result.passed is True

    def test_fails_partial_types(self):
        spans = [_span("a", span_type="llm_call")]
        result = check_span_types_present(
            spans, required_types=["llm_call", "tool_call"]
        )
        assert result.passed is False


# --- check_no_orphans ---


class TestCheckNoOrphans:
    def test_passes_no_orphans(self):
        parent = _span("p")
        child = _span("c")
        child.parent_id = "p"
        result = check_no_orphans([parent, child])
        assert result.passed is True

    def test_passes_no_parents(self):
        result = check_no_orphans([_span("a"), _span("b")])
        assert result.passed is True

    def test_fails_with_orphans(self):
        orphan = _span("c")
        orphan.parent_id = "nonexistent"
        result = check_no_orphans([_span("a"), orphan])
        assert result.passed is False
        assert "1" in result.details

    def test_empty_spans(self):
        result = check_no_orphans([])
        assert result.passed is True


# --- check_timestamps_valid ---


class TestCheckTimestampsValid:
    def test_passes_valid_timestamps(self):
        spans = [_span("a"), _span("b")]
        result = check_timestamps_valid(spans)
        assert result.passed is True

    def test_fails_reversed_time(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        s = Span(
            id="bad",
            name="bad-span",
            span_type="llm_call",
            parent_id=None,
            start_time=t,
            end_time=t - timedelta(seconds=1),
            status="ok",
            attributes={},
        )
        result = check_timestamps_valid([s])
        assert result.passed is False

    def test_passes_no_end_time(self):
        s = _span("a")
        s.end_time = None
        result = check_timestamps_valid([s])
        assert result.passed is True

    def test_empty_spans(self):
        result = check_timestamps_valid([])
        assert result.passed is True


# --- check_attributes_present ---


class TestCheckAttributesPresent:
    def test_passes_no_required(self):
        result = check_attributes_present([_span("a")])
        assert result.passed is True

    def test_passes_with_required(self):
        s = _span("a", model="gpt-4")
        result = check_attributes_present([s], required_keys=["model"])
        assert result.passed is True

    def test_fails_missing_key(self):
        s = _span("a")
        result = check_attributes_present([s], required_keys=["model"])
        assert result.passed is False

    def test_empty_required_keys(self):
        result = check_attributes_present([_span("a")], required_keys=[])
        assert result.passed is True


# --- run_health_check ---


class TestRunHealthCheck:
    def test_full_pipeline_passes(self):
        spans = [_span("a", span_type="llm_call", model="gpt-4")]
        report = run_health_check(
            spans, required_types=["llm_call"], required_attributes=["model"]
        )
        assert report.all_passed is True
        assert report.failed_count == 0
        assert report.passed_count == 5

    def test_full_pipeline_with_failures(self):
        report = run_health_check([], required_types=["llm_call"])
        assert report.all_passed is False
        assert report.failed_count > 0

    def test_default_required_types(self):
        spans = [_span("a", span_type="llm_call")]
        report = run_health_check(spans)
        assert report.all_passed is True

    def test_check_count(self):
        report = run_health_check([_span("a")])
        assert len(report.checks) == 5


# --- HealthCheckReport ---


class TestHealthCheckReport:
    def test_format_text_all_pass(self):
        report = run_health_check([_span("a", span_type="llm_call")])
        text = report.format_text()
        assert "PASS" in text
        assert "All checks passed" in text

    def test_format_text_with_failure(self):
        report = run_health_check([])
        text = report.format_text()
        assert "FAIL" in text
        assert "failed" in text

    def test_as_dict_roundtrip(self):
        report = run_health_check([_span("a", span_type="llm_call")])
        d = report.as_dict()
        restored = HealthCheckReport.from_dict(d)
        assert restored.all_passed == report.all_passed
        assert restored.passed_count == report.passed_count
        assert restored.failed_count == report.failed_count
        assert len(restored.checks) == len(report.checks)

    def test_from_dict_empty(self):
        report = HealthCheckReport.from_dict({})
        assert report.checks == []
        assert report.all_passed is True


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_spans_all_checks(self):
        report = run_health_check([])
        # spans_captured fails, span_types_present fails on empty set
        assert report.failed_count >= 1

    def test_single_root_span(self):
        report = run_health_check([_span("root", span_type="llm_call")])
        assert report.all_passed is True
