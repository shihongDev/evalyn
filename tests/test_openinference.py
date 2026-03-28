"""Tests for integration.openinference module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from evalyn_sdk.integration.openinference import (
    OPENINFERENCE_SPAN_KINDS,
    SPAN_TYPE_TO_OI,
    ComplianceReport,
    OpenInferenceSpan,
    OpenInferenceTrace,
    add_oi_attributes,
    check_compliance,
    check_trace_compliance,
    convert_span_to_oi,
    convert_trace_to_oi,
    export_as_openinference_json,
)
from evalyn_sdk.models import Span


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _span(sid, name="test", span_type="llm_call", duration_ms=100, **attrs):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        attributes=attrs,
    )


# ---------------------------------------------------------------------------
# convert_span_to_oi
# ---------------------------------------------------------------------------


class TestConvertSpanToOi:
    def test_basic(self):
        s = _span("s1", name="my-llm-call")
        oi = convert_span_to_oi(s)
        assert oi.name == "my-llm-call"
        assert oi.span_kind == "LLM"
        assert oi.span_id == "s1"
        assert oi.status_code == "OK"

    def test_span_kind_mapping_llm(self):
        oi = convert_span_to_oi(_span("s1", span_type="llm_call"))
        assert oi.span_kind == "LLM"

    def test_span_kind_mapping_tool(self):
        oi = convert_span_to_oi(_span("s1", span_type="tool_call"))
        assert oi.span_kind == "TOOL"

    def test_span_kind_mapping_retrieval(self):
        oi = convert_span_to_oi(_span("s1", span_type="retrieval"))
        assert oi.span_kind == "RETRIEVER"

    def test_span_kind_mapping_node(self):
        oi = convert_span_to_oi(_span("s1", span_type="node"))
        assert oi.span_kind == "CHAIN"

    def test_span_kind_mapping_agent(self):
        oi = convert_span_to_oi(_span("s1", span_type="agent"))
        assert oi.span_kind == "AGENT"

    def test_span_kind_mapping_embedding(self):
        oi = convert_span_to_oi(_span("s1", span_type="embedding"))
        assert oi.span_kind == "EMBEDDING"

    def test_span_kind_mapping_reranker(self):
        oi = convert_span_to_oi(_span("s1", span_type="reranker"))
        assert oi.span_kind == "RERANKER"

    def test_span_kind_mapping_scorer(self):
        oi = convert_span_to_oi(_span("s1", span_type="scorer"))
        assert oi.span_kind == "CHAIN"

    def test_span_kind_mapping_custom(self):
        oi = convert_span_to_oi(_span("s1", span_type="custom"))
        assert oi.span_kind == "CHAIN"

    def test_unknown_type_defaults_to_chain(self):
        oi = convert_span_to_oi(_span("s1", span_type="unknown_thing"))
        assert oi.span_kind == "CHAIN"

    def test_error_status(self):
        s = _span("s1")
        s.status = "error"
        oi = convert_span_to_oi(s)
        assert oi.status_code == "ERROR"

    def test_start_end_time_iso(self):
        s = _span("s1")
        oi = convert_span_to_oi(s)
        assert oi.start_time != ""
        assert oi.end_time != ""
        # Should be parseable ISO strings
        datetime.fromisoformat(oi.start_time)
        datetime.fromisoformat(oi.end_time)

    def test_attributes_include_oi_kind(self):
        oi = convert_span_to_oi(_span("s1", span_type="tool_call"))
        assert oi.attributes["openinference.span.kind"] == "TOOL"

    def test_parent_id_used_as_trace_id(self):
        s = _span("s1")
        s.parent_id = "root-1"
        oi = convert_span_to_oi(s)
        assert oi.trace_id == "root-1"
        assert oi.parent_id == "root-1"

    def test_no_parent_uses_own_id_as_trace_id(self):
        s = _span("s1")
        assert s.parent_id is None
        oi = convert_span_to_oi(s)
        assert oi.trace_id == "s1"


# ---------------------------------------------------------------------------
# convert_trace_to_oi
# ---------------------------------------------------------------------------


class TestConvertTraceToOi:
    def test_basic(self):
        spans = [_span("s1"), _span("s2")]
        trace = convert_trace_to_oi(spans)
        assert trace.trace_id == "s1"
        assert len(trace.spans) == 2
        assert trace.project_name == "evalyn"

    def test_custom_project_name(self):
        trace = convert_trace_to_oi([_span("s1")], project_name="my-project")
        assert trace.project_name == "my-project"

    def test_empty_spans(self):
        trace = convert_trace_to_oi([])
        assert trace.trace_id == ""
        assert trace.spans == []

    def test_consistent_trace_id(self):
        s1 = _span("s1")
        s2 = _span("s2")
        s2.parent_id = "s1"
        trace = convert_trace_to_oi([s1, s2])
        # All spans share the same trace_id
        assert all(s.trace_id == trace.trace_id for s in trace.spans)


# ---------------------------------------------------------------------------
# check_compliance
# ---------------------------------------------------------------------------


class TestCheckCompliance:
    def test_valid_span(self):
        oi = OpenInferenceSpan(
            name="test", span_kind="LLM", trace_id="t1", span_id="s1"
        )
        issues = check_compliance(oi)
        assert issues == []

    def test_invalid_span_kind(self):
        oi = OpenInferenceSpan(
            name="test", span_kind="INVALID", trace_id="t1", span_id="s1"
        )
        issues = check_compliance(oi)
        assert any("invalid span_kind" in i for i in issues)

    def test_missing_name(self):
        oi = OpenInferenceSpan(
            name="", span_kind="LLM", trace_id="t1", span_id="s1"
        )
        issues = check_compliance(oi)
        assert any("missing span name" in i for i in issues)

    def test_missing_span_id(self):
        oi = OpenInferenceSpan(
            name="test", span_kind="LLM", trace_id="t1", span_id=""
        )
        issues = check_compliance(oi)
        assert any("missing span_id" in i for i in issues)

    def test_missing_trace_id(self):
        oi = OpenInferenceSpan(
            name="test", span_kind="LLM", trace_id="", span_id="s1"
        )
        issues = check_compliance(oi)
        assert any("missing trace_id" in i for i in issues)

    def test_multiple_issues(self):
        oi = OpenInferenceSpan(
            name="", span_kind="BAD", trace_id="", span_id=""
        )
        issues = check_compliance(oi)
        assert len(issues) == 4


# ---------------------------------------------------------------------------
# check_trace_compliance
# ---------------------------------------------------------------------------


class TestCheckTraceCompliance:
    def test_all_compliant(self):
        trace = OpenInferenceTrace(
            trace_id="t1",
            spans=[
                OpenInferenceSpan(
                    name="a", span_kind="LLM", trace_id="t1", span_id="s1"
                ),
                OpenInferenceSpan(
                    name="b", span_kind="TOOL", trace_id="t1", span_id="s2"
                ),
            ],
        )
        report = check_trace_compliance(trace)
        assert report.total_spans == 2
        assert report.compliant_spans == 2
        assert report.compliance_rate == 1.0
        assert report.issues == []

    def test_partial_compliance(self):
        trace = OpenInferenceTrace(
            trace_id="t1",
            spans=[
                OpenInferenceSpan(
                    name="a", span_kind="LLM", trace_id="t1", span_id="s1"
                ),
                OpenInferenceSpan(
                    name="", span_kind="BAD", trace_id="t1", span_id="s2"
                ),
            ],
        )
        report = check_trace_compliance(trace)
        assert report.compliant_spans == 1
        assert report.compliance_rate == 0.5
        assert len(report.issues) > 0

    def test_empty_trace(self):
        trace = OpenInferenceTrace(trace_id="t1", spans=[])
        report = check_trace_compliance(trace)
        assert report.total_spans == 0
        assert report.compliance_rate == 0.0


# ---------------------------------------------------------------------------
# add_oi_attributes
# ---------------------------------------------------------------------------


class TestAddOiAttributes:
    def test_adds_keys(self):
        s = _span("s1", span_type="llm_call")
        result = add_oi_attributes(s)
        assert result.attributes["openinference.span.kind"] == "LLM"
        assert result.attributes["openinference.version"] == "1.0"

    def test_returns_new_span(self):
        s = _span("s1")
        result = add_oi_attributes(s)
        assert result is not s
        # Original should NOT have the new keys
        assert "openinference.span.kind" not in s.attributes

    def test_preserves_existing_attributes(self):
        s = _span("s1", foo="bar")
        result = add_oi_attributes(s)
        assert result.attributes["foo"] == "bar"


# ---------------------------------------------------------------------------
# export_as_openinference_json
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_valid_json(self):
        trace = convert_trace_to_oi([_span("s1")])
        raw = export_as_openinference_json(trace)
        data = json.loads(raw)
        assert "trace_id" in data
        assert "spans" in data
        assert len(data["spans"]) == 1

    def test_empty_trace(self):
        trace = OpenInferenceTrace(trace_id="t1", spans=[])
        raw = export_as_openinference_json(trace)
        data = json.loads(raw)
        assert data["spans"] == []


# ---------------------------------------------------------------------------
# Constants coverage
# ---------------------------------------------------------------------------


class TestConstants:
    def test_span_type_to_oi_covers_all_types(self):
        expected_keys = {
            "llm_call",
            "tool_call",
            "retrieval",
            "node",
            "agent",
            "embedding",
            "reranker",
            "scorer",
            "custom",
        }
        assert set(SPAN_TYPE_TO_OI.keys()) == expected_keys

    def test_all_mapped_values_are_valid(self):
        for val in SPAN_TYPE_TO_OI.values():
            assert val in OPENINFERENCE_SPAN_KINDS


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------


class TestDataclassRoundTrip:
    def test_oi_span_as_dict_from_dict(self):
        original = OpenInferenceSpan(
            name="test",
            span_kind="LLM",
            trace_id="t1",
            span_id="s1",
            parent_id="p1",
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-01T00:00:01+00:00",
            status_code="OK",
            attributes={"key": "val"},
        )
        d = original.as_dict()
        restored = OpenInferenceSpan.from_dict(d)
        assert restored.name == original.name
        assert restored.span_kind == original.span_kind
        assert restored.trace_id == original.trace_id
        assert restored.span_id == original.span_id
        assert restored.attributes == original.attributes

    def test_oi_trace_as_dict_from_dict(self):
        original = OpenInferenceTrace(
            trace_id="t1",
            spans=[
                OpenInferenceSpan(name="a", span_kind="LLM", span_id="s1"),
            ],
            project_name="proj",
        )
        d = original.as_dict()
        restored = OpenInferenceTrace.from_dict(d)
        assert restored.trace_id == "t1"
        assert len(restored.spans) == 1
        assert restored.project_name == "proj"


# ---------------------------------------------------------------------------
# ComplianceReport format_text
# ---------------------------------------------------------------------------


class TestComplianceReportFormat:
    def test_format_text_no_issues(self):
        report = ComplianceReport(
            total_spans=3, compliant_spans=3, compliance_rate=1.0
        )
        text = report.format_text()
        assert "Total spans: 3" in text
        assert "Compliant: 3" in text
        assert "100.0%" in text

    def test_format_text_with_issues(self):
        report = ComplianceReport(
            total_spans=2,
            compliant_spans=1,
            issues=["span s2: invalid span_kind: 'BAD'"],
            compliance_rate=0.5,
        )
        text = report.format_text()
        assert "Issues:" in text
        assert "invalid span_kind" in text
