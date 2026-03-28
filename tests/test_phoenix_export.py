"""Tests for integration.phoenix_export module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from evalyn_sdk.integration.phoenix_export import (
    ExportConfig,
    PhoenixSpan,
    PlatformExportResult,
    convert_to_langfuse_format,
    convert_to_langsmith_format,
    convert_to_phoenix_span,
    export_spans,
    format_as_langfuse_json,
    format_as_phoenix_json,
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
# PhoenixSpan
# ---------------------------------------------------------------------------


class TestPhoenixSpan:
    def test_defaults(self):
        s = PhoenixSpan("test")
        assert s.span_kind == "LLM"
        assert s.status == "OK"
        assert s.start_time_ns == 0

    def test_as_dict(self):
        s = PhoenixSpan("s", "CHAIN", {"trace_id": "t"}, {"k": "v"}, 100, 200, "OK")
        d = s.as_dict()
        assert d["name"] == "s"
        assert d["span_kind"] == "CHAIN"
        assert d["context"] == {"trace_id": "t"}
        assert d["attributes"] == {"k": "v"}
        assert d["start_time_ns"] == 100
        assert d["end_time_ns"] == 200

    def test_from_dict(self):
        d = {"name": "x", "span_kind": "TOOL", "status": "ERROR"}
        s = PhoenixSpan.from_dict(d)
        assert s.name == "x"
        assert s.span_kind == "TOOL"
        assert s.status == "ERROR"

    def test_from_dict_defaults(self):
        s = PhoenixSpan.from_dict({"name": "n"})
        assert s.span_kind == "LLM"
        assert s.status == "OK"
        assert s.context == {}

    def test_roundtrip(self):
        original = PhoenixSpan("rt", "CHAIN", {"a": "b"}, {"x": 1}, 10, 20, "OK")
        restored = PhoenixSpan.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.span_kind == original.span_kind
        assert restored.context == original.context
        assert restored.attributes == original.attributes
        assert restored.start_time_ns == original.start_time_ns


# ---------------------------------------------------------------------------
# ExportConfig
# ---------------------------------------------------------------------------


class TestExportConfig:
    def test_defaults(self):
        c = ExportConfig()
        assert c.platform == "phoenix"
        assert c.project_name == "evalyn"
        assert c.include_inputs is True
        assert c.include_outputs is True

    def test_as_dict(self):
        c = ExportConfig("langfuse", "proj", False, True)
        d = c.as_dict()
        assert d["platform"] == "langfuse"
        assert d["include_inputs"] is False

    def test_from_dict(self):
        d = {"platform": "langsmith", "project_name": "test", "include_inputs": False}
        c = ExportConfig.from_dict(d)
        assert c.platform == "langsmith"
        assert c.project_name == "test"
        assert c.include_inputs is False
        assert c.include_outputs is True

    def test_roundtrip(self):
        original = ExportConfig("langfuse", "p", False, False)
        restored = ExportConfig.from_dict(original.as_dict())
        assert restored.platform == original.platform
        assert restored.project_name == original.project_name
        assert restored.include_inputs == original.include_inputs


# ---------------------------------------------------------------------------
# PlatformExportResult
# ---------------------------------------------------------------------------


class TestPlatformExportResult:
    def test_as_dict(self):
        r = PlatformExportResult("phoenix", 10, "phoenix_json", "/tmp/out.json")
        d = r.as_dict()
        assert d["spans_exported"] == 10
        assert d["file_path"] == "/tmp/out.json"

    def test_format_text(self):
        r = PlatformExportResult("phoenix", 5, "phoenix_json", "/tmp/x.json")
        text = r.format_text()
        assert "phoenix" in text
        assert "5" in text
        assert "/tmp/x.json" in text

    def test_format_text_no_path(self):
        r = PlatformExportResult("langfuse", 0, "langfuse_json")
        text = r.format_text()
        assert "langfuse" in text
        assert "File:" not in text


# ---------------------------------------------------------------------------
# convert_to_phoenix_span
# ---------------------------------------------------------------------------


class TestConvertToPhoenixSpan:
    def test_basic(self):
        s = _span("s1", "call_llm", "llm_call")
        ps = convert_to_phoenix_span(s)
        assert ps.name == "call_llm"
        assert ps.span_kind == "LLM"
        assert ps.status == "OK"
        assert ps.context["span_id"] == "s1"

    def test_tool_call_kind(self):
        s = _span("s2", "run_tool", "tool_call")
        ps = convert_to_phoenix_span(s)
        assert ps.span_kind == "TOOL"

    def test_retrieval_kind(self):
        s = _span("s3", "fetch", "retrieval")
        ps = convert_to_phoenix_span(s)
        assert ps.span_kind == "RETRIEVER"

    def test_error_status(self):
        s = _span("s4", "fail", "llm_call")
        s.status = "error"
        ps = convert_to_phoenix_span(s)
        assert ps.status == "ERROR"

    def test_attributes_preserved(self):
        s = _span("s5", "t", "llm_call", model="gpt-4")
        ps = convert_to_phoenix_span(s)
        assert ps.attributes["model"] == "gpt-4"
        assert ps.attributes["evalyn.span_type"] == "llm_call"

    def test_timestamps_nonzero(self):
        s = _span("s6", "t", "llm_call")
        ps = convert_to_phoenix_span(s)
        assert ps.start_time_ns > 0
        assert ps.end_time_ns > ps.start_time_ns


# ---------------------------------------------------------------------------
# convert_to_langfuse_format
# ---------------------------------------------------------------------------


class TestConvertToLangfuseFormat:
    def test_basic(self):
        s = _span("lf1", "gen", "llm_call")
        d = convert_to_langfuse_format(s)
        assert d["id"] == "lf1"
        assert d["type"] == "generation"
        assert d["level"] == "DEFAULT"

    def test_non_llm_type(self):
        s = _span("lf2", "chain", "graph")
        d = convert_to_langfuse_format(s)
        assert d["type"] == "span"

    def test_parent_id(self):
        s = _span("lf3", "child", "llm_call")
        s.parent_id = "parent1"
        d = convert_to_langfuse_format(s)
        assert d["parentObservationId"] == "parent1"
        assert d["traceId"] == "parent1"

    def test_no_parent(self):
        s = _span("lf4", "root", "llm_call")
        d = convert_to_langfuse_format(s)
        assert "parentObservationId" not in d
        assert d["traceId"] == "lf4"


# ---------------------------------------------------------------------------
# convert_to_langsmith_format
# ---------------------------------------------------------------------------


class TestConvertToLangsmithFormat:
    def test_basic(self):
        s = _span("ls1", "run", "llm_call")
        d = convert_to_langsmith_format(s)
        assert d["id"] == "ls1"
        assert d["run_type"] == "llm"
        assert d["error"] is None

    def test_chain_type(self):
        s = _span("ls2", "chain", "graph")
        d = convert_to_langsmith_format(s)
        assert d["run_type"] == "chain"

    def test_error(self):
        s = _span("ls3", "fail", "llm_call")
        s.status = "error"
        d = convert_to_langsmith_format(s)
        assert d["error"] == "error"

    def test_parent_run_id(self):
        s = _span("ls4", "child", "llm_call")
        s.parent_id = "p1"
        d = convert_to_langsmith_format(s)
        assert d["parent_run_id"] == "p1"


# ---------------------------------------------------------------------------
# export_spans
# ---------------------------------------------------------------------------


class TestExportSpans:
    def test_phoenix_format(self):
        spans = [_span("e1"), _span("e2")]
        config = ExportConfig(platform="phoenix")
        result = export_spans(spans, config)
        assert result.platform == "phoenix"
        assert result.spans_exported == 2
        assert result.format == "phoenix_json"

    def test_langfuse_format(self):
        spans = [_span("e3")]
        config = ExportConfig(platform="langfuse")
        result = export_spans(spans, config)
        assert result.platform == "langfuse"
        assert result.spans_exported == 1
        assert result.format == "langfuse_json"

    def test_langsmith_format(self):
        spans = [_span("e4")]
        config = ExportConfig(platform="langsmith")
        result = export_spans(spans, config)
        assert result.platform == "langsmith"
        assert result.format == "langsmith_json"

    def test_unsupported_platform(self):
        with pytest.raises(ValueError):
            export_spans([], ExportConfig(platform="unknown"))

    def test_empty_spans(self):
        result = export_spans([], ExportConfig(platform="phoenix"))
        assert result.spans_exported == 0


# ---------------------------------------------------------------------------
# JSON formatting
# ---------------------------------------------------------------------------


class TestFormatAsPhoenixJson:
    def test_valid_json(self):
        ps = [PhoenixSpan("a"), PhoenixSpan("b")]
        raw = format_as_phoenix_json(ps)
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "a"

    def test_empty(self):
        raw = format_as_phoenix_json([])
        assert json.loads(raw) == []


class TestFormatAsLangfuseJson:
    def test_valid_json(self):
        spans = [_span("j1"), _span("j2")]
        raw = format_as_langfuse_json(spans)
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "j1"

    def test_empty(self):
        raw = format_as_langfuse_json([])
        assert json.loads(raw) == []
