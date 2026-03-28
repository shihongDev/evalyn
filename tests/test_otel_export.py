from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.otel_export import (
    ExportConfig,
    ExportResult,
    OTelSpan,
    convert_span_to_otel,
    convert_spans_to_otel,
    export_to_file,
    format_as_jaeger_json,
    format_as_otlp_json,
    format_as_zipkin_json,
)


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


# -- OTelSpan -----------------------------------------------------------------


class TestOTelSpan:
    def test_as_dict(self):
        s = OTelSpan(trace_id="t1", span_id="s1", operation_name="op")
        d = s.as_dict()
        assert d["trace_id"] == "t1"
        assert d["span_id"] == "s1"
        assert d["operation_name"] == "op"
        assert d["service_name"] == "evalyn"
        assert d["status"] == "ok"

    def test_from_dict(self):
        data = {
            "trace_id": "t2",
            "span_id": "s2",
            "parent_span_id": "p1",
            "operation_name": "call",
            "start_time_unix_ns": 1000,
            "duration_ns": 500,
            "service_name": "myservice",
            "tags": {"k": "v"},
            "status": "error",
        }
        s = OTelSpan.from_dict(data)
        assert s.trace_id == "t2"
        assert s.parent_span_id == "p1"
        assert s.start_time_unix_ns == 1000
        assert s.duration_ns == 500
        assert s.tags == {"k": "v"}
        assert s.status == "error"

    def test_from_dict_defaults(self):
        s = OTelSpan.from_dict({})
        assert s.trace_id == ""
        assert s.service_name == "evalyn"
        assert s.tags == {}

    def test_roundtrip(self):
        original = OTelSpan(
            trace_id="t", span_id="s", parent_span_id="p",
            operation_name="op", start_time_unix_ns=999, duration_ns=111,
            service_name="svc", tags={"a": "b"}, status="error",
        )
        rebuilt = OTelSpan.from_dict(original.as_dict())
        assert rebuilt.as_dict() == original.as_dict()


# -- ExportConfig --------------------------------------------------------------


class TestExportConfig:
    def test_as_dict(self):
        c = ExportConfig(format="jaeger", endpoint="http://localhost:14268")
        d = c.as_dict()
        assert d["format"] == "jaeger"
        assert d["endpoint"] == "http://localhost:14268"

    def test_from_dict(self):
        c = ExportConfig.from_dict({"format": "zipkin", "headers": {"auth": "tok"}})
        assert c.format == "zipkin"
        assert c.headers == {"auth": "tok"}

    def test_from_dict_defaults(self):
        c = ExportConfig.from_dict({})
        assert c.format == "otlp"
        assert c.service_name == "evalyn"


# -- ExportResult --------------------------------------------------------------


class TestExportResult:
    def test_as_dict(self):
        r = ExportResult(exported_count=5, failed_count=1, format="otlp", endpoint="/tmp/out.json")
        d = r.as_dict()
        assert d["exported_count"] == 5
        assert d["failed_count"] == 1

    def test_format_text(self):
        r = ExportResult(exported_count=3, failed_count=0, format="jaeger", endpoint="http://j")
        text = r.format_text()
        assert "Exported: 3" in text
        assert "Failed: 0" in text
        assert "Format: jaeger" in text
        assert "Endpoint: http://j" in text

    def test_format_text_no_endpoint(self):
        r = ExportResult(exported_count=1, format="otlp")
        text = r.format_text()
        assert "Endpoint" not in text


# -- convert_span_to_otel ------------------------------------------------------


class TestConvertSpanToOtel:
    def test_basic(self):
        s = _span("s1", name="generate")
        otel = convert_span_to_otel(s)
        assert otel.span_id == "s1"
        assert otel.operation_name == "generate"
        assert otel.service_name == "evalyn"
        assert otel.status == "ok"

    def test_timestamps(self):
        s = _span("s2", duration_ms=200)
        otel = convert_span_to_otel(s)
        assert otel.start_time_unix_ns > 0
        assert otel.duration_ns == 200 * 1_000_000

    def test_custom_service_name(self):
        s = _span("s3")
        otel = convert_span_to_otel(s, service_name="myapp")
        assert otel.service_name == "myapp"

    def test_attributes_mapped_to_tags(self):
        s = _span("s4", model="gemini-2.5-flash", temperature="0.7")
        otel = convert_span_to_otel(s)
        assert otel.tags["model"] == "gemini-2.5-flash"
        assert otel.tags["temperature"] == "0.7"
        assert otel.tags["span_type"] == "llm_call"

    def test_parent_id(self):
        s = _span("s5")
        s.parent_id = "parent1"
        otel = convert_span_to_otel(s)
        assert otel.parent_span_id == "parent1"

    def test_no_parent(self):
        s = _span("s6")
        otel = convert_span_to_otel(s)
        assert otel.parent_span_id == ""

    def test_no_end_time(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        s = Span(id="s7", name="running", span_type="llm_call", parent_id=None, start_time=t)
        otel = convert_span_to_otel(s)
        assert otel.duration_ns == 0


# -- convert_spans_to_otel (batch) ---------------------------------------------


class TestConvertSpansBatch:
    def test_batch(self):
        spans = [_span("a"), _span("b"), _span("c")]
        result = convert_spans_to_otel(spans)
        assert len(result) == 3
        assert result[0].span_id == "a"
        assert result[2].span_id == "c"

    def test_empty(self):
        assert convert_spans_to_otel([]) == []


# -- format_as_otlp_json -------------------------------------------------------


class TestFormatOtlpJson:
    def test_valid_json(self):
        otel = [convert_span_to_otel(_span("x"))]
        text = format_as_otlp_json(otel)
        parsed = json.loads(text)
        assert "resourceSpans" in parsed
        assert len(parsed["resourceSpans"]) == 1

    def test_empty(self):
        text = format_as_otlp_json([])
        parsed = json.loads(text)
        assert parsed["resourceSpans"] == []

    def test_span_fields(self):
        otel = [convert_span_to_otel(_span("f1", name="fetch"))]
        parsed = json.loads(format_as_otlp_json(otel))
        span_data = parsed["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert span_data["name"] == "fetch"
        assert span_data["spanId"] == "f1"


# -- format_as_jaeger_json -----------------------------------------------------


class TestFormatJaegerJson:
    def test_valid_json(self):
        otel = [convert_span_to_otel(_span("j1"))]
        text = format_as_jaeger_json(otel)
        parsed = json.loads(text)
        assert "data" in parsed
        assert len(parsed["data"]) == 1

    def test_empty(self):
        parsed = json.loads(format_as_jaeger_json([]))
        assert parsed["data"] == []

    def test_span_fields(self):
        otel = [convert_span_to_otel(_span("j2", name="reason"))]
        parsed = json.loads(format_as_jaeger_json(otel))
        jaeger_span = parsed["data"][0]["spans"][0]
        assert jaeger_span["operationName"] == "reason"
        assert jaeger_span["spanID"] == "j2"


# -- format_as_zipkin_json -----------------------------------------------------


class TestFormatZipkinJson:
    def test_valid_json_array(self):
        otel = [convert_span_to_otel(_span("z1"))]
        text = format_as_zipkin_json(otel)
        parsed = json.loads(text)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_empty(self):
        parsed = json.loads(format_as_zipkin_json([]))
        assert parsed == []

    def test_span_fields(self):
        otel = [convert_span_to_otel(_span("z2", name="query"))]
        parsed = json.loads(format_as_zipkin_json(otel))
        assert parsed[0]["name"] == "query"
        assert parsed[0]["id"] == "z2"

    def test_parent_id_present(self):
        s = _span("z3")
        s.parent_id = "zp"
        otel = [convert_span_to_otel(s)]
        parsed = json.loads(format_as_zipkin_json(otel))
        assert parsed[0]["parentId"] == "zp"

    def test_parent_id_absent(self):
        otel = [convert_span_to_otel(_span("z4"))]
        parsed = json.loads(format_as_zipkin_json(otel))
        assert "parentId" not in parsed[0]


# -- export_to_file ------------------------------------------------------------


class TestExportToFile:
    def test_otlp(self, tmp_path):
        otel = [convert_span_to_otel(_span("e1"))]
        path = str(tmp_path / "out.json")
        result = export_to_file(otel, path, format="otlp")
        assert result.exported_count == 1
        assert result.failed_count == 0
        assert result.format == "otlp"
        parsed = json.loads(open(path).read())
        assert "resourceSpans" in parsed

    def test_jaeger(self, tmp_path):
        otel = [convert_span_to_otel(_span("e2"))]
        path = str(tmp_path / "jaeger.json")
        result = export_to_file(otel, path, format="jaeger")
        assert result.exported_count == 1
        parsed = json.loads(open(path).read())
        assert "data" in parsed

    def test_zipkin(self, tmp_path):
        otel = [convert_span_to_otel(_span("e3"))]
        path = str(tmp_path / "zipkin.json")
        result = export_to_file(otel, path, format="zipkin")
        assert result.exported_count == 1
        parsed = json.loads(open(path).read())
        assert isinstance(parsed, list)

    def test_unknown_format(self, tmp_path):
        otel = [convert_span_to_otel(_span("e4"))]
        path = str(tmp_path / "bad.json")
        result = export_to_file(otel, path, format="unknown")
        assert result.exported_count == 0
        assert result.failed_count == 1
        assert not os.path.exists(path)

    def test_empty_spans(self, tmp_path):
        path = str(tmp_path / "empty.json")
        result = export_to_file([], path, format="otlp")
        assert result.exported_count == 0
        assert result.failed_count == 0
