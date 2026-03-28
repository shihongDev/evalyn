from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.storage.denormalized import (
    DenormalizationConfig,
    DenormalizationReport,
    FlattenedRecord,
    build_denormalization_report,
    export_flat_csv,
    flatten_span,
    flatten_trace,
    query_flat_records,
)


def _span(sid, name="s", span_type="llm_call", parent_id=None, duration_ms=100, **attrs):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        attributes=attrs,
    )


# ---------------------------------------------------------------------------
# flatten_span
# ---------------------------------------------------------------------------


class TestFlattenSpan:
    def test_basic(self):
        s = _span("s1", name="call_llm", span_type="llm_call")
        rec = flatten_span(s)
        assert rec.span_id == "s1"
        assert rec.span_name == "call_llm"
        assert rec.span_type == "llm_call"
        assert rec.depth == 0
        assert rec.parent_name == ""
        assert rec.status == "ok"
        assert rec.duration_ms == 100.0

    def test_with_parent_and_depth(self):
        s = _span("s2", name="child")
        rec = flatten_span(s, parent_name="parent_span", depth=3)
        assert rec.parent_name == "parent_span"
        assert rec.depth == 3

    def test_truncates_text(self):
        long_text = "x" * 1000
        s = _span("s3", input=long_text, output=long_text)
        cfg = DenormalizationConfig(max_text_length=50)
        rec = flatten_span(s, config=cfg)
        assert len(rec.input_text) == 50
        assert len(rec.output_text) == 50

    def test_excludes_input_output(self):
        s = _span("s4", input="hello", output="world")
        cfg = DenormalizationConfig(include_input=False, include_output=False)
        rec = flatten_span(s, config=cfg)
        assert rec.input_text == ""
        assert rec.output_text == ""

    def test_extracts_cost_tokens(self):
        s = _span("s5", cost=0.05, tokens=150)
        rec = flatten_span(s)
        assert rec.cost == 0.05
        assert rec.tokens == 150

    def test_record_id_format(self):
        s = _span("myspan")
        rec = flatten_span(s)
        assert rec.record_id == "myspan-flat"

    def test_default_config(self):
        s = _span("s6", input="hi")
        rec = flatten_span(s)
        assert rec.input_text == "hi"


# ---------------------------------------------------------------------------
# flatten_trace
# ---------------------------------------------------------------------------


class TestFlattenTrace:
    def test_preserves_hierarchy(self):
        root = _span("r", name="root", span_type="session")
        child = _span("c", name="child", span_type="llm_call", parent_id="r")
        records = flatten_trace([root, child])
        assert len(records) == 2
        assert records[0].span_name == "root"
        assert records[0].depth == 0
        assert records[1].span_name == "child"
        assert records[1].parent_name == "root"

    def test_computes_depth(self):
        r = _span("r", name="root")
        c1 = _span("c1", name="mid", parent_id="r")
        c2 = _span("c2", name="leaf", parent_id="c1")
        records = flatten_trace([r, c1, c2])
        depths = {rec.span_name: rec.depth for rec in records}
        assert depths["root"] == 0
        assert depths["mid"] == 1
        assert depths["leaf"] == 2

    def test_multiple_roots(self):
        r1 = _span("r1", name="root1")
        r2 = _span("r2", name="root2")
        records = flatten_trace([r1, r2])
        assert len(records) == 2
        assert all(r.depth == 0 for r in records)

    def test_empty_spans(self):
        records = flatten_trace([])
        assert records == []

    def test_single_span(self):
        s = _span("only", name="single")
        records = flatten_trace([s])
        assert len(records) == 1
        assert records[0].span_name == "single"
        assert records[0].depth == 0

    def test_passes_config(self):
        s = _span("s1", name="test", input="x" * 1000)
        cfg = DenormalizationConfig(max_text_length=10)
        records = flatten_trace([s], config=cfg)
        assert len(records[0].input_text) == 10


# ---------------------------------------------------------------------------
# query_flat_records
# ---------------------------------------------------------------------------


class TestQueryFlatRecords:
    def _records(self):
        return [
            FlattenedRecord(record_id="1", span_type="llm_call", status="ok", duration_ms=50, depth=0),
            FlattenedRecord(record_id="2", span_type="tool_call", status="ok", duration_ms=200, depth=1),
            FlattenedRecord(record_id="3", span_type="llm_call", status="error", duration_ms=300, depth=2),
        ]

    def test_by_span_type(self):
        recs = self._records()
        result = query_flat_records(recs, {"span_type": "llm_call"})
        assert len(result) == 2
        assert all(r.span_type == "llm_call" for r in result)

    def test_by_status(self):
        recs = self._records()
        result = query_flat_records(recs, {"status": "error"})
        assert len(result) == 1
        assert result[0].record_id == "3"

    def test_by_min_duration(self):
        recs = self._records()
        result = query_flat_records(recs, {"min_duration": 100})
        assert len(result) == 2

    def test_by_max_duration(self):
        recs = self._records()
        result = query_flat_records(recs, {"max_duration": 100})
        assert len(result) == 1

    def test_by_min_depth(self):
        recs = self._records()
        result = query_flat_records(recs, {"min_depth": 1})
        assert len(result) == 2

    def test_by_max_depth(self):
        recs = self._records()
        result = query_flat_records(recs, {"max_depth": 0})
        assert len(result) == 1

    def test_combined_filters(self):
        recs = self._records()
        result = query_flat_records(recs, {"span_type": "llm_call", "status": "ok"})
        assert len(result) == 1
        assert result[0].record_id == "1"

    def test_empty_filters(self):
        recs = self._records()
        result = query_flat_records(recs, {})
        assert len(result) == 3

    def test_no_match(self):
        recs = self._records()
        result = query_flat_records(recs, {"span_type": "nonexistent"})
        assert result == []

    def test_by_span_name(self):
        recs = [FlattenedRecord(record_id="a", span_name="generate")]
        result = query_flat_records(recs, {"span_name": "generate"})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# export_flat_csv
# ---------------------------------------------------------------------------


class TestExportFlatCsv:
    def test_valid_csv(self):
        recs = [
            FlattenedRecord(record_id="r1", span_name="test", span_type="llm_call"),
        ]
        csv_str = export_flat_csv(recs)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["record_id"] == "r1"
        assert rows[0]["span_name"] == "test"

    def test_multiple_records(self):
        recs = [
            FlattenedRecord(record_id="r1"),
            FlattenedRecord(record_id="r2"),
        ]
        csv_str = export_flat_csv(recs)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2

    def test_empty_records(self):
        csv_str = export_flat_csv([])
        assert csv_str == ""

    def test_has_header(self):
        recs = [FlattenedRecord(record_id="r1")]
        csv_str = export_flat_csv(recs)
        first_line = csv_str.split("\n")[0]
        assert "record_id" in first_line
        assert "span_type" in first_line


# ---------------------------------------------------------------------------
# build_denormalization_report
# ---------------------------------------------------------------------------


class TestBuildDenormalizationReport:
    def test_basic_report(self):
        spans = [_span("s1"), _span("s2", parent_id="s1")]
        records = flatten_trace(spans)
        report = build_denormalization_report(records, spans)
        assert report.total_spans == 2
        assert report.flattened_records == 2
        assert report.avg_depth == 0.5  # (0 + 1) / 2

    def test_empty_input(self):
        report = build_denormalization_report([], [])
        assert report.total_spans == 0
        assert report.flattened_records == 0
        assert report.avg_depth == 0.0

    def test_format_text(self):
        report = DenormalizationReport(total_spans=10, flattened_records=10, avg_depth=1.5)
        text = report.format_text()
        assert "Total spans: 10" in text
        assert "Avg depth: 1.5" in text


# ---------------------------------------------------------------------------
# Dataclass roundtrips
# ---------------------------------------------------------------------------


class TestDataclassRoundtrips:
    def test_flattened_record_as_dict_from_dict(self):
        rec = FlattenedRecord(record_id="r1", span_name="test", cost=0.1, tokens=50)
        d = rec.as_dict()
        restored = FlattenedRecord.from_dict(d)
        assert restored.record_id == "r1"
        assert restored.span_name == "test"
        assert restored.cost == 0.1
        assert restored.tokens == 50

    def test_denormalization_config_from_dict(self):
        cfg = DenormalizationConfig(include_input=False, max_text_length=100)
        d = cfg.as_dict()
        restored = DenormalizationConfig.from_dict(d)
        assert restored.include_input is False
        assert restored.max_text_length == 100

    def test_denormalization_report_as_dict(self):
        report = DenormalizationReport(total_spans=5, flattened_records=5, avg_depth=2.0)
        d = report.as_dict()
        assert d["total_spans"] == 5
        assert d["avg_depth"] == 2.0
