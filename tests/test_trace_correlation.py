"""Tests for trace correlation with external events."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.trace.correlation import (
    ExternalEvent,
    CorrelationMatch,
    CorrelationReport,
    correlate_by_time,
    find_events_in_window,
    build_correlation_report,
    annotate_trace_with_events,
)


def _now():
    return datetime.now(timezone.utc)


def _event(eid, etype="deployment", ts=None, desc="", meta=None):
    return ExternalEvent(
        event_id=eid,
        event_type=etype,
        timestamp=ts or _now(),
        description=desc,
        metadata=meta or {},
    )


def _trace(tid, ts=None):
    return {"id": tid, "timestamp": ts or _now()}


# -- ExternalEvent as_dict / from_dict --


class TestExternalEventSerialization:
    def test_as_dict_roundtrip(self):
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ev = ExternalEvent(
            event_id="ev1",
            event_type="deployment",
            timestamp=ts,
            description="v2.0 deploy",
            metadata={"version": "2.0"},
        )
        d = ev.as_dict()
        restored = ExternalEvent.from_dict(d)
        assert restored.event_id == "ev1"
        assert restored.event_type == "deployment"
        assert restored.timestamp == ts
        assert restored.description == "v2.0 deploy"
        assert restored.metadata == {"version": "2.0"}

    def test_from_dict_defaults(self):
        d = {
            "event_id": "ev2",
            "event_type": "incident",
            "timestamp": datetime(2026, 3, 1, tzinfo=timezone.utc).isoformat(),
        }
        ev = ExternalEvent.from_dict(d)
        assert ev.description == ""
        assert ev.metadata == {}

    def test_as_dict_fields(self):
        ev = _event("e1", "release", desc="hotfix")
        d = ev.as_dict()
        assert d["event_id"] == "e1"
        assert d["event_type"] == "release"
        assert d["description"] == "hotfix"
        assert "timestamp" in d


# -- CorrelationMatch --


class TestCorrelationMatch:
    def test_as_dict(self):
        m = CorrelationMatch(trace_id="t1", event_id="e1", time_delta_ms=150.0)
        d = m.as_dict()
        assert d["trace_id"] == "t1"
        assert d["event_id"] == "e1"
        assert d["time_delta_ms"] == 150.0
        assert d["correlation_type"] == "temporal"

    def test_custom_correlation_type(self):
        m = CorrelationMatch(
            trace_id="t1", event_id="e1", time_delta_ms=0.0, correlation_type="causal"
        )
        assert m.correlation_type == "causal"
        assert m.as_dict()["correlation_type"] == "causal"


# -- correlate_by_time --


class TestCorrelateByTime:
    def test_matches_close_events(self):
        base = _now()
        traces = [_trace("t1", base)]
        events = [_event("e1", ts=base + timedelta(milliseconds=500))]
        matches = correlate_by_time(traces, events, max_gap_ms=1000.0)
        assert len(matches) == 1
        assert matches[0].trace_id == "t1"
        assert matches[0].event_id == "e1"
        assert matches[0].time_delta_ms == pytest.approx(500.0, abs=1.0)

    def test_rejects_far_events(self):
        base = _now()
        traces = [_trace("t1", base)]
        events = [_event("e1", ts=base + timedelta(seconds=120))]
        matches = correlate_by_time(traces, events, max_gap_ms=60000.0)
        assert len(matches) == 0

    def test_picks_nearest_event(self):
        base = _now()
        traces = [_trace("t1", base)]
        events = [
            _event("e_far", ts=base + timedelta(milliseconds=900)),
            _event("e_near", ts=base + timedelta(milliseconds=100)),
        ]
        matches = correlate_by_time(traces, events, max_gap_ms=1000.0)
        assert len(matches) == 1
        assert matches[0].event_id == "e_near"

    def test_event_before_trace(self):
        base = _now()
        traces = [_trace("t1", base)]
        events = [_event("e1", ts=base - timedelta(milliseconds=200))]
        matches = correlate_by_time(traces, events, max_gap_ms=1000.0)
        assert len(matches) == 1
        assert matches[0].time_delta_ms == pytest.approx(200.0, abs=1.0)

    def test_exact_boundary(self):
        base = _now()
        traces = [_trace("t1", base)]
        events = [_event("e1", ts=base + timedelta(milliseconds=60000))]
        matches = correlate_by_time(traces, events, max_gap_ms=60000.0)
        assert len(matches) == 1

    def test_just_outside_boundary(self):
        base = _now()
        traces = [_trace("t1", base)]
        events = [_event("e1", ts=base + timedelta(milliseconds=60001))]
        matches = correlate_by_time(traces, events, max_gap_ms=60000.0)
        assert len(matches) == 0

    def test_multiple_traces(self):
        base = _now()
        traces = [
            _trace("t1", base),
            _trace("t2", base + timedelta(seconds=1)),
        ]
        events = [_event("e1", ts=base + timedelta(milliseconds=500))]
        matches = correlate_by_time(traces, events, max_gap_ms=1000.0)
        assert len(matches) == 2

    def test_empty_traces(self):
        events = [_event("e1")]
        matches = correlate_by_time([], events)
        assert matches == []

    def test_empty_events(self):
        traces = [_trace("t1")]
        matches = correlate_by_time(traces, [])
        assert matches == []

    def test_both_empty(self):
        matches = correlate_by_time([], [])
        assert matches == []


# -- find_events_in_window --


class TestFindEventsInWindow:
    def test_filters_correctly(self):
        base = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        events = [
            _event("e1", ts=base - timedelta(hours=2)),
            _event("e2", ts=base),
            _event("e3", ts=base + timedelta(hours=1)),
            _event("e4", ts=base + timedelta(hours=3)),
        ]
        result = find_events_in_window(
            events,
            start=base - timedelta(hours=1),
            end=base + timedelta(hours=2),
        )
        assert len(result) == 2
        ids = {e.event_id for e in result}
        assert ids == {"e2", "e3"}

    def test_inclusive_boundaries(self):
        base = datetime(2026, 6, 1, tzinfo=timezone.utc)
        events = [_event("e1", ts=base)]
        result = find_events_in_window(events, start=base, end=base)
        assert len(result) == 1

    def test_empty_events(self):
        base = _now()
        result = find_events_in_window([], base, base + timedelta(hours=1))
        assert result == []

    def test_no_match(self):
        base = datetime(2026, 6, 1, tzinfo=timezone.utc)
        events = [_event("e1", ts=base + timedelta(days=10))]
        result = find_events_in_window(events, start=base, end=base + timedelta(hours=1))
        assert result == []


# -- build_correlation_report --


class TestBuildCorrelationReport:
    def test_counts(self):
        base = _now()
        traces = [
            _trace("t1", base),
            _trace("t2", base + timedelta(minutes=10)),
        ]
        events = [_event("e1", ts=base + timedelta(milliseconds=100))]
        report = build_correlation_report(traces, events, max_gap_ms=1000.0)
        assert report.total_traces == 2
        assert report.total_events == 1
        assert report.correlated_traces == 1
        assert len(report.matches) == 1

    def test_no_correlations(self):
        base = _now()
        traces = [_trace("t1", base)]
        events = [_event("e1", ts=base + timedelta(hours=5))]
        report = build_correlation_report(traces, events, max_gap_ms=1000.0)
        assert report.correlated_traces == 0
        assert report.matches == []

    def test_report_as_dict_from_dict(self):
        report = CorrelationReport(
            matches=[CorrelationMatch("t1", "e1", 50.0)],
            total_traces=3,
            total_events=2,
            correlated_traces=1,
        )
        d = report.as_dict()
        restored = CorrelationReport.from_dict(d)
        assert restored.total_traces == 3
        assert restored.total_events == 2
        assert restored.correlated_traces == 1
        assert len(restored.matches) == 1
        assert restored.matches[0].trace_id == "t1"


# -- CorrelationReport format_text --


class TestCorrelationReportFormatText:
    def test_format_text_output(self):
        report = CorrelationReport(
            matches=[CorrelationMatch("t1", "e1", 123.4)],
            total_traces=5,
            total_events=2,
            correlated_traces=1,
        )
        text = report.format_text()
        assert "1/5" in text
        assert "2 events" in text
        assert "t1" in text
        assert "e1" in text
        assert "123.4" in text

    def test_format_text_empty(self):
        report = CorrelationReport()
        text = report.format_text()
        assert "0/0" in text


# -- annotate_trace_with_events --


class TestAnnotateTraceWithEvents:
    def test_basic_annotation(self):
        events = [
            _event("e1", "deployment", desc="deploy v2"),
            _event("e2", "incident", desc="outage"),
        ]
        matches = [
            CorrelationMatch("t1", "e1", 100.0),
            CorrelationMatch("t1", "e2", 200.0),
            CorrelationMatch("t2", "e1", 50.0),
        ]
        result = annotate_trace_with_events("t1", matches, events)
        assert result["trace_id"] == "t1"
        assert len(result["correlated_events"]) == 2
        event_ids = {e["event_id"] for e in result["correlated_events"]}
        assert event_ids == {"e1", "e2"}

    def test_no_matches_for_trace(self):
        events = [_event("e1")]
        matches = [CorrelationMatch("t2", "e1", 100.0)]
        result = annotate_trace_with_events("t1", matches, events)
        assert result["trace_id"] == "t1"
        assert result["correlated_events"] == []

    def test_event_details_included(self):
        events = [_event("e1", "config_change", desc="toggle feature X")]
        matches = [CorrelationMatch("t1", "e1", 42.0)]
        result = annotate_trace_with_events("t1", matches, events)
        ev = result["correlated_events"][0]
        assert ev["event_type"] == "config_change"
        assert ev["description"] == "toggle feature X"
        assert ev["time_delta_ms"] == 42.0
