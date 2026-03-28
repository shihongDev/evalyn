from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.analysis.density_heatmap import (
    HeatmapCell,
    HeatmapData,
    build_heatmap,
    render_ascii_heatmap,
    render_html_heatmap,
)


def _span(sid, name="s", span_type="custom", status="ok", start_time=None, **attrs):
    t = start_time or datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)  # Monday 14:30
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=None,
        start_time=t,
        status=status,
        attributes=attrs,
    )


def _child_span(sid, parent_id, **kwargs):
    """Non-root span (has parent_id)."""
    s = _span(sid, **kwargs)
    s.parent_id = parent_id
    return s


# --- HeatmapCell tests ---


class TestHeatmapCell:
    def test_as_dict(self):
        c = HeatmapCell(day=0, hour=14, count=5, value=5.0)
        d = c.as_dict()
        assert d == {"day": 0, "hour": 14, "count": 5, "value": 5.0}


# --- HeatmapData tests ---


class TestHeatmapData:
    def test_as_dict_empty(self):
        h = HeatmapData()
        d = h.as_dict()
        assert d["cells"] == []
        assert d["total_traces"] == 0
        assert d["metric"] == "count"

    def test_roundtrip(self):
        h = HeatmapData(
            cells=[HeatmapCell(day=1, hour=10, count=3, value=3.0)],
            total_traces=3,
            metric="count",
        )
        d = h.as_dict()
        h2 = HeatmapData.from_dict(d)
        assert h2.total_traces == 3
        assert len(h2.cells) == 1
        assert h2.cells[0].day == 1
        assert h2.cells[0].hour == 10
        assert h2.cells[0].count == 3
        assert h2.cells[0].value == 3.0
        assert h2.metric == "count"

    def test_from_dict_defaults(self):
        h = HeatmapData.from_dict({})
        assert h.total_traces == 0
        assert h.cells == []
        assert h.metric == "count"


# --- build_heatmap tests ---


class TestBuildHeatmap:
    def test_empty_spans(self):
        data = build_heatmap([])
        assert data.total_traces == 0
        assert data.cells == []

    def test_single_span(self):
        # Monday 14:30 UTC
        s = _span("1")
        data = build_heatmap([s])
        assert data.total_traces == 1
        assert len(data.cells) == 1
        assert data.cells[0].day == 0  # Monday
        assert data.cells[0].hour == 14
        assert data.cells[0].count == 1
        assert data.cells[0].value == 1.0

    def test_multiple_same_day_hour(self):
        t = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        spans = [
            _span("1", start_time=t),
            _span("2", start_time=t + timedelta(minutes=15)),
            _span("3", start_time=t + timedelta(minutes=45)),
        ]
        data = build_heatmap(spans)
        assert data.total_traces == 3
        assert len(data.cells) == 1
        assert data.cells[0].count == 3

    def test_different_days(self):
        mon = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)   # Monday
        wed = datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc)   # Wednesday
        spans = [_span("1", start_time=mon), _span("2", start_time=wed)]
        data = build_heatmap(spans)
        assert data.total_traces == 2
        assert len(data.cells) == 2
        days = {c.day for c in data.cells}
        assert days == {0, 2}  # Monday, Wednesday

    def test_different_hours(self):
        t1 = datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 2, 20, 0, tzinfo=timezone.utc)
        spans = [_span("1", start_time=t1), _span("2", start_time=t2)]
        data = build_heatmap(spans)
        assert len(data.cells) == 2
        hours = {c.hour for c in data.cells}
        assert hours == {8, 20}

    def test_ignores_child_spans(self):
        root = _span("1")
        child = _child_span("2", parent_id="1")
        data = build_heatmap([root, child])
        assert data.total_traces == 1

    def test_count_metric(self):
        t = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        spans = [_span("1", start_time=t), _span("2", start_time=t)]
        data = build_heatmap(spans, metric="count")
        assert data.metric == "count"
        assert data.cells[0].value == 2.0

    def test_cost_metric(self):
        t = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        spans = [
            _span("1", start_time=t, cost=0.05),
            _span("2", start_time=t, cost=0.10),
        ]
        data = build_heatmap(spans, metric="cost")
        assert data.metric == "cost"
        assert abs(data.cells[0].value - 0.15) < 1e-9

    def test_cost_metric_missing_cost(self):
        t = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        spans = [_span("1", start_time=t)]  # no cost attribute
        data = build_heatmap(spans, metric="cost")
        assert data.cells[0].value == 0.0

    def test_error_rate_metric(self):
        t = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        spans = [
            _span("1", start_time=t, status="error"),
            _span("2", start_time=t, status="ok"),
            _span("3", start_time=t, status="ok"),
            _span("4", start_time=t, status="error"),
        ]
        data = build_heatmap(spans, metric="error_rate")
        assert data.metric == "error_rate"
        assert abs(data.cells[0].value - 0.5) < 1e-9

    def test_error_rate_all_ok(self):
        t = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        spans = [_span("1", start_time=t, status="ok")]
        data = build_heatmap(spans, metric="error_rate")
        assert data.cells[0].value == 0.0

    def test_all_seven_days(self):
        # Create one span per day of the week
        base = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)  # Monday
        spans = [
            _span(str(i), start_time=base + timedelta(days=i))
            for i in range(7)
        ]
        data = build_heatmap(spans)
        assert data.total_traces == 7
        assert len(data.cells) == 7
        days = sorted(c.day for c in data.cells)
        assert days == [0, 1, 2, 3, 4, 5, 6]


# --- render tests ---


class TestRenderAscii:
    def test_produces_output(self):
        data = build_heatmap([_span("1")])
        text = render_ascii_heatmap(data)
        assert "Mon" in text
        assert "Sun" in text
        assert "Metric: count" in text

    def test_empty_data(self):
        data = build_heatmap([])
        text = render_ascii_heatmap(data)
        assert "Total traces: 0" in text

    def test_width_parameter(self):
        data = build_heatmap([_span("1")])
        text = render_ascii_heatmap(data, width=60)
        assert isinstance(text, str)

    def test_all_days_present_in_output(self):
        data = build_heatmap([_span("1")])
        text = render_ascii_heatmap(data)
        for label in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            assert label in text


class TestRenderHtml:
    def test_produces_table(self):
        data = build_heatmap([_span("1")])
        html = render_html_heatmap(data)
        assert "<table" in html
        assert "</table>" in html
        assert "Mon" in html

    def test_empty_data(self):
        data = build_heatmap([])
        html = render_html_heatmap(data)
        assert "<table" in html
        assert "Total traces: 0" in html

    def test_contains_metric_info(self):
        data = build_heatmap([_span("1")], metric="cost")
        html = render_html_heatmap(data)
        assert "cost" in html

    def test_all_day_labels(self):
        data = build_heatmap([_span("1")])
        html = render_html_heatmap(data)
        for label in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            assert label in html
