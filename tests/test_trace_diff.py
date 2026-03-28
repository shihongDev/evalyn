"""Tests for trace_diff module - side-by-side trace comparison."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.trace_diff import (
    SpanDiff,
    TraceDiff,
    diff_traces,
    format_diff_ascii,
    format_diff_html,
)


def _now():
    return datetime.now(timezone.utc)


def _span(sid, name="s", span_type="custom", parent_id=None, duration_ms=100, **attrs):
    t = _now()
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        attributes=attrs,
    )


# -- Empty traces --


class TestEmptyTraces:
    def test_both_empty(self):
        result = diff_traces([], [])
        assert result.added_count == 0
        assert result.removed_count == 0
        assert result.changed_count == 0
        assert result.unchanged_count == 0
        assert result.diffs == []

    def test_left_empty_right_has_spans(self):
        right = [_span("r1", name="fetch", span_type="tool_call")]
        result = diff_traces([], right)
        assert result.added_count == 1
        assert result.removed_count == 0
        assert len(result.diffs) == 1
        assert result.diffs[0].status == "added"
        assert result.diffs[0].right_id == "r1"

    def test_right_empty_left_has_spans(self):
        left = [_span("l1", name="fetch", span_type="tool_call")]
        result = diff_traces(left, [])
        assert result.removed_count == 1
        assert result.added_count == 0
        assert len(result.diffs) == 1
        assert result.diffs[0].status == "removed"
        assert result.diffs[0].left_id == "l1"


# -- Identical traces --


class TestIdenticalTraces:
    def test_single_identical_span(self):
        left = [_span("l1", name="generate", span_type="llm_call", duration_ms=100)]
        right = [_span("r1", name="generate", span_type="llm_call", duration_ms=100)]
        result = diff_traces(left, right)
        assert result.unchanged_count == 1
        assert result.changed_count == 0
        assert result.diffs[0].status == "unchanged"

    def test_multiple_identical_spans(self):
        left = [
            _span("l1", name="a", span_type="custom", duration_ms=50),
            _span("l2", name="b", span_type="llm_call", duration_ms=200),
        ]
        right = [
            _span("r1", name="a", span_type="custom", duration_ms=50),
            _span("r2", name="b", span_type="llm_call", duration_ms=200),
        ]
        result = diff_traces(left, right)
        assert result.unchanged_count == 2
        assert result.changed_count == 0

    def test_identical_with_same_output(self):
        left = [_span("l1", name="gen", span_type="llm_call", output="hello world")]
        right = [_span("r1", name="gen", span_type="llm_call", output="hello world")]
        result = diff_traces(left, right)
        assert result.unchanged_count == 1
        assert result.diffs[0].output_diff == ""


# -- Added spans --


class TestAddedSpans:
    def test_one_added(self):
        left = [_span("l1", name="a", span_type="custom")]
        right = [
            _span("r1", name="a", span_type="custom"),
            _span("r2", name="b", span_type="tool_call"),
        ]
        result = diff_traces(left, right)
        assert result.added_count == 1
        added = [d for d in result.diffs if d.status == "added"]
        assert len(added) == 1
        assert added[0].span_name == "b"
        assert added[0].left_id is None

    def test_multiple_added(self):
        right = [
            _span("r1", name="x", span_type="custom"),
            _span("r2", name="y", span_type="llm_call"),
        ]
        result = diff_traces([], right)
        assert result.added_count == 2


# -- Removed spans --


class TestRemovedSpans:
    def test_one_removed(self):
        left = [
            _span("l1", name="a", span_type="custom"),
            _span("l2", name="b", span_type="tool_call"),
        ]
        right = [_span("r1", name="a", span_type="custom")]
        result = diff_traces(left, right)
        assert result.removed_count == 1
        removed = [d for d in result.diffs if d.status == "removed"]
        assert len(removed) == 1
        assert removed[0].span_name == "b"
        assert removed[0].right_id is None

    def test_multiple_removed(self):
        left = [
            _span("l1", name="x", span_type="custom"),
            _span("l2", name="y", span_type="llm_call"),
        ]
        result = diff_traces(left, [])
        assert result.removed_count == 2


# -- Changed spans --


class TestChangedSpans:
    def test_duration_change(self):
        left = [_span("l1", name="gen", span_type="llm_call", duration_ms=100)]
        right = [_span("r1", name="gen", span_type="llm_call", duration_ms=250)]
        result = diff_traces(left, right)
        assert result.changed_count == 1
        d = result.diffs[0]
        assert d.status == "changed"
        assert d.duration_delta_ms == pytest.approx(150.0, abs=1)

    def test_output_change(self):
        left = [_span("l1", name="gen", span_type="llm_call", output="hello")]
        right = [_span("r1", name="gen", span_type="llm_call", output="goodbye")]
        result = diff_traces(left, right)
        assert result.changed_count == 1
        d = result.diffs[0]
        assert d.status == "changed"
        assert "hello" in d.output_diff
        assert "goodbye" in d.output_diff

    def test_cost_change(self):
        left = [_span("l1", name="gen", span_type="llm_call", cost=0.01)]
        right = [_span("r1", name="gen", span_type="llm_call", cost=0.05)]
        result = diff_traces(left, right)
        assert result.changed_count == 1
        d = result.diffs[0]
        assert d.cost_delta == pytest.approx(0.04)

    def test_no_change_zero_cost(self):
        """Both spans have zero cost - should be unchanged."""
        left = [_span("l1", name="gen", span_type="llm_call", duration_ms=100, cost=0)]
        right = [_span("r1", name="gen", span_type="llm_call", duration_ms=100, cost=0)]
        result = diff_traces(left, right)
        assert result.unchanged_count == 1


# -- Mixed scenarios --


class TestMixedScenarios:
    def test_add_remove_change_unchanged(self):
        left = [
            _span("l1", name="a", span_type="custom", duration_ms=100),
            _span("l2", name="b", span_type="tool_call", duration_ms=100),
            _span("l3", name="c", span_type="llm_call", duration_ms=100, output="old"),
        ]
        right = [
            _span("r1", name="a", span_type="custom", duration_ms=100),
            # b removed
            _span("r3", name="c", span_type="llm_call", duration_ms=100, output="new"),
            _span("r4", name="d", span_type="custom", duration_ms=50),  # added
        ]
        result = diff_traces(left, right)
        assert result.unchanged_count == 1
        assert result.removed_count == 1
        assert result.changed_count == 1
        assert result.added_count == 1

    def test_duplicate_keys_first_wins(self):
        """When multiple spans share (name, type), first occurrence wins."""
        left = [
            _span("l1", name="gen", span_type="llm_call", duration_ms=100),
            _span("l2", name="gen", span_type="llm_call", duration_ms=999),
        ]
        right = [_span("r1", name="gen", span_type="llm_call", duration_ms=100)]
        result = diff_traces(left, right)
        # Should match on first occurrence (l1 vs r1) - same duration
        assert result.unchanged_count == 1
        assert result.diffs[0].left_id == "l1"

    def test_same_name_different_type(self):
        """Spans with same name but different type are distinct."""
        left = [_span("l1", name="process", span_type="custom")]
        right = [_span("r1", name="process", span_type="llm_call")]
        result = diff_traces(left, right)
        assert result.removed_count == 1
        assert result.added_count == 1


# -- format_text --


class TestFormatText:
    def test_empty_diff(self):
        td = TraceDiff()
        text = td.format_text()
        assert "0 added" in text
        assert "0 removed" in text

    def test_format_includes_statuses(self):
        left = [
            _span("l1", name="a", span_type="custom"),
            _span("l2", name="b", span_type="tool_call"),
        ]
        right = [
            _span("r1", name="a", span_type="custom"),
            _span("r3", name="c", span_type="llm_call"),
        ]
        result = diff_traces(left, right)
        text = result.format_text()
        assert "[unchanged]" in text or "[removed]" in text
        assert "b" in text
        assert "c" in text

    def test_format_shows_duration_delta(self):
        left = [_span("l1", name="gen", span_type="llm_call", duration_ms=100)]
        right = [_span("r1", name="gen", span_type="llm_call", duration_ms=300)]
        result = diff_traces(left, right)
        text = result.format_text()
        assert "dt=" in text
        assert "ms" in text


# -- format_diff_ascii --


class TestFormatDiffAscii:
    def test_header_present(self):
        td = TraceDiff()
        out = format_diff_ascii(td)
        assert "Status" in out
        assert "Span Name" in out
        assert "Total:" in out

    def test_added_span_shown_with_plus(self):
        td = TraceDiff(
            diffs=[SpanDiff(span_name="new_span", span_type="custom", status="added", right_id="r1")],
            added_count=1,
        )
        out = format_diff_ascii(td)
        assert "[+]" in out
        assert "new_span" in out

    def test_removed_span_shown_with_minus(self):
        td = TraceDiff(
            diffs=[SpanDiff(span_name="old_span", span_type="custom", status="removed", left_id="l1")],
            removed_count=1,
        )
        out = format_diff_ascii(td)
        assert "[-]" in out
        assert "old_span" in out

    def test_changed_span_shown_with_tilde(self):
        td = TraceDiff(
            diffs=[
                SpanDiff(
                    span_name="gen",
                    span_type="llm_call",
                    status="changed",
                    left_id="l1",
                    right_id="r1",
                    duration_delta_ms=50.0,
                )
            ],
            changed_count=1,
        )
        out = format_diff_ascii(td)
        assert "[~]" in out
        assert "+50.0ms" in out


# -- format_diff_html --


class TestFormatDiffHtml:
    def test_html_table_structure(self):
        td = TraceDiff()
        html = format_diff_html(td)
        assert "<table" in html
        assert "</table>" in html
        assert "<th>" in html

    def test_added_row_green(self):
        td = TraceDiff(
            diffs=[SpanDiff(span_name="new", span_type="custom", status="added", right_id="r1")],
            added_count=1,
        )
        html = format_diff_html(td)
        assert "#d4edda" in html
        assert "added" in html

    def test_removed_row_red(self):
        td = TraceDiff(
            diffs=[SpanDiff(span_name="old", span_type="custom", status="removed", left_id="l1")],
            removed_count=1,
        )
        html = format_diff_html(td)
        assert "#f8d7da" in html
        assert "removed" in html

    def test_changed_row_yellow(self):
        td = TraceDiff(
            diffs=[
                SpanDiff(
                    span_name="gen",
                    span_type="llm_call",
                    status="changed",
                    left_id="l1",
                    right_id="r1",
                    cost_delta=0.03,
                )
            ],
            changed_count=1,
        )
        html = format_diff_html(td)
        assert "#fff3cd" in html
        assert "+0.0300" in html


# -- as_dict / from_dict roundtrip --


class TestRoundtrip:
    def test_span_diff_roundtrip(self):
        sd = SpanDiff(
            span_name="gen",
            span_type="llm_call",
            status="changed",
            left_id="l1",
            right_id="r1",
            duration_delta_ms=42.5,
            output_diff="--- left\n+++ right\n",
            cost_delta=0.005,
        )
        restored = SpanDiff.from_dict(sd.as_dict())
        assert restored.span_name == sd.span_name
        assert restored.span_type == sd.span_type
        assert restored.status == sd.status
        assert restored.left_id == sd.left_id
        assert restored.right_id == sd.right_id
        assert restored.duration_delta_ms == sd.duration_delta_ms
        assert restored.output_diff == sd.output_diff
        assert restored.cost_delta == sd.cost_delta

    def test_trace_diff_roundtrip(self):
        td = TraceDiff(
            diffs=[
                SpanDiff(span_name="a", span_type="custom", status="unchanged", left_id="l1", right_id="r1"),
                SpanDiff(span_name="b", span_type="tool_call", status="added", right_id="r2"),
            ],
            added_count=1,
            removed_count=0,
            changed_count=0,
            unchanged_count=1,
        )
        restored = TraceDiff.from_dict(td.as_dict())
        assert restored.added_count == 1
        assert restored.unchanged_count == 1
        assert len(restored.diffs) == 2
        assert restored.diffs[0].span_name == "a"
        assert restored.diffs[1].status == "added"

    def test_trace_diff_roundtrip_empty(self):
        td = TraceDiff()
        restored = TraceDiff.from_dict(td.as_dict())
        assert restored.diffs == []
        assert restored.added_count == 0


# -- Duration and cost deltas --


class TestDeltas:
    def test_duration_delta_positive(self):
        left = [_span("l1", name="gen", span_type="llm_call", duration_ms=100)]
        right = [_span("r1", name="gen", span_type="llm_call", duration_ms=200)]
        result = diff_traces(left, right)
        assert result.diffs[0].duration_delta_ms == pytest.approx(100.0, abs=1)

    def test_duration_delta_negative(self):
        left = [_span("l1", name="gen", span_type="llm_call", duration_ms=300)]
        right = [_span("r1", name="gen", span_type="llm_call", duration_ms=100)]
        result = diff_traces(left, right)
        assert result.diffs[0].duration_delta_ms == pytest.approx(-200.0, abs=1)

    def test_cost_delta_positive(self):
        left = [_span("l1", name="gen", span_type="llm_call", cost=0.01)]
        right = [_span("r1", name="gen", span_type="llm_call", cost=0.10)]
        result = diff_traces(left, right)
        assert result.diffs[0].cost_delta == pytest.approx(0.09)

    def test_cost_delta_negative(self):
        left = [_span("l1", name="gen", span_type="llm_call", cost=0.10)]
        right = [_span("r1", name="gen", span_type="llm_call", cost=0.02)]
        result = diff_traces(left, right)
        assert result.diffs[0].cost_delta == pytest.approx(-0.08)

    def test_cost_delta_none_when_both_zero(self):
        left = [_span("l1", name="gen", span_type="llm_call", duration_ms=100)]
        right = [_span("r1", name="gen", span_type="llm_call", duration_ms=100)]
        result = diff_traces(left, right)
        # No cost attribute at all - cost_delta should be None
        assert result.diffs[0].cost_delta is None

    def test_multiline_output_diff(self):
        left = [_span("l1", name="gen", span_type="llm_call", output="line1\nline2\nline3")]
        right = [_span("r1", name="gen", span_type="llm_call", output="line1\nchanged\nline3")]
        result = diff_traces(left, right)
        assert result.changed_count == 1
        d = result.diffs[0]
        assert "line2" in d.output_diff
        assert "changed" in d.output_diff
