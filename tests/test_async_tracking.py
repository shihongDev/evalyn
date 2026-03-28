from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.async_tracking import (
    ConcurrentCall,
    ConcurrencyGroup,
    ConcurrencyReport,
    detect_concurrent_calls,
    compute_max_concurrency,
    compute_wall_clock,
    analyze_concurrency,
    identify_sequential_bottlenecks,
)


def _span(sid, span_type="llm_call", start_offset_ms=0, duration_ms=100, **attrs):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=start_offset_ms)
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        attributes=attrs,
    )


# -- ConcurrentCall --


class TestConcurrentCall:
    def test_as_dict(self):
        cc = ConcurrentCall(
            call_id="c1", start_ms=0.0, end_ms=100.0, duration_ms=100.0, group_id="g1"
        )
        d = cc.as_dict()
        assert d["call_id"] == "c1"
        assert d["duration_ms"] == 100.0
        assert d["group_id"] == "g1"

    def test_defaults(self):
        cc = ConcurrentCall(call_id="c1", start_ms=0.0, end_ms=50.0, duration_ms=50.0)
        assert cc.group_id == ""


# -- ConcurrencyGroup --


class TestConcurrencyGroup:
    def test_parallelism_ratio_parallel(self):
        g = ConcurrencyGroup(
            group_id="g1",
            total_duration_ms=200.0,
            wall_clock_ms=100.0,
        )
        assert g.parallelism_ratio == 0.5

    def test_parallelism_ratio_sequential(self):
        g = ConcurrencyGroup(
            group_id="g1",
            total_duration_ms=200.0,
            wall_clock_ms=200.0,
        )
        assert g.parallelism_ratio == 1.0

    def test_parallelism_ratio_zero_total(self):
        g = ConcurrencyGroup(group_id="g1")
        assert g.parallelism_ratio == 0.0

    def test_speedup_parallel(self):
        g = ConcurrencyGroup(
            group_id="g1",
            total_duration_ms=200.0,
            wall_clock_ms=100.0,
        )
        assert g.speedup == 2.0

    def test_speedup_sequential(self):
        g = ConcurrencyGroup(
            group_id="g1",
            total_duration_ms=100.0,
            wall_clock_ms=100.0,
        )
        assert g.speedup == 1.0

    def test_speedup_zero_wall(self):
        g = ConcurrencyGroup(group_id="g1")
        assert g.speedup == 0.0

    def test_as_dict_includes_properties(self):
        g = ConcurrencyGroup(
            group_id="g1",
            total_duration_ms=200.0,
            wall_clock_ms=100.0,
        )
        d = g.as_dict()
        assert d["parallelism_ratio"] == 0.5
        assert d["speedup"] == 2.0

    def test_from_dict_roundtrip(self):
        calls = [
            ConcurrentCall(call_id="c1", start_ms=0.0, end_ms=100.0, duration_ms=100.0, group_id="g1"),
        ]
        g = ConcurrencyGroup(
            group_id="g1",
            calls=calls,
            max_concurrent=1,
            total_duration_ms=100.0,
            wall_clock_ms=100.0,
        )
        rebuilt = ConcurrencyGroup.from_dict(g.as_dict())
        assert rebuilt.group_id == "g1"
        assert len(rebuilt.calls) == 1
        assert rebuilt.calls[0].call_id == "c1"
        assert rebuilt.total_duration_ms == 100.0


# -- ConcurrencyReport --


class TestConcurrencyReport:
    def test_as_dict(self):
        report = ConcurrencyReport(total_calls=5, max_concurrent=3, avg_parallelism=0.6)
        d = report.as_dict()
        assert d["total_calls"] == 5
        assert d["max_concurrent"] == 3

    def test_from_dict_roundtrip(self):
        report = ConcurrencyReport(total_calls=2, max_concurrent=2, avg_parallelism=0.5)
        rebuilt = ConcurrencyReport.from_dict(report.as_dict())
        assert rebuilt.total_calls == 2
        assert rebuilt.max_concurrent == 2

    def test_format_text(self):
        g = ConcurrencyGroup(
            group_id="g1",
            calls=[],
            max_concurrent=2,
            total_duration_ms=200.0,
            wall_clock_ms=100.0,
        )
        report = ConcurrencyReport(
            groups=[g], total_calls=4, max_concurrent=2, avg_parallelism=0.5
        )
        text = report.format_text()
        assert "Total calls: 4" in text
        assert "Max concurrent: 2" in text
        assert "Group g1" in text

    def test_format_text_empty(self):
        report = ConcurrencyReport()
        text = report.format_text()
        assert "Total calls: 0" in text


# -- detect_concurrent_calls --


class TestDetectConcurrentCalls:
    def test_overlapping_spans(self):
        # Two spans that overlap: [0, 100] and [50, 150]
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=50, duration_ms=100)
        groups = detect_concurrent_calls([s1, s2])
        assert len(groups) == 1
        assert len(groups[0].calls) == 2

    def test_no_overlap_sequential(self):
        # Two sequential spans: [0, 100] and [100, 200]
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=100, duration_ms=100)
        groups = detect_concurrent_calls([s1, s2])
        assert len(groups) == 0

    def test_three_overlapping(self):
        # [0, 100], [50, 150], [80, 180] - all transitively overlap
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=50, duration_ms=100)
        s3 = _span("c", start_offset_ms=80, duration_ms=100)
        groups = detect_concurrent_calls([s1, s2, s3])
        assert len(groups) == 1
        assert len(groups[0].calls) == 3

    def test_empty_spans(self):
        groups = detect_concurrent_calls([])
        assert groups == []

    def test_single_span(self):
        groups = detect_concurrent_calls([_span("a")])
        assert groups == []

    def test_two_separate_groups(self):
        # Group 1: [0, 100] and [50, 150]
        # Group 2: [500, 600] and [550, 650]
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=50, duration_ms=100)
        s3 = _span("c", start_offset_ms=500, duration_ms=100)
        s4 = _span("d", start_offset_ms=550, duration_ms=100)
        groups = detect_concurrent_calls([s1, s2, s3, s4])
        assert len(groups) == 2


# -- compute_max_concurrency --


class TestComputeMaxConcurrency:
    def test_sequential_is_1(self):
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=100, duration_ms=100)
        assert compute_max_concurrency([s1, s2]) == 1

    def test_parallel_is_2(self):
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=50, duration_ms=100)
        assert compute_max_concurrency([s1, s2]) == 2

    def test_three_parallel(self):
        s1 = _span("a", start_offset_ms=0, duration_ms=200)
        s2 = _span("b", start_offset_ms=50, duration_ms=200)
        s3 = _span("c", start_offset_ms=100, duration_ms=200)
        assert compute_max_concurrency([s1, s2, s3]) == 3

    def test_empty(self):
        assert compute_max_concurrency([]) == 0

    def test_single_span(self):
        assert compute_max_concurrency([_span("a")]) == 1


# -- compute_wall_clock --


class TestComputeWallClock:
    def test_non_overlapping(self):
        calls = [
            ConcurrentCall(call_id="a", start_ms=0, end_ms=100, duration_ms=100),
            ConcurrentCall(call_id="b", start_ms=200, end_ms=300, duration_ms=100),
        ]
        assert compute_wall_clock(calls) == 200.0

    def test_overlapping_merges(self):
        calls = [
            ConcurrentCall(call_id="a", start_ms=0, end_ms=100, duration_ms=100),
            ConcurrentCall(call_id="b", start_ms=50, end_ms=150, duration_ms=100),
        ]
        assert compute_wall_clock(calls) == 150.0

    def test_fully_contained(self):
        calls = [
            ConcurrentCall(call_id="a", start_ms=0, end_ms=200, duration_ms=200),
            ConcurrentCall(call_id="b", start_ms=50, end_ms=100, duration_ms=50),
        ]
        assert compute_wall_clock(calls) == 200.0

    def test_empty(self):
        assert compute_wall_clock([]) == 0.0

    def test_single(self):
        calls = [ConcurrentCall(call_id="a", start_ms=10, end_ms=50, duration_ms=40)]
        assert compute_wall_clock(calls) == 40.0


# -- analyze_concurrency --


class TestAnalyzeConcurrency:
    def test_full_analysis(self):
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=50, duration_ms=100)
        s3 = _span("c", start_offset_ms=300, duration_ms=100)
        report = analyze_concurrency([s1, s2, s3])
        assert report.total_calls == 3
        assert report.max_concurrent == 2
        assert len(report.groups) == 1

    def test_empty(self):
        report = analyze_concurrency([])
        assert report.total_calls == 0
        assert report.max_concurrent == 0

    def test_all_sequential(self):
        s1 = _span("a", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", start_offset_ms=100, duration_ms=100)
        report = analyze_concurrency([s1, s2])
        assert report.max_concurrent == 1
        assert len(report.groups) == 0


# -- identify_sequential_bottlenecks --


class TestIdentifySequentialBottlenecks:
    def test_same_type_sequential(self):
        s1 = _span("a", span_type="llm_call", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", span_type="llm_call", start_offset_ms=100, duration_ms=100)
        ids = identify_sequential_bottlenecks([s1, s2])
        assert "a" in ids
        assert "b" in ids

    def test_different_type_not_flagged(self):
        s1 = _span("a", span_type="llm_call", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", span_type="tool_call", start_offset_ms=100, duration_ms=100)
        ids = identify_sequential_bottlenecks([s1, s2])
        assert ids == []

    def test_overlapping_not_flagged(self):
        s1 = _span("a", span_type="llm_call", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", span_type="llm_call", start_offset_ms=50, duration_ms=100)
        ids = identify_sequential_bottlenecks([s1, s2])
        assert ids == []

    def test_empty(self):
        assert identify_sequential_bottlenecks([]) == []

    def test_single_span(self):
        assert identify_sequential_bottlenecks([_span("a")]) == []

    def test_chain_of_three(self):
        s1 = _span("a", span_type="llm_call", start_offset_ms=0, duration_ms=100)
        s2 = _span("b", span_type="llm_call", start_offset_ms=100, duration_ms=100)
        s3 = _span("c", span_type="llm_call", start_offset_ms=200, duration_ms=100)
        ids = identify_sequential_bottlenecks([s1, s2, s3])
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
